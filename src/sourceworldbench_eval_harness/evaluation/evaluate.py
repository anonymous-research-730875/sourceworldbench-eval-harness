"""uv run python -m sourceworldbench_eval_harness.evaluation.evaluate configs/<task>.yaml

Ground truth is always read from the dataset, never from the prediction file, so a
submission cannot supply its own answer key.
"""

import logging
from pathlib import Path
from typing import Any

import typer

from sourceworldbench_eval_harness import tasks
from sourceworldbench_eval_harness.artifacts import load, write, write_config_beside
from sourceworldbench_eval_harness.config import EvalConfig, load_config
from sourceworldbench_eval_harness.hub import load_token
from sourceworldbench_eval_harness.tasks.core.base import Groups, Task

NESTED_KEYS = ("details",)


def main(config: Path) -> None:
    load_token()
    cfg: EvalConfig = load_config(config)
    task: Task[Any] = tasks.resolve(cfg, config)

    output_dir = Path(cfg.shared.output_dir)
    predictions_path = output_dir / cfg.evaluate.input

    try:
        rows = load(predictions_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Evaluation input file not found: {predictions_path}. Run `predict` first.") from None

    reference = task.reference()
    logging.info("[evaluate] task=%s  %d prediction row(s)", task.name, len(rows))

    task.validate(rows, reference)
    groups = task.score(rows, reference)
    results = {"task": task.name, **Task.flatten(groups)}

    out_path = output_dir / cfg.evaluate.output
    write(out_path, results)
    write_config_beside(out_path, cfg)
    _log(task.name, groups)
    logging.info("[evaluate] -> %s", out_path)


SCORED_SUFFIX = "_scored"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"  # unmeasurable, not zero — e.g. MCC over a single class
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list | tuple):
        return ", ".join(str(v) for v in value)
    return str(value)


def _rows(metrics: dict[str, Any]) -> list[tuple[str, str, str]]:
    """`<metric>_scored` becomes a column beside its metric rather than a row of its
    own: it is that metric's coverage, not a separate result."""
    return [
        (key, _fmt(value), _fmt(scored) if (scored := metrics.get(f"{key}{SCORED_SUFFIX}")) is not None else "")
        for key, value in metrics.items()
        if key not in NESTED_KEYS and not key.endswith(SCORED_SUFFIX)
    ]


def _log(task_name: str, by_group: Groups) -> None:
    """One table per group; nested detail stays in the file."""
    groups = {group: _rows(metrics) for group, metrics in by_group.items()}
    # The run's own header row, alongside the metrics it produced.
    groups[None] = [("task", task_name, "")] + groups.get(None, [])
    # One width across every table, so columns line up down the whole output.
    label_w = max((len(label) for rows in groups.values() for label, _, _ in rows), default=6)
    value_w = max((len(value) for rows in groups.values() for _, value, _ in rows), default=5)
    value_w = max(value_w, len("value"))

    # Grouped tables first, so the ungrouped summary reads as a footer.
    ordered: list[tuple[str | None, list[tuple[str, str, str]]]] = [(g, r) for g, r in groups.items() if g is not None]
    ordered += [(g, r) for g, r in groups.items() if g is None]

    for group, rows in ordered:
        has_scored = any(scored for _, _, scored in rows)
        scored_w = max((len(s) for _, _, s in rows), default=0)
        scored_w = max(scored_w, len("scored")) if has_scored else 0

        heading = group.upper() if group else ("summary" if len(groups) > 1 else "metrics")
        header = f"  {heading:<{label_w}}  {'value':>{value_w}}"
        if has_scored:
            header += f"  {'scored':>{scored_w}}"
        logging.info("")
        logging.info(header)
        logging.info("  " + "─" * (len(header) - 2))
        for label, value, scored in rows:
            line = f"  {label:<{label_w}}  {value:>{value_w}}"
            if has_scored:
                line += f"  {scored:>{scored_w}}"
            logging.info(line)
    logging.info("")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    typer.run(main)
