"""uv run python -m sourceworldbench_eval_harness.prediction.predict configs/<task>.yaml

Rows come from `predict.model` or from `predict.external_input`. What an external
file *means* is the task's call, not this driver's.
"""

import logging
from pathlib import Path
from typing import Any

import typer

from sourceworldbench_eval_harness.artifacts import load, write, write_config_beside
from sourceworldbench_eval_harness.config import EvalConfig, load_config
from sourceworldbench_eval_harness.hub import load_token
from sourceworldbench_eval_harness.models.core.base import PredictionFileError
from sourceworldbench_eval_harness.tasks import resolve
from sourceworldbench_eval_harness.tasks.core.base import Task


def _external_rows(task: Task[Any]) -> list[dict[str, Any]] | None:
    if (source := task.external_path) is None:
        return None
    try:
        rows = load(source)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration specifies external input for predictions, but "
            f"input file with predictions is not found: {source}."
        )
    if not isinstance(rows, list):
        raise PredictionFileError(f"External input {source} must be a JSON list of objects; got {type(rows).__name__}.")
    offender = next((index for index, row in enumerate(rows) if not isinstance(row, dict)), None)
    if offender is not None:
        raise PredictionFileError(
            f"External input {source} must be a JSON list of objects; "
            f"row {offender} is {type(rows[offender]).__name__}."
        )
    logging.info("[predict] external input %s (%d row(s))", source, len(rows))
    return rows


def main(config: Path) -> None:
    load_token()
    cfg: EvalConfig = load_config(config)
    task: Task[Any] = resolve(cfg, config)

    rows = task.predict(_external_rows(task))
    task.validate_shape(rows)

    out = Path(cfg.shared.output_dir) / cfg.predict.output
    write(out, rows)
    task.write_extras(out, rows)
    write_config_beside(out, cfg)
    logging.info("[predict] task=%s, %d row(s) -> %s", task.name, len(rows), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    typer.run(main)
