"""python -m sourceworldbench_eval_harness.leaderboard.push <config.yaml>

Uploads the local metrics file to the HF results dataset, at
`submissions/<slug>/<model>__<stamp>.json`. A known dataset version makes the
grouping key `<slug>@<version>`, so each version gets its own leaderboard.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
import yaml
from huggingface_hub import HfApi, whoami

from sourceworldbench_eval_harness.config import EvalConfig, InvalidConfigError, LeaderboardConfig, load_config
from sourceworldbench_eval_harness.hub import require_token, resolve_sha
from sourceworldbench_eval_harness.leaderboard import DEFAULT_RESULTS_REPO


def _metrics_path(cfg: EvalConfig, sub: LeaderboardConfig) -> Path:
    """`leaderboard.input`, falling back to `evaluate.output`."""
    output_dir = Path(cfg.shared.output_dir)
    return output_dir / (sub.input or cfg.evaluate.output)


def _build_record(metrics: dict[str, Any], config: EvalConfig, sub: LeaderboardConfig) -> dict[str, Any]:
    slug = sub.dataset.rsplit("/", 1)[-1]
    version = sub.revision
    return {
        "dataset": f"{slug}@{version}" if version else slug,
        "dataset_version": version,
        "dataset_sha": resolve_sha(config.load.name, version),
        "model": sub.model,
        "metrics": metrics,
        "submitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": yaml.safe_dump(config.model_dump(exclude_none=True), sort_keys=False),
    }


def main(config: Path) -> None:
    require_token("push results to the leaderboard")

    cfg = load_config(config)
    sub = cfg.leaderboard
    if sub is None:
        raise InvalidConfigError(
            f"{config} has no `leaderboard` block, so there is nothing to submit. Add one naming the "
            "dataset the run scored and the model to show on the leaderboard:\n\n"
            "leaderboard:\n"
            "  dataset: anonymous-research-730875/sourceworldbench-test-status-anchored\n"
            "  model: my-model\n"
        )
    repo = sub.repo or DEFAULT_RESULTS_REPO

    metrics_path = _metrics_path(cfg, sub)
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics not found at {metrics_path} — run `evaluate` first")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    record = _build_record(metrics, cfg, sub)

    api = HfApi()
    try:
        record["submitter"] = sub.submitter or whoami(token=api.token)["name"]
    except Exception:
        record["submitter"] = sub.submitter or "anonymous"

    stamp = record["submitted_at"].replace(":", "").replace("-", "")
    remote_path = f"submissions/{record['dataset']}/{record['model']}__{stamp}.json"
    api.upload_file(
        path_or_fileobj=json.dumps(record, indent=2).encode("utf-8"),
        path_in_repo=remote_path,
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"add {record['dataset']}/{record['model']} submission",
    )
    logging.info("[push] %s -> %s/%s", metrics_path, repo, remote_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    typer.run(main)
