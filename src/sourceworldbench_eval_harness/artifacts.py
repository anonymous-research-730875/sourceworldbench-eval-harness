import json
from pathlib import Path
from typing import Any

import yaml

from sourceworldbench_eval_harness.config import EvalConfig


def load(path: Path) -> Any:
    """Read whatever is at `path`, picking format from the extension."""
    if path.suffix == "":
        from datasets import Dataset

        ds = Dataset.load_from_disk(str(path))
        return [dict(row) for row in ds]
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == "":
        from datasets import Dataset

        ds = data if isinstance(data, Dataset) else Dataset.from_list(list(data))
        ds.save_to_disk(str(path))
        return

    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for row in data:
                fh.write(json.dumps(row, ensure_ascii=False, default=str))
                fh.write("\n")
        return

    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def write_config_beside(result_path: Path, config: EvalConfig) -> None:
    """Drop the config that produced `result_path` at `<result>.config.yaml`.

    Recorded as that step saw it: task defaults appear only once `tasks.resolve` has
    run, and `load.resolved_sha` only when the step went through `hub.load_split`.
    """
    sidecar = result_path.with_name(result_path.name + ".config.yaml")
    dumped = config.model_dump(exclude_none=True)
    sidecar.write_text(yaml.safe_dump(dumped, sort_keys=False), encoding="utf-8")
