"""Task registry. `task.name` in a config selects one of these, which leaves
`predict` and `evaluate` as drivers that know nothing about questions."""

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sourceworldbench_eval_harness.config import EvalConfig, InvalidConfigError
from sourceworldbench_eval_harness.tasks.core.base import Task
from sourceworldbench_eval_harness.tasks.memory.hotspot import MemoryHotspot
from sourceworldbench_eval_harness.tasks.memory.prediction import MemoryPrediction
from sourceworldbench_eval_harness.tasks.status.anchored import TestStatusAnchored
from sourceworldbench_eval_harness.tasks.status.single_test import SingleTestStatus
from sourceworldbench_eval_harness.tasks.time.hotspot import TimeHotspot
from sourceworldbench_eval_harness.tasks.time.prediction import TimePrediction

TASKS: dict[str, type[Task[Any]]] = {
    "single_test_status": SingleTestStatus,
    "test_status_anchored": TestStatusAnchored,
    "time_prediction": TimePrediction,
    "time_hotspot": TimeHotspot,
    "memory_prediction": MemoryPrediction,
    "memory_hotspot": MemoryHotspot,
}


def resolve(cfg: EvalConfig, config_path: Path | None = None) -> Task[Any]:
    """`load_config` validates the `task` block only as far as every task shares it,
    since the rest is the named task's own schema. This completes the validation and
    installs the result on the config, so the task and the sidecar read the same
    settings with defaults filled in.
    """
    name = cfg.task.name
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; valid names: {sorted(TASKS)}")
    task = TASKS[name]
    try:
        settings = task.Config.model_validate(cfg.task.model_dump(exclude_unset=True))
    except ValidationError as exc:
        raise InvalidConfigError(f"Invalid `task` block for {name!r}:\n{exc}") from exc
    cfg.task = settings
    return task(cfg, settings, config_path)


__all__ = ["TASKS", "resolve"]
