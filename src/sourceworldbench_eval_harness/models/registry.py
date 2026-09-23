"""Model registry, keyed by `task.name` first so a model name means whatever that
task needs it to mean."""

from typing import Any

from sourceworldbench_eval_harness.models.core.base import BasePredictor
from sourceworldbench_eval_harness.models.memory import hotspot as mh
from sourceworldbench_eval_harness.models.memory import prediction as mp
from sourceworldbench_eval_harness.models.status import anchored as tsa
from sourceworldbench_eval_harness.models.status import single_test as sts
from sourceworldbench_eval_harness.models.time import hotspot as th
from sourceworldbench_eval_harness.models.time import prediction as tp

MODELS: dict[str, dict[str, type[BasePredictor]]] = {
    "single_test_status": {
        "all_passed": sts.AllPassed,
        "random_status": sts.RandomStatus,
        "oracle": sts.Oracle,
        "from_file": sts.FromFile,
    },
    "test_status_anchored": {
        "all_passed": tsa.AllPassed,
        "random_status": tsa.RandomStatus,
        "oracle": tsa.Oracle,
        "from_file": tsa.FromFile,
    },
    "time_prediction": {
        "constant_seconds": tp.ConstantSeconds,
        "random_seconds": tp.RandomSeconds,
        "oracle": tp.Oracle,
        "from_file": tp.FromFile,
    },
    "time_hotspot": {
        "most_called": th.MostCalled,
        "random_functions": th.RandomFunctions,
        "oracle": th.Oracle,
        "from_file": th.FromFile,
    },
    "memory_prediction": {
        "constant_bytes": mp.ConstantBytes,
        "random_bytes": mp.RandomBytes,
        "oracle": mp.Oracle,
        "from_file": mp.FromFile,
    },
    "memory_hotspot": {
        "most_called": mh.MostCalled,
        "random_functions": mh.RandomFunctions,
        "oracle": mh.Oracle,
        "from_file": mh.FromFile,
    },
}


def resolve(name: str, task: str, params: dict[str, Any] | None = None) -> BasePredictor:
    """Naming another task's model is caught here rather than downstream as a
    prediction nothing can read."""
    models = MODELS.get(task)
    if models is None:
        raise ValueError(f"no models registered for task {task!r}; tasks: {sorted(MODELS)}")
    if name not in models:
        elsewhere = sorted(t for t, m in MODELS.items() if name in m)
        whose = f"; {name!r} is a {' / '.join(elsewhere)} model" if elsewhere else ""
        raise ValueError(f"unknown model {name!r} for {task}{whose}; valid: {sorted(models)}")
    model = models[name](**(params or {}))
    model.name = name
    return model


__all__ = ["MODELS", "resolve"]
