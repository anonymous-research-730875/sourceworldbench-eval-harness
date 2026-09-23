"""Scaffolding every task's test file needs: a valid config, a resolved task, and a dataset
double standing in for the Hub.

One copy rather than six, so a new required config block is one edit rather than six.

Importable from anywhere under `tests/` because `pythonpath` puts this directory on
`sys.path`; a `conftest.py` would not be, the suite running under `--import-mode=importlib`.
"""

import pytest

from sourceworldbench_eval_harness.config import EvalConfig
from sourceworldbench_eval_harness.tasks import resolve
from sourceworldbench_eval_harness.tasks.core.base import InvalidSubmissionError


def eval_config(name, **task_settings):
    """A minimal valid config whose `task` block carries `task_settings`.

    The `leaderboard` block is always present: it is optional to the schema, and a
    file that omitted it was not testing that, only failing to mention it.
    """
    return EvalConfig.model_validate(
        {
            "shared": {"output_root": "outputs/test"},
            "task": {"name": name, **task_settings},
            "load": {"name": "owner/dataset"},
            "predict": {"output": "predictions.jsonl"},
            "evaluate": {"input": "predictions.jsonl", "output": "metrics.json"},
            "leaderboard": {"dataset": "owner/dataset", "model": "baseline"},
        }
    )


def build_task(name, require_full_coverage=True, model=None, **settings):
    """A task with no dataset behind it — `reference` is supplied directly.

    Built through `resolve` so the settings are validated and defaulted exactly as a
    real run would have them. `model` names `predict.model`, which only the generating
    path needs: an external file defaults to `from_file`.
    """
    task = resolve(eval_config(name, require_full_coverage=require_full_coverage, **settings))
    if model is not None:
        task.cfg.predict.model = model
    return task


def row(instance_id, **questions):
    """One prediction row: `row(PASSED_ID, outcome="PASSED")`."""
    return {"instance_id": instance_id, **questions}


class FakeDataset(list):
    """A stand-in for a HF Dataset: rows that also name their columns.

    `_dataset_rows` checks `column_names` when present, so this exercises that path; passing
    `columns` explicitly takes a column away without taking away the rows' key.
    """

    def __init__(self, rows, columns=None):
        super().__init__(rows)
        self.column_names = columns if columns is not None else sorted({key for r in rows for key in r})


def attach_dataset(task, rows):
    """Point a task at in-memory rows instead of the Hub."""
    task.load_dataset_rows = lambda: rows  # type: ignore[method-assign]
    return task


def expect_duplicate_instance_rejected(task, rows):
    """A dataset repeating an instance id must fail the run rather than silently overwrite.

    The check belongs to `QuestionTask.once_per_instance`, so every task is asked the same
    question here rather than each file asserting its own version.
    """
    repeated = FakeDataset([*rows, dict(rows[0])], rows.column_names)
    with pytest.raises(InvalidSubmissionError, match="duplicate instance_id"):
        attach_dataset(task, repeated).reference()


def expect_missing_column_named(task, rows, column):
    """A column the dataset lacks must be named up front by `_dataset_rows`, rather than
    surfacing as a `KeyError` on an opaque row once scoring is already under way. Which
    column is load-bearing is each task's own, so the caller names it."""
    without = FakeDataset([{key: value for key, value in row.items() if key != column} for row in rows])
    with pytest.raises(ValueError, match="has no column"):
        attach_dataset(task, without).reference()
