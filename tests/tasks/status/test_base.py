"""What the status tasks add to the shared question scaffolding: a label vocabulary.

The generic checks live in `tasks/core/test_questions.py`, which is parametrized over every
registered task; this file covers only the status tasks, found by class so a new one
cannot be registered without appearing here.
"""

import pytest

import helpers
from sourceworldbench_eval_harness import tasks as tasks_registry
from sourceworldbench_eval_harness.tasks.status.base import StatusTask

# Per task: a two-instance reference in that task's own shape, and an answer for
# each, so the shared checks can be driven identically against all of them.
CASES = {
    "test_status_anchored": {
        "reference": {"a": {"t1": "PASSED"}, "b": {"u1": "PASSED"}},
        "a": {"tests": ["t1"], "labels": ["PASSED"]},
        "b": {"tests": ["u1"], "labels": ["PASSED"]},
    },
    "single_test_status": {
        "reference": {"a": {"label": "PASSED", "test": "t1"}, "b": {"label": "PASSED", "test": "u1"}},
        "a": "PASSED",
        "b": "PASSED",
    },
}

TASKS = sorted(name for name, cls in tasks_registry.TASKS.items() if issubclass(cls, StatusTask))


def task(name, **settings):
    return helpers.build_task(name, **settings)


def rows_for(name, *instance_ids):
    case = CASES[name]
    return [{"instance_id": i, "outcome": case[i]} for i in instance_ids]


def test_every_registered_status_task_is_covered_here():
    """Guards this file against the drift that hid `test_status_anchored` from it."""
    assert set(TASKS) == set(CASES)


@pytest.mark.parametrize("name", TASKS)
def test_the_questions_come_from_the_tasks_own_schema(name):
    status = task(name)
    assert status.questions == status.schema.questions
    # `all_pass` asks about a set of tests, so it belongs to the multi-test tasks only.
    assert ("all_pass" in status.questions) == (name != "single_test_status")


def test_the_id_always_travels_and_single_test_status_carries_the_test_too():
    assert task("test_status_anchored").carried_fields == ("instance_id",)
    assert task("single_test_status").carried_fields == ("instance_id", "test")


@pytest.mark.parametrize("name", TASKS)
def test_scoring_a_correct_submission_gives_a_perfect_classification(name):
    groups = task(name).score(rows_for(name, "a", "b"), CASES[name]["reference"])
    assert groups["outcome"]["macro_f1"] == 1.0
    assert groups["outcome"]["tests"] == 2
