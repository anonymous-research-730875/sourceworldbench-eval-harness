"""What the status models add to the shared question scaffolding: a label vocabulary.

The generic checks live in `models/core/test_questions.py`, which is parametrized over every
registered task; this file covers only the status models, found by class so a new one
cannot be registered without appearing here.
"""

import pytest

from sourceworldbench_eval_harness import models
from sourceworldbench_eval_harness.models.status.base import StatusModel
from sourceworldbench_eval_harness.schemas.status import SINGLE_TEST_STATUS, TEST_STATUS_ANCHORED

# Written out rather than read off the models, so the assertion has something
# independent to compare against.
SCHEMAS = {
    "test_status_anchored": TEST_STATUS_ANCHORED,
    "single_test_status": SINGLE_TEST_STATUS,
}

TASKS = sorted(
    task
    for task, registered in models.MODELS.items()
    if all(issubclass(cls, StatusModel) for cls in registered.values())
)


def model(task, name, **params):
    return models.resolve(name, task, params)


def test_every_registered_status_task_is_covered_here():
    """Guards this file against the drift that hid `test_status_anchored` from it."""
    assert set(TASKS) == set(SCHEMAS)


@pytest.mark.parametrize("task", TASKS)
def test_every_task_registers_the_same_four_models(task):
    assert set(models.MODELS[task]) == {"all_passed", "random_status", "oracle", "from_file"}


@pytest.mark.parametrize("task", TASKS)
def test_every_registered_model_carries_its_own_tasks_vocabulary(task):
    """A model resolved under a task must carry that task's schema, not the other's."""
    for name in models.MODELS[task]:
        instance = model(task, name)
        assert isinstance(instance, StatusModel)
        assert instance.schema is SCHEMAS[task]
        assert instance.questions == SCHEMAS[task].questions


def test_all_pass_is_a_multi_test_question_only():
    assert model("test_status_anchored", "all_passed", questions=["all_pass"]).questions == ("all_pass",)
    with pytest.raises(ValueError, match="unknown question"):
        model("single_test_status", "all_passed", questions=["all_pass"])
