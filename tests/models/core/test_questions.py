"""The model scaffolding every question task shares.

Parametrized over every *registered* task, read from the registry so a new task is
covered the moment it is registered. Nothing here may assume what an answer looks
like: the shared reader must move one around without inspecting it.
"""

import pytest

from sourceworldbench_eval_harness import models, tasks
from sourceworldbench_eval_harness.models.core.base import PredictionFileError
from sourceworldbench_eval_harness.models.core.questions import QuestionModel, SubmissionReader

TASKS = sorted(models.MODELS)

# One valid answer per task, in that task's own shape.
ANSWER = {
    "single_test_status": "PASSED",
    "test_status_anchored": {"tests": ["t1"], "labels": ["PASSED"]},
    "time_prediction": 12.5,
    "time_hotspot": ["a.py:10:f"],
    "memory_prediction": 1.25e9,
    "memory_hotspot": ["a.py:10:f"],
}


def model(task, name, **params):
    return models.resolve(name, task, params)


def questions_of(task):
    """This task's questions, read off its reader — which answers all of them."""
    return model(task, "from_file").answerable


def two_questions(task):
    """One question to keep and one to drop, whichever this task happens to ask."""
    asked = questions_of(task)
    return asked[-1], asked[0]


def test_every_registered_task_is_covered_here():
    """Guards this file against drifting behind the registry as a task is added."""
    assert set(TASKS) == set(ANSWER)


def test_the_task_and_model_registries_name_the_same_tasks():
    """Two parallel registries, so nothing may be registered in only one of them."""
    assert set(tasks.TASKS) == set(models.MODELS)


@pytest.mark.parametrize("task", TASKS)
def test_every_task_registers_a_reader_and_an_oracle(task):
    """Without `from_file` a task could not score a submission at all; without
    `oracle` there is nothing to prove a new config is wired up correctly."""
    assert {"from_file", "oracle"} <= set(models.MODELS[task])


@pytest.mark.parametrize("task", TASKS)
def test_every_registered_model_is_bound_to_its_own_task(task):
    """A model resolved under a task must answer that task's questions, not another's."""
    for name in models.MODELS[task]:
        instance = model(task, name)
        assert isinstance(instance, QuestionModel)
        assert instance.task == task
        assert instance.answerable == questions_of(task)


@pytest.mark.parametrize("task", TASKS)
def test_every_model_answers_its_own_tasks_questions_by_default(task):
    for name in models.MODELS[task]:
        assert model(task, name).questions == questions_of(task)


@pytest.mark.parametrize("task", TASKS)
def test_questions_can_be_narrowed(task):
    kept, _ = two_questions(task)
    assert model(task, "oracle", questions=[kept]).questions == (kept,)


@pytest.mark.parametrize("task", TASKS)
def test_a_question_this_task_does_not_ask_is_rejected(task):
    """`q9` is asked by no task."""
    with pytest.raises(ValueError, match="unknown question"):
        model(task, "oracle", questions=["q9"])


@pytest.mark.parametrize("task", TASKS)
def test_only_the_submission_reader_declares_it_reads_one(task):
    """The flag the tasks check before running a model over an external file."""
    assert model(task, "from_file").reads_submission is True
    generating = [name for name in models.MODELS[task] if name != "from_file"]
    assert generating and all(not model(task, name).reads_submission for name in generating)


@pytest.mark.parametrize("task", TASKS)
def test_the_reader_moves_an_answer_without_inspecting_it(task):
    kept, _ = two_questions(task)
    rows = [{kept: ANSWER[task]}]
    assert model(task, "from_file", questions=[kept]).predict(rows) == [{kept: ANSWER[task]}]


@pytest.mark.parametrize("task", TASKS)
def test_a_renamed_column_is_read_under_the_canonical_question(task):
    kept, _ = two_questions(task)
    rows = [{"my_answer": ANSWER[task]}]
    reader = model(task, "from_file", questions=[kept], question_fields={kept: "my_answer"})
    assert reader.predict(rows) == [{kept: ANSWER[task]}]


@pytest.mark.parametrize("task", TASKS)
def test_questions_not_kept_never_reach_the_extract(task):
    kept, dropped = two_questions(task)
    # The same answer under both keys: the reader never looks inside one, so its shape
    # does not have to suit the question it is filed under.
    rows = [{dropped: ANSWER[task], kept: ANSWER[task]}]
    assert model(task, "from_file", questions=[kept]).predict(rows) == [{kept: ANSWER[task]}]


@pytest.mark.parametrize("task", TASKS)
def test_the_whole_file_is_rejected_before_any_of_it_is_read(task):
    """A row answering nothing fails the run rather than scoring nothing."""
    kept, _ = two_questions(task)
    rows = [{kept: ANSWER[task]}, {"something_else": ANSWER[task]}]
    with pytest.raises(PredictionFileError, match="1 row\\(s\\) answer none of"):
        model(task, "from_file").predict(rows)


@pytest.mark.parametrize("task", TASKS)
def test_an_unknown_question_field_names_the_valid_questions(task):
    with pytest.raises(ValueError, match="unknown question.*in question_fields"):
        model(task, "from_file", question_fields={"q9": "col"})


@pytest.mark.parametrize("task", TASKS)
def test_the_shared_reader_is_the_one_implementation(task):
    """Every task's `from_file` inherits it, rather than each carrying a copy."""
    assert issubclass(type(model(task, "from_file")), SubmissionReader)
