"""The task scaffolding every question task shares.

Parametrized over every registered task: the id, duplicate, coverage and no-answer
messages must read the same whichever task a file was submitted to. Nothing here may
assume what an answer looks like — that is each task's own test file.
"""

import pytest

import helpers
from sourceworldbench_eval_harness import tasks as tasks_registry
from sourceworldbench_eval_harness.config import InvalidConfigError
from sourceworldbench_eval_harness.tasks.core.base import InvalidSubmissionError
from sourceworldbench_eval_harness.tasks.core.questions import QuestionTask

# Per task: one question it asks, a two-instance reference in that task's own shape,
# and an answer for each, so the shared checks can be driven identically against all
# of them.
CASES = {
    "single_test_status": {
        "question": "outcome",
        "typo_setting": {"lable_field": "label"},
        "reference": {"a": {"label": "PASSED", "test": "t1"}, "b": {"label": "PASSED", "test": "u1"}},
        "answers": {"a": "PASSED", "b": "FAILED"},
    },
    "test_status_anchored": {
        "question": "outcome",
        "typo_setting": {"ks": [1, 5]},
        "reference": {"a": {"t1": "PASSED"}, "b": {"u1": "PASSED"}},
        "answers": {
            "a": {"tests": ["t1"], "labels": ["PASSED"]},
            "b": {"tests": ["u1"], "labels": ["FAILED"]},
        },
    },
    "time_prediction": {
        "question": "seconds",
        "typo_setting": {"id_fields": "instance_id"},
        "reference": {"a": {"seconds": 4.0}, "b": {"seconds": 40.0}},
        "answers": {"a": 4.0, "b": 400.0},
    },
    "time_hotspot": {
        "question": "exclusive_top1",
        "typo_setting": {"hotspots_field": "hotspots"},
        "reference": {
            "a": {"index": {"f1": "f1", "f2": "f2"}, "ranked": ["f1"], "weights": {"f1": 9.0, "f2": 1.0}},
            "b": {"index": {"g1": "g1"}, "ranked": ["g1"], "weights": {"g1": 5.0}},
        },
        "answers": {"a": ["f1"], "b": ["g1"]},
    },
    "memory_prediction": {
        "question": "bytes",
        "typo_setting": {"id_fields": "instance_id"},
        "reference": {"a": {"bytes": 4.0e6}, "b": {"bytes": 4.0e7}},
        "answers": {"a": 4.0e6, "b": 4.0e8},
    },
    "memory_hotspot": {
        "question": "cumulative_top1",
        "typo_setting": {"hotspots_field": "hotspots_cumulative"},
        "reference": {
            "a": {
                "index": {"f1": "f1", "f2": "f2"},
                "axes": {"cumulative": {"ranked": ["f2"], "weights": {"f1": 1.0, "f2": 9.0}}},
            },
            "b": {
                "index": {"g1": "g1"},
                "axes": {"cumulative": {"ranked": ["g1"], "weights": {"g1": 5.0}}},
            },
        },
        "answers": {"a": ["f2"], "b": ["g1"]},
    },
}

TASKS = sorted(CASES)


def task(name, model=None, **settings):
    return helpers.build_task(name, model=model, **settings)


def rows_for(name, *instance_ids):
    case = CASES[name]
    return [{"instance_id": i, case["question"]: case["answers"][i]} for i in instance_ids]


@pytest.mark.parametrize("name", TASKS)
def test_every_task_is_a_question_task_with_every_hook_filled_in(name):
    """An unimplemented hook would make the class abstract and `resolve` would fail."""
    instance = task(name)
    assert isinstance(instance, QuestionTask)
    for hook in ("check_shape", "cross_check", "score_question", "generated_rows", "reference"):
        assert not getattr(getattr(type(instance), hook), "__isabstractmethod__", False)


def test_every_registered_task_is_covered_here():
    """Guards this file against drifting behind the registry as a task is added."""
    assert set(TASKS) == set(tasks_registry.TASKS)


@pytest.mark.parametrize("name", TASKS)
def test_an_unknown_task_setting_is_rejected(name):
    """A near-miss of a real setting name, which is the mistake `extra="forbid"` exists
    for: accepted silently it would run with the default and read as if it had not."""
    typo = CASES[name]["typo_setting"]
    with pytest.raises(InvalidConfigError, match=next(iter(typo))):
        task(name, **typo)


@pytest.mark.parametrize("name", TASKS)
def test_an_unknown_model_names_the_valid_options(name):
    with pytest.raises(ValueError, match=f"unknown model 'nope' for {name}"):
        task(name, model="nope").configured_model()


@pytest.mark.parametrize("name", TASKS)
def test_a_generating_model_paired_with_an_external_file_is_refused(name):
    """Scoring a file with a model that builds its own answers would silently ignore the
    file. `oracle` is the one generating model every task registers."""
    with pytest.raises(ValueError, match="would ignore predict.external_input"):
        task(name, model="oracle").predict(rows_for(name, "a"))


@pytest.mark.parametrize("name", TASKS)
def test_every_question_is_usable_as_a_metric_prefix(name):
    """Metrics are reported as `{question}_{metric}` and the leaderboard splits them
    back apart on that underscore, so a name that is empty, repeated, or not a plain
    lowercase identifier would not survive the round trip."""
    questions = task(name).questions
    assert questions
    assert len(set(questions)) == len(questions)
    assert all(q.isidentifier() and q.islower() for q in questions)


@pytest.mark.parametrize("name", TASKS)
def test_the_id_always_travels(name):
    assert task(name).carried_fields[0] == "instance_id"


@pytest.mark.parametrize("name", TASKS)
def test_a_complete_submission_passes(name):
    instance = task(name)
    rows = rows_for(name, "a", "b")
    instance.validate_shape(rows)
    instance.validate(rows, CASES[name]["reference"])


@pytest.mark.parametrize("name", TASKS)
def test_a_row_without_an_id_is_named_by_its_index(name):
    case = CASES[name]
    with pytest.raises(InvalidSubmissionError, match="row 0: missing 'instance_id'"):
        task(name).validate([{case["question"]: case["answers"]["a"]}], case["reference"])


@pytest.mark.parametrize("name", TASKS)
def test_a_duplicate_row_is_rejected(name):
    rows = rows_for(name, "a", "a")
    with pytest.raises(InvalidSubmissionError, match="a: duplicate row for this instance"):
        task(name, require_full_coverage=False).validate(rows, CASES[name]["reference"])


@pytest.mark.parametrize("name", TASKS)
def test_an_instance_not_in_the_dataset_is_rejected(name):
    case = CASES[name]
    rows = [{"instance_id": "ghost", case["question"]: case["answers"]["a"]}]
    with pytest.raises(InvalidSubmissionError, match="ghost: not present in the dataset"):
        task(name, require_full_coverage=False).validate(rows, case["reference"])


@pytest.mark.parametrize("name", TASKS)
def test_a_row_answering_nothing_is_rejected(name):
    with pytest.raises(InvalidSubmissionError, match="a: no question answers"):
        task(name, require_full_coverage=False).validate([{"instance_id": "a"}], CASES[name]["reference"])


@pytest.mark.parametrize("name", TASKS)
def test_partial_coverage_is_rejected_and_can_be_allowed(name):
    case = CASES[name]
    rows = rows_for(name, "a")
    expected = rf"{case['question']}: 1 dataset instance\(s\) have no prediction, e.g. \['b'\]"
    with pytest.raises(InvalidSubmissionError, match=expected):
        task(name).validate(rows, case["reference"])
    task(name, require_full_coverage=False).validate(rows, case["reference"])


@pytest.mark.parametrize("name", TASKS)
def test_scoring_groups_by_question_and_counts_instances(name):
    case = CASES[name]
    groups = task(name).score(rows_for(name, "a", "b"), case["reference"])
    assert groups[None]["instances"] == 2
    assert groups[None]["questions_answered"] == [case["question"]]
    unanswered = [q for q in task(name).questions if q != case["question"]]
    assert all(q not in groups for q in unanswered)  # unanswered, so unscored


def test_flat_classification_keeps_tables_out_of_the_columns():
    pooled = {
        "macro_f1": 0.5,
        "macro_precision": 0.5,
        "macro_recall": 0.5,
        "balanced_accuracy": 0.5,
        "accuracy": 0.5,
        "per_class": {"PASSED": {"f1": 1.0}, "FAILED": {"f1": 0.0}},
        "confusion": {"PASSED": {"PASSED": 1}},
    }
    flat, details = QuestionTask.flat_classification(pooled, count=7)

    assert flat["tests"] == 7
    assert flat["passed_f1"] == 1.0 and flat["failed_f1"] == 0.0
    assert "mcc" not in flat  # absent from the report, so absent from the columns
    assert set(details) == {"per_class", "confusion"}
    assert all(not isinstance(value, dict) for value in flat.values())


def test_mcc_is_carried_only_when_the_report_defines_it():
    pooled = {
        "macro_f1": 1.0,
        "macro_precision": 1.0,
        "macro_recall": 1.0,
        "balanced_accuracy": 1.0,
        "accuracy": 1.0,
        "mcc": 0.75,
        "per_class": {},
        "confusion": {},
    }
    flat, _ = QuestionTask.flat_classification(pooled, count=1)
    assert flat["mcc"] == 0.75
    # A question asked about instances rather than tests names its own unit.
    instance_flat, _ = QuestionTask.flat_classification(pooled, count=304, count_key="instances")
    assert instance_flat["instances"] == 304 and "tests" not in instance_flat
    # Ordering is what `evaluate`'s tables and the metrics file key order follow.
    assert list(flat) == [
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "accuracy",
        "tests",
        "mcc",
    ]
