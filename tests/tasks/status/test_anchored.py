"""Anchored test status: `all_pass`, and the three-status vocabulary. What it shares with
`test_status` is covered by `tasks/status/test_base.py`."""

import pytest

import helpers
from helpers import FakeDataset
from sourceworldbench_eval_harness.config import InvalidConfigError
from sourceworldbench_eval_harness.tasks import resolve
from sourceworldbench_eval_harness.tasks.core.base import InvalidSubmissionError, Task
from sourceworldbench_eval_harness.tasks.status.anchored import TestStatusAnchored

# Two instances: one whose patch broke something, one still clean — the split `all_pass`
# is asked about. Every test here passed on the base state, by construction.
REFERENCE = {
    "pandas__aaaaaaaa__pred_x": {"t1": "PASSED", "t2": "FAILED", "t3": "PASSED", "t4": "ERROR"},
    "pandas__bbbbbbbb__pred_y": {"u1": "PASSED", "u2": "PASSED"},
}

BROKEN, CLEAN = REFERENCE


def eval_config(name="test_status_anchored", **task_settings):
    return helpers.eval_config(name, **task_settings)


def task(require_full_coverage=True, model=None, **settings):
    return helpers.build_task(
        "test_status_anchored", require_full_coverage=require_full_coverage, model=model, **settings
    )


def flat_score(rows, **settings):
    return Task.flatten(task(**settings).score(rows, REFERENCE))


def labeled(tests, labels):
    return {"tests": tests, "labels": labels}


def oracle_rows():
    """A perfect submission: every question, every instance, from `REFERENCE`."""
    rows = []
    for instance_id, truth in REFERENCE.items():
        tests = list(truth)
        labels = [truth[t] for t in tests]
        rows.append(
            {
                "instance_id": instance_id,
                "binary": labeled(tests, ["PASSED" if truth[t] == "PASSED" else "NOT_PASSED" for t in tests]),
                "outcome": labeled(tests, labels),
                "all_pass": all(label == "PASSED" for label in labels),
            }
        )
    return rows


def dataset_rows():
    return FakeDataset(
        [
            {
                "instance_id": instance_id,
                "tests": list(truth),
                "labels": [truth[t] for t in truth],
                "repo": "pandas-dev/pandas",
                "patch": "--- a/x\n+++ b/x\n",
            }
            for instance_id, truth in REFERENCE.items()
        ]
    )


def with_dataset(task_, rows=None):
    return helpers.attach_dataset(task_, dataset_rows() if rows is None else rows)


def test_config_selects_the_task():
    assert isinstance(resolve(eval_config()), TestStatusAnchored)


def test_a_duplicate_instance_in_the_dataset_is_rejected():
    """The other task files have always asserted this; this one did not, and the guard
    it rests on is now shared, so all four ask for it."""
    helpers.expect_duplicate_instance_rejected(task(), dataset_rows())


def test_it_asks_four_questions_including_all_pass():
    assert task().questions == ("binary", "outcome", "all_pass")


def test_it_scores_the_three_status_vocabulary():
    """SKIPPED and OTHER are filtered out of this dataset, as in the single-test task."""
    assert task().schema.labels == ("PASSED", "FAILED", "ERROR")
    assert task().schema.label_vocabulary["outcome"] == ("PASSED", "FAILED", "ERROR")


def test_there_is_one_test_column_and_nothing_to_configure():
    """The dataset ships a single test list, so the columns are not a config choice."""
    assert task().test_columns == ("tests", "labels")
    with pytest.raises(InvalidConfigError, match="tests_field"):
        resolve(eval_config(tests_field="tests"))


def test_all_pass_is_not_cross_referenced_against_test_names():
    assert task().per_test_questions == ("binary", "outcome")


def test_a_correct_answer_on_both_instances_scores_one():
    rows = [{"instance_id": BROKEN, "all_pass": False}, {"instance_id": CLEAN, "all_pass": True}]
    results = flat_score(rows)

    assert results["all_pass_macro_f1"] == 1.0
    assert results["all_pass_mcc"] == 1.0
    assert results["all_pass_instances"] == 2
    assert "all_pass_tests" not in results  # the unit is the instance, and says so


def test_an_always_yes_answer_finds_no_regression():
    """The constant "the patch broke nothing" answer: the floor `all_pass` is read against."""
    rows = [{"instance_id": i, "all_pass": True} for i in REFERENCE]
    results = flat_score(rows)

    assert results["all_pass_accuracy"] == 0.5
    assert results["all_pass_passed_f1"] == pytest.approx(2 / 3)
    assert results["all_pass_not_passed_f1"] == 0.0  # missed the only broken instance
    assert results["all_pass_mcc"] is None  # a single predicted class has no correlation


def test_all_pass_is_scored_with_non_passed_as_the_positive_class():
    """`all_pass_not_passed_f1` is the number that matters: spotting a broken instance."""
    rows = [{"instance_id": BROKEN, "all_pass": False}, {"instance_id": CLEAN, "all_pass": False}]
    results = flat_score(rows)

    assert results["all_pass_not_passed_f1"] == pytest.approx(2 / 3)  # caught it, plus a false alarm
    assert results["all_pass_passed_f1"] == 0.0


def test_a_non_boolean_answer_is_rejected_without_the_dataset():
    for bad in ("true", 1, None, {"tests": ["t1"]}):
        with pytest.raises(InvalidSubmissionError, match="must be true or false"):
            task().validate_shape([{"instance_id": BROKEN, "all_pass": bad}])


def test_all_pass_needs_no_test_names_to_validate():
    """`all_pass` has nothing to cross-reference, so true or false alone is a complete answer."""
    rows = [{"instance_id": i, "all_pass": True} for i in REFERENCE]
    task().validate(rows, REFERENCE)


def test_a_missing_all_pass_is_still_caught_by_the_coverage_check():
    rows = [{"instance_id": BROKEN, "all_pass": False}]
    with pytest.raises(InvalidSubmissionError, match=r"all_pass: 1 dataset instance\(s\) have no prediction"):
        task().validate(rows, REFERENCE)


def test_the_oracle_scores_perfectly_on_every_question_in_one_run():
    results = flat_score(oracle_rows())
    assert results["binary_macro_f1"] == 1.0
    assert results["outcome_macro_f1"] == 1.0
    assert results["all_pass_macro_f1"] == 1.0
    assert results["questions_answered"] == ["binary", "outcome", "all_pass"]


def test_outcome_has_three_class_columns_and_no_mcc():
    results = flat_score(oracle_rows())
    assert {"outcome_passed_f1", "outcome_failed_f1", "outcome_error_f1"} <= set(results)
    assert not any(key.startswith("outcome_skipped") or key.startswith("outcome_other") for key in results)
    assert "outcome_mcc" not in results  # defined for two classes only
    assert results["binary_mcc"] == 1.0


def test_outcome_rejects_a_status_this_dataset_filters_out():
    """SKIPPED belongs to `test_status`, so it means the wrong dataset or prompt."""
    rows = [{"instance_id": BROKEN, "outcome": labeled(["t1"], ["SKIPPED"])}]
    with pytest.raises(InvalidSubmissionError, match="unknown label"):
        task().validate_shape(rows)


def test_partial_coverage_within_an_instance_is_rejected():
    rows = [{"instance_id": BROKEN, "outcome": labeled(["t1", "t2"], ["PASSED", "FAILED"])}]
    with pytest.raises(InvalidSubmissionError, match="covers 2 of 4"):
        task(require_full_coverage=False).validate(rows, REFERENCE)


def test_a_baseline_answers_every_question_including_all_pass():
    rows = with_dataset(task(model="all_passed")).predict(None)

    assert len(rows) == 2
    assert {"instance_id", "binary", "outcome", "all_pass"} == set(rows[0])
    assert rows[0]["all_pass"] is True  # all_passed claims nothing broke, on both instances
    assert rows[0]["outcome"]["labels"] == ["PASSED"] * 4


def test_the_oracle_all_pass_answer_follows_its_labels():
    rows = with_dataset(task(model="oracle")).predict(None)
    by_id = {row["instance_id"]: row for row in rows}

    assert by_id[BROKEN]["all_pass"] is False
    assert by_id[CLEAN]["all_pass"] is True


def test_the_oracle_round_trips_through_validation_and_scoring():
    status = with_dataset(task(model="oracle"))

    rows = status.predict(None)
    reference = status.reference()
    status.validate_shape(rows)
    status.validate(rows, reference)
    results = Task.flatten(status.score(rows, reference))

    assert results["outcome_macro_f1"] == 1.0
    assert results["all_pass_macro_f1"] == 1.0


def test_reference_is_keyed_by_test_name():
    assert with_dataset(task()).reference() == REFERENCE


def test_a_dataset_label_outside_the_vocabulary_is_rejected_before_scoring():
    """`multiclass_confusion` indexes by the true label, so this must not reach it."""
    rows = dataset_rows()
    rows[0]["labels"] = ["SKIPPED"] + rows[0]["labels"][1:]

    with pytest.raises(InvalidSubmissionError, match=r"outside \['PASSED', 'FAILED', 'ERROR'\]"):
        with_dataset(task(), rows).reference()


def test_questions_can_be_narrowed_to_all_pass_alone():
    status = with_dataset(task(model="all_passed"))
    status.cfg.predict.params = {"questions": ["all_pass"]}
    rows = status.predict(None)

    assert set(rows[0]) == {"instance_id", "all_pass"}
    results = Task.flatten(status.score(rows, REFERENCE))
    assert results["questions_answered"] == ["all_pass"]
    assert not any(key.startswith(("binary_", "outcome_")) for key in results)
