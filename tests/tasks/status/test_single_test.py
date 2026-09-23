import json

import pytest

import helpers
from helpers import FakeDataset, row
from sourceworldbench_eval_harness.config import InvalidConfigError
from sourceworldbench_eval_harness.models.core.base import PredictionFileError
from sourceworldbench_eval_harness.tasks import resolve
from sourceworldbench_eval_harness.tasks.core.base import InvalidSubmissionError, Task
from sourceworldbench_eval_harness.tasks.status.single_test import SingleTestStatus

# One instance per label — the smallest set on which macro F1 over the three-way
# vocabulary is fully defined, and the shape `reference` returns.
REFERENCE = {
    "pandas__aaaaaaaa__pred_x": {"label": "PASSED", "test": "pandas/tests/test_a.py::test_one"},
    "pandas__bbbbbbbb__pred_y": {"label": "FAILED", "test": "pandas/tests/test_b.py::test_two"},
    "astropy__cccccccc__pred_z": {"label": "ERROR", "test": "astropy/tests/test_c.py::test_three"},
}

PASSED_ID, FAILED_ID, ERROR_ID = REFERENCE


def eval_config(name="single_test_status", **task_settings):
    return helpers.eval_config(name, **task_settings)


def task(require_full_coverage=True, model=None, **settings):
    return helpers.build_task(
        "single_test_status", require_full_coverage=require_full_coverage, model=model, **settings
    )


def flat_score(rows, **settings):
    """`score`'s groups as the flat `{question}_{metric}` file `evaluate` writes."""
    return Task.flatten(task(**settings).score(rows, REFERENCE))


def oracle_rows():
    """A perfect submission: every instance, both questions, from `REFERENCE`."""
    return [
        row(instance_id, binary="PASSED" if truth["label"] == "PASSED" else "NOT_PASSED", outcome=truth["label"])
        for instance_id, truth in REFERENCE.items()
    ]


def dataset_rows():
    """Dataset-shaped rows matching `REFERENCE`, with the columns a model is given.

    `single_test_state_id` is carried because v0.2.0 ships it and the harness reads nothing
    from it, so the reference has to come out without it.
    """
    return FakeDataset(
        [
            {
                "instance_id": instance_id,
                "single_test_state_id": f"{instance_id}__{truth['test']}",
                "test": truth["test"],
                "label": truth["label"],
                "repo": instance_id.split("__")[0],
                "base_commit": "deadbeef",
                "patch": "--- a/x\n+++ b/x\n",
                "container": "registry/image@sha256:abc",
                "command": "pytest -v",
            }
            for instance_id, truth in REFERENCE.items()
        ]
    )


def with_dataset(task_, rows=None):
    return helpers.attach_dataset(task_, dataset_rows() if rows is None else rows)


def test_oracle_scores_perfectly_on_every_question_in_one_run():
    results = flat_score(oracle_rows())
    assert results["binary_macro_f1"] == 1.0
    assert results["outcome_macro_f1"] == 1.0
    assert results["outcome_accuracy"] == 1.0
    assert results["outcome_tests"] == 3
    assert results["instances"] == 3
    assert results["questions_answered"] == ["binary", "outcome"]


def test_constant_prediction_scores_zero_on_every_class_it_never_predicts():
    """The floor: a class that occurs but is never predicted scores 0.0, not None."""
    rows = [row(instance_id, binary="PASSED", outcome="PASSED") for instance_id in REFERENCE]
    results = flat_score(rows)

    assert results["outcome_accuracy"] == pytest.approx(1 / 3)
    assert results["outcome_passed_f1"] == pytest.approx(0.5)
    assert results["outcome_failed_f1"] == 0.0
    assert results["outcome_error_f1"] == 0.0
    assert results["outcome_macro_f1"] == pytest.approx(0.5 / 3)
    assert results["binary_not_passed_f1"] == 0.0


def test_binary_collapses_the_three_way_truth_to_two_labels():
    """FAILED and ERROR are both NOT_PASSED, so a binary answer is right on both."""
    rows = [
        row(PASSED_ID, binary="PASSED"),
        row(FAILED_ID, binary="NOT_PASSED"),
        row(ERROR_ID, binary="NOT_PASSED"),
    ]
    assert flat_score(rows)["binary_macro_f1"] == 1.0


def test_mcc_is_reported_for_the_binary_question_only():
    """MCC is defined for two classes, so `outcome`'s five-way vocabulary has none."""
    results = flat_score(oracle_rows())
    assert results["binary_mcc"] == 1.0
    assert "outcome_mcc" not in results


def test_metrics_are_flat_and_prefixed_by_question():
    """The leaderboard turns columns into metrics, so nothing may nest."""
    results = flat_score([row(PASSED_ID, outcome="PASSED")], require_full_coverage=False)
    scalars = {key: value for key, value in results.items() if key != "details"}

    assert all(not isinstance(value, dict) for value in scalars.values())
    assert {"outcome_macro_f1", "outcome_accuracy", "outcome_tests"} <= set(scalars)
    assert not any(key.startswith("binary_") for key in scalars)  # unanswered, so unscored


def test_unanswered_questions_are_skipped_not_zeroed():
    results = flat_score([row(PASSED_ID, outcome="PASSED")], require_full_coverage=False)
    assert results["questions_answered"] == ["outcome"]
    assert not any(key.startswith("binary_") for key in results)


def test_there_is_no_per_instance_aggregate():
    """One test per instance, so a per-instance macro would score a single label."""
    assert not any("per_instance" in key for key in flat_score(oracle_rows()))


def test_confusion_and_per_class_tables_stay_out_of_the_columns():
    results = flat_score(oracle_rows())
    assert set(results["details"]) == {"binary", "outcome"}
    assert {"per_class", "confusion"} == set(results["details"]["outcome"])
    assert results["outcome_error_f1"] == 1.0  # per-class F1 is still promoted to a column


def test_valid_submission_passes():
    task().validate(oracle_rows(), REFERENCE)


def test_partial_coverage_is_rejected_and_can_be_allowed():
    rows = [row(PASSED_ID, outcome="PASSED")]
    with pytest.raises(InvalidSubmissionError, match="no prediction"):
        task().validate(rows, REFERENCE)
    task(require_full_coverage=False).validate(rows, REFERENCE)


def test_unknown_instance_is_rejected():
    with pytest.raises(InvalidSubmissionError, match="not present in the dataset"):
        task(require_full_coverage=False).validate([row("ghost", outcome="PASSED")], REFERENCE)


def test_duplicate_instance_rows_are_rejected():
    rows = [row(PASSED_ID, outcome="PASSED"), row(PASSED_ID, outcome="FAILED")]
    with pytest.raises(InvalidSubmissionError, match="duplicate row"):
        task(require_full_coverage=False).validate(rows, REFERENCE)


def test_a_row_answering_nothing_is_rejected():
    with pytest.raises(InvalidSubmissionError, match="no question answers"):
        task(require_full_coverage=False).validate([row(PASSED_ID)], REFERENCE)


def test_a_mismatched_test_name_is_rejected():
    """The check that catches a file built against another dataset revision."""
    rows = [row(PASSED_ID, test="pandas/tests/test_a.py::test_renamed", outcome="PASSED")]
    with pytest.raises(InvalidSubmissionError, match="built against different rows"):
        task(require_full_coverage=False).validate(rows, REFERENCE)


def test_a_matching_test_name_passes_and_an_absent_one_is_fine():
    matching = [row(PASSED_ID, test=REFERENCE[PASSED_ID]["test"], outcome="PASSED")]
    task(require_full_coverage=False).validate(matching, REFERENCE)
    task(require_full_coverage=False).validate([row(PASSED_ID, outcome="PASSED")], REFERENCE)


def test_all_problems_are_reported_together():
    rows = [
        row("ghost", outcome="PASSED"),
        row(PASSED_ID, test="pandas/tests/test_a.py::test_renamed", outcome="PASSED"),
    ]
    with pytest.raises(InvalidSubmissionError) as excinfo:
        task(require_full_coverage=False).validate(rows, REFERENCE)

    message = str(excinfo.value)
    assert "not present in the dataset" in message
    assert "built against different rows" in message


def test_shape_validation_needs_no_ground_truth():
    task().validate_shape(oracle_rows())


def test_a_wrapped_answer_is_named_as_the_multi_test_shape():
    """The mistake anyone arriving from `test_status` makes, so it gets its own message."""
    with pytest.raises(InvalidSubmissionError, match="the label itself, not an object"):
        task().validate_shape([row(PASSED_ID, outcome={"tests": ["t"], "labels": ["PASSED"]})])


def test_a_non_string_label_is_rejected():
    with pytest.raises(InvalidSubmissionError, match="label must be a string"):
        task().validate_shape([row(PASSED_ID, outcome=1)])


def test_binary_rejects_the_three_way_vocabulary():
    """FAILED is an `outcome` answer, so it means the wrong prompt was used."""
    with pytest.raises(InvalidSubmissionError, match="unknown label 'FAILED'"):
        task().validate_shape([row(PASSED_ID, binary="FAILED")])


def test_outcome_rejects_the_binary_vocabulary():
    with pytest.raises(InvalidSubmissionError, match="unknown label 'NOT_PASSED'"):
        task().validate_shape([row(PASSED_ID, outcome="NOT_PASSED")])


def test_a_row_without_an_id_is_named_by_its_index():
    with pytest.raises(InvalidSubmissionError, match="row 0: missing 'instance_id'"):
        task().validate_shape([{"outcome": "PASSED"}])


def test_reference_keys_the_label_and_test_by_instance():
    assert with_dataset(task()).reference() == REFERENCE


def test_a_label_column_with_holes_is_refused_as_an_answer_key():
    """A revision that introduced nulls in `label` must fail rather than score around
    them, so the run cannot quietly cover fewer instances than it reports."""
    rows = dataset_rows()
    rows[1]["label"] = None

    with pytest.raises(InvalidSubmissionError, match="cannot be scored against"):
        with_dataset(task(), rows).reference()


def test_a_label_outside_the_vocabulary_is_rejected_before_scoring():
    """`multiclass_confusion` indexes by the true label, so this must not reach it."""
    rows = dataset_rows()
    rows[0]["label"] = "SKIPPED"

    with pytest.raises(InvalidSubmissionError, match="outside \\['PASSED', 'FAILED', 'ERROR'\\]"):
        with_dataset(task(), rows).reference()


def test_a_duplicate_instance_in_the_dataset_is_rejected():
    helpers.expect_duplicate_instance_rejected(task(), dataset_rows())


def test_a_dataset_missing_a_needed_column_names_what_is_available():
    helpers.expect_missing_column_named(task(), dataset_rows(), "label")


def test_the_answer_key_is_not_a_config_choice():
    """`label` is a property of the dataset, so the setting does not exist to be named."""
    with pytest.raises(InvalidConfigError, match="label_field"):
        task(label_field="labl")


def test_a_baseline_writes_one_row_per_instance_with_the_test_name():
    rows = with_dataset(task(model="all_passed")).predict(None)

    assert len(rows) == 3
    assert rows[0]["instance_id"] == PASSED_ID
    assert rows[0]["test"] == REFERENCE[PASSED_ID]["test"]
    assert rows[0]["binary"] == "PASSED" and rows[0]["outcome"] == "PASSED"


def test_the_oracle_round_trips_through_validation_and_scoring():
    """The sanity check to run against a new config: it must come out at 1.0."""
    status = with_dataset(task(model="oracle"))

    rows = status.predict(None)
    reference = status.reference()
    status.validate_shape(rows)
    status.validate(rows, reference)
    results = Task.flatten(status.score(rows, reference))

    assert results["binary_macro_f1"] == 1.0
    assert results["outcome_macro_f1"] == 1.0


def test_an_external_file_is_normalised_to_the_id_test_and_questions():
    external = [
        {"instance_id": PASSED_ID, "test": REFERENCE[PASSED_ID]["test"], "outcome": "PASSED", "notes": "dropped"},
    ]
    rows = task().predict(external)

    assert rows == [{"instance_id": PASSED_ID, "test": REFERENCE[PASSED_ID]["test"], "outcome": "PASSED"}]


def test_an_external_column_can_be_renamed_and_questions_narrowed():
    status = task()
    status.cfg.predict.params = {"questions": ["outcome"], "question_fields": {"outcome": "predicted_label"}}
    rows = status.predict([{"instance_id": PASSED_ID, "predicted_label": "FAILED", "binary": "PASSED"}])

    assert rows == [{"instance_id": PASSED_ID, "outcome": "FAILED"}]  # binary was not kept, so it is not scored


def test_an_external_row_answering_nothing_fails_the_run():
    with pytest.raises(PredictionFileError, match="answer none of"):
        task().predict([{"instance_id": PASSED_ID}])


def calls_file(tmp_path, content):
    path = tmp_path / "function_called.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_no_calls_file_means_no_calls():
    status = task()
    assert status._calls() is None
    assert status._model_row(dataset_rows()[0], None)["calls"] is None


def test_calls_reach_the_model_row_and_a_null_entry_is_passed_through(tmp_path):
    path = calls_file(tmp_path, {PASSED_ID: ["pandas.read_csv", "pandas.DataFrame"], FAILED_ID: None})
    status = task(calls_file=str(path))
    calls = status._calls()

    rows = dataset_rows()
    assert status._model_row(rows[0], calls)["calls"] == ["pandas.read_csv", "pandas.DataFrame"]
    assert status._model_row(rows[1], calls)["calls"] is None  # null: extraction failed
    assert status._model_row(rows[2], calls)["calls"] is None  # absent entirely


def test_a_calls_file_that_is_not_a_mapping_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be a JSON object"):
        task(calls_file=str(calls_file(tmp_path, ["pandas.read_csv"])))._calls()


def test_a_calls_entry_that_is_not_a_list_of_strings_is_rejected(tmp_path):
    path = calls_file(tmp_path, {PASSED_ID: {"call": "pandas.read_csv"}})
    with pytest.raises(InvalidSubmissionError, match="expected a list of strings or null"):
        task(calls_file=str(path))._calls()


def test_a_missing_calls_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="task.calls_file not found"):
        task(calls_file=str(tmp_path / "absent.json"))._calls()


def test_coverage_is_logged_and_unknown_ids_are_ignored(tmp_path, caplog):
    path = calls_file(tmp_path, {PASSED_ID: ["pandas.read_csv"], FAILED_ID: None, "ghost": ["x"]})
    status = with_dataset(task(model="all_passed", calls_file=str(path)))

    with caplog.at_level("INFO"):
        rows = status.predict(None)

    assert len(rows) == 3  # an unknown id changes nothing about what is predicted
    assert "1 of 3 instance(s) have a list, 1 null, 1 absent" in caplog.text
    assert "not in the dataset" in caplog.text


def test_config_selects_the_task():
    assert isinstance(resolve(eval_config()), SingleTestStatus)


def test_the_id_field_comes_from_the_config():
    assert task().id_field == "instance_id"


def test_an_unknown_question_is_rejected():
    """`all_pass` exists for the multi-test task, so asking for it here must not pass
    silently."""
    status = task(model="all_passed")
    status.cfg.predict.params = {"questions": ["all_pass"]}

    with pytest.raises(ValueError, match="unknown question"):
        status.configured_model()
