"""The shared vocabulary, and where the datasets legitimately differ."""

import dataclasses

import pytest

from sourceworldbench_eval_harness.schemas.status import (
    BINARY_LABELS,
    NOT_PASSED,
    PASSED,
    SINGLE_TEST_STATUS,
    TEST_STATUS_ANCHORED,
    StatusSchema,
    to_binary,
)


@pytest.mark.parametrize("schema", [TEST_STATUS_ANCHORED, SINGLE_TEST_STATUS])
def test_every_status_collapses_into_the_binary_vocabulary(schema):
    """`binary`'s definition, not a property of either dataset: PASSED, or everything else."""
    assert to_binary(PASSED) == PASSED
    assert {to_binary(label) for label in schema.labels} == set(BINARY_LABELS)
    assert all(to_binary(label) == NOT_PASSED for label in schema.labels if label != PASSED)


@pytest.mark.parametrize("schema", [TEST_STATUS_ANCHORED, SINGLE_TEST_STATUS])
def test_binary_asks_two_labels_and_outcome_the_datasets_own(schema):
    vocabulary = schema.label_vocabulary
    assert vocabulary["binary"] == BINARY_LABELS
    assert vocabulary["outcome"] == schema.labels


def test_all_pass_carries_no_labels_so_it_is_absent_from_the_vocabulary():
    """A true/false has no labels to check, and `check_shape` relies on that absence."""
    assert "all_pass" in TEST_STATUS_ANCHORED.questions
    assert "all_pass" not in TEST_STATUS_ANCHORED.label_vocabulary


def test_a_question_a_dataset_does_not_ask_gets_no_vocabulary():
    """Single-test-status has no `all_pass` at all, so nothing may map one."""
    assert "all_pass" not in SINGLE_TEST_STATUS.questions
    assert set(SINGLE_TEST_STATUS.label_vocabulary) == {"binary", "outcome"}


def test_the_registered_datasets_differ_only_in_the_questions_they_ask():
    """Same statuses; the anchored task additionally asks `all_pass`."""
    assert TEST_STATUS_ANCHORED.labels == SINGLE_TEST_STATUS.labels
    assert set(TEST_STATUS_ANCHORED.questions) - set(SINGLE_TEST_STATUS.questions) == {"all_pass"}
    # PASSED leads, which is what makes the binary collapse and the
    # `multiclass_mcc` positive-class-last convention hold.
    assert TEST_STATUS_ANCHORED.labels[0] == SINGLE_TEST_STATUS.labels[0] == PASSED


def test_a_vocabulary_cannot_be_edited_after_a_run_reads_it():
    """A sidecar is dumped after scoring, so it must describe what was computed."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        TEST_STATUS_ANCHORED.labels = ("PASSED",)


def test_a_new_dataset_declares_only_its_statuses_and_questions():
    """The seam: adding a task means one `StatusSchema`, not another module."""
    schema = StatusSchema(labels=(PASSED, "FLAKY"), questions=("outcome",))
    assert schema.label_vocabulary == {"outcome": (PASSED, "FLAKY")}
    assert to_binary("FLAKY") == NOT_PASSED
