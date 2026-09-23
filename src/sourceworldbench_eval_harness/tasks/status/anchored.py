"""`sourceworldbench-test-status-anchored`: a fixed subset of tests per row, all of which PASSED
on the base commit, so a non-PASSED label is a regression the patch introduced.

Adds `all_pass` (`Q-all-pass`, "are all of these tests expected to pass?"), answered with a
single true or false, to the multi-test questions. See `docs/TEST_STATUS_ANCHORED_FORMAT.md`.
"""

from typing import Any, Literal

from pydantic import ConfigDict

from sourceworldbench_eval_harness import metrics
from sourceworldbench_eval_harness.schemas.status import (
    ALL_PASS,
    BINARY_LABELS,
    NOT_PASSED,
    PASSED,
    TEST_STATUS_ANCHORED,
)
from sourceworldbench_eval_harness.tasks.status.multi_test import MultiTestStatusTask, MultiTestTaskConfig

__all__ = ["TEST_COLUMNS", "TestStatusAnchored"]

TEST_COLUMNS = ("tests", "labels")


class TestStatusAnchoredTaskConfig(MultiTestTaskConfig):
    """A `calls_file` setting belongs here once that file exists for this dataset —
    see `single_test_status` for the pattern."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["test_status_anchored"]


class TestStatusAnchored(MultiTestStatusTask[TestStatusAnchoredTaskConfig]):
    name = "test_status_anchored"
    Config = TestStatusAnchoredTaskConfig
    schema = TEST_STATUS_ANCHORED

    @property
    def test_columns(self) -> tuple[str, str]:
        return TEST_COLUMNS

    @property
    def per_test_questions(self) -> tuple[str, ...]:
        """`all_pass` asks about the instance, so it names no tests to cross-reference."""
        return tuple(question for question in self.questions if question != ALL_PASS)

    def check_shape(self, where: str, question: str, answer: Any) -> list[str]:
        if question != ALL_PASS:
            return super().check_shape(where, question, answer)
        if isinstance(answer, bool):
            return []
        return [
            f"{where}/{question}: answer must be true or false — true if every test "
            f"still passes — got {type(answer).__name__}"
        ]

    def score_question(
        self, question: str, rows: list[dict[str, Any]], reference: Any
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if question != ALL_PASS:
            return super().score_question(question, rows, reference)
        return self.all_pass_metrics(rows, reference)

    def all_pass_metrics(
        self, rows: list[dict[str, Any]], reference: dict[str, dict[str, str]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pooled over instances rather than tests, hence `all_pass_instances`. Scored
        in the `binary` vocabulary with PASSED meaning "nothing broke", so non-PASSED
        stays the positive class and `all_pass_not_passed_f1` reads as: does the model
        notice a patch broke something?
        """
        pooled_true: list[str] = []
        pooled_pred: list[str] = []
        for row in rows:
            labels = reference[row[self.id_field]].values()
            pooled_true.append(PASSED if all(label == PASSED for label in labels) else NOT_PASSED)
            pooled_pred.append(PASSED if row[ALL_PASS] else NOT_PASSED)

        pooled = metrics.classification_report(pooled_true, pooled_pred, BINARY_LABELS)
        return self.flat_classification(pooled, len(pooled_true), count_key="instances")
