"""Single-test-status models: one label per question, for one test."""

from abc import ABC, abstractmethod

from sourceworldbench_eval_harness.models.core.base import Row, SeededModel
from sourceworldbench_eval_harness.models.status.base import StatusModel, SubmissionReader
from sourceworldbench_eval_harness.schemas.status import BINARY, OUTCOME, PASSED, SINGLE_TEST_STATUS, to_binary


class SingleTestStatusModel(StatusModel, ABC):
    task = "single_test_status"
    schema = SINGLE_TEST_STATUS

    def answers_for(self, label: str) -> dict[str, str]:
        answers: dict[str, str] = {}
        if BINARY in self.questions:
            answers[BINARY] = to_binary(label)
        if OUTCOME in self.questions:
            answers[OUTCOME] = label
        return answers


class LabelPredictor(SingleTestStatusModel, ABC):
    """The row is the repository state plus `{"test", "label", "calls"}`, normalized
    by the task so no model names a dataset column. `label` is ground truth, which is
    what makes `Oracle` possible.
    """

    @abstractmethod
    def label_for(self, row: Row) -> str: ...

    def predict_single(self, row: Row) -> dict[str, str]:
        return self.answers_for(self.label_for(row))


class AllPassed(LabelPredictor):
    """The majority-class floor. At this class balance `random_status` can beat it on
    macro F1, so both are floors a model has to clear."""

    def label_for(self, row: Row) -> str:
        return PASSED


class RandomStatus(SeededModel, LabelPredictor):
    def label_for(self, row: Row) -> str:
        return self.rng.choice(self.schema.labels)


class Oracle(LabelPredictor):
    """Copies ground truth — a self-test, not a model.

    Must score 1.0 on every metric; anything else means the join or the label
    column is wrong, so run it first against a new config.
    """

    def label_for(self, row: Row) -> str:
        return str(row["label"])


class FromFile(SubmissionReader, SingleTestStatusModel):
    """This task's reader for an external submission — see `models.status`.

    An answer here is a bare label per question, as
    `docs/SINGLE_TEST_STATUS_FORMAT.md` documents, but nothing in the reading of
    it is specific to that, so the implementation is shared.
    """
