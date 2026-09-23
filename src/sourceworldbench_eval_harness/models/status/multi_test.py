"""Models for a task whose row holds many tests. Each task binds these to itself
with a one-line subclass, since a model has to know which task it belongs to."""

from abc import ABC, abstractmethod
from typing import Any

from sourceworldbench_eval_harness.models.core.base import Row, SeededModel
from sourceworldbench_eval_harness.models.status.base import StatusModel
from sourceworldbench_eval_harness.schemas.status import ALL_PASS, BINARY, OUTCOME, PASSED, to_binary


class MultiTestModel(StatusModel, ABC):
    def answers_for(self, tests: list[str], labels: list[str]) -> dict[str, Any]:
        """Keyed off the schema, so a task gets exactly the questions it declares."""
        answers: dict[str, Any] = {}
        if BINARY in self.questions:
            answers[BINARY] = {"tests": list(tests), "labels": [to_binary(x) for x in labels]}
        if OUTCOME in self.questions:
            answers[OUTCOME] = {"tests": list(tests), "labels": list(labels)}
        if ALL_PASS in self.questions:
            answers[ALL_PASS] = all(label == PASSED for label in labels)
        return answers


class MultiTestLabelPredictor(MultiTestModel, ABC):
    """The row is `{"tests": [...], "labels": [...]}`, normalized by the task so no model names a
    dataset column. `labels` is ground truth, which is what makes `Oracle` possible.

    Answering every question from one set of labels is right for a baseline and wrong for a
    real submission — see `tasks.status.multi_test`.
    """

    @abstractmethod
    def labels_for(self, tests: list[str], truth: list[str]) -> list[str]: ...

    def predict_single(self, row: Row) -> dict[str, Any]:
        tests, truth = list(row["tests"]), list(row["labels"])
        return self.answers_for(tests, self.labels_for(tests, truth))


class AllPassed(MultiTestLabelPredictor, ABC):
    def labels_for(self, tests: list[str], truth: list[str]) -> list[str]:
        return [PASSED] * len(tests)


class RandomStatus(SeededModel, MultiTestLabelPredictor, ABC):
    def labels_for(self, tests: list[str], truth: list[str]) -> list[str]:
        return [self.rng.choice(self.schema.labels) for _ in tests]


class Oracle(MultiTestLabelPredictor, ABC):
    """A self-test, not a model: must score 1.0 on every metric that can reach it."""

    def labels_for(self, tests: list[str], truth: list[str]) -> list[str]:
        return list(truth)


__all__ = [
    "AllPassed",
    "MultiTestLabelPredictor",
    "MultiTestModel",
    "Oracle",
    "RandomStatus",
]
