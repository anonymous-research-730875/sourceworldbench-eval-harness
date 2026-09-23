"""Memory-prediction baselines.

`constant_bytes` is also the majority answer to every budget question, since one guess
either fits a threshold or does not.
"""

from abc import ABC, abstractmethod
from typing import Any

from sourceworldbench_eval_harness.models.core.base import Row, SeededModel
from sourceworldbench_eval_harness.models.core.questions import QuestionModel, SubmissionReader
from sourceworldbench_eval_harness.schemas.memory import (
    BUDGETS,
    BYTES,
    GB,
    PREDICTION_QUESTIONS,
)


class MemoryPredictionModel(QuestionModel, ABC):
    task = "memory_prediction"

    @property
    def answerable(self) -> tuple[str, ...]:
        return PREDICTION_QUESTIONS

    def answers_for(self, size: float) -> dict[str, Any]:
        """Deriving the budget answers from one guess suits a baseline, which has a single
        behaviour to express, and defeats a real submission: the questions exist because
        different prompts elicit different answers."""
        answers: dict[str, Any] = {}
        if BYTES in self.questions:
            answers[BYTES] = size
        for question, budget in BUDGETS.items():
            if question in self.questions:
                answers[question] = size <= budget
        return answers


class BytesPredictor(MemoryPredictionModel, ABC):
    """Rows are normalized by the task so no model names a dataset column. They carry ground
    truth, which is what makes `Oracle` possible."""

    @abstractmethod
    def bytes_for(self, row: Row) -> float:
        """The footprint this model predicts, in bytes."""

    def predict_single(self, row: Row) -> dict[str, Any]:
        return self.answers_for(self.bytes_for(row))


class Oracle(BytesPredictor):
    """A self-test, not a model: must score perfectly, so run it first against a new config."""

    def bytes_for(self, row: Row) -> float:
        return float(row["bytes"])


class ConstantBytes(BytesPredictor):
    """The floor every real score is read against. Set `params.bytes` to a median, which
    minimises the log-ratio error a constant can achieve."""

    def __init__(self, bytes: float = 1.0 * GB, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.constant = float(bytes)

    def bytes_for(self, row: Row) -> float:
        return self.constant


class RandomBytes(SeededModel, BytesPredictor):
    """Log-uniform, so a guess is as likely to be kilobytes as gigabytes."""

    def __init__(self, low: float = 100.0e3, high: float = 10.0e9, **kwargs: Any) -> None:
        super().__init__(**kwargs)  # `seed` is `SeededModel`'s
        self.low, self.high = float(low), float(high)

    def bytes_for(self, row: Row) -> float:
        return float(self.low * pow(self.high / self.low, self.rng.random()))


class FromFile(SubmissionReader, MemoryPredictionModel):
    """This task's reader for an external submission — see `models.core.questions`."""
