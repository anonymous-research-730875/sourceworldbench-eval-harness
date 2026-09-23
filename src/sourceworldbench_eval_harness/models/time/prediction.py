"""Time-prediction models: one guess at the runtime, answered four ways.

`constant_seconds` is the floor, and also the majority answer to each budget question,
since a single guess either fits a threshold or does not.
"""

from abc import ABC, abstractmethod
from typing import Any

from sourceworldbench_eval_harness.models.core.base import Row, SeededModel
from sourceworldbench_eval_harness.models.core.questions import QuestionModel, SubmissionReader
from sourceworldbench_eval_harness.schemas.time import (
    BUDGETS,
    PREDICTION_QUESTIONS,
    SECONDS,
)


class TimePredictionModel(QuestionModel, ABC):
    task = "time_prediction"

    @property
    def answerable(self) -> tuple[str, ...]:
        return PREDICTION_QUESTIONS

    def answers_for(self, seconds: float) -> dict[str, Any]:
        """One predicted runtime, shaped as each question wants it.

        Deriving the budget answers from `seconds` is right for a baseline, which has a single behaviour to
        express, and wrong for a real submission: the questions exist because different
        prompts elicit different answers.
        """
        answers: dict[str, Any] = {}
        if SECONDS in self.questions:
            answers[SECONDS] = seconds
        for question, budget in BUDGETS.items():
            if question in self.questions:
                answers[question] = seconds <= budget
        return answers


class SecondsPredictor(TimePredictionModel, ABC):
    """The row is `{"seconds": ...}` plus the repository state,
    normalized by the task so no model names a dataset column. `seconds` is ground
    truth, which is what makes `Oracle` possible.
    """

    @abstractmethod
    def seconds_for(self, row: Row) -> float:
        """The runtime this model predicts, in seconds."""

    def predict_single(self, row: Row) -> dict[str, Any]:
        return self.answers_for(self.seconds_for(row))


class Oracle(SecondsPredictor):
    """Copies ground truth — a self-test, not a model.

    Exact on `seconds`, so `log10_error` is 0, every `within_factor` is 1 and the fit is the
    identity line — slope 1, intercept 0; the rest are then correct by construction. Run
    it first against a new config.
    """

    def seconds_for(self, row: Row) -> float:
        return float(row["seconds"])


class ConstantSeconds(SecondsPredictor):
    """Always answer the same runtime — the floor every real score is read against.

    `params.seconds` sets it; the dataset's median is the informed choice, since it
    minimises the log-ratio error a constant can achieve.
    """

    def __init__(self, seconds: float = 5.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.constant = float(seconds)

    def seconds_for(self, row: Row) -> float:
        return self.constant


class RandomSeconds(SeededModel, SecondsPredictor):
    """Draw a runtime log-uniformly between `params.low` and `params.high`, so the
    guess is as likely to be seconds as minutes."""

    def __init__(self, low: float = 1.0, high: float = 600.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)  # `seed` is `SeededModel`'s
        self.low, self.high = float(low), float(high)

    def seconds_for(self, row: Row) -> float:
        return float(self.low * pow(self.high / self.low, self.rng.random()))


class FromFile(SubmissionReader, TimePredictionModel):
    """This task's reader for an external submission — see `models.core.questions`."""
