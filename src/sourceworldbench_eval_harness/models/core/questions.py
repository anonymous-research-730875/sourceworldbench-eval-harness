"""Model scaffolding for any task that asks a fixed set of questions per instance."""

from abc import ABC, abstractmethod
from typing import Any

from sourceworldbench_eval_harness.models.core.base import SAMPLE, BasePredictor, PredictionFileError, Row


class QuestionModel(BasePredictor, ABC):
    # Does this model read a finished submission rather than work an answer out? A
    # task whose `external_input` *is* the submission refuses to pair a generating
    # model with it, which would otherwise silently answer from nothing.
    reads_submission: bool = False

    @property
    @abstractmethod
    def answerable(self) -> tuple[str, ...]:
        """Every question this model's task asks, in report order."""

    def __init__(self, questions: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        valid = self.answerable
        self.questions = tuple(questions) if questions else valid
        unknown = [question for question in self.questions if question not in valid]
        if unknown:
            raise ValueError(f"unknown question(s) {unknown}; valid: {list(valid)}")


class SubmissionReader(QuestionModel, ABC):
    """Take the answers already in the row, moving each under its canonical question name and
    leaving the task's shape checks to judge it. What lands is a normalised extract, not a copy.

    Abstract because knowing which questions exist is the task's own `FromFile`'s job.
    """

    reads_submission = True

    def __init__(self, question_fields: dict[str, str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        valid = self.answerable
        unknown = sorted(set(question_fields or {}) - set(valid))
        if unknown:
            raise ValueError(f"unknown question(s) {unknown} in question_fields; valid: {list(valid)}")
        self.fields = {q: (question_fields or {}).get(q, q) for q in self.questions}

    def predict(self, rows: list[Row]) -> list[Any]:
        """Reject the whole file before predicting any of it."""
        blank = [i for i, row in enumerate(rows) if not any(f in row for f in self.fields.values())]
        if blank:
            raise PredictionFileError(
                f"{len(blank)} row(s) answer none of {sorted(self.fields.values())}, e.g. rows {blank[:SAMPLE]}"
            )
        return super().predict(rows)

    def predict_single(self, row: Row) -> dict[str, Any]:
        return {question: row[field] for question, field in self.fields.items() if field in row}


__all__ = ["QuestionModel", "SubmissionReader"]
