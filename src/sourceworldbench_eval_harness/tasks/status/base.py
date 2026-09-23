"""What the status tasks add to `tasks.core.questions`: a label vocabulary."""

from abc import ABC
from collections.abc import Sequence
from typing import Any, ClassVar, TypeVar

from sourceworldbench_eval_harness import metrics
from sourceworldbench_eval_harness.schemas.status import StatusSchema
from sourceworldbench_eval_harness.tasks.core.questions import QuestionTask, QuestionTaskConfig


class StatusTaskConfig(QuestionTaskConfig):
    """Adds nothing yet; named so a status-only setting has somewhere to land without
    every status task changing base class to get it."""


S = TypeVar("S", bound=StatusTaskConfig)


class StatusTask(QuestionTask[S], ABC):
    # Also, what this task's models validate `params.questions` against, so a task
    # and its models cannot disagree about the vocabulary.
    schema: ClassVar[StatusSchema]

    @property
    def questions(self) -> tuple[str, ...]:
        return self.schema.questions

    def score_labels(self, y_true: Sequence[str], y_pred: Sequence[str], question: str) -> dict[str, Any]:
        vocabulary = self.schema.label_vocabulary
        try:
            allowed = vocabulary[question]
        except KeyError:
            raise ValueError(f"{question} carries no labels; labeled questions: {sorted(vocabulary)}") from None
        return metrics.classification_report(y_true, y_pred, allowed)


__all__ = ["StatusTask", "StatusTaskConfig"]
