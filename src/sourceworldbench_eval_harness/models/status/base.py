"""What the status-task models add to `models.core.questions`: a label vocabulary."""

from abc import ABC
from typing import ClassVar

from sourceworldbench_eval_harness.models.core.questions import QuestionModel, SubmissionReader
from sourceworldbench_eval_harness.schemas.status import StatusSchema

__all__ = ["StatusModel", "SubmissionReader"]


class StatusModel(QuestionModel, ABC):
    # Set by the per-task base class, as `task` is.
    schema: ClassVar[StatusSchema]

    @property
    def answerable(self) -> tuple[str, ...]:
        return self.schema.questions
