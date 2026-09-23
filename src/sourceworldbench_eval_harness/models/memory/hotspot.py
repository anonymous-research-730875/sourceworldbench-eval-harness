"""Memory-hotspot baselines.

`most_called` is the one worth reporting: it measures how much of the task falls out of
call count alone.
"""

from abc import ABC, abstractmethod

from sourceworldbench_eval_harness.models.core.base import Row, SeededModel
from sourceworldbench_eval_harness.models.core.questions import QuestionModel, SubmissionReader
from sourceworldbench_eval_harness.schemas.memory import (
    FUNCTIONS_FIELD,
    HOTSPOT_AXES,
    HOTSPOT_QUESTIONS,
    TOP_K,
    HotspotAxis,
)


class MemoryHotspotModel(QuestionModel, ABC):
    task = "memory_hotspot"

    @property
    def answerable(self) -> tuple[str, ...]:
        return HOTSPOT_QUESTIONS


class RankingPredictor(MemoryHotspotModel, ABC):
    """Rows are normalized by the task so no model names a dataset column. They carry ground
    truth, which is what makes `Oracle` possible."""

    @abstractmethod
    def rank(self, row: Row, axis: HotspotAxis) -> list[str]:
        """Most expensive first on `axis`, by this model's reckoning."""

    def predict_single(self, row: Row) -> dict[str, list[str]]:
        answers: dict[str, list[str]] = {}
        for axis in HOTSPOT_AXES:
            asked = [question for question in axis.questions if question in self.questions]
            if not asked:
                continue
            # Ranked once then sliced, so one axis cannot contradict itself across the
            # questions asked about it. Separate axes may differ, because they genuinely do.
            ranked = self.rank(row, axis)
            for question in asked:
                answers[question] = ranked[: TOP_K[question]]
        return answers


class Oracle(RankingPredictor):
    """A self-test, not a model: must score 1.0."""

    def rank(self, row: Row, axis: HotspotAxis) -> list[str]:
        return [hotspot["key"] for hotspot in row[axis.hotspots_field]]


class MostCalled(RankingPredictor):
    """A heuristic rather than a floor: how much is answerable without reasoning about
    allocation per call."""

    def rank(self, row: Row, axis: HotspotAxis) -> list[str]:
        ordered = sorted(row[FUNCTIONS_FIELD], key=lambda f: f["call_count"], reverse=True)
        return [function["key"] for function in ordered]


class RandomFunctions(SeededModel, RankingPredictor):
    """The floor every real score is read against."""

    def rank(self, row: Row, axis: HotspotAxis) -> list[str]:
        keys = [function["key"] for function in row[FUNCTIONS_FIELD]]
        return self.rng.sample(keys, len(keys))


class FromFile(SubmissionReader, MemoryHotspotModel):
    """This task's reader for an external submission — see `models.core.questions`."""
