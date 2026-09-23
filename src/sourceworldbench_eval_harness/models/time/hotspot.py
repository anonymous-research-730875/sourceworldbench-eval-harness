"""Time-hotspot models: a ranking of the codebase's functions, sliced per question.

`most_called` is the one worth reporting: if call count alone ranks hotspots well, the
task is easier than it looks.
"""

from abc import ABC, abstractmethod

from sourceworldbench_eval_harness.models.core.base import Row, SeededModel
from sourceworldbench_eval_harness.models.core.questions import QuestionModel, SubmissionReader
from sourceworldbench_eval_harness.schemas.time import (
    FUNCTIONS_FIELD,
    HOTSPOT_QUESTIONS,
    HOTSPOTS_FIELD,
    TOP_K,
)


class TimeHotspotModel(QuestionModel, ABC):
    task = "time_hotspot"

    @property
    def answerable(self) -> tuple[str, ...]:
        return HOTSPOT_QUESTIONS


class RankingPredictor(TimeHotspotModel, ABC):
    """The row is `{"functions_complete_list": [...], "hotspots": [...]}` plus the repository
    state, normalized by the task so no model names a dataset column. `hotspots` is ground
    truth, which is what makes `Oracle` possible.
    """

    @abstractmethod
    def rank(self, row: Row) -> list[str]:
        """The codebase's functions, most expensive first, by this model's reckoning."""

    def predict_single(self, row: Row) -> dict[str, list[str]]:
        # Ranked once and then sliced, so `top5` is `top20`'s first five by construction:
        # one ranking cannot contradict itself between the questions asked about it.
        ranked = self.rank(row)
        return {question: ranked[: TOP_K[question]] for question in self.questions}


class Oracle(RankingPredictor):
    """Copies the ranked ground truth — a self-test, not a model. Must score 1.0."""

    def rank(self, row: Row) -> list[str]:
        return [hotspot["key"] for hotspot in row[HOTSPOTS_FIELD]]


class MostCalled(RankingPredictor):
    """Rank by call count: a real heuristic rather than a floor, measuring how much of the task is
    answerable without reasoning about cost per call.
    """

    def rank(self, row: Row) -> list[str]:
        ordered = sorted(row[FUNCTIONS_FIELD], key=lambda f: f["call_count"], reverse=True)
        return [function["key"] for function in ordered]


class RandomFunctions(SeededModel, RankingPredictor):
    """Draw from the profiled functions — the floor every real score is read against."""

    def rank(self, row: Row) -> list[str]:
        keys = [function["key"] for function in row[FUNCTIONS_FIELD]]
        return self.rng.sample(keys, len(keys))


class FromFile(SubmissionReader, TimeHotspotModel):
    """This task's reader for an external submission — see `models.core.questions`."""
