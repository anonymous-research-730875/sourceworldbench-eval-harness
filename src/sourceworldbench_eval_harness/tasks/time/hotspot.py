"""`sourceworldbench-time-hotspots`: which functions dominate a workload's runtime.

The top 1/5/20 functions, most expensive first, by exclusive time — a function's own
frame, excluding the calls the profiler recorded separately.

`functions_complete_list` holds every profiled function, the candidate pool and what makes
"from the codebase, not a built-in" checkable. See `docs/TIME_HOTSPOT_FORMAT.md`.
"""

import logging
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import ConfigDict

from sourceworldbench_eval_harness import metrics
from sourceworldbench_eval_harness.models.core.questions import QuestionModel
from sourceworldbench_eval_harness.schemas.time import (
    FUNCTIONS_FIELD,
    HOTSPOT_QUESTIONS,
    HOTSPOT_TIME_FIELD,
    HOTSPOTS_FIELD,
    STATE_FIELDS,
    TOP_K,
    alias,
    lookup_index,
    resolve,
)
from sourceworldbench_eval_harness.tasks.core.base import SAMPLE, fail
from sourceworldbench_eval_harness.tasks.core.questions import QuestionTask, QuestionTaskConfig

__all__ = ["TimeHotspot"]

# The dataset columns every row must carry: the candidate pool and the ranked ground truth.
REQUIRED_FIELDS: tuple[str, ...] = (FUNCTIONS_FIELD, HOTSPOTS_FIELD)


class TimeHotspotTaskConfig(QuestionTaskConfig):
    model_config = ConfigDict(extra="forbid")

    name: Literal["time_hotspot"]


class TimeHotspot(QuestionTask[TimeHotspotTaskConfig]):
    name = "time_hotspot"
    Config = TimeHotspotTaskConfig

    @property
    def questions(self) -> tuple[str, ...]:
        return HOTSPOT_QUESTIONS

    @staticmethod
    def scoreable(row: Any) -> bool:
        """Whether this workload profiled enough functions for every question to be asked and scored.

        Applied on both sides, so the prediction file and the reference stay aligned and every
        question is always scored over the same instances. See `reference`.
        """
        return len(row[HOTSPOTS_FIELD]) >= max(TOP_K.values())

    def generated_rows(self, model: QuestionModel) -> list[dict[str, Any]]:
        id_field = self.id_field
        return [
            {
                id_field: row[id_field],
                **model.predict_single(
                    {
                        FUNCTIONS_FIELD: [dict(f) for f in row[FUNCTIONS_FIELD]],
                        HOTSPOTS_FIELD: [dict(h) for h in row[HOTSPOTS_FIELD]],
                        **{field: row[field] for field in STATE_FIELDS if field in row},
                    }
                ),
            }
            for row in self._dataset_rows(id_field, *REQUIRED_FIELDS)
            if self.scoreable(row)
        ]

    def check_shape(self, where: str, question: str, answer: Any) -> list[str]:
        at = f"{where}/{question}"
        wanted = TOP_K[question]
        if not isinstance(answer, list):
            return [f"{at}: answer must be a list of {wanted} function(s), most expensive first"]
        if len(answer) != wanted:
            return [f"{at}: {len(answer)} function(s) but {question} asks for exactly {wanted}"]

        aliases = [alias(entry) for entry in answer]
        malformed = [position for position, key in enumerate(aliases) if key is None]
        if malformed:
            return [
                f"{at}: {len(malformed)} entry(s) do not identify a function by its `key`, or by "
                f"`filename`, `firstlineno` and `qualname`, e.g. at index {malformed[:SAMPLE]}"
            ]
        if len(set(aliases)) != len(aliases):
            return [f"{at}: the same function is listed more than once"]
        return []

    def cross_check(self, instance_id: str, row: dict[str, Any], truth: Any) -> list[str]:
        """Every function named must be one this workload actually profiled, which is
        what keeps an answer inside the codebase rather than naming a built-in."""
        problems: list[str] = []
        for question in (q for q in HOTSPOT_QUESTIONS if q in row):
            found = [resolve(entry, truth["index"]) for entry in row[question]]
            unknown = [entry for entry, key in zip(row[question], found, strict=True) if key is None]
            if unknown:
                problems.append(
                    f"{instance_id}/{question}: {len(unknown)} function(s) are not in this "
                    f"instance's profile, e.g. {[alias(e) for e in unknown[:SAMPLE]]}"
                )
                continue
            # `check_shape` can only compare the spellings as submitted, and a function's
            # `key` never equals its own file/line/qualname triple as a string, so one
            # function listed under both spellings gets past it. Both entries resolve to
            # the same function, which leaves the answer really `k - 1` long: a shorter
            # ranking has one fewer placement to get wrong, and `somers_d` would score it
            # out of a smaller denominator. After resolution the two spellings are one
            # name again, so the duplicate is caught here instead.
            if len(set(found)) != len(found):
                problems.append(
                    f"{instance_id}/{question}: the same function is listed more than once, "
                    f"under two different spellings"
                )
        return problems

    def reference(self) -> dict[str, dict[str, Any]]:
        """Per instance: an index resolving either accepted spelling to one canonical key, and the
        ranked true hotspots with the times they are ranked by.

        A workload profiling fewer functions than the largest question asks for is dropped rather
        than failing the run — `top20` is unanswerable from a 12-function profile, and `validate`
        wants every reference instance answered for every question.
        """
        id_field = self.id_field
        reference: dict[str, dict[str, Any]] = {}
        problems: list[str] = []
        unscoreable: list[str] = []
        wanted = max(TOP_K.values())

        rows = self._dataset_rows(id_field, *REQUIRED_FIELDS)
        for instance_id, row in self.once_per_instance(rows, problems):
            if not self.scoreable(row):
                unscoreable.append(f"{instance_id} ({len(row[HOTSPOTS_FIELD])})")
                continue
            functions = [dict(f) for f in row[FUNCTIONS_FIELD]]
            index = lookup_index(functions)

            weights = {f["key"]: f[HOTSPOT_TIME_FIELD] for f in functions}
            ranked = [h["key"] for h in row[HOTSPOTS_FIELD]]
            if defects := self.ranking_defects(instance_id, ranked, weights):
                problems.extend(defects)
                continue
            reference[instance_id] = {"index": index, "ranked": ranked, "weights": weights}
        if unscoreable:
            logging.warning(
                "[reference] %d instance(s) profiled fewer than %d function(s), so no question is scored on them: %s",
                len(unscoreable),
                wanted,
                ", ".join(unscoreable[:SAMPLE]),
            )
        fail(problems)
        return reference

    @staticmethod
    def ranking_defects(instance_id: str, ranked: list[str], weights: dict[str, float]) -> list[str]:
        """Ways the ranked ground truth can break the contract `ndcg_macro` is scored under,
        checked rather than assumed.

        `weighted_ndcg_at_k` caps at 1.0 only if the expected functions are the heaviest there are,
        in descending order. A profile that breaks that scores *above* 1.0, where the leaderboard
        silently drops the column; naming the instance here fails the run instead.
        """
        field = HOTSPOTS_FIELD
        unknown = [key for key in ranked if key not in weights]
        if unknown:
            return [
                f"{instance_id}: {len(unknown)} {field} entry(s) are absent from "
                f"{FUNCTIONS_FIELD}, e.g. {unknown[:SAMPLE]}"
            ]
        if len(set(ranked)) != len(ranked):
            return [f"{instance_id}: {field} lists the same function more than once"]
        # Compared by weight rather than by key, so that functions tied on time may be
        # ranked either way round.
        heaviest = sorted(weights.values(), reverse=True)[: len(ranked)]
        if [weights[key] for key in ranked] != heaviest:
            return [
                f"{instance_id}: {field} is not the {len(ranked)} costliest function(s) in "
                f"descending {HOTSPOT_TIME_FIELD} order, which ndcg_macro needs to cap at 1.0"
            ]
        return []

    def score_question(
        self, question: str, rows: list[dict[str, Any]], reference: Any
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Per-instance retrieval scores, averaged. `*_scored` carries the coverage,
        since a metric is undefined on an instance with nothing to find. `ndcg_micro`
        cannot be averaged that way, so it comes through `across`."""
        k = TOP_K[question]
        scored = [self.retrieval_scores(row[question], reference[row[self.id_field]], k) for row in rows]
        pooled = metrics.micro_ndcg([gains for _, gains in scored])
        return self.flat_aggregate([per_instance for per_instance, _ in scored], ndcg_micro=pooled), None

    @staticmethod
    def retrieval_scores(
        answer: Sequence[Any], truth: dict[str, Any], k: int
    ) -> tuple[dict[str, float | None], tuple[float, float]]:
        """One instance's answer to one question, with the gain pair `ndcg_micro` needs.

        `hit_rate` and `time_captured` read the answer as a set, counting functions and weighing
        them by the time they cost. `ndcg_macro` and `somers_d` read it as a ranking and differ in
        how: top-heavy and gain-weighted versus position-blind and order-only, which a perfectly
        reversed answer separates — it still scores well on the first, and -1 on the second.

        Both are dropped at `k = 1`, with `time_captured`: one function has no order to correlate,
        and one expected function makes `time_captured` a second name for `hit_rate`.
        """
        weights = truth["weights"]
        found = [key for key in (resolve(entry, truth["index"]) for entry in answer) if key is not None]
        expected = metrics.tie_tolerant(found, weights, truth["ranked"][:k])
        scores: dict[str, float | None] = {
            "hit_rate": metrics.hit_rate(found, expected),
            "ndcg_macro": metrics.weighted_ndcg_at_k(found, weights, expected),
        }
        if k > 1:
            scores["time_captured"] = metrics.weight_captured(found, weights, expected)
            scores["somers_d"] = metrics.weight_rank_somers_d(found, weights)
        return scores, metrics.weighted_dcg_at_k(found, weights, expected)
