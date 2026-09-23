"""`sourceworldbench-memory-hotspots`: which functions dominate a workload's memory use.

See `docs/MEMORY_HOTSPOT_FORMAT.md`.
"""

import logging
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import ConfigDict

from sourceworldbench_eval_harness import metrics
from sourceworldbench_eval_harness.models.core.questions import QuestionModel
from sourceworldbench_eval_harness.schemas.memory import (
    AXIS_FOR,
    FUNCTIONS_FIELD,
    HOTSPOT_AXES,
    HOTSPOT_QUESTIONS,
    HOTSPOTS_FIELDS,
    STATE_FIELDS,
    TOP_K,
    HotspotAxis,
    alias,
    lookup_index,
    resolve,
)
from sourceworldbench_eval_harness.tasks.core.base import SAMPLE, fail
from sourceworldbench_eval_harness.tasks.core.questions import QuestionTask, QuestionTaskConfig

__all__ = ["MemoryHotspot"]

# A row missing any of these cannot be scored, so reading them fails on it by name.
REQUIRED_FIELDS: tuple[str, ...] = (FUNCTIONS_FIELD, *HOTSPOTS_FIELDS)


class MemoryHotspotTaskConfig(QuestionTaskConfig):
    model_config = ConfigDict(extra="forbid")

    name: Literal["memory_hotspot"]


class MemoryHotspot(QuestionTask[MemoryHotspotTaskConfig]):
    name = "memory_hotspot"
    Config = MemoryHotspotTaskConfig

    @property
    def questions(self) -> tuple[str, ...]:
        return HOTSPOT_QUESTIONS

    @staticmethod
    def scoreable(row: Any) -> bool:
        """Every axis must be deep enough, not just the one asked about: `validate` wants each
        answered question to cover every reference instance, so an instance kept for one axis
        and dropped for another would read as a missing prediction."""
        return all(len(row[axis.hotspots_field]) >= axis.deepest for axis in HOTSPOT_AXES)

    def generated_rows(self, model: QuestionModel) -> list[dict[str, Any]]:
        id_field = self.id_field
        return [
            {
                id_field: row[id_field],
                **model.predict_single(
                    {
                        FUNCTIONS_FIELD: [dict(f) for f in row[FUNCTIONS_FIELD]],
                        **{field: [dict(h) for h in row[field]] for field in HOTSPOTS_FIELDS},
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
        """Keeps an answer inside the codebase rather than naming a built-in."""
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
            # A `key` never equals its own triple as a string, so one function under both
            # spellings gets past `check_shape` and leaves the answer really `k - 1` long —
            # fewer placements to get wrong. Only resolution makes the two one name again.
            if len(set(found)) != len(found):
                problems.append(
                    f"{instance_id}/{question}: the same function is listed more than once, "
                    f"under two different spellings"
                )
        return problems

    def reference(self) -> dict[str, dict[str, Any]]:
        """A profile too short for its deepest question is dropped rather than failing the run,
        since `top20` is genuinely unanswerable from twelve functions."""
        id_field = self.id_field
        reference: dict[str, dict[str, Any]] = {}
        problems: list[str] = []
        unscoreable: list[str] = []

        rows = self._dataset_rows(id_field, *REQUIRED_FIELDS)
        for instance_id, row in self.once_per_instance(rows, problems):
            if not self.scoreable(row):
                depths = ", ".join(f"{axis.name}={len(row[axis.hotspots_field])}" for axis in HOTSPOT_AXES)
                unscoreable.append(f"{instance_id} ({depths})")
                continue
            functions = [dict(f) for f in row[FUNCTIONS_FIELD]]
            index = lookup_index(functions)

            axes: dict[str, dict[str, Any]] = {}
            defects: list[str] = []
            for axis in HOTSPOT_AXES:
                weights = {f["key"]: f[axis.weight_field] for f in functions}
                ranked = [h["key"] for h in row[axis.hotspots_field]]
                defects.extend(self.ranking_defects(instance_id, axis, ranked, weights))
                axes[axis.name] = {"ranked": ranked, "weights": weights}
            if defects:
                problems.extend(defects)
                continue
            reference[instance_id] = {"index": index, "axes": axes}
        if unscoreable:
            logging.warning(
                "[reference] %d instance(s) profiled too few function(s) on some axis, so no question "
                "is scored on them: %s",
                len(unscoreable),
                ", ".join(unscoreable[:SAMPLE]),
            )
        fail(problems)
        return reference

    @staticmethod
    def ranking_defects(instance_id: str, axis: HotspotAxis, ranked: list[str], weights: dict[str, float]) -> list[str]:
        """`ndcg_macro` caps at 1.0 only while ground truth is the heaviest functions in
        descending order. Break that and it scores above 1.0, which the leaderboard silently
        drops — so fail the run here instead."""
        field = axis.hotspots_field
        unknown = [key for key in ranked if key not in weights]
        if unknown:
            return [
                f"{instance_id}: {len(unknown)} {field} entry(s) are absent from "
                f"{FUNCTIONS_FIELD}, e.g. {unknown[:SAMPLE]}"
            ]
        if len(set(ranked)) != len(ranked):
            return [f"{instance_id}: {field} lists the same function more than once"]
        # By weight, not by key, so functions tied on size may be ranked either way round.
        heaviest = sorted(weights.values(), reverse=True)[: len(ranked)]
        if [weights[key] for key in ranked] != heaviest:
            return [
                f"{instance_id}: {field} is not the {len(ranked)} costliest function(s) in "
                f"descending {axis.weight_field} order, which ndcg_macro needs to cap at 1.0"
            ]
        return []

    def score_question(
        self, question: str, rows: list[dict[str, Any]], reference: Any
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """`*_scored` carries coverage, since a metric is undefined where there is nothing to
        find. `ndcg_micro` cannot be averaged per instance, so it comes through `across`."""
        axis = AXIS_FOR[question]
        k = TOP_K[question]
        scored = [self.retrieval_scores(row[question], reference[row[self.id_field]], axis, k) for row in rows]
        pooled = metrics.micro_ndcg([gains for _, gains in scored])
        return self.flat_aggregate([per_instance for per_instance, _ in scored], ndcg_micro=pooled), None

    @staticmethod
    def retrieval_scores(
        answer: Sequence[Any], truth: dict[str, Any], axis: HotspotAxis, k: int
    ) -> tuple[dict[str, float | None], tuple[float, float]]:
        """Four scores because each is blind to something: a reversed answer still scores well
        on `ndcg_macro` and -1 on `somers_d`, and a single heavy hit separates `hit_rate` from
        `bytes_captured`.

        At `k = 1` there is no order to correlate and `bytes_captured` would restate `hit_rate`,
        so both are dropped.
        """
        ranking = truth["axes"][axis.name]
        weights = ranking["weights"]
        found = [key for key in (resolve(entry, truth["index"]) for entry in answer) if key is not None]
        expected = metrics.tie_tolerant(found, weights, ranking["ranked"][:k])
        scores: dict[str, float | None] = {
            "hit_rate": metrics.hit_rate(found, expected),
            "ndcg_macro": metrics.weighted_ndcg_at_k(found, weights, expected),
        }
        if k > 1:
            scores["bytes_captured"] = metrics.weight_captured(found, weights, expected)
            scores["somers_d"] = metrics.weight_rank_somers_d(found, weights)
        return scores, metrics.weighted_dcg_at_k(found, weights, expected)
