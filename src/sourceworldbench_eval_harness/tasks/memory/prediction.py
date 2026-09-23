"""`sourceworldbench-memory-prediction`: how much memory a workload allocates.

See `docs/MEMORY_PREDICTION_FORMAT.md`.
"""

from typing import Any, Literal

from pydantic import ConfigDict

from sourceworldbench_eval_harness import metrics
from sourceworldbench_eval_harness.models.core.questions import QuestionModel
from sourceworldbench_eval_harness.schemas.memory import (
    BUDGET_FIELDS,
    BUDGETS,
    BYTES,
    BYTES_FIELD,
    FACTORS,
    PREDICTION_QUESTIONS,
    STATE_FIELDS,
    is_bytes,
    spell_bytes,
    trim,
)
from sourceworldbench_eval_harness.tasks.core.base import SAMPLE, fail
from sourceworldbench_eval_harness.tasks.core.questions import QuestionTask, QuestionTaskConfig

__all__ = ["MemoryPrediction"]


class MemoryPredictionTaskConfig(QuestionTaskConfig):
    model_config = ConfigDict(extra="forbid")

    name: Literal["memory_prediction"]


class MemoryPrediction(QuestionTask[MemoryPredictionTaskConfig]):
    name = "memory_prediction"
    Config = MemoryPredictionTaskConfig

    @property
    def questions(self) -> tuple[str, ...]:
        return PREDICTION_QUESTIONS

    def generated_rows(self, model: QuestionModel) -> list[dict[str, Any]]:
        id_field = self.id_field
        return [
            {
                id_field: row[id_field],
                **model.predict_single(
                    {
                        "bytes": row[BYTES_FIELD],
                        **{field: row[field] for field in STATE_FIELDS if field in row},
                    }
                ),
            }
            for row in self._dataset_rows(id_field, BYTES_FIELD)
        ]

    def check_shape(self, where: str, question: str, answer: Any) -> list[str]:
        at = f"{where}/{question}"
        if question == BYTES:
            if not is_bytes(answer):
                return [f"{at}: answer must be a positive, finite number of bytes, not {answer!r}"]
            return []
        if isinstance(answer, bool):
            return []
        return [f"{at}: answer must be true or false — is it at most {spell_bytes(BUDGETS[question])}?"]

    def cross_check(self, instance_id: str, row: dict[str, Any], truth: Any) -> list[str]:
        """Every answer is judged without ground truth: the budgets are fixed thresholds
        and `bytes` is shape-checked, so there is nothing row-specific to reject."""
        return []

    def reference(self) -> dict[str, dict[str, Any]]:
        id_field = self.id_field
        reference: dict[str, dict[str, Any]] = {}
        problems: list[str] = []
        missing: list[str] = []
        misbudgeted: list[str] = []

        rows = self._dataset_rows(id_field, BYTES_FIELD)
        for instance_id, row in self.once_per_instance(rows, problems):
            size = row[BYTES_FIELD]
            if not is_bytes(size):  # the same bar a submitted answer has to clear
                missing.append(instance_id)
                continue
            mismatch = self._budget_mismatch(row, size)
            if mismatch:
                misbudgeted.append(f"{instance_id} ({mismatch})")
                continue
            reference[instance_id] = {"bytes": size}

        if missing:
            problems.append(
                f"{len(missing)} row(s) have no usable {BYTES_FIELD!r}, e.g. {missing[:SAMPLE]}; "
                "an answer key with missing values cannot be scored against"
            )
        if misbudgeted:
            problems.append(
                f"{len(misbudgeted)} row(s) answer a budget question differently than their "
                f"{BYTES_FIELD!r} does, e.g. {misbudgeted[:SAMPLE]}; the thresholds this harness "
                "scores against are not the ones the dataset was built with"
            )
        fail(problems)
        return reference

    def _budget_mismatch(self, row: dict[str, Any], size: float) -> str | None:
        """The dataset's own budget answers, as a guard: a revision that moved a threshold
        would otherwise silently mark right answers wrong."""
        for question, field in BUDGET_FIELDS.items():
            if field not in row or row[field] is None:
                continue
            if bool(row[field]) != (size <= BUDGETS[question]):
                return f"{field}={row[field]!r} but {BYTES_FIELD}={size!r}"
        return None

    def score_question(
        self, question: str, rows: list[dict[str, Any]], reference: Any
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if question == BYTES:
            return self.bytes_metrics(rows, reference), None
        return self.budget_metrics(question, rows, reference)

    def bytes_metrics(self, rows: list[dict[str, Any]], reference: Any) -> dict[str, Any]:
        """Scale-free, because the target spans several orders of magnitude. The debiased
        error and the fit need every instance at once, so neither has a per-instance
        companion."""
        predicted = [float(row[BYTES]) for row in rows]
        expected = [reference[row[self.id_field]]["bytes"] for row in rows]

        per_instance: list[dict[str, float | None]] = []
        for guess, truth in zip(predicted, expected, strict=True):
            scores: dict[str, float | None] = {"log10_error": metrics.log10_error(guess, truth)}
            for factor in FACTORS:
                scores[f"within_factor_{trim(factor)}"] = metrics.within_factor(guess, truth, factor)
            per_instance.append(scores)

        slope, intercept = metrics.log_fit(predicted, expected)
        return self.flat_aggregate(
            per_instance,
            debiased_log10_error=metrics.debiased_log10_error(predicted, expected),
            log_slope=slope,
            log_intercept=intercept,
        )

    def budget_metrics(
        self, question: str, rows: list[dict[str, Any]], reference: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """`fits` is the positive class, so `fits_f1` reads as: does it know what stays small?"""
        budget = BUDGETS[question]
        labels = ("exceeds", "fits")
        pooled_true = ["fits" if reference[row[self.id_field]]["bytes"] <= budget else "exceeds" for row in rows]
        pooled_pred = ["fits" if row[question] else "exceeds" for row in rows]
        pooled = metrics.classification_report(pooled_true, pooled_pred, labels)
        return self.flat_classification(pooled, len(rows), count_key="instances")
