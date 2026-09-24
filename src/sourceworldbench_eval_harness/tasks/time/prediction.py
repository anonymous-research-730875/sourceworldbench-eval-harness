"""`sourceworldbench-time-prediction`: how long a workload takes to run.

Four questions from one number of seconds; only seconds has a ground-truth column,
the budget answers being derived. See `docs/TIME_PREDICTION_FORMAT.md`.
"""

from typing import Any, Literal

from pydantic import ConfigDict

from sourceworldbench_eval_harness import metrics
from sourceworldbench_eval_harness.models.core.questions import QuestionModel
from sourceworldbench_eval_harness.schemas.time import (
    BUDGETS,
    FACTORS,
    PREDICTION_QUESTIONS,
    SECONDS,
    STATE_FIELDS,
    TIME_FIELD,
    is_seconds,
    trim,
)
from sourceworldbench_eval_harness.tasks.core.base import SAMPLE, fail
from sourceworldbench_eval_harness.tasks.core.questions import QuestionTask, QuestionTaskConfig

__all__ = ["TimePrediction"]


class TimePredictionTaskConfig(QuestionTaskConfig):
    model_config = ConfigDict(extra="forbid")

    name: Literal["time_prediction"]


class TimePrediction(QuestionTask[TimePredictionTaskConfig]):
    name = "time_prediction"
    Config = TimePredictionTaskConfig

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
                        "seconds": row[TIME_FIELD],
                        **{field: row[field] for field in STATE_FIELDS if field in row},
                    }
                ),
            }
            for row in self._dataset_rows(id_field, TIME_FIELD)
        ]

    def check_shape(self, where: str, question: str, answer: Any) -> list[str]:
        at = f"{where}/{question}"
        if question == SECONDS:
            if not is_seconds(answer):
                return [f"{at}: answer must be a positive, finite number of seconds, not {answer!r}"]
            return []
        if isinstance(answer, bool):
            return []
        return [f"{at}: answer must be true or false — does it finish within {trim(BUDGETS[question])}s?"]

    def cross_check(self, instance_id: str, row: dict[str, Any], truth: Any) -> list[str]:
        """Every answer is judged without ground truth: the budgets are fixed thresholds
        and `seconds` is shape-checked, so there is nothing row-specific to reject."""
        return []

    def reference(self) -> dict[str, dict[str, Any]]:
        id_field = self.id_field
        reference: dict[str, dict[str, Any]] = {}
        problems: list[str] = []
        missing: list[str] = []

        rows = self._dataset_rows(id_field, TIME_FIELD)
        for instance_id, row in self.once_per_instance(rows, problems):
            seconds = row[TIME_FIELD]
            if not is_seconds(seconds):  # the same bar a submitted answer has to clear
                missing.append(instance_id)
                continue
            reference[instance_id] = {"seconds": seconds}

        if missing:
            problems.append(
                f"{len(missing)} row(s) have no usable {TIME_FIELD!r}, e.g. {missing[:SAMPLE]}; "
                "an answer key with missing values cannot be scored against"
            )
        fail(problems)
        return reference

    def score_question(
        self, question: str, rows: list[dict[str, Any]], reference: Any
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if question == SECONDS:
            return self.seconds_metrics(rows, reference), None
        return self.budget_metrics(question, rows, reference)

    def seconds_metrics(self, rows: list[dict[str, Any]], reference: Any) -> dict[str, Any]:
        """Scale-free per-instance error, plus the figures needing every instance at once:
        what error survives correcting an offset, what survives correcting the scale too,
        and whether the magnitudes track the real ones."""
        predicted = [float(row[SECONDS]) for row in rows]
        expected = [reference[row[self.id_field]]["seconds"] for row in rows]

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
            calibrated_log10_error=metrics.calibrated_log10_error(predicted, expected),
            log_slope=slope,
            log_intercept=intercept,
        )

    def budget_metrics(
        self, question: str, rows: list[dict[str, Any]], reference: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pooled binary classification over instances. `fits` is the positive class,
        so `fits_f1` reads as: does the model know what will come back quickly?"""
        budget = BUDGETS[question]
        labels = ("misses", "fits")
        pooled_true = ["fits" if reference[row[self.id_field]]["seconds"] <= budget else "misses" for row in rows]
        pooled_pred = ["fits" if row[question] else "misses" for row in rows]
        pooled = metrics.classification_report(pooled_true, pooled_pred, labels)
        return self.flat_classification(pooled, len(rows), count_key="instances")
