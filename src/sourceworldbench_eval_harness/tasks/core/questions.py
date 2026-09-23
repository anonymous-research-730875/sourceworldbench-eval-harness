"""Shared pipeline steps for any task that asks a fixed set of questions per instance.

A question key is the dataset's own name for it and metrics are reported as
`{question}_{metric}`, which is how the leaderboard groups a tab — so a key must be a plain
lowercase identifier, and no key may prefix another's metric column.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, TypeVar, cast

from pydantic import Field

from sourceworldbench_eval_harness import metrics
from sourceworldbench_eval_harness.config import TaskConfig
from sourceworldbench_eval_harness.models.core.questions import QuestionModel
from sourceworldbench_eval_harness.tasks.core.base import SAMPLE, Groups, Task, fail


class QuestionTaskConfig(TaskConfig):
    id_field: str = Field(default="instance_id", min_length=1)
    require_full_coverage: bool = True


Q = TypeVar("Q", bound=QuestionTaskConfig)


class QuestionTask(Task[Q], ABC):
    @property
    @abstractmethod
    def questions(self) -> tuple[str, ...]:
        """Every question this task asks, in report order."""

    def configured_model(self, default: str | None = None) -> QuestionModel:
        return cast(QuestionModel, super().configured_model(default))

    @property
    def id_field(self) -> str:
        return self.settings.id_field

    @property
    def carried_fields(self) -> tuple[str, ...]:
        """Columns copied from a submission row alongside its answers.

        A field the file omits is left absent rather than set to None, so
        `validate_shape` names it.
        """
        return (self.id_field,)

    @abstractmethod
    def check_shape(self, where: str, question: str, answer: Any) -> list[str]:
        """Problems with one question's answer, judged without the dataset."""

    @abstractmethod
    def cross_check(self, instance_id: str, row: dict[str, Any], truth: Any) -> list[str]:
        """Problems with one row that only ground truth can reveal."""

    @abstractmethod
    def score_question(
        self, question: str, rows: list[dict[str, Any]], reference: Any
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """One question's metrics, plus its `details` entry when it has one."""

    @abstractmethod
    def generated_rows(self, model: QuestionModel) -> list[dict[str, Any]]:
        """Prediction rows built by running `model` over the dataset."""

    def predict(self, external: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if external is not None:
            model = self.configured_model("from_file")
            if not model.reads_submission:
                raise ValueError(
                    f"predict.model {model.name!r} builds its own predictions and would ignore "
                    f"predict.external_input; drop one, or use `from_file` to score the file"
                )
            carried = self.carried_fields
            rows = [
                {field: row[field] for field in carried if field in row} | answer
                for row, answer in zip(external, model.predict(external), strict=True)
            ]
        else:
            model = self.configured_model()
            rows = self.generated_rows(model)
        logging.info("[predict] model=%s (%d row(s))", model.name, len(rows))
        logging.info("[predict] questions=%s", ",".join(q for q in self.questions if any(q in row for row in rows)))
        return rows

    def validate_shape(self, rows: list[dict[str, Any]]) -> None:
        """Checks needing only the submitted file: a failure means it is malformed."""
        id_field = self.id_field
        questions = self.questions
        problems: list[str] = []
        for index, row in enumerate(rows):
            where = row.get(id_field, f"row {index}")
            if id_field not in row:
                problems.append(f"row {index}: missing {id_field!r}")
                continue
            present = [question for question in questions if question in row]
            if not present:
                problems.append(f"{where}: no question answers ({', '.join(questions)})")
                continue
            for question in present:
                problems.extend(self.check_shape(where, question, row[question]))
        fail(problems)

    def validate(self, rows: list[dict[str, Any]], reference: Any) -> None:
        """Checks needing ground truth: a failure means the file describes another dataset."""
        problems: list[str] = []
        seen: set[str] = set()
        id_field = self.id_field
        questions = self.questions

        for index, row in enumerate(rows):
            instance_id = row.get(id_field)
            if instance_id is None:
                problems.append(f"row {index}: missing {id_field!r}")
                continue
            if instance_id in seen:
                problems.append(f"{instance_id}: duplicate row for this instance")
                continue
            seen.add(instance_id)
            if instance_id not in reference:
                problems.append(f"{instance_id}: not present in the dataset")
                continue
            if not any(question in row for question in questions):
                problems.append(f"{instance_id}: no question answers ({', '.join(questions)})")
                continue
            problems.extend(self.cross_check(instance_id, row, reference[instance_id]))

        answered = {question for row in rows for question in questions if question in row}
        for question in (question for question in questions if question in answered):
            covered = {row[id_field] for row in rows if question in row and id_field in row}
            absent = sorted(set(reference) - covered)
            if absent and self.settings.require_full_coverage:
                problems.append(
                    f"{question}: {len(absent)} dataset instance(s) have no prediction, e.g. {absent[:SAMPLE]}"
                )
        fail(problems)

    def score(self, rows: list[dict[str, Any]], reference: Any) -> Groups:
        groups: Groups = {}
        details: dict[str, Any] = {}

        for question in self.questions:
            answered = [row for row in rows if question in row]
            if not answered:
                continue
            groups[question], detail = self.score_question(question, answered, reference)
            if detail is not None:
                details[question] = detail

        summary: dict[str, Any] = {
            "instances": len(rows),
            # Declared order, not sorted: the names carry no ordering of their own.
            "questions_answered": [q for q in self.questions if any(q in row for row in rows)],
        }
        if details:
            summary["details"] = details
        groups[None] = summary
        return groups

    def _dataset_rows(self, *required: str) -> Any:
        """A column the dataset does not have would otherwise surface as a `KeyError`
        on an opaque row rather than naming the column and what is available."""
        rows = self.load_dataset_rows()
        columns = getattr(rows, "column_names", None)
        if columns is not None:
            missing = [field for field in required if field not in columns]
            if missing:
                raise ValueError(
                    f"{self.cfg.load.name} ({self.cfg.load.split}) has no column(s) {missing}; "
                    f"available: {sorted(columns)}"
                )
        return rows

    def once_per_instance(
        self, rows: Iterable[dict[str, Any]], problems: list[str]
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Dataset rows paired with their instance id, a repeat reported rather than silently
        overwriting the row already keyed under it.

        Every task's `reference` needs this guard, and four hand-written copies could drift. An id
        repeated after its first row failed some later check still counts as duplicated.
        """
        seen: set[str] = set()
        id_field = self.id_field
        for row in rows:
            instance_id = row[id_field]
            if instance_id in seen:
                problems.append(f"dataset contains duplicate {id_field} {instance_id!r}")
                continue
            seen.add(instance_id)
            yield instance_id, row

    @staticmethod
    def flat_aggregate(per_instance: Sequence[dict[str, float | None]], **across: Any) -> dict[str, Any]:
        """Macro-averaged columns with the instance count last, since the metrics file
        and `evaluate`'s tables follow key order and a count reads as a footer.

        `across` carries a figure computed over every instance at once rather than
        averaged over them one at a time, which is why it cannot come from `aggregate`.
        """
        aggregated = metrics.aggregate(per_instance)
        flat: dict[str, Any] = {key: value for key, value in aggregated.items() if key != "instances"}
        flat.update(across)
        flat["instances"] = aggregated["instances"]
        return flat

    @staticmethod
    def flat_classification(
        pooled: dict[str, Any], count: int, count_key: str = "tests"
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Flat columns plus a `details` entry, since the leaderboard turns every
        column into a metric and cannot show a nested table.

        `count_key` names the unit pooled: tests, or instances for a question asked
        about a whole instance.
        """
        flat: dict[str, Any] = {
            "macro_f1": pooled["macro_f1"],
            "macro_precision": pooled["macro_precision"],
            "macro_recall": pooled["macro_recall"],
            "balanced_accuracy": pooled["balanced_accuracy"],
            "accuracy": pooled["accuracy"],
            count_key: count,
        }
        if "mcc" in pooled:  # two-class vocabularies only
            flat["mcc"] = pooled["mcc"]
        for label, scores in pooled["per_class"].items():
            flat[f"{label.lower()}_f1"] = scores["f1"]
        return flat, {"per_class": pooled["per_class"], "confusion": pooled["confusion"]}


__all__ = ["QuestionTask", "QuestionTaskConfig"]
