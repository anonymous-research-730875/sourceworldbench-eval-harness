"""Shared scoring for a task whose row holds many tests.

Absent questions are never derived from the others: `outcome`'s labels mechanically
yield `binary` and a true/false for `all_pass`, but the questions exist because
different prompts elicit different behavior.
"""

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, TypeVar

from sourceworldbench_eval_harness import metrics
from sourceworldbench_eval_harness.models.core.questions import QuestionModel
from sourceworldbench_eval_harness.schemas.status import BINARY, to_binary
from sourceworldbench_eval_harness.tasks.core.base import SAMPLE, fail
from sourceworldbench_eval_harness.tasks.status.base import StatusTask, StatusTaskConfig


class MultiTestTaskConfig(StatusTaskConfig):
    """Adds nothing yet; named so a multi-test-only setting has somewhere to land
    without every multi-test task changing base class to get it."""


Q = TypeVar("Q", bound=MultiTestTaskConfig)


class MultiTestStatusTask(StatusTask[Q], ABC):
    @property
    @abstractmethod
    def test_columns(self) -> tuple[str, str]:
        """The dataset columns holding this task's tests and their labels."""

    @property
    def per_test_questions(self) -> tuple[str, ...]:
        """Questions whose answer names tests, so can be cross-referenced. A task
        also asking about the instance as a whole narrows this."""
        return self.questions

    def generated_rows(self, model: QuestionModel) -> list[dict[str, Any]]:
        id_field = self.id_field
        tests_field, labels_field = self.test_columns  # fail on a bad column choice before downloading
        return [
            {
                id_field: row[id_field],
                **model.predict_single({"tests": list(row[tests_field]), "labels": list(row[labels_field])}),
            }
            for row in self.load_dataset_rows()
        ]

    @staticmethod
    def answer_tests(at: str, answer: Any) -> tuple[list[Any] | None, list[str]]:
        """The `{tests: [...]}` envelope both validation entry points open before they diverge; `None`
        meaning it could not be opened.

        `check_shape` and `check_answer` are reached by different paths and could otherwise drift
        into two different messages for one malformed answer.
        """
        if not isinstance(answer, dict):
            return None, [f"{at}: answer must be an object with a 'tests' list"]
        tests = answer.get("tests")
        if not isinstance(tests, list):
            return None, [f"{at}: 'tests' must be a list"]
        return tests, []

    def check_shape(self, where: str, question: str, answer: Any) -> list[str]:
        at = f"{where}/{question}"
        tests, malformed = self.answer_tests(at, answer)
        if tests is None:
            return malformed
        if not tests:
            return [f"{at}: 'tests' is empty"]
        allowed = self.schema.label_vocabulary.get(question)
        if allowed is None:
            return []  # a question whose answer carries no labels: the test list is all of it

        labels = answer.get("labels")
        if not isinstance(labels, list):
            return [f"{at}: 'labels' must be a list for {question}"]
        if len(labels) != len(tests):
            return [f"{at}: {len(tests)} tests but {len(labels)} labels"]
        bad = sorted({label for label in labels if label not in allowed})
        if bad:
            return [f"{at}: unknown label(s) {bad[:SAMPLE]}; allowed: {sorted(allowed)}"]
        return []

    def cross_check(self, instance_id: str, row: dict[str, Any], truth: Any) -> list[str]:
        problems: list[str] = []
        for question in (q for q in self.per_test_questions if q in row):
            problems.extend(self.check_answer(instance_id, question, row[question], truth))
        return problems

    @classmethod
    def check_answer(cls, instance_id: str, question: str, answer: Any, expected: dict[str, str]) -> list[str]:
        where = f"{instance_id}/{question}"
        tests, malformed = cls.answer_tests(where, answer)
        if tests is None:
            return malformed
        if len(set(tests)) != len(tests):
            duplicated = sorted({t for t, n in Counter(tests).items() if n > 1})
            return [f"{where}: duplicate test names, e.g. {duplicated[:SAMPLE]}"]
        unknown = [t for t in tests if t not in expected]
        if unknown:
            return [f"{where}: {len(unknown)} unknown test name(s), e.g. {unknown[:SAMPLE]}"]
        if len(tests) != len(expected):
            return [f"{where}: covers {len(tests)} of {len(expected)} tests ({len(expected) - len(tests)} missing)"]
        return []

    def reference(self) -> dict[str, dict[str, str]]:
        """`{instance_id: {test_name: label}}` — keyed by name, never position, since one tests column
        is not a positional subset of another.

        Labels are checked against the vocabulary first: `metrics.multiclass_confusion` indexes its
        matrix by the true label, so an unexpected status would be a `KeyError` mid-scoring instead
        of a message naming the rows.
        """
        tests_field, labels_field = self.test_columns
        allowed = set(self.schema.labels)

        reference: dict[str, dict[str, str]] = {}
        problems: list[str] = []
        unknown_labels: list[str] = []
        for instance_id, row in self.once_per_instance(self.load_dataset_rows(), problems):
            tests, labels = row[tests_field], row[labels_field]
            if len(tests) != len(labels):
                problems.append(f"{instance_id}: {len(tests)} tests but {len(labels)} labels")
                continue
            if len(set(tests)) != len(tests):
                problems.append(f"{instance_id}: duplicate test names in {tests_field}")
                continue
            outside = sorted({label for label in labels if label not in allowed})
            if outside:
                unknown_labels.append(f"{instance_id} {outside[:SAMPLE]}")
                continue
            reference[instance_id] = dict(zip(tests, labels, strict=True))

        if unknown_labels:
            problems.append(
                f"{len(unknown_labels)} row(s) carry a {labels_field} outside {list(self.schema.labels)}, "
                f"e.g. {unknown_labels[:SAMPLE]}"
            )
        fail(problems)
        return reference

    def score_question(
        self, question: str, rows: list[dict[str, Any]], reference: Any
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        return self.label_metrics(rows, reference, question)

    def label_metrics(
        self, rows: list[dict[str, Any]], reference: dict[str, dict[str, str]], question: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pooled rather than per-instance: macro-F1 divides by the classes present in
        that instance, and an instance can hold only PASSED, so the divisor collapses to
        1 and a single-class F1 becomes a perfect score."""
        pooled_true: list[str] = []
        pooled_pred: list[str] = []
        per_instance: list[dict[str, float | None]] = []

        for row in rows:
            truth = reference[row[self.id_field]]
            answer = row[question]
            expected = [truth[test] for test in answer["tests"]]
            if question == BINARY:
                expected = [to_binary(label) for label in expected]
            report = self.score_labels(expected, answer["labels"], question)
            per_instance.append({k: v for k, v in report.items() if isinstance(v, float | int | type(None))})
            pooled_true.extend(expected)
            pooled_pred.extend(answer["labels"])

        pooled = self.score_labels(pooled_true, pooled_pred, question)
        flat, details = self.flat_classification(pooled, len(pooled_true))
        # Reference only; not comparable, and never the headline.
        flat["per_instance_macro_f1"] = metrics.mean([s["macro_f1"] for s in per_instance])

        return flat, details


__all__ = ["MultiTestStatusTask", "MultiTestTaskConfig"]
