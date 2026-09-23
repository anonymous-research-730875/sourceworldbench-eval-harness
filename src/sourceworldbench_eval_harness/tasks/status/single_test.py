"""`sourceworldbench-single-test-status`: one patched repo state and one test per row, asked for that test's
`binary` and `outcome` label. An answer is the label itself, not a list.

The class balance is mild enough for two floors rather than one: `all_passed` on accuracy
and `random_status` on macro F1. See `docs/SINGLE_TEST_STATUS_FORMAT.md`.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ConfigDict

from sourceworldbench_eval_harness.models.core.questions import QuestionModel
from sourceworldbench_eval_harness.schemas.status import BINARY, SINGLE_TEST_STATUS, to_binary
from sourceworldbench_eval_harness.tasks.core.base import SAMPLE, fail
from sourceworldbench_eval_harness.tasks.status.base import StatusTask, StatusTaskConfig

__all__ = ["SingleTestStatus"]

TEST_FIELD = "test"

# The answer key: the test's status after the patch, which is the question this task
# asks. Not configurable — the eval is bound to this dataset, so the column is a
# property of it rather than a choice.
LABEL_FIELD = "label"

# Repository state handed to a model under these names. Baselines ignore them; a
# real predictor needs them to answer at all.
STATE_FIELDS: tuple[str, ...] = ("repo", "base_commit", "patch", "container", "command")


class SingleTestStatusTaskConfig(StatusTaskConfig):
    model_config = ConfigDict(extra="forbid")

    name: Literal["single_test_status"]
    # `{instance_id: [called function, ...] | null}`. Never scored — context a
    # predictor may use — and read from where it is, not under `shared.output_root`.
    calls_file: str | None = None


class SingleTestStatus(StatusTask[SingleTestStatusTaskConfig]):
    name = "single_test_status"
    Config = SingleTestStatusTaskConfig
    schema = SINGLE_TEST_STATUS

    @property
    def carried_fields(self) -> tuple[str, ...]:
        """The test name travels with the id, so `cross_check` can verify it."""
        return self.id_field, TEST_FIELD

    def generated_rows(self, model: QuestionModel) -> list[dict[str, Any]]:
        id_field = self.id_field
        calls = self._calls()  # read before downloading, so a bad path fails fast
        dataset_rows = self._dataset_rows(id_field, LABEL_FIELD, TEST_FIELD)
        rows = [
            {
                id_field: row[id_field],
                TEST_FIELD: row[TEST_FIELD],
                **model.predict_single(self._model_row(row, calls)),
            }
            for row in dataset_rows
        ]
        if calls is not None:
            self._log_calls_coverage(calls, {row[id_field] for row in rows})
        return rows

    def check_shape(self, where: str, question: str, answer: Any) -> list[str]:
        """The wrapped `{tests, labels}` shape is the multi-test task's, and the
        natural mistake to make coming from there, so it earns its own message."""
        at = f"{where}/{question}"
        allowed = self.schema.label_vocabulary[question]
        if isinstance(answer, dict):
            return [f'{at}: the answer is the label itself, not an object — write "{question}": "{allowed[0]}"']
        if not isinstance(answer, str):
            return [f"{at}: label must be a string, got {type(answer).__name__}"]
        if answer not in allowed:
            return [f"{at}: unknown label {answer!r}; allowed: {list(allowed)}"]
        return []

    def reference(self) -> dict[str, dict[str, str]]:
        """`{instance_id: {"label": ..., "test": ...}}`.

        Labels are checked against the vocabulary first: `metrics.multiclass_confusion`
        indexes its matrix by the true label, so an unexpected status would otherwise
        be a `KeyError` mid-scoring. The test name comes along for `cross_check`.
        """
        id_field = self.id_field
        labels = self.schema.labels
        rows = self._dataset_rows(id_field, LABEL_FIELD, TEST_FIELD)

        reference: dict[str, dict[str, str]] = {}
        problems: list[str] = []
        missing_label: list[str] = []
        unknown_label: list[str] = []
        for instance_id, row in self.once_per_instance(rows, problems):
            label = row[LABEL_FIELD]
            if label is None:
                missing_label.append(instance_id)
                continue
            if label not in labels:
                unknown_label.append(f"{instance_id} ({label!r})")
                continue
            reference[instance_id] = {"label": label, TEST_FIELD: row[TEST_FIELD]}

        if missing_label:
            problems.append(
                f"{len(missing_label)} row(s) have no {LABEL_FIELD!r}, e.g. {missing_label[:SAMPLE]}; "
                "an answer key with missing values cannot be scored against"
            )
        if unknown_label:
            problems.append(
                f"{len(unknown_label)} row(s) carry a {LABEL_FIELD} outside {list(labels)}, "
                f"e.g. {unknown_label[:SAMPLE]}"
            )
        fail(problems)
        return reference

    def cross_check(self, instance_id: str, row: dict[str, Any], truth: Any) -> list[str]:
        """Optional, and the only check that catches a file built against another
        revision: the ids overlap, so it would otherwise score against different tests.
        """
        expected_test = truth[TEST_FIELD]
        submitted_test = row.get(TEST_FIELD)
        if submitted_test is not None and submitted_test != expected_test:
            return [
                f"{instance_id}: {TEST_FIELD} is {submitted_test!r}, but this instance holds "
                f"{expected_test!r} — the file was built against different rows"
            ]
        return []

    def score_question(
        self, question: str, rows: list[dict[str, Any]], reference: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pooled is the only option: one test per instance means a per-instance figure
        would be computed over a single label. The count is still named `tests`, which
        is what the leaderboard reads to caption a question's panel."""
        pooled_true: list[str] = []
        pooled_pred: list[str] = []
        for row in rows:
            label = reference[row[self.id_field]]["label"]
            pooled_true.append(to_binary(label) if question == BINARY else label)
            pooled_pred.append(row[question])

        pooled = self.score_labels(pooled_true, pooled_pred, question)
        return self.flat_classification(pooled, len(pooled_true))

    def _model_row(self, row: Any, calls: dict[str, list[str] | None] | None) -> dict[str, Any]:
        """Generic keys, so no model names a dataset column."""
        return {
            "test": row[TEST_FIELD],
            "label": row[LABEL_FIELD],
            "calls": calls.get(row[self.id_field]) if calls else None,
            **{field: row[field] for field in STATE_FIELDS if field in row},
        }

    def _calls(self) -> dict[str, list[str] | None] | None:
        """Read leniently: the file is known to be incomplete, a null meaning the
        extraction failed, so a missing or null entry is a logged gap rather than a
        failed run. The *shape* is still checked — accepting a different file quietly
        would leave every model context-free with no sign why.
        """
        configured = self.settings.calls_file
        if not configured:
            return None
        path = Path(configured).expanduser()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"task.calls_file not found: {path}") from None

        if not isinstance(raw, dict):
            raise ValueError(
                f"{path} must be a JSON object mapping {self.id_field} to a list of calls or null; "
                f"got {type(raw).__name__}"
            )
        problems = [
            f"{key!r}: expected a list of strings or null, got {type(value).__name__}"
            for key, value in raw.items()
            if value is not None and not (isinstance(value, list) and all(isinstance(call, str) for call in value))
        ]
        fail(problems[:SAMPLE])
        logging.info("[predict] calls file %s (%d entr(y|ies))", path, len(raw))
        return cast(dict[str, list[str] | None], raw)

    @staticmethod
    def _log_calls_coverage(calls: dict[str, list[str] | None], instance_ids: set[str]) -> None:
        with_calls = sum(1 for instance_id in instance_ids if calls.get(instance_id) is not None)
        null = sum(1 for instance_id in instance_ids if instance_id in calls and calls[instance_id] is None)
        logging.info(
            "[predict] calls: %d of %d instance(s) have a list, %d null, %d absent",
            with_calls,
            len(instance_ids),
            null,
            len(instance_ids) - with_calls - null,
        )
        unknown = sorted(set(calls) - instance_ids)
        if unknown:
            logging.warning(
                "[predict] calls file has %d id(s) not in the dataset, ignored, e.g. %s",
                len(unknown),
                unknown[:SAMPLE],
            )
