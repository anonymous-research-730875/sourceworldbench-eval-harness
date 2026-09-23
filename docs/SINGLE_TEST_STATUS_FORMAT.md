# Single-test-status prediction format

[`anonymous-research-730875/sourceworldbench-single-test-status`](https://huggingface.co/datasets/anonymous-research-730875/sourceworldbench-single-test-status),
`test` split. Configs pin `load.revision: v0.2.0`.

Each row is **one agentic patch and one test**; the question is what that test does once the
patch is applied. Sibling task:
[`TEST_STATUS_ANCHORED_FORMAT.md`](TEST_STATUS_ANCHORED_FORMAT.md) asks the same about many
tests at once.

**One file, answering as many questions as you have run** — one evaluation scores all of them
and produces one leaderboard row.

## The file

JSONL, one object per dataset instance:

```json
{"instance_id": "pandas__b73c38e2__prediction_oh_gpt52", "binary": "NOT_PASSED", "outcome": "ERROR"}
```

| Field | Required | Meaning |
|---|---|---|
| `instance_id` | always | Must match the dataset; echo verbatim. |
| `binary` / `outcome` | at least one | The answer, as the label itself. Absent keys are not scored. |
| `test` | optional | Never scored, but checked against the dataset when present. |

| Dataset column | Asks | Key | Vocabulary |
|---|---|---|---|
| `Q-binary` | Is the test PASSED or not? | `binary` | `PASSED`, `NOT_PASSED` |
| `Q-outcome` | What is the test's status? | `outcome` | `PASSED`, `FAILED`, `ERROR` |

`SKIPPED` and `OTHER` are filtered out, so three classes rather than five, and `label`
(status after the patch) is the answer key and is not configurable. v0.2.0 adds
`single_test_state_id`, exactly `instance_id` + `__` + `test`; nothing reads it, so a
submission need not carry it.

Three things that are easy to get wrong:

- **The answer is the label string, not an object.** A row holds one test, so there is
  nothing to align: `"outcome": "FAILED"`, not `{"tests": [...], "labels": [...]}`. That
  wrapped shape belongs to the multi-test task.
- **Include `test` if you can.** It is the only check that catches a file built against a
  different revision — `instance_id`s overlap across revisions, so without it a stale
  submission scores silently against different tests.
- **Answer each question with its own run.** `outcome` mechanically gives you `binary`, but
  deriving it makes both columns report one capability measured once.

## Optional call list

`task.calls_file: ~/function_called.json` maps each instance to the functions its test calls
(`{id: list | null}`; `null` means extraction failed). **Nothing is scored from it** — it is
context handed to models at predict time. Missing and null entries are logged rather than
fatal, but the file's shape is checked, since a malformed one would leave every model
context-free with no sign why.

## Validation

Every problem is reported at once.

- **`predict` — shape only, no dataset needed:** missing `instance_id`, a row answering no
  question, a wrapped answer, a non-string label, a label outside the vocabulary. *Your file
  is malformed.*
- **`evaluate` — against ground truth:** unknown or duplicate `instance_id`, a `test` that is
  not this instance's, missing instances (unless `task.require_full_coverage: false`). *Your
  file describes a different dataset.*

## Scoring

Both questions are **pooled** multiclass classification: every answered instance in one
confusion matrix, macro F1 computed once. Pooling is the only option here — one test per
instance means a per-instance figure would read 1.0 or 0.0 for every row.

Metrics are flat columns named `{question}_{metric}` (the leaderboard turns each column into
a metric); per-class tables and the confusion matrix go under `details`. Per question:
`macro_f1`, per-class F1s, `macro_precision`, `macro_recall`, `balanced_accuracy`
(= `macro_recall` renamed), `accuracy`, `tests`, plus `mcc` for `binary` only (defined for
two classes). Macro figures average only over classes with support.

**Read against two floors, not one.** The balance is milder here than on the anchored task,
which changes which trivial baseline is hardest to beat. `random_status` predicts minority
classes sometimes and so beats `all_passed` on **macro F1**; `all_passed` never does (scoring
`not_passed_f1` and `failed_f1` of exactly 0.0) but wins on **accuracy**. A model must clear
the constant on accuracy *and* random on macro F1 — beating one is not evidence of anything.
`all_passed`'s `binary_mcc` is `n/a`: one predicted class has no correlation to report.

`oracle` must come out at 1.0 on every metric. Run it before submitting, to confirm your
config and revision line up.

## Running it

```bash
uv run python -m sourceworldbench_eval_harness.prediction.predict  configs/baselines/status/single_test_all_passed.yaml
uv run python -m sourceworldbench_eval_harness.evaluation.evaluate configs/baselines/status/single_test_all_passed.yaml
uv run python -m sourceworldbench_eval_harness.leaderboard.push    configs/baselines/status/single_test_all_passed.yaml
```

`predict.model` takes `oracle`, `all_passed` or `random_status` — one config per model in
`configs/baselines/status/`; narrow questions with `params: { questions: [outcome] }`.

**Your own predictions:** copy `configs/templates/status/single_test_external.yaml`, point
`predict.external_input` at your `.jsonl`, run `predict` → `evaluate`. That path is not
resolved under `shared.output_root` (though `~` expands), so your file can stay where it is.
The `from_file` model is the default, since the file *is* the submission; pairing
`external_input` with a generating baseline is an error. Two `params`: `questions` keeps only
those answers (a row answering none of them fails the run), and `question_fields`
(e.g. `{outcome: predicted_label}`) names the column holding a question when your file calls
it something else.

What lands at `predict.output` is **not a copy of your file** — only `instance_id`, `test`
when supplied, and the labels under canonical names, every other column dropped. That extract
is what `evaluate` scores, so it is worth a look after your first `predict`.
