# Anchored test-status prediction format

[`anonymous-research-730875/sourceworldbench-test-status-anchored`](https://huggingface.co/datasets/anonymous-research-730875/sourceworldbench-test-status-anchored),
`test` split. Configs pin `load.revision: v0.2.0`.

Each row is one agentic patch and a **fixed subset of tests**. Sibling task:
[`SINGLE_TEST_STATUS_FORMAT.md`](SINGLE_TEST_STATUS_FORMAT.md) asks the same about one test
at a time.

**The anchor: every one of those tests PASSED on the base commit.** So a non-PASSED label is
not a test that was already broken — it is a regression the patch introduced. You are being
asked what the patch broke.

**One file, answering as many questions as you have run** — one evaluation scores all of them
and produces one leaderboard row.

## The questions

Each key is the dataset's prompt name minus the `Q-` prefix, with dashes as underscores.

| Dataset column | Asks | Key | Answer |
|---|---|---|---|
| `Q-binary` | Label each test PASSED / NOT_PASSED | `binary` | `tests` + aligned `labels` |
| `Q-outcome` | Label each test PASSED / FAILED / ERROR | `outcome` | `tests` + aligned `labels` |
| `Q-all-pass` | **Are all of these tests expected to pass?** | `all_pass` | a single `true` / `false` |

## The file

JSONL, one object per dataset instance:

```json
{"instance_id": "pandas__b73c38e2__prediction_oh_gpt51",
 "binary": {"tests": ["..."], "labels": ["PASSED", "NOT_PASSED", "..."]},
 "outcome": {"tests": ["..."], "labels": ["PASSED", "FAILED", "..."]},
 "all_pass": true}
```

`instance_id` is always required and echoed verbatim; at least one question key must be
present, and absent keys are simply not scored. `tests` holds names copied verbatim from the
dataset's `tests` column, and `labels` is aligned element-wise with it.

`SKIPPED` and `OTHER` are filtered out, so three classes rather than five.

Four things that are easy to get wrong:

- **`all_pass` is the one answer that is not a list**, being asked about the instance rather
  than a test: `"all_pass": true` — not an object, not the string `"true"`.
- Predictions are matched **by test name, never by position**, so `tests` may be in any order.
- Every question present must cover **every** test in the instance — partial coverage is
  rejected.
- **Answer each question with its own run.** `outcome`'s labels mechanically give you
  `binary` and a true/false for `all_pass`, but deriving them makes three columns report one
  capability measured once.

**Some instances contain at least one regression and the rest are clean.** That split is
what `all_pass` asks about, and it is why `all_pass` is near balanced while `binary` and
`outcome` remain minority-class problems.

The `function_called.json` companion is not published for this dataset yet. When it is, it
plugs in as it does for `single_test_status`: a `task.calls_file` pointing at your local copy,
handed to models as context and never scored.

## Validation

Every problem is reported at once.

- **`predict` — shape only, no dataset needed:** missing `instance_id`, a row answering no
  question, misaligned `labels`, a label outside the vocabulary, a non-boolean `all_pass`.
  *Your file is malformed.*
- **`evaluate` — against ground truth:** unknown or duplicate `instance_id`, unknown or
  duplicate test names, partial coverage of an instance, missing instances (unless
  `task.require_full_coverage: false`). *Your file describes a different dataset.*

## Scoring

**Non-PASSED is the positive class throughout.** The great majority of tests pass, so
anything scored against the PASSED class saturates.

| Question | Reported |
|---|---|
| `binary` | pooled `binary_macro_f1`, `binary_passed_f1`, `binary_not_passed_f1`, `binary_mcc`, `binary_balanced_accuracy`, `binary_accuracy`, `binary_tests` |
| `outcome` | as `binary` over three classes, minus `mcc` (two classes only) |
| `all_pass` | `all_pass_macro_f1`, `all_pass_passed_f1`, `all_pass_not_passed_f1`, `all_pass_mcc`, `all_pass_accuracy`, **`all_pass_instances`** |

`binary` and `outcome` are **pooled** — every test from every instance in one confusion
matrix. `binary_per_instance_macro_f1` exists for reference only and is **not comparable**:
a clean instance contains only PASSED, where macro-F1 divides by one class and a
single-class F1 becomes a perfect score.

**`all_pass` pools over instances, not tests** — hence `all_pass_instances`. It is scored in
the `binary` vocabulary with `PASSED` meaning "nothing broke", so **`all_pass_not_passed_f1`
is the number that matters**: how reliably the model notices that a patch broke something.

### A trap when reading the numbers

**On `binary`/`outcome` the constant beats random here**, unlike the single-test task: with
so few tests failing, guessing costs more than it gains, so `all_passed` is the baseline to
beat. On `all_pass` the two constants split and neither is informative — `all_passed` answers
"nothing broke" everywhere so `all_pass_not_passed_f1` is 0.0, while `random_status`, drawing
each of an instance's tests from three statuses, essentially never finds an all-PASSED
instance, so it answers "something broke" every time and `all_pass_passed_f1` is 0.0. Both
report `all_pass_mcc` as `n/a`. **A model must beat the constant's `all_pass` macro-F1 *and*
produce a defined MCC.**

## Running it

```bash
uv run python -m sourceworldbench_eval_harness.prediction.predict  configs/baselines/status/anchored_all_passed.yaml
uv run python -m sourceworldbench_eval_harness.evaluation.evaluate configs/baselines/status/anchored_all_passed.yaml
uv run python -m sourceworldbench_eval_harness.leaderboard.push    configs/baselines/status/anchored_all_passed.yaml
```

`predict.model` takes `oracle`, `all_passed` or `random_status` — one config per model in
`configs/baselines/status/`; narrow questions with `params: { questions: [outcome, all_pass] }`.

**Your own predictions:** copy `configs/templates/status/anchored_external.yaml`, point
`predict.external_input` at your `.jsonl`, run `predict` → `evaluate`. That path is not
resolved under `shared.output_root` (though `~` expands), so your file can stay where it is.
The `from_file` model is the default, since the file *is* the submission; pairing
`external_input` with a generating baseline is an error. Two `params`: `questions` keeps only
those answers (a row answering none of them fails the run), and `question_fields`
(e.g. `{all_pass: no_regressions}`) names the column holding a question when your file calls
it something else.

What lands at `predict.output` is **not a copy of your file** — only `instance_id` plus the
answers under canonical names, every other column dropped. That extract is what `evaluate`
scores.
