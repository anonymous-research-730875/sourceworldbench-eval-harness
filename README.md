# sourceworldbench-eval

Minimal eval harness

## Install

```bash
uv sync --group dev
cp .env.example .env   # set HF_TOKEN if needed
```

## Run

```bash
uv run python -m sourceworldbench_eval_harness.prediction.predict  configs/baselines/status/single_test_all_passed.yaml
uv run python -m sourceworldbench_eval_harness.evaluation.evaluate configs/baselines/status/single_test_all_passed.yaml
uv run python -m sourceworldbench_eval_harness.leaderboard.push    configs/baselines/status/single_test_all_passed.yaml
```

Each step reads one top-level YAML key, writes an artifact, and drops a
`<result>.config.yaml` sidecar with the full config that produced it.

One prediction file answers as many questions as you have run, and one evaluation
scores all of them into flat `binary_macro_f1`, `outcome_failed_f1` columns — one
leaderboard row per model rather than one per question.

### Single-test-status task

`anonymous-research-730875/sourceworldbench-single-test-status` asks the same question one test at a
time: each row is an agentic patch plus a single test, and the answer is that
test's status in the patched state. Two scored questions —
`PASSED`/`NOT_PASSED` (`binary`) and `PASSED`/`FAILED`/`ERROR` (`outcome`).

A submission is one row per instance carrying the label itself per question:

```json
{"instance_id": "pandas__b73c38e2__prediction_oh_gpt52", "binary": "NOT_PASSED", "outcome": "ERROR"}
```

```bash
uv run python -m sourceworldbench_eval_harness.prediction.predict  configs/baselines/status/single_test_all_passed.yaml
uv run python -m sourceworldbench_eval_harness.evaluation.evaluate configs/baselines/status/single_test_all_passed.yaml
```

One setting is specific to it: `calls_file` optionally points at a local
`function_called.json`, `{instance_id: [call, ...] | null}`, which is handed to
models as context and never scored. The answer key is the dataset's `label` column
and is not configurable.

The class balance is milder here than on the anchored task, so there are **two
trivial floors, not one**: `all_passed` bounds accuracy while `random_status` bounds
macro F1 above the constant, since guessing predicts the minority classes sometimes
and a constant never does. A model has to clear both. The format, the metrics and
both floors are documented in
[`docs/SINGLE_TEST_STATUS_FORMAT.md`](docs/SINGLE_TEST_STATUS_FORMAT.md).

### Anchored test-status task

`anonymous-research-730875/sourceworldbench-test-status-anchored` asks about a fixed subset of tests
per row, **all of which passed on the base commit** — so a non-PASSED label is a
regression the patch introduced, not a test that was already broken. It uses the same
three statuses as the single-test task.

It carries three prompts. Two are the familiar ones, and the third is asked about
the instance rather than a test:

| Dataset column | Asks | Key |
|---|---|---|
| `Q-binary` | `PASSED` / `NOT_PASSED` per test | `binary` |
| `Q-outcome` | `PASSED` / `FAILED` / `ERROR` per test | `outcome` |
| `Q-all-pass` | are *all* of these tests expected to pass? | `all_pass` |

```json
{"instance_id": "pandas__b73c38e2__prediction_oh_gpt51",
 "outcome": {"tests": ["..."], "labels": ["PASSED", "FAILED", "..."]},
 "all_pass": true}
```

```bash
uv run python -m sourceworldbench_eval_harness.prediction.predict  configs/baselines/status/anchored_all_passed.yaml
uv run python -m sourceworldbench_eval_harness.evaluation.evaluate configs/baselines/status/anchored_all_passed.yaml
```

`all_pass` is the only answer that is a bare boolean, and the only one pooled
over instances rather than tests — so its count column is `all_pass_instances`,
and `all_pass_not_passed_f1` is the figure that matters: how reliably a model
notices that a patch broke something. Both trivial baselines score 0.0000 on one
of `all_pass`'s two classes and an undefined MCC, so a model has to beat the
constant's macro-F1 there *and* produce a defined MCC. The floors and every
metric are in
[`docs/TEST_STATUS_ANCHORED_FORMAT.md`](docs/TEST_STATUS_ANCHORED_FORMAT.md).

### Time-prediction task

`anonymous-research-730875/sourceworldbench-time-prediction` asks how long a workload takes once the
patch is applied, four ways: the seconds (`seconds`), and whether it fits a 5s, 60s or
300s budget (`budget_5s`, `budget_60s`, `budget_300s`).

```json
{"instance_id": "...", "seconds": 6.89, "budget_5s": false, "budget_60s": true, "budget_300s": true}
```

```bash
uv run python -m sourceworldbench_eval_harness.prediction.predict  configs/baselines/time/prediction_constant_seconds.yaml
uv run python -m sourceworldbench_eval_harness.evaluation.evaluate configs/baselines/time/prediction_constant_seconds.yaml
```

`seconds` is scored **scale-free** — the target spans seconds to minutes, so an absolute
error would be set by the slowest instance alone. `seconds_log10_error` is the figure
it is ranked on, with `seconds_within_factor_2` reading how often the answer was
usable at all. `seconds_debiased_log10_error` and `seconds_calibrated_log10_error` read
what error would survive correcting a systematic offset, and correcting the scale on top
of it. Why each metric is the one chosen:
[`docs/TIME_PREDICTION_FORMAT.md`](docs/TIME_PREDICTION_FORMAT.md).

### Time-hotspot task

`anonymous-research-730875/sourceworldbench-time-hotspots` asks where that time goes. It ranks each profile by
**exclusive** time — a function's own frame, excluding the calls the profiler recorded
separately — and asks for the top 1, 5 and 20, most expensive first. That is three
questions: `exclusive_top1`, `exclusive_top5` and `exclusive_top20`.
`functions_complete_list` holds the whole profile, which is both the candidate pool and
what makes "from the codebase, not a built-in" checkable rather than assumed.

```json
{"instance_id": "...", "exclusive_top1": ["lib/matplotlib/_afm.py:240 _parse_kern_pairs"]}
```

```bash
uv run python -m sourceworldbench_eval_harness.prediction.predict  configs/baselines/time/hotspot_most_called.yaml
uv run python -m sourceworldbench_eval_harness.evaluation.evaluate configs/baselines/time/hotspot_most_called.yaml
```

Counting functions, valuing them and ordering them differ, so `exclusive_top5` and
`exclusive_top20` each report `hit_rate` for how much of the expected set was found,
`time_captured` for how much of its time was accounted for, `ndcg_macro` for whether the
ranking put the expensive ones first, and `somers_d` (with Kendall's `tau_b` beside it, for
comparison with outside numbers) for whether what was named is in the
right order. `exclusive_top1` reports only the first and third: one function has no order
to correlate, and one expected function makes `time_captured` a second name for `hit_rate`.
Time is skewed enough that finding one of five can capture most of the workload or almost 
none of it, and only `time_captured` says which. `ndcg_macro` is top-heavy by construction,
so `somers_d` sits beside it — spanning **-1 to 1**, it goes negative on a backwards answer
and tells you whether a good NDCG came from ordering or from set membership. Rank on
`hit_rate`, with `exclusive_top20` as the headline question.
[`docs/TIME_HOTSPOT_FORMAT.md`](docs/TIME_HOTSPOT_FORMAT.md) covers why.

### Memory-prediction task

`anonymous-research-730875/sourceworldbench-memory-prediction` asks how much memory a workload needs once the
patch is applied, four ways that mirror the timing task: the bytes (`bytes`), and whether
it is at most 500MB, 1GB or 5GB (`budget_500mb`, `budget_1gb`, `budget_5gb`).

```json
{"instance_id": "...", "bytes": 632205358, "budget_500mb": false, "budget_1gb": true, "budget_5gb": true}
```

```bash
uv run python -m sourceworldbench_eval_harness.prediction.predict  configs/baselines/memory/prediction_constant_bytes.yaml
uv run python -m sourceworldbench_eval_harness.evaluation.evaluate configs/baselines/memory/prediction_constant_bytes.yaml
```

The target is a `tracemalloc` **peak increase** on the Python heap — not RSS, and not a
total, so memory allocated and freed again during the test still counts. Budget names are
decimal, so `500MB` is 500 million bytes. `bytes` is scored **scale-free**, since the target
spans megabytes to tens of gigabytes: `bytes_log10_error` is the figure it is ranked on, with
`bytes_within_factor_1.25` and `_2` reading how often the answer was calibrated and how often
it sized a machine correctly at all. `bytes_debiased_log10_error` and
`bytes_calibrated_log10_error` read what error would survive correcting a systematic offset,
and correcting the scale on top of it. Why each metric is the one chosen:
[`docs/MEMORY_PREDICTION_FORMAT.md`](docs/MEMORY_PREDICTION_FORMAT.md).

### Memory-hotspot task

`anonymous-research-730875/sourceworldbench-memory-hotspots` asks where that memory goes. It ranks each
profile by `cumulative`: the net memory a function's calls add, summed over all of them,
exclusive of the repository functions it calls. Two depths are asked — `cumulative_top1`
and `cumulative_top5`.

```json
{"instance_id": "...", "cumulative_top1": ["astropy/utils/parsing.py:118 yacc"]}
```

```bash
uv run python -m sourceworldbench_eval_harness.prediction.predict  configs/baselines/memory/hotspot_most_called.yaml
uv run python -m sourceworldbench_eval_harness.evaluation.evaluate configs/baselines/memory/hotspot_most_called.yaml
```

Each question reports the same four figures as the timing hotspots, with `bytes_captured` in
place of `time_captured`, and is ranked the same way: on `hit_rate`, with `cumulative_top5` as the
headline question.
[`docs/MEMORY_HOTSPOT_FORMAT.md`](docs/MEMORY_HOTSPOT_FORMAT.md) covers the rest.

## `shared` block

A required `shared` block holds settings used across steps:

```yaml
shared:
  output_root: outputs/single_test_status  # all step paths resolve under here
  output_tag: null                  # optional; artifacts go to `<output_root>_<output_tag>` when set
```

Every step's `input`/`output` is a path **relative to** the resolved output
directory (`output_root`, plus `_<output_tag>` when a tag is set), so one run's
artifacts stay together and you can keep parallel runs apart with a tag.

## `task` block

A required `task` block names the evaluation and carries everything specific to
it. Nothing task-specific goes in a step block, so the step blocks are the same
shape for every task and you never need to pair a config with a script:

```yaml
task:
  name: test_status_anchored
  id_field: instance_id         # column identifying one scored item
  require_full_coverage: true   # every dataset instance must be answered
```

`predict` and `evaluate` both dispatch on `task.name`: the task decides which models it
can run, how ground truth is built, which checks apply and which metrics are computed.
`single_test_status`, `test_status_anchored`, `time_prediction`, `time_hotspot`,
`memory_prediction` and `memory_hotspot` are registered today; adding another means adding
a module under `sourceworldbench_eval_harness/tasks/` and a line to `tasks/registry.py`, not editing the
steps. A task block carries only its own task's settings — `calls_file` belongs to `single_test_status` alone, and naming it
under `test_status_anchored` is an error rather than something quietly ignored.

Any task asking a fixed set of questions per instance extends
`tasks.core.questions.QuestionTask`, which owns what none of them should re-decide — the
id and coverage checks, the question loop, the question-name keying the leaderboard
groups on — and asks for four hooks: how a row is generated, how an answer is shaped,
what ground truth can reveal about it, and how a question is scored. It knows nothing
about what an answer *is*, which is why the timing and memory tasks fit it without owning a
label vocabulary.

`tasks.status.StatusTask` adds the part only status tasks share (the flat classification
columns), and `tasks.status.multi_test.MultiTestStatusTask` fills in all four hooks from
the columns it names, so two multi-test tasks differ only in their dataset, vocabulary
and columns — and their models differ only in which task they are bound to. A task
shaped differently can still implement `tasks.core.base.Task` directly.

## Pipeline

Each step reads its `input` from a path on disk, so you can enter the pipeline at any stage by pointing that stage's
`input:` at an artifact you already have, instead of running the earlier steps.
See [Bring your own predictions](#bring-your-own-predictions) for evaluating predictions produced outside this harness.

### 1. `load`

Names the Hugging Face dataset split the task scores against:

```yaml
load:
  name: anonymous-research-730875/sourceworldbench-test-status-anchored
  split: test
  revision: v0.2.0            # optional: pin a dataset version (tag, branch, or commit)
```

`predict` and `evaluate` read this block directly — there is no separate step to
run first, and both resolve it exactly the way the `load` step does, so they
cannot end up on a different split or revision. They reuse the Hugging Face
cache; set `HF_HUB_OFFLINE=1` to run with no network at all once it is warm.

`load` is also runnable on its own, to archive the split as an on-disk HF Dataset
folder. Fetching is the reason to run it, so unlike every other reader of this
block it ignores the cache and always redownloads. It writes to a folder named by
an `output` key in the `load` block, which none of the shipped configs carry —
add one before running it:

```yaml
load:
  name: anonymous-research-730875/sourceworldbench-test-status-anchored
  split: test
  revision: v0.2.0
  output: dataset
```

```bash
uv run python -m sourceworldbench_eval_harness.data.load <your-config>.yaml
```

Nothing reads that folder back automatically — it is for inspection and
archival, not an input to the rest of the pipeline.

Task datasets are versioned with git tags on the HF repo (`v<major>.<minor>.<patch>`).
Omitting `revision` loads the latest `main`. Either way, the exact commit pulled is
recorded as `load.resolved_sha` in the `.config.yaml` sidecar, so a run stays
reproducible even if a tag later moves. A pinned `revision` also labels the
submission — see [pinning a version](#pinning-a-dataset-version).

### 2. `predict`

Runs the model named by `predict.model` over the task's input, writes the rows to `predict.output` and shape-checks
them, so a malformed submission fails before an artifact is written. `params` are passed to the model as keyword
arguments. `from_file` resolves per task, so it means whatever reading a file means there.

What a row looks like is the task's business: the anchored task writes one row per instance with an answer per question (see
[`docs/TEST_STATUS_ANCHORED_FORMAT.md`](docs/TEST_STATUS_ANCHORED_FORMAT.md)).

```yaml
predict:
  model: all_passed         # all_passed | random_status | oracle
  params: { seed: 42 }      # add `questions: [binary, outcome]` to answer a subset
  output: predictions.jsonl
```

### 3. `evaluate`

Computes the task's metrics. Ground truth is rebuilt from the dataset named in `load` and matched to predictions on
`task.id_field` — it is **not** taken from the prediction file, so predictions can't smuggle in their own labels.
Unknown or duplicate ids are errors, and a prediction that answers no question the task recognises is rejected.

The metrics file holds flat scalars at the top level — one per leaderboard column — with anything that isn't a single
number (per-class breakdowns, confusion matrices) nested under `details`.

```yaml
evaluate:
  input: predictions.jsonl
  output: metrics.json
```

## Bring your own predictions

Point `predict.external_input` at your `.jsonl` and run `predict` as usual — it is the one path not resolved under
`shared.output_root`, though `~` still expands. For test status the file *is* the submission, so the `from_file` model
is the default: it keeps the id and the question answers from each row, under canonical `binary`/`outcome` names,
and shape-checks them. What lands at `predict.output` is therefore a normalised extract, not a copy of your file — every
other column is dropped. Ground truth still comes from the dataset.

The row shape, the per-question answers and every validation message are documented in
[`docs/TEST_STATUS_ANCHORED_FORMAT.md`](docs/TEST_STATUS_ANCHORED_FORMAT.md). A ready-to-edit
config lives at `configs/templates/status/anchored_external.yaml`:

```yaml
shared:
  output_root: outputs/test_status_anchored_external
  output_tag: <tag>

task:
  name: test_status_anchored
  id_field: instance_id
  require_full_coverage: true     # set false to score a partial submission

load:
  name: anonymous-research-730875/sourceworldbench-test-status-anchored
  split: test
  revision: v0.2.0

predict:
  external_input: <path/to/predictions.jsonl>    # your file; `~` expands
  output: predictions.jsonl                      # the extract that gets scored

evaluate:
  input: predictions.jsonl
  output: metrics.json

leaderboard:
  input: metrics.json
  dataset: anonymous-research-730875/sourceworldbench-test-status-anchored   # task grouping key (shown as its slug)
  model: <model>                                # name shown on the leaderboard
```

`leaderboard.revision` records which dataset version the predictions were made on;
see [pinning a version](#pinning-a-dataset-version).

Then run predict → evaluate → push:

```bash
uv run python -m sourceworldbench_eval_harness.prediction.predict  <your-config>.yaml
uv run python -m sourceworldbench_eval_harness.evaluation.evaluate <your-config>.yaml
uv run python -m sourceworldbench_eval_harness.leaderboard.push    <your-config>.yaml
```

## Pinning a dataset version

A version can be pinned in either of two blocks:

| Key                    | Pins                                     | Use when                                    |
|------------------------|------------------------------------------|---------------------------------------------|
| `load.revision`        | the dataset the run scores against       | the run goes through the full pipeline      |
| `leaderboard.revision` | the version reported to the leaderboard  | predictions were made outside this harness  |

Set **one** of them and it is applied to both, so the version a run loads and the
version it reports can't drift apart. Setting both to different values is a config
error. When a version is known, submissions group under `<slug>@<version>`
(e.g. `sourceworldbench-test-status-anchored@v0.1.2`), so each dataset version appears as its own
leaderboard; unversioned submissions stay under the plain slug.

Pinning only affects which dataset version is read, so re-running a later step alone against
artifacts produced at another version won't be caught — check `load.resolved_sha` in
the artifact's `.config.yaml` sidecar, or re-run from `predict`, when changing versions.

## Storage formats

`load`, when given an `output`, writes an on-disk HF Dataset folder. `predict`
writes a predictions `.jsonl`; `evaluate` reads that `.jsonl` and writes a
`.json` metrics file. All paths are relative to `shared.output_root`.

## Config

Shipped configs come in two kinds, and the directory says which:

| Directory | Holds | Meant to be |
|---|---|---|
| `configs/templates/` | one annotated config per task, plus an `_external` variant | copied and edited — the comments explain every setting |
| `configs/baselines/` | one config per (task, model) baseline already on the leaderboard | run as-is, to reproduce a leaderboard row |

A baseline config is a template with the comments stripped and the model, its
params and `shared.output_tag` fixed, since nothing is overridable from the command
line and each leaderboard row needs its own `leaderboard.model`.

Full example (`configs/templates/status/anchored.yaml`).

```yaml
shared:
  output_root: outputs/test_status_anchored
  output_tag: <tag>

task:
  name: test_status_anchored
  id_field: instance_id
  require_full_coverage: true

load:
  name: anonymous-research-730875/sourceworldbench-test-status-anchored
  split: test
  revision: v0.2.0

predict:
  model: <model>
  params: {}
  output: predictions.jsonl

evaluate:
  input: predictions.jsonl
  output: metrics.json

leaderboard:
  input: metrics.json
  dataset: anonymous-research-730875/sourceworldbench-test-status-anchored
  model: <model>
```

A `<placeholder>` is a choice the copy has to make: which model to run, what to
call it on the leaderboard, and which output directory to keep its artifacts in.
Everything else is fixed by the task. `configs/baselines/` holds those choices
already made, one file per model.

