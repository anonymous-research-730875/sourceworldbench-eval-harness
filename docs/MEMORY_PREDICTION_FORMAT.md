# Memory-prediction prediction format

Submission format for [`anonymous-research-730875/sourceworldbench-memory-prediction`](https://huggingface.co/datasets/anonymous-research-730875/sourceworldbench-memory-prediction),
`test` split. One row per instance, one bare value per question. `predict` reports anything
malformed, so this covers only what the error messages cannot.

```json
{"instance_id": "...", "bytes": 632205358, "budget_500mb": false, "budget_1gb": true, "budget_5gb": true}
```

| Key | Prompt | Answer |
|---|---|---|
| `bytes` | `Q-bytes` | positive number of bytes |
| `budget_500mb` / `budget_1gb` / `budget_5gb` | `Q-budget-500MB` / `-1GB` / `-5GB` | `true` if it is at most the budget |

**Answer each with its own run.** Deriving the budgets from a byte count makes four columns
report one capability measured once. The built-in baselines do derive them, which is exactly
why they are not a model of how to submit.

## What the target measures

These ship no upstream `PROFILING.md`; the `Q-*` columns are the specification. Three points
are worth stating because none of them is what "memory usage" usually means:

- A **`tracemalloc` figure for the Python heap**, not RSS and not the process total.
- Only what goes through the **Python allocator**, so a workload dominated by NumPy or BLAS
  buffers looks far cheaper here than it does to the operating system.
- A **peak, not a total**: a loop allocating and freeing 80 MB a hundred times peaks at 80 MB.
- The window includes the test's **setup**, where the timing datasets charge nothing before
  the call.

## Byte units are decimal

`500MB` is 500 million bytes. Scoring as mebibytes would mark correct answers wrong either
side of every threshold.

## `bytes` is scored scale-free

The target spans megabytes to tens of gigabytes, so an absolute error would be set by the
largest instance alone: 200 MB out on a 20 GB workload is a good prediction, on a 5 MB workload
it is not a prediction at all. Hence `bytes_log10_error` and the bounded
`bytes_within_factor_1.25` / `_2`. **`bytes_log10_error` is the figure the question is ranked
on** — lower is better, and it is this dataset's headline.

**Read both.** One wild prediction can set `bytes_log10_error`, while a single instance moves
`bytes_within_factor_2` by at most 1/n. The gap between them is the distribution's shape —
0.35/0.40 is precise or hopeless with little in between, 0.05/0.60 is vague but rarely wild.

**Why these two bands.** A factor of 2 is the loosest ratio at which a prediction still sizes a
machine correctly; 1.25 is where it is calibrated rather than merely the right order of magnitude.
Both leave headroom on a target spanning nearly four orders of magnitude — a constant median guess
reaches 29% at factor 2 and 10% at 1.25, where a looser band such as 10x would be nearly free at
87%. A model uniformly 1.5x out clears factor 2 and fails 1.25, and that gap is what the pair
reports.

`bytes_debiased_log10_error` is the same mean absolute error after the corpus's median
log-ratio is taken out of every prediction: what would remain if the model's systematic over-
or under-prediction were corrected. A model uniformly 3x high and one scattered by 3x either
way score alike on `bytes_log10_error`, and only the first is one constant away from accurate.
It never exceeds `bytes_log10_error`, and the gap between them is what recalibration alone
would buy. Computed over every instance at once, so there is no `_scored` companion. **It is
not the ranking metric**: a submission is judged on the predictions it made, not on the ones a
correction it never applied would have made.

`bytes_log_slope` and `bytes_log_intercept` separate two failures the error metrics report
identically: slope near 1 with non-zero intercept tracks the target and is uniformly out — wrong
but calibratable — while slope near 0 is a model returning the same number whatever the input.
**Neither is ranked**; their best values sit mid-range, so sorting by them is meaningless.

## Budget questions: read the per-class F1s

`fits` is the positive class, so `fits_f1` reads as "does the model know what runs in a small
heap?". At 5GB almost everything fits, so accuracy is nearly free and **`exceeds_f1` is the
column showing whether the exceptions were noticed**. A single-class model reports `mcc` as
`n/a`.

Budget answers are derived from the byte count and cross-checked against the dataset's own
`within-budget-*` columns, so a revision that redefined a threshold stops the run instead of
silently marking right answers wrong. A revision that drops those columns is still scoreable.

## Why the revision is pinned

Footprints are regenerated, not extended, so scoring against a different revision gives quietly
wrong `bytes` errors. The tag has **no `v`
prefix** here (`0.2.0`), unlike the timing datasets (`v0.2.0`).

## Baselines

`oracle` copies ground truth and **must score 1.0** — run it first against a new config.
`constant_bytes` is the floor to beat: the median minimises the log-ratio error a constant can
achieve, and one guess either fits a threshold or does not, so it is also the majority answer to
every budget question. Prefer it over `random_bytes`, which draws log-uniformly so a guess is as
likely to be kilobytes as gigabytes.

There is no CLI override for a config field, so to compare baselines copy
`configs/templates/memory/prediction.yaml` per model and give each a distinct `shared.output_tag`.
