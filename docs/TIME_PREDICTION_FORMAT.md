# Time-prediction prediction format

Submission format for [`anonymous-research-730875/sourceworldbench-time-prediction`](https://huggingface.co/datasets/anonymous-research-730875/sourceworldbench-time-prediction),
`test` split. One row per instance, one bare value per question. `predict` reports anything
malformed, so this covers only what the error messages cannot.

```json
{"instance_id": "...", "seconds": 6.89, "budget_5s": false, "budget_60s": true, "budget_300s": true}
```

| Key | Prompt | Answer |
|---|---|---|
| `seconds` | `Q-seconds` | positive number of seconds |
| `budget_5s` / `budget_60s` / `budget_300s` | `Q-budget-5s` / `-60s` / `-300s` | `true` if it fits the budget |

A number of seconds mechanically gives you all three budgets, but deriving
them makes four columns report one capability measured once. **Answer each with its own
run.** The built-in baselines *do* derive them, which is exactly why they are not a model
of how to submit.

What the target measures is specified upstream in the dataset's own `PROFILING.md`, which
v0.2.0 tags: one clock reading either side of the test function call, elapsed rather than
CPU time, and nothing before the call — not process startup, collection, or fixtures.

## `seconds` is scored scale-free

The target spans seconds to minutes, so an absolute error would be set by the slowest
instance alone: 30s out on a 400s workload is a good prediction, on a 2s workload it is not
a prediction at all. Hence `seconds_log10_error` (`|log10(predicted/actual)|`, symmetric in
over- and under-prediction) and the bounded `seconds_within_factor_2`. No error in the
target's own units is reported. **`seconds_log10_error` is the figure the question is ranked
on** — lower is better, and it is this dataset's headline.

**Read both.** `seconds_log10_error` averages error magnitude, so one wild prediction can
set it (0.001s for a 442s workload contributes 5.6 alone); `seconds_within_factor_2` counts
how often the answer was usable, and a single instance can move it by at most 1/n. The gap
between them is the distribution's shape — 0.35/0.40 is precise or hopeless with little in
between, 0.05/0.60 is vague but rarely wild.

`seconds_debiased_log10_error` is the same mean absolute error after the corpus's median
log-ratio is taken out of every prediction: what would remain if the model's systematic
over- or under-prediction were corrected. A model uniformly 3x high and one scattered by 3x
either way score alike on `seconds_log10_error`, and only the first is one constant away
from accurate — this is the figure that tells them apart. It never exceeds
`seconds_log10_error`, and the gap between them is what recalibration alone would buy.
Computed over every instance at once, so there is no `_scored` companion. **It is not the
ranking metric**: a submission is judged on the predictions it made, not on the ones a
correction it never applied would have made.

`seconds_calibrated_log10_error` frees the scale as well as the offset: the mean absolute
residual after the best affine remapping of the predictions in log space, intercept fitted
freely and slope fitted but floored at zero. The three errors are nested — the raw one fixes
the map at identity, the debiased one frees the intercept, this one frees the slope too — so

    calibrated <= debiased <= raw

always holds, and the differences are what to read. Raw minus debiased is what correcting a
uniform offset would buy; debiased minus calibrated is what correcting a compressed or
stretched dynamic range would buy on top; what remains is scatter no remapping can reach. A
model whose predictions track the target but span too narrow a range — the common failure on
this task — is charged by the first two and excused by the third. **Also not ranked**, for the
same reason.

**The slope cannot go negative.** A negative slope reverses the predictions, so without the
floor a model that ranks workloads backwards would be handed the score of the reversal it
never submitted. The constraint caps such a model at slope 0 — the best constant — alongside
every other model carrying no usable signal. Where a model is right-way-round the floor costs
nothing, the unconstrained optimum being positive already, and slope 1 stays in the family
regardless, which is why the nesting above survives the constraint.

Freeing the slope gives this figure a **ceiling**: a model with no signal at all is remapped
to the best constant, so nothing can score worse than the best-constant error, and models
that differ widely on the raw figure can land on the same calibrated one. A value at that
ceiling is the finding — it says the predictions carry no usable information about which
instance is larger — but it is also why the calibrated column compresses a field rather than
ordering it.

`seconds_log_slope` and `seconds_log_intercept` come from a least-squares fit of
`log10(predicted)` on `log10(actual)` over every instance at once (so no `_scored`
companion). They separate two failures the error metrics report identically: slope near 1
with non-zero intercept tracks the target and is uniformly out by `10 ** intercept` — wrong
but calibratable — while slope near 0 is a model returning the same number whatever the
input. **Neither is ranked**: their best values sit mid-range, so sorting by them is
meaningless.

## Budget questions: read the per-class F1s

`fits` is the positive class, so `fits_f1` reads as "does the model know what comes back
quickly?". The imbalance moves with the threshold: at 300s almost everything fits, so
accuracy is nearly free and **`misses_f1` is the column showing whether the exceptions were
noticed**. A model predicting a single class reports `mcc` as `n/a`.

## Why the revision is pinned

Timings are regenerated, not extended. Predicting against one revision and scoring against
another gives quietly wrong `seconds` errors.

## Baselines

`oracle` copies ground truth and **must score 1.0** — run it first against a new config.
`constant_seconds` is the floor to beat: the dataset median minimises the log-ratio error a
constant can achieve, and since one guess either fits a threshold or does not, it is also
the majority answer to every budget question. Prefer it over `random_seconds`, which draws
log-uniformly so a guess is as likely to be seconds as minutes.

There is no CLI override for a config field, so to compare baselines copy
`configs/templates/time/prediction.yaml` per model and give each a distinct `shared.output_tag`.
