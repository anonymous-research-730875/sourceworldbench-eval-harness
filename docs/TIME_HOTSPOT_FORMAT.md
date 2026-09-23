# Time-hotspot prediction format

Submission format for [`anonymous-research-730875/sourceworldbench-time-hotspots`](https://huggingface.co/datasets/anonymous-research-730875/sourceworldbench-time-hotspots),
`test` split. `predict` reports anything malformed, so this covers only what the error
messages cannot.

```json
{"instance_id": "...", "exclusive_top1": ["lib/matplotlib/_afm.py:240 _parse_kern_pairs"]}
```

## The three questions

The dataset ranks each profile by **exclusive** time — a function's own frame, excluding the
calls the profiler recorded separately — and asks for the top 1, 5 and 20:

| Question | Answer |
|---|---|
| `exclusive_top1` / `_top5` / `_top20` | exactly 1 / 5 / 20 functions, most expensive first |

Answer whichever you can; an unanswered question is left unscored rather than counted wrong,
and each deserves its own model run.

How those times are produced is specified upstream in the dataset's own `PROFILING.md`,
which v0.2.0 tags. Two points bear on a prediction: the times are elapsed, not CPU, so a
function that waits is charged for the wait; and a function's self time keeps whatever
compiled work the profiler was never notified about, so a three-line function can hold
seconds it never spent on its own bytecode.

A function may be named by the dataset's `key`, by a `filename:firstlineno qualname` triple —
the spelling `functions_input_list` uses, so it is what a prompt shows a model — or by an
object with those three fields. All resolve identically and may be mixed. Every function
named must be one this workload profiled. Naming one function twice is rejected **even under
two different spellings** — the answer would be shorter than the question asks.

## The four metrics

| Metric | Reads |
|---|---|
| `hit_rate` | how much of the expected set was found |
| `time_captured` | how much of its *time* was accounted for |
| `ndcg_macro` | how much cost was surfaced early (top-heavy) |
| `somers_d` | whether what was named is in the right order (**-1 to 1**) |

Rank on `hit_rate`, with `exclusive_top20` as the headline question; quote `time_captured`,
which states a share of the workload directly. `hit_rate` and `time_captured` are read as a
pair because time is skewed enough that finding 1 of 5 can capture most of the workload or
almost none.

All four accept a tie at the cutoff: where the `k`-th and `k+1`-th functions cost exactly the same,
which one the reference kept is arbitrary, so naming either scores alike.

`ndcg_macro` grades gain by time rather than a flat 1 for being expected, so a reversed
answer does not score 1.0 the way a binary gain would, and a function just outside the
top-`k` still earns its own time. But it stays top-heavy by construction: a fully reversed
answer still scores well clear of zero, because naming the right functions at all is
credited for the time they hold. So `somers_d` sits beside it, weighing every pair equally
and going negative on a backwards answer. **Read `somers_d` to see whether a good
`ndcg_macro` came from ordering or from set membership.**

Precision@k is absent (the answer is exactly `k` long, so it would be `hit_rate`
renamed). The `top1` questions report neither `time_captured` nor `somers_d` — with one
expected function there is no share to weigh and no order to correlate.

**Why Somers' D and not Kendall's tau-b.** Same numerator (concordant minus discordant
pairs), different denominator: tau-b uses `sqrt((n0-Tx)(n0-Ty))`, Somers' D uses `n0 - Ty`,
dropping only pairs the *ground truth* ties. Two consequences this task needs — truth ties
are microsecond rounding and unrankable, so either order is accepted, while a tie in the
*prediction* stays a wrong answer rather than the discount tau-b gives it; and a flawless
answer can reach 1.0 at all, where tau-b caps it at `sqrt(1 - Ty/n0)`, a ceiling that moves
with each instance's tie count. Argument order therefore carries meaning:
`somers_d(truth, predicted)`, matching `scipy.stats.somersd`.

`ndcg_micro` is reported too: macro is the mean of per-instance ratios, counting every
instance equally though they hold unequal time to win; micro divides once over the whole set,
weighting an instance by what is at stake in it.

## Instances with short profiles

`top20` is unanswerable for a workload that profiled fewer than 20 functions, so such an
instance is dropped from the reference entirely (with a warning naming it) rather than
failing the run — which is what keeps all three columns averaged over the same instances
and therefore comparable. The guard stays even when no workload in a revision trips it,
because a regenerated profile may be shorter.

## Why the revision is pinned

Profiles are regenerated, not extended, and line numbers move when a file changes. A stale
triple surfaces as "not in this instance's profile" or, worse, resolves to whatever function
now occupies that line.

## Baselines

`oracle` copies ground truth and **must score 1.0** — run it first against a new config.
`random_functions` is the floor. `most_called` is the one worth reading: call count is free
to compute and correlates with cost, so the gap between it and `oracle` is the part needing
actual reasoning about what a function costs *per* call.

There is no CLI override for a config field, so to compare baselines copy
`configs/templates/time/hotspot.yaml` per model and give each a distinct `shared.output_tag`.
