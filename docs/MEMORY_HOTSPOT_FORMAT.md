# Memory-hotspot prediction format

Submission format for [`anonymous-research-730875/sourceworldbench-memory-hotspots`](https://huggingface.co/datasets/anonymous-research-730875/sourceworldbench-memory-hotspots),
`test` split. `predict` reports anything malformed, so this covers only what the error
messages cannot.

```json
{"instance_id": "...", "cumulative_top1": ["astropy/utils/parsing.py:118 yacc"]}
```

## The two questions

| Question | Answer |
|---|---|
| `cumulative_top1` / `_top5` | exactly 1 / 5 functions, largest first |

There is no `cumulative_top20`; an answer to it is ignored. An unanswered question is left
unscored rather than counted wrong, and each deserves its own model run.

## What the ranking measures

**`cumulative`** is the net memory a call's frame adds between entry and return, **summed
over every call**.

It is **exclusive**, which excludes less than it sounds like: a function keeps what the
library code it calls allocates, and only the candidate-list functions it calls are excluded.
It is a `tracemalloc` figure, so it counts no memory allocated outside the Python
allocator. These ship no upstream `PROFILING.md`; the `Q-*` columns are the specification, and
state the measure longhand — the word *exclusive* survives only in the column names.

## Naming a function

A function may be named by the dataset's `key`, by a `filename:firstlineno qualname` triple —
the spelling a prompt shows a model — or by an object with those three fields. All resolve
identically and may be mixed. Every function named must be one this workload profiled. Naming
one twice is rejected **even under two different spellings**, since the answer would then be
shorter than the question asks.

## The four metrics

| Metric | Reads |
|---|---|
| `hit_rate` | how much of the expected set was found |
| `bytes_captured` | how much of its *memory* was accounted for |
| `ndcg_macro` | how much cost was surfaced early (top-heavy) |
| `somers_d` | whether what was named is in the right order (**-1 to 1**) |

Rank on `hit_rate`, with `cumulative_top5` as the headline question; quote `bytes_captured`, which
states a share of the profile directly. Read `hit_rate` and `bytes_captured` as a pair, because
allocation is skewed enough that finding 1 of 5 can capture most of the workload's memory or
almost none.

All four accept a tie at the cutoff: where the `k`-th and `k+1`-th functions allocate byte-identical
amounts, which one the reference kept is arbitrary, so naming either scores alike.

`ndcg_macro` grades gain by bytes, so a reversed answer does not score 1.0 the way a binary
gain would. But it stays top-heavy: a fully reversed answer still scores well clear of zero,
because naming the right functions at all is credited for the memory they hold. **Read
`somers_d` to see whether a good `ndcg_macro` came from ordering or from set membership.**

Precision@k would be `hit_rate` renamed, since the answer is exactly `k` long. The `top1`
questions report neither `bytes_captured` nor `somers_d` — with one expected function there is
no share to weigh and no order to correlate.

**Why Somers' D and not Kendall's tau-b.** Same numerator, different denominator: Somers' D
drops only pairs the *ground truth* ties. Two consequences this task needs — truth ties are
byte-level coincidences and unrankable, so either order is accepted, while a tie in the
*prediction* stays wrong rather than getting the discount tau-b gives it; and a flawless answer
can reach 1.0 at all, where tau-b caps it at a ceiling that moves with each instance's tie
count. Argument order therefore carries meaning: `somers_d(truth, predicted)`, matching
`scipy.stats.somersd`.

`ndcg_micro` is reported too: macro counts every instance equally though they hold unequal
memory to win, micro weights an instance by what is at stake in it.

## Instances with short profiles

A profile ranking fewer than 5 functions cannot answer `cumulative_top5`, so the instance is
dropped from the reference (with a warning) rather than failing the run — which keeps every
column averaged over the same instances and therefore comparable. The guard stays even when no
workload in a revision trips it, because a regenerated profile may be shorter.

## Why the revision is pinned

Profiles are regenerated, not extended, and line numbers move when a file changes. A stale
triple surfaces as "not in this instance's profile" or, worse, resolves to whatever function
now occupies that line. The tag has **no `v` prefix** here (`0.2.0`), unlike the timing
datasets (`v0.2.0`).

## Baselines

`oracle` copies ground truth and **must score 1.0** — run it first against a new config.
`random_functions` is the floor. `most_called` is the one worth reading: call count carries
real signal on `cumulative`, which sums over calls, so its gap to `oracle` is the part needing
actual reasoning about a function's footprint rather than its call count.

There is no CLI override for a config field, so to compare baselines copy
`configs/templates/memory/hotspot.yaml` per model and give each a distinct `shared.output_tag`.
