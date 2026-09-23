"""Pure math: nothing here knows about a task, a question or a dataset column.

`None` means undefined and is dropped by `aggregate`; `0.0` is a real score of zero.
Collapsing the two would let a model that never predicts the minority class sit out the
minority metrics instead of scoring zero on them.
"""

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from statistics import median
from typing import Any


def harmonic_f1(p: float | None, r: float | None) -> float | None:
    """F1 from a precision/recall pair, propagating `None`."""
    if p is None or r is None:
        return None
    return 2 * p * r / (p + r) if (p + r) else 0.0


def mean(values: Sequence[float | None]) -> float | None:
    """Mean of the defined values; `None` when nothing is defined."""
    usable = [value for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def multiclass_confusion(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Nested `{true_label: {predicted_label: count}}` over a fixed vocabulary."""
    matrix = {truth: dict.fromkeys(labels, 0) for truth in labels}
    for truth, pred in zip(y_true, y_pred, strict=True):
        matrix[truth][pred] += 1
    return matrix


def classification_report(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    """Per-class and macro figures over a fixed label vocabulary.

    Macro figures average only over classes with support in `y_true`: SKIPPED is rare
    enough to be absent from many instances, and averaging over absent classes would let
    a class with no examples swing the headline number.
    """
    matrix = multiclass_confusion(y_true, y_pred, labels)
    per_class: dict[str, dict[str, float | None]] = {}
    for label in labels:
        true_positive = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[other][label] for other in labels)
        # A class that occurs but was never predicted scores 0, not None.
        p: float | None = true_positive / predicted if predicted else (0.0 if support else None)
        r: float | None = true_positive / support if support else None
        per_class[label] = {"precision": p, "recall": r, "f1": harmonic_f1(p, r), "support": support}

    present = [label for label in labels if per_class[label]["support"]]
    report: dict[str, Any] = {
        "accuracy": sum(matrix[label][label] for label in labels) / len(y_true) if y_true else None,
        "macro_f1": mean([per_class[label]["f1"] for label in present]),
        "macro_precision": mean([per_class[label]["precision"] for label in present]),
        "macro_recall": mean([per_class[label]["recall"] for label in present]),
        "per_class": per_class,
        "confusion": matrix,
    }
    report["balanced_accuracy"] = report["macro_recall"]
    if len(labels) == 2:
        report["mcc"] = multiclass_mcc(matrix, labels)
    return report


def multiclass_mcc(matrix: dict[str, dict[str, int]], labels: Sequence[str]) -> float | None:
    """`labels[1]` is the positive class; MCC is symmetric, so the magnitude does not
    depend on that choice."""
    negative, positive = labels
    true_positive = matrix[positive][positive]
    true_negative = matrix[negative][negative]
    false_positive = matrix[negative][positive]
    false_negative = matrix[positive][negative]
    denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    if denominator == 0:
        return None
    return (true_positive * true_negative - false_positive * false_negative) / denominator


def aggregate(per_instance: Sequence[dict[str, float | None]]) -> dict[str, Any]:
    """Macro-average per-instance scores. `<metric>_scored` records how many instances
    contributed, since undefined metrics are skipped rather than counted as zero: most
    instances of these datasets have no failing test, so a macro figure read without its
    `_scored` count is misleading.
    """
    if not per_instance:
        return {"instances": 0}
    keys = sorted({key for scores in per_instance for key in scores})
    out: dict[str, Any] = {"instances": len(per_instance)}
    for key in keys:
        values = [scores.get(key) for scores in per_instance]
        out[key] = mean(values)
        out[f"{key}_scored"] = sum(1 for value in values if value is not None)
    return out


# `log10_error` and `within_factor` are the scale-free pair, and deliberately the only
# pair offered: a target spanning orders of magnitude makes an error in the target's own
# units meaningless, since one slow instance dominates the mean, while a ratio treats
# "2s predicted as 4s" and "200s as 400s" alike. `log_fit` then asks whether the
# predictions move with the target at all, which no average of per-instance errors can
# answer.


def log10_error(predicted: float, expected: float) -> float | None:
    """`|log10(predicted / expected)|`: 0 when exact, 1.0 when out by an order of magnitude,
    symmetric in over- and under-prediction. `None` unless both are positive.

    Base 10 so a value reads as orders of magnitude and shares a scale with `log_fit`.
    """
    if predicted <= 0 or expected <= 0:
        return None
    return abs(math.log10(predicted / expected))


def within_factor(predicted: float, expected: float, factor: float = 2.0) -> float | None:
    """1.0 when `predicted` is within a multiplicative `factor` of `expected`, else 0.0.

    A float so `aggregate` averages it into an accuracy, and bounded to [0, 1] so the
    leaderboard treats it as a rankable score — which an error in seconds is not.
    """
    if predicted <= 0 or expected <= 0:
        return None
    return 1.0 if 1 / factor <= predicted / expected <= factor else 0.0


def log_fit(predicted: Sequence[float], expected: Sequence[float]) -> tuple[float | None, float | None]:
    """Least-squares fit of `log10(predicted)` against `log10(expected)`, as (slope, intercept).
    `(None, None)` below two usable pairs, or when the target is constant. Non-positive pairs
    are dropped, having no logarithm.

    Separates two failures a mean error reports identically: slope near 1 with a non-zero
    intercept is a model that tracks the target and is uniformly out by `10 ** intercept` —
    calibratable, and it has understood relative cost — while slope near 0 is a model
    returning much the same number whatever the input.
    """
    if len(predicted) != len(expected):
        raise ValueError(f"{len(predicted)} predictions but {len(expected)} targets")
    points = [
        (math.log10(truth), math.log10(guess))
        for guess, truth in zip(predicted, expected, strict=True)
        if guess > 0 and truth > 0
    ]
    if len(points) < 2:
        return None, None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    spread = sum((x - mean_x) ** 2 for x, _ in points)
    if not spread:
        return None, None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / spread
    return slope, mean_y - slope * mean_x


def debiased_log10_error(predicted: Sequence[float], expected: Sequence[float]) -> float | None:
    """`log10_error` again, after the corpus's median log-ratio is taken out of every
    prediction: the error that would remain if the model's systematic over- or
    under-prediction were corrected. `None` with no usable pair. Non-positive pairs are
    dropped, having no logarithm.

    Reports the scatter `log10_error` confounds with offset — a model uniformly 3x high and
    one scattered by 3x either way score alike there, and only the first is one constant
    away from accurate. The median is an offset minimising the mean absolute residual, so
    this never exceeds `log10_error`, and the gap between them is what recalibration alone
    would buy.

    Computed over every instance at once, so it has no per-instance companion: the offset is
    a property of the corpus, and subtracting a per-instance bias would score nothing.
    """
    if len(predicted) != len(expected):
        raise ValueError(f"{len(predicted)} predictions but {len(expected)} targets")
    residuals = [
        math.log10(guess) - math.log10(truth)
        for guess, truth in zip(predicted, expected, strict=True)
        if guess > 0 and truth > 0
    ]
    if not residuals:
        return None
    bias = median(residuals)
    return sum(abs(residual - bias) for residual in residuals) / len(residuals)


# The set-retrieval metrics judge an answer against an expected set of items rather than
# against a relevance vector. The last of these reads order too, which it can only do
# because the weights make one expected item worth more than another.


def tie_tolerant(predicted: Sequence[Any], weights: dict[Any, float], expected: Sequence[Any]) -> list[Any]:
    """`expected` with each item the answer missed swapped for an equally heavy one it named.

    Which of two equally heavy items a top-`k` truncation kept is arbitrary, so charging for
    the choice would tax a coin flip. The order-reading metrics already score either alike,
    because they weigh what was named rather than test it for membership; this is what lets
    `hit_rate` and `weight_captured` agree with them.

    Substitution is capped by the slots actually missed, so naming several tied items cannot
    buy credit for an item that was genuinely overlooked.
    """
    named, wanted = set(predicted), set(expected)
    spare = [item for item in predicted if item not in wanted]
    resolved: list[Any] = []
    for item in expected:
        if item in named:
            resolved.append(item)
            continue
        swap = next((c for c in spare if weights.get(c, 0.0) == weights.get(item, 0.0)), None)
        if swap is None:
            resolved.append(item)
        else:
            spare.remove(swap)
            resolved.append(swap)
    return resolved


def hit_rate(predicted: Sequence[Any], expected: Sequence[Any]) -> float | None:
    """Fraction of `expected` that `predicted` contains; `None` when nothing is expected.
    With equal-sized sets this is precision and recall at once, hence the single name.
    """
    wanted = set(expected)
    if not wanted:
        return None
    return len(wanted & set(predicted)) / len(wanted)


def weight_captured(
    predicted: Sequence[Any], weights: dict[Any, float], expected: Sequence[Any] | None = None
) -> float | None:
    """Share of the expected items' total weight that `predicted` accounts for; `None` when that
    weight is zero. Where `hit_rate` counts items, this counts what they are worth.
    """
    wanted = set(expected) if expected is not None else set(weights)
    total = sum(weights.get(item, 0.0) for item in wanted)
    if total <= 0:
        return None
    return sum(weights.get(item, 0.0) for item in set(predicted) & wanted) / total


def _dcg(gains: Iterable[float]) -> float:
    return sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, start=1))


def weighted_dcg_at_k(
    predicted: Sequence[Any], weights: dict[Any, float], expected: Sequence[Any]
) -> tuple[float, float]:
    """This answer's discounted gain and the most it could have been — the two halves of
    an NDCG, kept separate so `micro_ndcg` can sum them before dividing."""
    ideal = sorted((weights.get(item, 0.0) for item in set(expected)), reverse=True)
    unique = list(dict.fromkeys(predicted))[: len(ideal)]
    return _dcg(weights.get(item, 0.0) for item in unique), _dcg(ideal)


def _tie_pairs(values: Sequence[float]) -> int:
    """Pairs tied within one ordering — what a tie correction takes out of a denominator."""
    return sum(size * (size - 1) // 2 for size in Counter(values).values())


def somers_d(truth: Sequence[float], predicted: Sequence[float]) -> float | None:
    """Somers' D of `predicted` against `truth`: how much more often the two orderings agree than
    disagree, over the pairs `truth` actually orders. `None` when `truth` is constant or below
    two items.

    Order-only and position-blind where NDCG is top-heavy and gain-weighted, and it spans -1
    to 1, so an answer ordered backwards scores -1 rather than merely low.

    The denominator `n0 - Ty` is what makes this Somers' D rather than Kendall's tau: it drops
    only the pairs `truth` ties. Times separated by profiler rounding are unrankable, so
    scoring either order as an error would tax noise, while a pair the truth does separate is
    always asked — a predicted tie there is a wrong answer, not a discount. It is also what
    lets a flawless answer reach 1.0, where `tau-b` caps at `sqrt(1 - Ty/n0)`, a ceiling that
    moves with each instance's tie count and makes columns incomparable.

    Argument order carries meaning, as in `scipy.stats.somersd`: this conditions on the first.
    """
    if len(truth) != len(predicted):
        raise ValueError(f"paired vectors must be the same length, got {len(truth)} and {len(predicted)}")
    if len(truth) < 2:
        return None

    concordant = discordant = 0
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            actual, claimed = truth[i] - truth[j], predicted[i] - predicted[j]
            if actual == 0 or claimed == 0:
                # Neither agrees nor disagrees. Whether the pair still counts against the
                # answer is the denominator's business: a truth tie leaves it entirely, a
                # predicted tie stays in and so costs the model the credit it forwent.
                continue
            if (actual > 0) == (claimed > 0):
                concordant += 1
            else:
                discordant += 1

    orderable = len(truth) * (len(truth) - 1) // 2 - _tie_pairs(truth)
    if orderable == 0:  # the truth orders nothing, so there was nothing to be right about
        return None
    return (concordant - discordant) / orderable


def weight_rank_somers_d(predicted: Sequence[Any], weights: dict[Any, float]) -> float | None:
    """Somers' D between the order `predicted` lists items in and the order their weights imply;
    `None` below two distinct items, or when they all weigh the same.

    Judges the sequence alone — which items were chosen is `hit_rate`'s question, what they
    were worth is `weight_captured`'s. Unknown items are weightless; repeats keep the first.
    """
    unique = list(dict.fromkeys(predicted))
    if len(unique) < 2:
        return None
    # Negated so that listed-earlier is the larger value, matching weighs-more.
    positions = [-index for index in range(len(unique))]
    return somers_d([weights.get(item, 0.0) for item in unique], positions)


def micro_ndcg(gains: Sequence[tuple[float, float]]) -> float | None:
    """One NDCG over every instance at once, from `weighted_dcg_at_k` pairs.

    Weights each instance by what is at stake in it where a macro mean counts them equally,
    and so moves when the dataset's weights change. Neither dominates; both are reported.
    """
    ideal = sum(best for _, best in gains)
    if ideal <= 0:
        return None
    return sum(actual for actual, _ in gains) / ideal


def weighted_ndcg_at_k(predicted: Sequence[Any], weights: dict[Any, float], expected: Sequence[Any]) -> float | None:
    """NDCG cut at `len(expected)`, an item's gain being its own weight rather than a flat 1.
    `None` when the expected weight is zero.

    A binary gain leaves the DCG unchanged under permutation, scoring a perfectly reversed
    answer 1.0; grading the gain by weight is what makes "most expensive first" scorable.
    Repeats drop to the first occurrence so the ideal ranking stays the ceiling.

    1.0 is the ceiling only if `expected` holds the heaviest items there are — the caller's
    contract, enforced by `TimeHotspot.reference`, since a violation scores above 1.0 rather
    than failing visibly.
    """
    actual, ideal = weighted_dcg_at_k(predicted, weights, expected)
    return actual / ideal if ideal > 0 else None
