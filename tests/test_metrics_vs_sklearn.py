"""Cross-check every metric against scikit-learn on randomised inputs.

These decide public leaderboard numbers, so agreeing with our own arithmetic is not
enough. Includes the degenerate shapes these datasets are full of: all-PASSED
instances, absent classes, single-class predictions.
"""

import math
import random

import pytest
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    ndcg_score,
    precision_score,
    recall_score,
)

from sourceworldbench_eval_harness import metrics
from sourceworldbench_eval_harness.schemas import status as schema

# A five-class vocabulary of this module's own. These tests exercise `metrics`, which
# is vocabulary-agnostic, and want more classes than any registered dataset declares —
# the names only need to be distinct, not to belong to a task.
VOCABULARY = ("PASSED", "FAILED", "ERROR", "SKIPPED", "OTHER")

CASES = 200


def label_pairs(seed: int, vocabulary: tuple[str, ...]):
    rng = random.Random(seed)
    for _ in range(CASES):
        n = rng.randint(1, 40)
        # Skew heavily toward PASSED, like the real data, and sometimes emit a
        # constant prediction — the always-PASSED baseline shape.
        truth = [rng.choices(vocabulary, weights=[90, 4, 3, 1, 2][: len(vocabulary)])[0] for _ in range(n)]
        if rng.random() < 0.3:
            pred = [schema.PASSED] * n
        else:
            pred = [rng.choice(vocabulary) for _ in range(n)]
        yield truth, pred


@pytest.mark.parametrize("vocabulary", [VOCABULARY, schema.BINARY_LABELS])
def test_per_class_prf_matches_sklearn(vocabulary):
    if vocabulary == schema.BINARY_LABELS:
        pairs = (
            (
                ["PASSED" if x == "PASSED" else "NOT_PASSED" for x in t],
                ["PASSED" if p == "PASSED" else "NOT_PASSED" for p in pr],
            )
            for t, pr in label_pairs(4, VOCABULARY)
        )
    else:
        pairs = label_pairs(4, vocabulary)

    for truth, pred in pairs:
        report = metrics.classification_report(truth, pred, vocabulary)
        for label in vocabulary:
            support = report["per_class"][label]["support"]
            if support == 0:
                continue  # sklearn cannot distinguish absent from missed
            expected_recall = recall_score(truth, pred, labels=[label], average="micro", zero_division=0)
            expected_precision = precision_score(truth, pred, labels=[label], average="micro", zero_division=0)
            assert report["per_class"][label]["recall"] == pytest.approx(expected_recall)
            assert report["per_class"][label]["precision"] == pytest.approx(expected_precision)


def test_macro_f1_over_present_classes_matches_sklearn():
    for truth, pred in label_pairs(5, VOCABULARY):
        report = metrics.classification_report(truth, pred, VOCABULARY)
        present = sorted({label for label in VOCABULARY if report["per_class"][label]["support"]})
        expected = f1_score(truth, pred, labels=present, average="macro", zero_division=0)
        assert report["macro_f1"] == pytest.approx(expected)


def test_balanced_accuracy_matches_sklearn():
    for truth, pred in label_pairs(6, VOCABULARY):
        report = metrics.classification_report(truth, pred, VOCABULARY)
        assert report["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(truth, pred))


def test_binary_mcc_matches_sklearn():
    checked = 0
    for truth, pred in label_pairs(7, VOCABULARY):
        truth = [schema.to_binary(x) for x in truth]
        pred = [schema.to_binary(x) for x in pred]
        report = metrics.classification_report(truth, pred, schema.BINARY_LABELS)
        if report["mcc"] is None:
            continue  # sklearn returns 0.0 where the denominator vanishes
        assert report["mcc"] == pytest.approx(matthews_corrcoef(truth, pred))
        checked += 1
    assert checked > 50


def test_accuracy_matches_sklearn():
    for truth, pred in label_pairs(8, VOCABULARY):
        report = metrics.classification_report(truth, pred, VOCABULARY)
        expected = sum(t == p for t, p in zip(truth, pred, strict=True)) / len(truth)
        assert report["accuracy"] == pytest.approx(expected)


def test_log10_error_is_symmetric_and_scale_free():
    assert metrics.log10_error(4.0, 2.0) == pytest.approx(math.log10(2))
    assert metrics.log10_error(1.0, 2.0) == pytest.approx(math.log10(2))
    assert metrics.log10_error(400.0, 200.0) == pytest.approx(math.log10(2))
    assert metrics.log10_error(20.0, 2.0) == pytest.approx(1.0)  # one order of magnitude
    assert metrics.log10_error(2.0, 2.0) == 0.0
    assert metrics.log10_error(0.0, 2.0) is None
    assert metrics.log10_error(2.0, -1.0) is None


def test_log_fit_recovers_a_known_line():
    truth = [1.0, 10.0, 100.0, 1000.0]

    slope, intercept = metrics.log_fit(truth, truth)
    assert (slope, intercept) == (pytest.approx(1.0), pytest.approx(0.0))

    # Uniformly 100x high: the line shifts, its slope does not.
    slope, intercept = metrics.log_fit([t * 100 for t in truth], truth)
    assert (slope, intercept) == (pytest.approx(1.0), pytest.approx(2.0))

    # Magnitudes compressed to the square root: ordering intact, scaling halved, which
    # the slope reports and no average of per-instance errors can.
    slope, _ = metrics.log_fit([t**0.5 for t in truth], truth)
    assert slope == pytest.approx(0.5)

    # A constant prediction has no slope at all.
    slope, intercept = metrics.log_fit([5.0] * 4, truth)
    assert (slope, intercept) == (pytest.approx(0.0), pytest.approx(math.log10(5)))


def test_log_fit_is_none_when_undetermined():
    assert metrics.log_fit([2.0], [1.0]) == (None, None)
    assert metrics.log_fit([], []) == (None, None)
    assert metrics.log_fit([1.0, 2.0], [3.0, 3.0]) == (None, None)  # constant target
    assert metrics.log_fit([0.0, -1.0], [1.0, 10.0]) == (None, None)  # no usable pair
    with pytest.raises(ValueError):
        metrics.log_fit([1.0, 2.0], [1.0])


def test_debiased_log10_error_removes_a_constant_offset():
    truth = [1.0, 10.0, 100.0, 1000.0]

    # Uniformly 100x high: every residual is the same, so nothing survives debiasing,
    # while `log10_error` charges the full two orders of magnitude.
    high = [t * 100 for t in truth]
    assert metrics.debiased_log10_error(high, truth) == pytest.approx(0.0)
    assert metrics.mean([metrics.log10_error(g, t) for g, t in zip(high, truth)]) == pytest.approx(2.0)

    # Scattered by the same factor either way: the median offset is 0, so debiasing
    # takes nothing out and the two figures agree.
    scattered = [1.0 * 100, 10.0 / 100, 100.0 * 100, 1000.0 / 100]
    assert metrics.debiased_log10_error(scattered, truth) == pytest.approx(2.0)

    assert metrics.debiased_log10_error(truth, truth) == pytest.approx(0.0)


def test_debiased_log10_error_never_exceeds_the_biased_one():
    """The median minimises the mean absolute residual, so recalibration can only help."""
    rng = random.Random(23)
    for _ in range(CASES):
        n = rng.randint(1, 25)
        truth = [10 ** rng.uniform(-1, 3) for _ in range(n)]
        guess = [t ** rng.uniform(0, 1.5) * 10 ** rng.uniform(-1, 1) for t in truth]
        biased = metrics.mean([metrics.log10_error(g, t) for g, t in zip(guess, truth)])
        assert metrics.debiased_log10_error(guess, truth) <= biased + 1e-12


def test_debiased_log10_error_is_none_without_a_usable_pair():
    assert metrics.debiased_log10_error([], []) is None
    assert metrics.debiased_log10_error([0.0, -1.0], [1.0, 10.0]) is None
    # One pair is enough: its own residual is the median, leaving an error of zero.
    assert metrics.debiased_log10_error([2.0], [1.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        metrics.debiased_log10_error([1.0, 2.0], [1.0])


def test_log_fit_matches_scipy():
    """Cross-checked like the rest: the least-squares fit is hand-rolled, so it is
    compared against an independent implementation over the scale the target spans."""
    from scipy.stats import linregress

    rng = random.Random(17)
    for _ in range(CASES):
        n = rng.randint(2, 25)
        truth = [10 ** rng.uniform(-1, 3) for _ in range(n)]  # 0.1s to 1000s
        guess = [t ** rng.uniform(0, 1.5) * 10 ** rng.uniform(-1, 1) for t in truth]
        slope, intercept = metrics.log_fit(guess, truth)
        if len({round(t, 12) for t in truth}) < 2:
            assert (slope, intercept) == (None, None)  # no slope from a constant target
            continue
        theirs = linregress([math.log10(t) for t in truth], [math.log10(g) for g in guess])
        assert slope == pytest.approx(theirs.slope, abs=1e-9)
        assert intercept == pytest.approx(theirs.intercept, abs=1e-9)


def test_somers_d_matches_scipy():
    """Including the ties that are the reason for choosing this variant: `truth` is drawn
    from a small pool so equal values occur constantly, which is the real dataset's shape
    where several functions cost the same."""
    from scipy.stats import somersd

    rng = random.Random(18)
    tied = checked = 0
    for _ in range(CASES):
        n = rng.randint(2, 20)
        # A small pool on the truth side and a strict ranking on the prediction side, so
        # ties land where equal exclusive times do.
        predicted = list(range(n))
        rng.shuffle(predicted)
        truth = [float(rng.randint(1, max(2, n // 3))) for _ in range(n)]
        theirs = somersd(truth, predicted)
        ours = metrics.somers_d(truth, predicted)
        if math.isnan(theirs.statistic):
            assert ours is None  # a constant truth orders nothing
            continue
        assert ours == pytest.approx(theirs.statistic)
        tied += len(set(truth)) < n
        checked += 1
    assert checked > 100
    assert tied > 50, "the tie handling went untested"


def test_somers_d_conditions_on_its_first_argument():
    """The asymmetry is the whole point, so swapping the arguments must change the answer.
    Three items, two of them tied in the truth: conditioning on the truth drops that pair
    and scores 2/2, conditioning on the prediction keeps it and scores 2/3. `scipy` splits
    the same way, which is why our parameter order copies its own."""
    from scipy.stats import somersd

    truth, predicted = [9.0, 5.0, 5.0], [3, 2, 1]

    assert metrics.somers_d(truth, predicted) == 1.0
    assert metrics.somers_d(predicted, truth) == pytest.approx(2 / 3)
    assert somersd(truth, predicted).statistic == 1.0
    assert somersd(predicted, truth).statistic == pytest.approx(2 / 3)


def test_somers_d_spans_minus_one_to_one():
    ascending = [1.0, 2.0, 3.0, 4.0]
    assert metrics.somers_d(ascending, ascending) == 1.0
    assert metrics.somers_d(ascending, list(reversed(ascending))) == -1.0
    # Half the pairs each way sits at zero, which is the floor NDCG has no way to express.
    assert metrics.somers_d([1, 2, 3, 4], [1, 4, 3, 2]) == pytest.approx(0.0)


def test_somers_d_reaches_one_despite_truth_ties_where_tau_b_could_not():
    """Why the denominator is `n0 - Ty` and not tau-b's `sqrt((n0 - Tx)(n0 - Ty))`.

    Two functions are tied in the truth, so that pair is unrankable and not asked about;
    ordering the other two correctly is a flawless answer and scores 1.0. Tau-b caps such an
    answer at `sqrt(1 - Ty/n0)`, a ceiling that moves with each instance's tie count.
    """
    from scipy.stats import kendalltau

    predicted, truth = [3, 2, 1], [9.0, 5.0, 5.0]

    assert metrics.somers_d(truth, predicted) == 1.0
    assert metrics.somers_d(truth, [3, 1, 2]) == 1.0  # the tied pair ordered the other way

    ceiling = kendalltau(predicted, truth, variant="b").statistic
    assert ceiling == pytest.approx(math.sqrt(2 / 3))  # sqrt(1 - Ty/n0), with Ty=1, n0=3
    assert ceiling < 1.0


def test_somers_d_scores_a_prediction_that_ties_everything_rather_than_dropping_it():
    """The other half of the asymmetry. A truth that orders nothing is `None` — nothing to get
    right, which `aggregate` drops.

    A *prediction* that orders nothing keeps every pair in the denominator with none
    concordant, so it scores 0.0: declining to rank is a wrong answer, not an excuse. Tau-b
    would call this undefined too and let the instance leave the average.
    """
    assert metrics.somers_d([1.0, 1.0, 1.0], [3.0, 1.0, 2.0]) is None
    assert metrics.somers_d([3.0, 1.0, 2.0], [1.0, 1.0, 1.0]) == 0.0
    assert metrics.somers_d([1.0], [2.0]) is None
    assert metrics.somers_d([], []) is None
    with pytest.raises(ValueError):
        metrics.somers_d([1.0, 2.0], [1.0])


def test_weight_rank_somers_d_reads_the_order_and_not_the_choice():
    """`weight_rank_somers_d` is scored on sequence alone, so an answer naming the cheapest
    functions still earns 1.0 for putting them in the right order — the complement of
    `hit_rate`, which scores the choice and ignores the order."""
    weights = {"a": 100.0, "b": 10.0, "c": 1.0, "cheap1": 0.2, "cheap2": 0.1}
    assert metrics.weight_rank_somers_d(["a", "b", "c"], weights) == 1.0
    assert metrics.weight_rank_somers_d(["c", "b", "a"], weights) == -1.0
    assert metrics.weight_rank_somers_d(["cheap1", "cheap2"], weights) == 1.0
    # Repeats drop to the first occurrence, and a name absent from `weights` is weightless.
    assert metrics.weight_rank_somers_d(["a", "a", "b"], weights) == 1.0
    assert metrics.weight_rank_somers_d(["a", "ghost"], weights) == 1.0
    assert metrics.weight_rank_somers_d(["a"], weights) is None
    assert metrics.weight_rank_somers_d([], weights) is None


def test_weight_rank_somers_d_does_not_charge_for_ordering_a_tie():
    """Functions that cost the same are unrankable — in the real profiles the equal values
    are microsecond rounding — so either order of a tied pair is accepted, and an answer
    whose named functions all cost the same is dropped rather than scored."""
    weights = {"a": 10.0, "b": 5.0, "c": 5.0}
    assert metrics.weight_rank_somers_d(["a", "b", "c"], weights) == 1.0
    assert metrics.weight_rank_somers_d(["a", "c", "b"], weights) == 1.0
    assert metrics.weight_rank_somers_d(["b", "c"], weights) is None


def test_weight_rank_somers_d_separates_a_reversal_that_ndcg_scores_perfectly():
    """The gap that earns this metric its place. With the right functions chosen, a
    gain-weighted NDCG still rewards a reversed answer highly, because the gains are all
    present and only the discount moves. The correlation calls it -1 outright."""
    weights = {"a": 10.0, "b": 9.0, "c": 8.0}
    expected = ["a", "b", "c"]
    reversed_answer = ["c", "b", "a"]

    ndcg = metrics.weighted_ndcg_at_k(reversed_answer, weights, expected)
    assert ndcg is not None and ndcg > 0.9
    assert metrics.weight_rank_somers_d(reversed_answer, weights) == -1.0


def test_within_factor_is_a_bounded_score():
    assert metrics.within_factor(4.0, 2.0, 2.0) == 1.0
    assert metrics.within_factor(1.0, 2.0, 2.0) == 1.0
    assert metrics.within_factor(4.1, 2.0, 2.0) == 0.0
    assert metrics.within_factor(2.0, 2.0, 1.25) == 1.0
    assert metrics.within_factor(0.0, 2.0) is None


def test_weighted_ndcg_exceeds_one_when_the_caller_breaks_its_contract():
    """Why `TimeHotspot.reference` validates the ranking rather than trusting it: the
    metric cannot see that `expected` is not the heaviest set there is, so it scores
    above 1.0 instead of failing."""
    weights = {"a": 1.0, "b": 0.5, "c": 0.25}
    assert metrics.weighted_ndcg_at_k(["a"], weights, ["a"]) == pytest.approx(1.0)
    assert metrics.weighted_ndcg_at_k(["a"], weights, ["b"]) == pytest.approx(2.0)


def test_hit_rate_counts_what_was_found():
    assert metrics.hit_rate(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert metrics.hit_rate(["a", "x", "y"], ["a", "b", "c"]) == pytest.approx(1 / 3)
    assert metrics.hit_rate(["x"], ["a"]) == 0.0
    assert metrics.hit_rate(["a"], []) is None  # nothing expected


def test_weight_captured_values_items_rather_than_counting_them():
    """One dominant item is worth more than several trivial ones, which is the whole
    reason to report this beside `hit_rate`."""
    weights = {"heavy": 9.0, "light1": 0.5, "light2": 0.5}
    expected = list(weights)
    assert metrics.weight_captured(["heavy"], weights, expected) == pytest.approx(0.9)
    assert metrics.weight_captured(["light1", "light2"], weights, expected) == pytest.approx(0.1)
    # same count, ten times the value
    assert metrics.hit_rate(["heavy"], expected) == metrics.hit_rate(["light1"], expected)
    assert metrics.weight_captured([], weights, expected) == 0.0
    assert metrics.weight_captured(["heavy"], {"a": 0.0}, ["a"]) is None


def test_a_tie_at_the_cutoff_scores_the_same_either_way():
    """The k-th and (k+1)-th items are often equally heavy, and which one a truncation kept is
    arbitrary. `weighted_ndcg_at_k` never charged for the choice because it weighs what was
    named; without `tie_tolerant` these two would."""
    weights = {"a": 900.0, "b": 500.0, "c": 300.0, "d": 200.0, "e": 100.0, "f": 100.0}
    expected = ["a", "b", "c", "d", "e"]
    answer = ["a", "b", "c", "d", "f"]

    assert metrics.hit_rate(answer, expected) == pytest.approx(0.8)
    forgiving = metrics.tie_tolerant(answer, weights, expected)
    assert metrics.hit_rate(answer, forgiving) == 1.0
    assert metrics.weight_captured(answer, weights, forgiving) == 1.0
    assert metrics.weighted_ndcg_at_k(answer, weights, forgiving) == pytest.approx(
        metrics.weighted_ndcg_at_k(answer, weights, expected)
    )


def test_a_near_miss_at_the_cutoff_is_still_a_miss():
    """Only exact equality is a coin flip, so a lighter item stays the wrong answer."""
    weights = {"a": 900.0, "b": 500.0, "c": 300.0, "d": 200.0, "e": 100.0, "g": 50.0}
    expected = ["a", "b", "c", "d", "e"]
    answer = ["a", "b", "c", "d", "g"]

    assert metrics.tie_tolerant(answer, weights, expected) == expected
    assert metrics.hit_rate(answer, expected) == pytest.approx(0.8)


def test_tied_items_cannot_cover_one_genuinely_missed():
    """Substitution is capped by the slots missed, so naming both light items does not excuse
    overlooking the heavy one."""
    weights = {"a": 900.0, "b": 100.0, "c": 100.0, "d": 100.0}
    expected = ["a", "b"]
    answer = ["c", "d"]

    forgiving = metrics.tie_tolerant(answer, weights, expected)
    assert metrics.hit_rate(answer, forgiving) == pytest.approx(0.5)


def test_weighted_ndcg_reads_an_order_that_binary_ndcg_cannot():
    """The reason the hotspot task grades its gain by time: under a binary gain every
    expected item is worth the same, so reversing them costs nothing at all."""
    weights = {"heavy": 10.0, "mid": 3.0, "light": 1.0}
    expected = ["heavy", "mid", "light"]

    # A binary gain is what uniform weights reduce to: the reversal scores perfect.
    flat = dict.fromkeys(expected, 1.0)
    assert metrics.weighted_ndcg_at_k(list(reversed(expected)), flat, expected) == 1.0
    assert metrics.weighted_ndcg_at_k(expected, weights, expected) == 1.0

    ideal = 10 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    worst = 1 / math.log2(2) + 3 / math.log2(3) + 10 / math.log2(4)
    reversed_score = metrics.weighted_ndcg_at_k(list(reversed(expected)), weights, expected)
    assert reversed_score == pytest.approx(worst / ideal)
    assert reversed_score < 1.0


def test_weighted_ndcg_scores_zero_rather_than_none_when_nothing_was_found():
    """A binary-gain NDCG is `None` on a miss, so its mean silently covers only the
    instances that hit — which made the k=1 figure read ~1.0 for every model. A graded
    gain has a defined value for a wrong answer, so the mean is over every instance."""
    weights = {"a": 4.0, "b": 2.0, "c": 1.0}
    assert metrics.weighted_ndcg_at_k(["c"], weights, ["a"]) == pytest.approx(0.25)
    assert metrics.weighted_ndcg_at_k([], weights, ["a"]) == 0.0
    assert metrics.weighted_ndcg_at_k(["a"], {"a": 0.0}, ["a"]) is None  # nothing to find


def test_micro_ndcg_weighs_an_instance_by_its_ideal_rather_than_equally():
    """Two instances, one worth 100x the other. Perfect on the heavy and nothing on the
    light averages to 0.5 either way round, which is the figure `micro_ndcg` replaces.
    """
    heavy, light = (10.0, 10.0), (0.1, 0.1)  # (achieved, ideal)
    nails_heavy, nails_light = [heavy, (0.0, 0.1)], [(0.0, 10.0), light]

    assert metrics.mean([1.0, 0.0]) == metrics.mean([0.0, 1.0]) == 0.5
    assert metrics.micro_ndcg(nails_heavy) == pytest.approx(10 / 10.1)
    assert metrics.micro_ndcg(nails_light) == pytest.approx(0.1 / 10.1)
    assert metrics.micro_ndcg([heavy, light]) == 1.0
    assert metrics.micro_ndcg([(0.0, 0.0)]) is None  # nothing to find anywhere
    assert metrics.micro_ndcg([]) is None


def test_weighted_dcg_at_k_splits_the_ratio_weighted_ndcg_returns():
    weights = {"a": 4.0, "b": 2.0, "c": 1.0}
    achieved, ideal = metrics.weighted_dcg_at_k(["b", "a"], weights, ["a", "b"])

    assert ideal == pytest.approx(4 / math.log2(2) + 2 / math.log2(3))
    assert achieved == pytest.approx(2 / math.log2(2) + 4 / math.log2(3))
    assert achieved / ideal == pytest.approx(metrics.weighted_ndcg_at_k(["b", "a"], weights, ["a", "b"]))


def test_weighted_ndcg_keeps_the_ideal_ranking_as_its_ceiling():
    weights = {"a": 4.0, "b": 2.0, "c": 1.0}
    expected = ["a", "b"]
    # A repeat cannot buy a second position, which would put the score above 1.0.
    assert metrics.weighted_ndcg_at_k(["a", "a"], weights, expected) == metrics.weighted_ndcg_at_k(
        ["a"], weights, expected
    )
    # Naming the item just outside the expected set is not the mistake naming a free one is.
    assert metrics.weighted_ndcg_at_k(["a", "c"], weights, expected) > metrics.weighted_ndcg_at_k(
        ["a", "unprofiled"], weights, expected
    )


@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_weighted_ndcg_matches_sklearn(k):
    rng = random.Random(11 + k)
    checked = 0
    for _ in range(CASES):
        size = rng.randint(max(k, 2), 20)  # sklearn rejects a single-document ranking
        items = [f"f{index}" for index in range(size)]
        weights = {item: rng.uniform(0.01, 100.0) for item in items}
        expected = sorted(items, key=lambda item: -weights[item])[:k]
        predicted = items[:]
        rng.shuffle(predicted)

        rank = {item: position for position, item in enumerate(predicted)}
        sklearn_ndcg = ndcg_score(
            [[weights[item] for item in items]],
            [[size - rank[item] for item in items]],
            k=k,
        )
        assert metrics.weighted_ndcg_at_k(predicted, weights, expected) == pytest.approx(sklearn_ndcg)
        checked += 1
    assert checked == CASES
