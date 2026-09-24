import copy
import logging

import pytest

import helpers
from helpers import FakeDataset, row
from sourceworldbench_eval_harness.models.core.base import PredictionFileError
from sourceworldbench_eval_harness.schemas.memory import (
    AXIS_FOR,
    FUNCTIONS_FIELD,
    HOTSPOT_AXES,
    HOTSPOT_QUESTIONS,
    TOP_K,
    triple,
)
from sourceworldbench_eval_harness.tasks import resolve
from sourceworldbench_eval_harness.tasks.core.base import InvalidSubmissionError, Task
from sourceworldbench_eval_harness.tasks.memory.hotspot import MemoryHotspot

REPOS = {
    "matplotlib__aaaaaaaa__prediction_x": "matplotlib",
    "pandas__bbbbbbbb__prediction_y": "pandas",
}

FIRST_ID, SECOND_ID = REPOS

(CUMULATIVE,) = HOTSPOT_AXES

# The dataset ranks deeper than the deepest question asks, and the pool is a little larger
# still, so there are functions that are real but not expensive.
PROFILED = 25
HOTSPOTS = 20


def function(repo, index):
    """One profiled function.

    Weight doubles with the index, so the ranking is unambiguous and the shares a top-`k`
    captures are easy to reason about. Call count runs with it, so `most_called` is aligned
    with the ranking in this fixture.
    """
    return {
        "key": f"{repo}.mod.f{index:02d}",
        "filename": f"{repo}/mod.py",
        "firstlineno": 10 * (index + 1),
        "qualname": f"f{index:02d}",
        "exclusive_alloc_delta_sum": 2.0 ** (index - (PROFILED - 1)),
        "call_count": index + 1,
    }


def functions(instance_id):
    return [function(REPOS[instance_id], i) for i in range(PROFILED)]


def ordered(instance_id, axis):
    """This instance's profile, largest first on `axis`."""
    return sorted(functions(instance_id), key=lambda f: f[axis.weight_field], reverse=True)


def keys(instance_id, axis, count):
    """The `count` largest functions on `axis`, largest first."""
    return [f["key"] for f in ordered(instance_id, axis)[:count]]


def weights(instance_id, axis):
    return {f["key"]: f[axis.weight_field] for f in functions(instance_id)}


# The shape `reference` returns: one index of every accepted spelling, and per axis the
# ranked hotspots with the sizes they are ranked by.
REFERENCE = {
    instance_id: {
        "index": {
            spelling: f["key"]
            for f in functions(instance_id)
            for spelling in (f["key"], triple(f["filename"], f["firstlineno"], f["qualname"]))
        },
        "axes": {
            axis.name: {"ranked": keys(instance_id, axis, HOTSPOTS), "weights": weights(instance_id, axis)}
            for axis in HOTSPOT_AXES
        },
    }
    for instance_id in REPOS
}


def eval_config(name="memory_hotspot", **task_settings):
    return helpers.eval_config(name, **task_settings)


def task(require_full_coverage=True, model=None, **settings):
    return helpers.build_task("memory_hotspot", require_full_coverage=require_full_coverage, model=model, **settings)


def flat_score(rows, **settings):
    return Task.flatten(task(**settings).score(rows, REFERENCE))


def answers(instance_id):
    """A perfect answer to every question, on whichever axis each one asks about."""
    return {q: keys(instance_id, AXIS_FOR[q], TOP_K[q]) for q in HOTSPOT_QUESTIONS}


def oracle_rows():
    return [row(instance_id, **answers(instance_id)) for instance_id in REPOS]


def dataset_rows():
    return FakeDataset(
        [
            {
                "instance_id": instance_id,
                FUNCTIONS_FIELD: functions(instance_id),
                **{axis.hotspots_field: ordered(instance_id, axis)[:HOTSPOTS] for axis in HOTSPOT_AXES},
                "repo": REPOS[instance_id],
                "base_commit": "deadbeef",
                "patch": "--- a/x\n+++ b/x\n",
                "container": "registry/image@sha256:abc",
                "workload": "tests/test_x.py",
                "test_command": "pytest -v",
                "full_command": "pytest -v tests/test_x.py",
                "scale": "medium",
            }
            for instance_id in REPOS
        ]
    )


def with_dataset(task_, rows=None):
    return helpers.attach_dataset(task_, dataset_rows() if rows is None else rows)


def test_oracle_scores_perfectly_on_every_question_in_one_run():
    results = flat_score(oracle_rows())

    for question in HOTSPOT_QUESTIONS:
        assert results[f"{question}_hit_rate"] == 1.0
        assert results[f"{question}_ndcg_macro"] == 1.0
        assert results[f"{question}_ndcg_micro"] == 1.0
    assert results["cumulative_top5_instances"] == 2
    assert results["instances"] == 2
    assert results["questions_answered"] == list(HOTSPOT_QUESTIONS)


def test_only_the_questions_the_dataset_asks_are_scored():
    """The ranking is deeper than any question, so `cumulative_top20` is not a question at
    all and must not be invented from the available depth."""
    assert "cumulative_top20" not in HOTSPOT_QUESTIONS
    assert set(TOP_K) == set(HOTSPOT_QUESTIONS)
    assert [TOP_K[q] for q in CUMULATIVE.questions] == [1, 5]


def test_one_question_can_be_answered_without_the_others():
    """`require_full_coverage` is per question, so a model that only ranks a top 5 is
    scored on `cumulative_top5` rather than failed for the columns it left out."""
    top5_only = [row(instance_id, cumulative_top5=keys(instance_id, CUMULATIVE, 5)) for instance_id in REPOS]
    results = flat_score(top5_only)

    assert results["questions_answered"] == ["cumulative_top5"]
    assert results["cumulative_top5_ndcg_macro"] == 1.0
    assert not any(key.startswith("cumulative_top1_") for key in results)


def test_finding_the_dominant_function_is_worth_more_than_finding_a_trivial_one():
    """The reason `bytes_captured` is reported next to `hit_rate`: both answers below find
    exactly one of the five expected functions."""
    filler = keys(FIRST_ID, CUMULATIVE, PROFILED)[-4:]
    heaviest = [row(FIRST_ID, cumulative_top5=[keys(FIRST_ID, CUMULATIVE, 1)[0], *filler])]
    lightest = [row(FIRST_ID, cumulative_top5=[keys(FIRST_ID, CUMULATIVE, 5)[4], *filler])]
    dominant = flat_score(heaviest, require_full_coverage=False)
    trivial = flat_score(lightest, require_full_coverage=False)

    assert dominant["cumulative_top5_hit_rate"] == trivial["cumulative_top5_hit_rate"] == pytest.approx(0.2)
    # Weights 1, 1/2, 1/4, 1/8, 1/16 -> the largest is 16/31 of the total.
    assert dominant["cumulative_top5_bytes_captured"] == pytest.approx(16 / 31)
    assert trivial["cumulative_top5_bytes_captured"] == pytest.approx(1 / 31)


def test_ndcg_penalises_the_right_functions_in_the_wrong_order():
    """The set metrics are blind to ordering, so a reversed answer is what decides whether
    `ndcg_macro` grades gain by size or by mere membership: a binary gain scores this 1.0."""
    expected = keys(FIRST_ID, CUMULATIVE, 5)
    perfect = flat_score([row(FIRST_ID, cumulative_top5=expected)], require_full_coverage=False)
    backwards = flat_score([row(FIRST_ID, cumulative_top5=list(reversed(expected)))], require_full_coverage=False)

    assert perfect["cumulative_top5_hit_rate"] == backwards["cumulative_top5_hit_rate"] == 1.0
    assert perfect["cumulative_top5_bytes_captured"] == backwards["cumulative_top5_bytes_captured"] == 1.0
    assert perfect["cumulative_top5_ndcg_macro"] == 1.0
    assert backwards["cumulative_top5_ndcg_macro"] < 0.7


def test_somers_d_calls_a_reversal_what_ndcg_still_scores_generously():
    """Why `somers_d` is reported beside `ndcg_macro`. The reversed answer above keeps a
    respectable NDCG because the set is right and only the discount moves; the correlation
    reads the ordering alone and bottoms out at -1."""
    expected = keys(FIRST_ID, CUMULATIVE, 5)
    perfect = flat_score([row(FIRST_ID, cumulative_top5=expected)], require_full_coverage=False)
    backwards = flat_score([row(FIRST_ID, cumulative_top5=list(reversed(expected)))], require_full_coverage=False)

    assert perfect["cumulative_top5_somers_d"] == 1.0
    assert backwards["cumulative_top5_somers_d"] == -1.0
    assert backwards["cumulative_top5_ndcg_macro"] > 0.5  # the NDCG floor is not a floor


def test_tau_b_is_reported_beside_somers_d_and_never_above_it():
    """The same ordering read two ways. They agree when nothing ties; where the truth ties,
    tau-b takes the discount that keeps it off 1.0, which is why it is not the ranked figure.
    """
    expected = keys(FIRST_ID, CUMULATIVE, 5)
    perfect = flat_score([row(FIRST_ID, cumulative_top5=expected)], require_full_coverage=False)
    backwards = flat_score([row(FIRST_ID, cumulative_top5=list(reversed(expected)))], require_full_coverage=False)

    assert perfect["cumulative_top5_tau_b"] <= perfect["cumulative_top5_somers_d"] + 1e-12
    assert backwards["cumulative_top5_tau_b"] >= backwards["cumulative_top5_somers_d"] - 1e-12
    assert backwards["cumulative_top5_tau_b"] < 0  # a reversal is negative either way


def test_top1_reports_neither_bytes_captured_nor_somers_d():
    """With one expected function the weighted share is 1 exactly when the unweighted one is,
    so `bytes_captured` would be `hit_rate` under a second name; and a single function has no
    order for a correlation to read."""
    results = flat_score(oracle_rows())

    for axis in HOTSPOT_AXES:
        assert f"{axis.name}_top1_bytes_captured" not in results
        assert f"{axis.name}_top1_somers_d" not in results
        assert results[f"{axis.name}_top5_bytes_captured"] == 1.0
        assert results[f"{axis.name}_top5_somers_d"] == 1.0


def test_top1_ndcg_is_defined_even_when_the_answer_is_wrong():
    """A binary gain is `None` on a miss, so the metric would average only over hits and read
    ~1.0 for every model. Graded by size, a wrong pick has a real value."""
    third = [row(FIRST_ID, cumulative_top1=[keys(FIRST_ID, CUMULATIVE, 3)[2]])]
    missed = flat_score(third, require_full_coverage=False)

    assert missed["cumulative_top1_hit_rate"] == 0.0
    assert missed["cumulative_top1_ndcg_macro"] == pytest.approx(0.25)  # weights halve, so the third is a quarter
    assert missed["cumulative_top1_ndcg_macro_scored"] == 1


def test_ndcg_micro_weights_an_instance_by_the_bytes_it_has_to_win():
    """The mean counts an instance holding a few kilobytes the same as one holding a gigabyte.
    `ndcg_micro` divides the summed gains instead, which the mean cannot express at all."""
    heavy, light = FIRST_ID, SECOND_ID
    scaled = {
        **REFERENCE[light],
        "axes": {
            axis.name: {
                "ranked": REFERENCE[light]["axes"][axis.name]["ranked"],
                "weights": {k: w / 100 for k, w in REFERENCE[light]["axes"][axis.name]["weights"].items()},
            }
            for axis in HOTSPOT_AXES
        },
    }
    reference = {heavy: REFERENCE[heavy], light: scaled}
    nails_heavy = [
        row(heavy, cumulative_top5=keys(heavy, CUMULATIVE, 5)),
        row(light, cumulative_top5=keys(light, CUMULATIVE, PROFILED)[-5:]),
    ]
    nails_light = [
        row(heavy, cumulative_top5=keys(heavy, CUMULATIVE, PROFILED)[-5:]),
        row(light, cumulative_top5=keys(light, CUMULATIVE, 5)),
    ]
    scored = [
        Task.flatten(task(require_full_coverage=False).score(rows, reference)) for rows in (nails_heavy, nails_light)
    ]

    assert scored[0]["cumulative_top5_ndcg_macro"] == pytest.approx(scored[1]["cumulative_top5_ndcg_macro"])
    assert scored[0]["cumulative_top5_ndcg_micro"] > 0.98
    assert scored[1]["cumulative_top5_ndcg_micro"] < 0.02


def test_ndcg_reads_the_order_that_hit_rate_cannot():
    """The same functions, the relevant one first or last."""
    expected, filler = keys(FIRST_ID, CUMULATIVE, 1)[0], keys(FIRST_ID, CUMULATIVE, PROFILED)[-4:]
    first = flat_score([row(FIRST_ID, cumulative_top5=[expected, *filler])], require_full_coverage=False)
    last = flat_score([row(FIRST_ID, cumulative_top5=[*filler, expected])], require_full_coverage=False)

    assert first["cumulative_top5_hit_rate"] == last["cumulative_top5_hit_rate"]
    assert first["cumulative_top5_ndcg_macro"] > last["cumulative_top5_ndcg_macro"]


def test_either_spelling_of_a_function_scores_the_same():
    """A model answers with the file/line/qualname triple the prompt asks for; the dataset's
    own `key` is accepted too, and neither is worth more."""
    spelled_out = [
        row(
            instance_id,
            cumulative_top1=[
                {"filename": f["filename"], "firstlineno": f["firstlineno"], "qualname": f["qualname"]}
                for f in ordered(instance_id, CUMULATIVE)[:1]
            ],
        )
        for instance_id in REPOS
    ]
    by_key = [row(instance_id, cumulative_top1=keys(instance_id, CUMULATIVE, 1)) for instance_id in REPOS]

    assert flat_score(spelled_out) == flat_score(by_key)
    assert flat_score(spelled_out)["cumulative_top1_hit_rate"] == 1.0


def test_a_function_that_exists_but_is_not_a_hotspot_scores_zero_not_none():
    """Measured and missed, which `aggregate` must not confuse with unmeasurable."""
    coldest = keys(FIRST_ID, CUMULATIVE, PROFILED)[-1]
    results = flat_score([row(FIRST_ID, cumulative_top1=[coldest])], require_full_coverage=False)

    assert results["cumulative_top1_hit_rate"] == 0.0
    assert results["cumulative_top1_hit_rate_scored"] == 1


def test_metrics_are_flat_and_no_question_carries_a_table():
    results = flat_score([row(FIRST_ID, cumulative_top1=keys(FIRST_ID, CUMULATIVE, 1))], require_full_coverage=False)

    assert all(not isinstance(value, dict) for value in results.values())
    assert "details" not in results
    assert not any(key.startswith("cumulative_top5_") for key in results)  # unanswered, so unscored


def test_valid_submission_passes():
    task().validate_shape(oracle_rows())
    task().validate(oracle_rows(), REFERENCE)


def test_a_function_outside_this_instances_profile_is_rejected():
    """What keeps an answer inside the codebase rather than naming a built-in — and catches a
    file whose rows were built for another instance."""
    rows = [row(FIRST_ID, cumulative_top1=["builtins.len"])]
    with pytest.raises(InvalidSubmissionError, match="not in this instance's profile"):
        task(require_full_coverage=False).validate(rows, REFERENCE)

    other = [row(FIRST_ID, cumulative_top1=keys(SECOND_ID, CUMULATIVE, 1))]
    with pytest.raises(InvalidSubmissionError, match="not in this instance's profile"):
        task(require_full_coverage=False).validate(other, REFERENCE)


def test_each_question_asks_for_exactly_its_own_number_of_functions():
    for question in HOTSPOT_QUESTIONS:
        task().validate_shape([row(FIRST_ID, **{question: keys(FIRST_ID, AXIS_FOR[question], TOP_K[question])})])

    with pytest.raises(InvalidSubmissionError, match="2 function\\(s\\) but cumulative_top1 asks for exactly 1"):
        task().validate_shape([row(FIRST_ID, cumulative_top1=keys(FIRST_ID, CUMULATIVE, 2))])
    with pytest.raises(InvalidSubmissionError, match="4 function\\(s\\) but cumulative_top5 asks for exactly 5"):
        task().validate_shape([row(FIRST_ID, cumulative_top5=keys(FIRST_ID, CUMULATIVE, 4))])


def test_an_answer_that_is_not_a_list_is_rejected():
    with pytest.raises(InvalidSubmissionError, match="must be a list of 1 function"):
        task().validate_shape([row(FIRST_ID, cumulative_top1=keys(FIRST_ID, CUMULATIVE, 1)[0])])


def test_the_same_function_listed_twice_is_rejected():
    """Five slots filled with one function would otherwise read as five found."""
    repeated = [keys(FIRST_ID, CUMULATIVE, 1)[0]] * 5
    with pytest.raises(InvalidSubmissionError, match="listed more than once"):
        task().validate_shape([row(FIRST_ID, cumulative_top5=repeated)])


def test_one_function_under_two_spellings_is_rejected_too():
    """The hedge the shape check cannot see: a `key` never equals its own triple as a string,
    so a function named both ways gets past it and collapses to one during scoring — leaving a
    short ranking that a rank correlation happily scores 1.0.
    """
    first, *rest = ordered(FIRST_ID, CUMULATIVE)[:5]
    hedged = [
        first["key"],
        {"filename": first["filename"], "firstlineno": first["firstlineno"], "qualname": first["qualname"]},
        *[f["key"] for f in rest[:3]],
    ]

    task().validate_shape([row(FIRST_ID, cumulative_top5=hedged)])  # the shape gate lets it through

    with pytest.raises(InvalidSubmissionError, match="under two different spellings"):
        task(require_full_coverage=False).validate([row(FIRST_ID, cumulative_top5=hedged)], REFERENCE)


def test_an_entry_missing_part_of_the_triple_names_both_accepted_spellings():
    partial = [{"filename": "matplotlib/mod.py", "qualname": "f00"}]  # no firstlineno
    with pytest.raises(InvalidSubmissionError, match="identify a function by its `key`"):
        task().validate_shape([row(FIRST_ID, cumulative_top1=partial)])
    with pytest.raises(InvalidSubmissionError, match="identify a function by its `key`"):
        task().validate_shape([row(FIRST_ID, cumulative_top1=[42])])


def test_reference_carries_the_ranking_the_weights_and_both_spellings():
    assert with_dataset(task()).reference() == REFERENCE


def test_an_instance_too_shallow_for_its_deepest_question_is_dropped_not_fatal(caplog):
    """A workload can rank fewer functions than a question asks for. Failing the run would
    make one short workload cost the whole dataset, so the instance leaves the reference and
    the others still score."""
    rows = dataset_rows()
    rows[0][CUMULATIVE.hotspots_field] = rows[0][CUMULATIVE.hotspots_field][:4]

    with caplog.at_level(logging.WARNING):
        reference = with_dataset(task(), rows).reference()

    assert set(reference) == {SECOND_ID}
    assert f"{FIRST_ID} (cumulative=4)" in caplog.text


def test_a_depth_that_still_covers_every_question_is_kept():
    """The deepest question asks for five, so five is enough — the dataset shipping a longer
    ranking must not become a requirement."""
    rows = dataset_rows()
    rows[0][CUMULATIVE.hotspots_field] = rows[0][CUMULATIVE.hotspots_field][:5]

    reference = with_dataset(task(), rows).reference()
    assert set(reference) == set(REPOS)
    assert reference[FIRST_ID]["axes"][CUMULATIVE.name]["ranked"] == keys(FIRST_ID, CUMULATIVE, 5)


def test_dropping_a_short_instance_leaves_the_rest_scoring_perfectly():
    """The dropped instance must not be quietly required of a submission either: `validate`
    wants every reference instance answered, so a file covering only the survivor passes."""
    rows = dataset_rows()
    rows[0][CUMULATIVE.hotspots_field] = rows[0][CUMULATIVE.hotspots_field][:4]
    scored = with_dataset(task(), rows)
    reference = scored.reference()
    survivor = [r for r in oracle_rows() if r["instance_id"] == SECOND_ID]

    scored.validate(survivor, reference)
    results = Task.flatten(scored.score(survivor, reference))

    assert results["cumulative_top5_ndcg_macro"] == 1.0


@pytest.mark.parametrize("axis", HOTSPOT_AXES, ids=[axis.name for axis in HOTSPOT_AXES])
def test_a_hotspot_absent_from_the_profile_is_rejected(axis):
    """The defect that would inflate `ndcg_macro` past 1.0: a pool trimmed after the rankings
    were computed leaves the ideal DCG below the real top-k. Either axis can be the one that
    breaks.
    """
    rows = dataset_rows()
    kept = {f["key"] for f in ordered(FIRST_ID, axis)[: HOTSPOTS - 1]}
    rows[0][FUNCTIONS_FIELD] = [f for f in rows[0][FUNCTIONS_FIELD] if f["key"] in kept]

    with pytest.raises(InvalidSubmissionError, match=f"absent from {FUNCTIONS_FIELD}"):
        with_dataset(task(), rows).reference()


@pytest.mark.parametrize("axis", HOTSPOT_AXES, ids=[axis.name for axis in HOTSPOT_AXES])
def test_hotspots_in_the_wrong_order_are_rejected(axis):
    rows = dataset_rows()
    hotspots = [dict(h) for h in rows[0][axis.hotspots_field]]
    hotspots[0], hotspots[1] = hotspots[1], hotspots[0]
    rows[0][axis.hotspots_field] = hotspots

    with pytest.raises(InvalidSubmissionError, match=f"descending {axis.weight_field} order"):
        with_dataset(task(), rows).reference()


def test_hotspots_that_are_not_the_costliest_functions_are_rejected():
    """Descending order is not enough on its own — the ranking also has to hold the largest
    functions there are, or a submission naming a larger one scores above 1.0."""
    rows = dataset_rows()
    hotspots = [dict(h) for h in rows[0][CUMULATIVE.hotspots_field]]
    hotspots[-1] = function(REPOS[FIRST_ID], 0)  # the smallest profiled one
    rows[0][CUMULATIVE.hotspots_field] = hotspots

    with pytest.raises(InvalidSubmissionError, match=f"descending {CUMULATIVE.weight_field} order"):
        with_dataset(task(), rows).reference()


def test_a_hotspot_repeated_in_the_dataset_is_rejected():
    rows = dataset_rows()
    hotspots = [dict(h) for h in rows[0][CUMULATIVE.hotspots_field]]
    hotspots[1] = dict(hotspots[0])
    rows[0][CUMULATIVE.hotspots_field] = hotspots

    with pytest.raises(InvalidSubmissionError, match="lists the same function more than once"):
        with_dataset(task(), rows).reference()


def test_functions_tied_on_size_may_be_ranked_either_way():
    """The ordering check compares weights rather than keys, so a profile with ties is not
    forced into one arbitrary order."""
    rows = dataset_rows()
    heaviest, second = ordered(FIRST_ID, CUMULATIVE)[:2]
    profile = [dict(f) for f in rows[0][FUNCTIONS_FIELD]]
    for entry in profile:
        if entry["key"] == second["key"]:
            entry[CUMULATIVE.weight_field] = heaviest[CUMULATIVE.weight_field]
    rows[0][FUNCTIONS_FIELD] = profile
    hotspots = [dict(h) for h in rows[0][CUMULATIVE.hotspots_field]]
    hotspots[0], hotspots[1] = hotspots[1], hotspots[0]
    rows[0][CUMULATIVE.hotspots_field] = hotspots

    reference = with_dataset(task(), rows).reference()
    assert reference[FIRST_ID]["axes"][CUMULATIVE.name]["ranked"][:2] == [second["key"], heaviest["key"]]


def test_a_tie_at_a_cutoff_is_not_charged_to_the_answer():
    """Naming either of two equally heavy functions must score alike, as `ndcg_macro` already
    did — the reference keeps only one of them inside the top `k`, arbitrarily."""
    reference = copy.deepcopy(REFERENCE)
    axis = reference[FIRST_ID]["axes"][CUMULATIVE.name]
    ranked, k = axis["ranked"], TOP_K["cumulative_top5"]
    axis["weights"][ranked[k]] = axis["weights"][ranked[k - 1]]

    substituted = [row(FIRST_ID, cumulative_top5=[*ranked[: k - 1], ranked[k]])]
    scored = Task.flatten(task().score(substituted, reference))

    assert scored["cumulative_top5_hit_rate"] == 1.0
    assert scored["cumulative_top5_bytes_captured"] == 1.0


def test_a_duplicate_instance_in_the_dataset_is_rejected():
    helpers.expect_duplicate_instance_rejected(task(), dataset_rows())


def test_a_dataset_missing_a_needed_column_names_what_is_available():
    helpers.expect_missing_column_named(task(), dataset_rows(), FUNCTIONS_FIELD)


def test_a_dataset_missing_the_ranking_column_names_it():
    helpers.expect_missing_column_named(task(), dataset_rows(), CUMULATIVE.hotspots_field)


def test_a_baseline_ranks_once_and_slices_per_question():
    rows = with_dataset(task(model="most_called")).predict(None)

    assert len(rows) == 2
    assert [len(rows[0][q]) for q in HOTSPOT_QUESTIONS] == [TOP_K[q] for q in HOTSPOT_QUESTIONS]
    assert rows[0]["cumulative_top1"] == rows[0]["cumulative_top5"][:1]  # one ranking, sliced


def test_most_called_is_a_heuristic_and_is_scored_like_any_other():
    """Call count runs with allocation in this fixture, so the baseline tops out — the point
    is that it goes through the same scoring as any submission."""
    hotspot = with_dataset(task(model="most_called"))
    results = Task.flatten(hotspot.score(hotspot.predict(None), REFERENCE))

    assert results["cumulative_top1_hit_rate"] == 1.0
    assert results["cumulative_top5_hit_rate"] == 1.0


def test_random_functions_is_seeded_and_draws_from_the_profile():
    def predict(seed):
        hotspot = with_dataset(task(model="random_functions"))
        hotspot.cfg.predict.params = {"seed": seed}
        return hotspot.predict(None)

    first = predict(7)
    assert [r["cumulative_top5"] for r in predict(7)] == [r["cumulative_top5"] for r in first]
    assert set(first[0]["cumulative_top5"]) <= set(keys(FIRST_ID, CUMULATIVE, PROFILED))


def test_the_oracle_round_trips_through_validation_and_scoring():
    """The sanity check to run against a new config: it must come out at 1.0."""
    hotspot = with_dataset(task(model="oracle"))

    rows = hotspot.predict(None)
    reference = hotspot.reference()
    hotspot.validate_shape(rows)
    hotspot.validate(rows, reference)
    results = Task.flatten(hotspot.score(rows, reference))

    assert results["cumulative_top1_hit_rate"] == 1.0
    assert results["cumulative_top5_bytes_captured"] == 1.0
    assert results["cumulative_top5_somers_d"] == 1.0


def test_an_external_file_is_normalised_to_the_id_and_questions():
    external = [row(FIRST_ID, cumulative_top1=keys(FIRST_ID, CUMULATIVE, 1), notes="dropped")]
    assert task().predict(external) == [{"instance_id": FIRST_ID, "cumulative_top1": keys(FIRST_ID, CUMULATIVE, 1)}]


def test_an_external_column_can_be_renamed_and_questions_narrowed():
    hotspot = task()
    hotspot.cfg.predict.params = {
        "questions": ["cumulative_top1"],
        "question_fields": {"cumulative_top1": "top_function"},
    }
    rows = hotspot.predict([{"instance_id": FIRST_ID, "top_function": ["x"], "cumulative_top5": ["y"]}])

    # cumulative_top5 was not kept, so it is not scored
    assert rows == [{"instance_id": FIRST_ID, "cumulative_top1": ["x"]}]


def test_an_external_row_answering_nothing_fails_the_run():
    with pytest.raises(PredictionFileError, match="answer none of"):
        task().predict([{"instance_id": FIRST_ID}])


def test_config_selects_the_task():
    assert isinstance(resolve(eval_config()), MemoryHotspot)
