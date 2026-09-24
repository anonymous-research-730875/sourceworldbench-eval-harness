import logging

import pytest

import helpers
from helpers import FakeDataset, row
from sourceworldbench_eval_harness.models.core.base import PredictionFileError
from sourceworldbench_eval_harness.schemas.time import (
    FUNCTIONS_FIELD,
    HOTSPOT_QUESTIONS,
    HOTSPOT_TIME_FIELD,
    HOTSPOTS_FIELD,
    TOP_K,
    triple,
)
from sourceworldbench_eval_harness.tasks import resolve
from sourceworldbench_eval_harness.tasks.core.base import InvalidSubmissionError, Task
from sourceworldbench_eval_harness.tasks.time.hotspot import TimeHotspot

REPOS = {
    "matplotlib__aaaaaaaa__prediction_x": "matplotlib",
    "pandas__bbbbbbbb__prediction_y": "pandas",
}

FIRST_ID, SECOND_ID = REPOS

# `top20` asks for 20, so an instance needs at least that many profiled hotspots; the pool
# is a little larger so there are functions that are real but not expensive.
PROFILED = 25
HOTSPOTS = 20


def function(repo, index):
    """One profiled function.

    Exclusive cost halves with each one while call count runs the other way, so the profile
    and a call-count ranking are exact opposites: `most_called` cannot look good by accident,
    and a test that reads one for the other cannot pass by accident either.
    """
    return {
        "key": f"{repo}.mod.f{index:02d}",
        "filename": f"{repo}/mod.py",
        "firstlineno": 10 * (index + 1),
        "qualname": f"f{index:02d}",
        "exclusive_time_s": 2.0 ** (-index),
        "call_count": index + 1,
    }


def functions(instance_id):
    return [function(REPOS[instance_id], i) for i in range(PROFILED)]


def ordered(instance_id):
    """This instance's profile, most expensive first."""
    return sorted(functions(instance_id), key=lambda f: f[HOTSPOT_TIME_FIELD], reverse=True)


def keys(instance_id, count):
    """The `count` most expensive functions, most expensive first."""
    return [f["key"] for f in ordered(instance_id)[:count]]


# The shape `reference` returns: one index of every accepted spelling, the ranked hotspots,
# and the weights they are ranked by.
REFERENCE = {
    instance_id: {
        "index": {
            spelling: f["key"]
            for f in functions(instance_id)
            for spelling in (f["key"], triple(f["filename"], f["firstlineno"], f["qualname"]))
        },
        "ranked": keys(instance_id, HOTSPOTS),
        "weights": {f["key"]: f[HOTSPOT_TIME_FIELD] for f in functions(instance_id)},
    }
    for instance_id in REPOS
}


def eval_config(name="time_hotspot", **task_settings):
    return helpers.eval_config(name, **task_settings)


def task(require_full_coverage=True, model=None, **settings):
    return helpers.build_task("time_hotspot", require_full_coverage=require_full_coverage, model=model, **settings)


def flat_score(rows, **settings):
    return Task.flatten(task(**settings).score(rows, REFERENCE))


def answers(instance_id):
    """A perfect answer to every question."""
    return {q: keys(instance_id, TOP_K[q]) for q in HOTSPOT_QUESTIONS}


def oracle_rows():
    return [row(instance_id, **answers(instance_id)) for instance_id in REPOS]


def dataset_rows():
    return FakeDataset(
        [
            {
                "instance_id": instance_id,
                FUNCTIONS_FIELD: functions(instance_id),
                HOTSPOTS_FIELD: ordered(instance_id)[:HOTSPOTS],
                "repo": REPOS[instance_id],
                "base_commit": "deadbeef",
                "patch": "--- a/x\n+++ b/x\n",
                "container": "registry/image@sha256:abc",
                "workload": "tests/test_x.py",
                "test_command": "pytest -v",
                "full_command": "pytest -v tests/test_x.py",
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
    assert results["exclusive_top5_instances"] == 2
    assert results["instances"] == 2
    assert results["questions_answered"] == list(HOTSPOT_QUESTIONS)


def test_one_question_can_be_answered_without_the_others():
    """`require_full_coverage` is per question, so a model that only ranks a top 20 is scored
    on `top20` rather than failed for the columns it left out.
    """
    top20_only = [row(instance_id, exclusive_top20=keys(instance_id, 20)) for instance_id in REPOS]
    results = flat_score(top20_only)

    assert results["questions_answered"] == ["exclusive_top20"]
    assert results["exclusive_top20_ndcg_macro"] == 1.0
    assert not any(key.startswith("exclusive_top1_") for key in results)


def test_finding_the_dominant_function_is_worth_more_than_finding_a_trivial_one():
    """The reason `time_captured` is reported next to `hit_rate`: both answers below
    find exactly one of the five expected functions."""
    filler = keys(FIRST_ID, PROFILED)[-4:]
    dominant = flat_score([row(FIRST_ID, exclusive_top5=[keys(FIRST_ID, 1)[0], *filler])], require_full_coverage=False)
    trivial = flat_score([row(FIRST_ID, exclusive_top5=[keys(FIRST_ID, 5)[4], *filler])], require_full_coverage=False)

    assert dominant["exclusive_top5_hit_rate"] == trivial["exclusive_top5_hit_rate"] == pytest.approx(0.2)
    # Weights 1, 1/2, 1/4, 1/8, 1/16 -> the most expensive is 16/31 of the total.
    assert dominant["exclusive_top5_time_captured"] == pytest.approx(16 / 31)
    assert trivial["exclusive_top5_time_captured"] == pytest.approx(1 / 31)


def test_ndcg_penalises_the_right_functions_in_the_wrong_order():
    """The set metrics are blind to ordering by construction, so a perfectly reversed
    answer is the case that decides whether `ndcg_macro` grades its gain by time or by mere
    membership. A binary gain scores this 1.0; graded by exclusive time it is ~0.57."""
    expected = keys(FIRST_ID, 5)
    perfect = flat_score([row(FIRST_ID, exclusive_top5=expected)], require_full_coverage=False)
    backwards = flat_score([row(FIRST_ID, exclusive_top5=list(reversed(expected)))], require_full_coverage=False)

    assert perfect["exclusive_top5_hit_rate"] == backwards["exclusive_top5_hit_rate"] == 1.0
    assert perfect["exclusive_top5_time_captured"] == backwards["exclusive_top5_time_captured"] == 1.0
    assert perfect["exclusive_top5_ndcg_macro"] == 1.0
    assert backwards["exclusive_top5_ndcg_macro"] < 0.7


def test_somers_d_calls_a_reversal_what_ndcg_still_scores_generously():
    """Why `somers_d` is reported beside `ndcg_macro`. The reversed answer above keeps a
    respectable NDCG because the set is right and only the discount moves; the correlation
    reads the ordering alone and bottoms out at -1.
    """
    expected = keys(FIRST_ID, 5)
    perfect = flat_score([row(FIRST_ID, exclusive_top5=expected)], require_full_coverage=False)
    backwards = flat_score([row(FIRST_ID, exclusive_top5=list(reversed(expected)))], require_full_coverage=False)

    assert perfect["exclusive_top5_somers_d"] == 1.0
    assert backwards["exclusive_top5_somers_d"] == -1.0
    assert backwards["exclusive_top5_ndcg_macro"] > 0.5  # the NDCG floor is not a floor


def test_tau_b_is_reported_beside_somers_d_and_never_above_it():
    """The same ordering read two ways. They agree when nothing ties; where the truth ties,
    tau-b takes the discount that keeps it off 1.0, which is why it is not the ranked figure.
    """
    expected = keys(FIRST_ID, 5)
    perfect = flat_score([row(FIRST_ID, exclusive_top5=expected)], require_full_coverage=False)
    backwards = flat_score([row(FIRST_ID, exclusive_top5=list(reversed(expected)))], require_full_coverage=False)

    assert perfect["exclusive_top5_tau_b"] <= perfect["exclusive_top5_somers_d"] + 1e-12
    assert backwards["exclusive_top5_tau_b"] >= backwards["exclusive_top5_somers_d"] - 1e-12
    assert backwards["exclusive_top5_tau_b"] < 0  # a reversal is negative either way


def test_top1_reports_neither_time_captured_nor_somers_d():
    """With one expected function the weighted share is 1 exactly when the unweighted one
    is, so `time_captured` would be `hit_rate` under a second name; and a single function
    has no order for a correlation to read."""
    results = flat_score(oracle_rows())

    assert "exclusive_top1_time_captured" not in results
    assert "exclusive_top1_somers_d" not in results
    assert results["exclusive_top5_time_captured"] == results["exclusive_top20_time_captured"] == 1.0
    assert results["exclusive_top5_somers_d"] == 1.0


def test_top1_ndcg_is_defined_even_when_the_answer_is_wrong():
    """A binary gain is `None` on a miss, so `top1_ndcg_macro` averaged only over the instances
    that hit and read ~1.0 for every model. Graded by time, a wrong single answer has a
    real value: the share of the hottest function's cost that the pick accounted for."""
    missed = flat_score([row(FIRST_ID, exclusive_top1=[keys(FIRST_ID, 3)[2]])], require_full_coverage=False)

    assert missed["exclusive_top1_hit_rate"] == 0.0
    # weights halve, so the third is a quarter
    assert missed["exclusive_top1_ndcg_macro"] == pytest.approx(0.25)
    assert missed["exclusive_top1_ndcg_macro_scored"] == 1


def test_ndcg_micro_weights_an_instance_by_the_time_it_has_to_win():
    """The mean counts an instance holding 0.02s the same as one holding 1.5s.
    `ndcg_micro` divides the summed gains instead, so getting the heavy one right is
    worth ~100x getting the light one right — which the mean cannot express at all."""
    heavy, light = FIRST_ID, SECOND_ID
    reference = {
        heavy: REFERENCE[heavy],
        light: {
            **REFERENCE[light],
            "weights": {key: weight / 100 for key, weight in REFERENCE[light]["weights"].items()},
        },
    }
    nails_heavy = [
        row(heavy, exclusive_top5=keys(heavy, 5)),
        row(light, exclusive_top5=keys(light, PROFILED)[-5:]),
    ]
    nails_light = [
        row(heavy, exclusive_top5=keys(heavy, PROFILED)[-5:]),
        row(light, exclusive_top5=keys(light, 5)),
    ]
    scored = [
        Task.flatten(task(require_full_coverage=False).score(rows, reference)) for rows in (nails_heavy, nails_light)
    ]

    assert scored[0]["exclusive_top5_ndcg_macro"] == pytest.approx(scored[1]["exclusive_top5_ndcg_macro"])
    assert scored[0]["exclusive_top5_ndcg_micro"] > 0.98
    assert scored[1]["exclusive_top5_ndcg_micro"] < 0.02


def test_ndcg_reads_the_order_that_hit_rate_cannot():
    """The same functions, the relevant one first or last."""
    expected, filler = keys(FIRST_ID, 1)[0], keys(FIRST_ID, PROFILED)[-4:]
    first = flat_score([row(FIRST_ID, exclusive_top5=[expected, *filler])], require_full_coverage=False)
    last = flat_score([row(FIRST_ID, exclusive_top5=[*filler, expected])], require_full_coverage=False)

    assert first["exclusive_top5_hit_rate"] == last["exclusive_top5_hit_rate"]
    assert first["exclusive_top5_ndcg_macro"] > last["exclusive_top5_ndcg_macro"]


def test_either_spelling_of_a_function_scores_the_same():
    """A model answers with the file/line/qualname triple the prompt asks for; the
    dataset's own `key` is accepted too, and neither is worth more."""
    spelled_out = [
        row(
            instance_id,
            exclusive_top1=[
                {"filename": f["filename"], "firstlineno": f["firstlineno"], "qualname": f["qualname"]}
                for f in ordered(instance_id)[:1]
            ],
        )
        for instance_id in REPOS
    ]
    by_key = [row(instance_id, exclusive_top1=keys(instance_id, 1)) for instance_id in REPOS]

    assert flat_score(spelled_out) == flat_score(by_key)
    assert flat_score(spelled_out)["exclusive_top1_hit_rate"] == 1.0


def test_a_function_that_exists_but_is_not_a_hotspot_scores_zero_not_none():
    """Measured and missed, which `aggregate` must not confuse with unmeasurable."""
    results = flat_score([row(FIRST_ID, exclusive_top1=[keys(FIRST_ID, PROFILED)[-1]])], require_full_coverage=False)

    assert results["exclusive_top1_hit_rate"] == 0.0
    assert results["exclusive_top1_hit_rate_scored"] == 1


def test_metrics_are_flat_and_no_question_carries_a_table():
    results = flat_score([row(FIRST_ID, exclusive_top1=keys(FIRST_ID, 1))], require_full_coverage=False)

    assert all(not isinstance(value, dict) for value in results.values())
    assert "details" not in results
    assert not any(key.startswith("exclusive_top5_") for key in results)  # unanswered, so unscored


def test_valid_submission_passes():
    task().validate_shape(oracle_rows())
    task().validate(oracle_rows(), REFERENCE)


def test_a_function_outside_this_instances_profile_is_rejected():
    """What keeps an answer inside the codebase rather than naming a built-in — and
    catches a file whose rows were built for another instance."""
    rows = [row(FIRST_ID, exclusive_top1=["builtins.len"])]
    with pytest.raises(InvalidSubmissionError, match="not in this instance's profile"):
        task(require_full_coverage=False).validate(rows, REFERENCE)

    other = [row(FIRST_ID, exclusive_top1=keys(SECOND_ID, 1))]
    with pytest.raises(InvalidSubmissionError, match="not in this instance's profile"):
        task(require_full_coverage=False).validate(other, REFERENCE)


def test_each_question_asks_for_exactly_its_own_number_of_functions():
    for question in HOTSPOT_QUESTIONS:
        task().validate_shape([row(FIRST_ID, **{question: keys(FIRST_ID, TOP_K[question])})])

    with pytest.raises(InvalidSubmissionError, match="2 function\\(s\\) but exclusive_top1 asks for exactly 1"):
        task().validate_shape([row(FIRST_ID, exclusive_top1=keys(FIRST_ID, 2))])
    with pytest.raises(InvalidSubmissionError, match="4 function\\(s\\) but exclusive_top5 asks for exactly 5"):
        task().validate_shape([row(FIRST_ID, exclusive_top5=keys(FIRST_ID, 4))])


def test_an_answer_that_is_not_a_list_is_rejected():
    with pytest.raises(InvalidSubmissionError, match="must be a list of 1 function"):
        task().validate_shape([row(FIRST_ID, exclusive_top1=keys(FIRST_ID, 1)[0])])


def test_the_same_function_listed_twice_is_rejected():
    """Five slots filled with one function would otherwise read as five found."""
    repeated = [keys(FIRST_ID, 1)[0]] * 5
    with pytest.raises(InvalidSubmissionError, match="listed more than once"):
        task().validate_shape([row(FIRST_ID, exclusive_top5=repeated)])


def test_one_function_under_two_spellings_is_rejected_too():
    """The hedge the shape check cannot see. `validate_shape` compares the entries as
    submitted and a `key` never equals its own triple as a string, so a function named
    both ways gets past it and then collapses to one during scoring — leaving a four-long
    ranking that a rank correlation happily scores 1.0. The duplicate is caught after
    resolution instead.
    """
    first, *rest = ordered(FIRST_ID)[:5]
    hedged = [
        first["key"],
        {
            "filename": first["filename"],
            "firstlineno": first["firstlineno"],
            "qualname": first["qualname"],
        },
        *[f["key"] for f in rest[:3]],
    ]

    task().validate_shape([row(FIRST_ID, exclusive_top5=hedged)])  # the shape gate lets it through

    with pytest.raises(InvalidSubmissionError, match="under two different spellings"):
        task(require_full_coverage=False).validate([row(FIRST_ID, exclusive_top5=hedged)], REFERENCE)


def test_an_entry_missing_part_of_the_triple_names_both_accepted_spellings():
    partial = [{"filename": "matplotlib/mod.py", "qualname": "f00"}]  # no firstlineno
    with pytest.raises(InvalidSubmissionError, match="identify a function by its `key`"):
        task().validate_shape([row(FIRST_ID, exclusive_top1=partial)])
    with pytest.raises(InvalidSubmissionError, match="identify a function by its `key`"):
        task().validate_shape([row(FIRST_ID, exclusive_top1=[42])])


def test_reference_carries_the_ranking_the_weights_and_both_spellings():
    assert with_dataset(task()).reference() == REFERENCE


def test_an_instance_with_too_few_hotspots_to_score_is_dropped_not_fatal(caplog):
    """A workload can profile fewer than 20 functions, and `top20` is then unanswerable
    for it. Failing the run would make one short workload cost the whole dataset, so the
    instance leaves the reference and the others still score."""
    rows = dataset_rows()
    rows[0][HOTSPOTS_FIELD] = rows[0][HOTSPOTS_FIELD][:19]

    with caplog.at_level(logging.WARNING):
        reference = with_dataset(task(), rows).reference()

    assert set(reference) == {SECOND_ID}
    assert f"{FIRST_ID} (19)" in caplog.text


def test_dropping_a_short_instance_leaves_the_rest_scoring_perfectly():
    """The dropped instance must not be quietly required of a submission either: `validate`
    wants every reference instance answered, so a file covering only the survivor passes."""
    rows = dataset_rows()
    rows[0][HOTSPOTS_FIELD] = rows[0][HOTSPOTS_FIELD][:19]
    scored = with_dataset(task(), rows)
    reference = scored.reference()
    survivor = [r for r in oracle_rows() if r["instance_id"] == SECOND_ID]

    scored.validate(survivor, reference)
    results = Task.flatten(scored.score(survivor, reference))

    assert results["exclusive_top20_ndcg_macro"] == 1.0


def test_a_hotspot_absent_from_the_profile_is_rejected():
    """The defect that would inflate `ndcg_macro` past 1.0: `functions_complete_list` trimmed
    after `hotspots` was computed, so an expected function has no weight and the ideal DCG
    falls below the real top-k."""
    rows = dataset_rows()
    rows[0][FUNCTIONS_FIELD] = rows[0][FUNCTIONS_FIELD][: HOTSPOTS - 1]

    with pytest.raises(InvalidSubmissionError, match=f"absent from {FUNCTIONS_FIELD}"):
        with_dataset(task(), rows).reference()


def test_hotspots_in_the_wrong_order_are_rejected():
    rows = dataset_rows()
    hotspots = [dict(h) for h in rows[0][HOTSPOTS_FIELD]]
    hotspots[0], hotspots[1] = hotspots[1], hotspots[0]
    rows[0][HOTSPOTS_FIELD] = hotspots

    with pytest.raises(InvalidSubmissionError, match=f"descending {HOTSPOT_TIME_FIELD} order"):
        with_dataset(task(), rows).reference()


def test_hotspots_that_are_not_the_costliest_functions_are_rejected():
    """Descending order is not enough on its own — the ranking also has to hold the
    heaviest functions there are, or a submission naming a costlier one scores above 1.0.
    """
    rows = dataset_rows()
    hotspots = [dict(h) for h in rows[0][HOTSPOTS_FIELD]]
    hotspots[-1] = function(REPOS[FIRST_ID], PROFILED - 1)  # the cheapest profiled one
    rows[0][HOTSPOTS_FIELD] = hotspots

    with pytest.raises(InvalidSubmissionError, match="descending exclusive_time_s order"):
        with_dataset(task(), rows).reference()


def test_a_hotspot_repeated_in_the_dataset_is_rejected():
    rows = dataset_rows()
    hotspots = [dict(h) for h in rows[0][HOTSPOTS_FIELD]]
    hotspots[1] = dict(hotspots[0])
    rows[0][HOTSPOTS_FIELD] = hotspots

    with pytest.raises(InvalidSubmissionError, match="lists the same function more than once"):
        with_dataset(task(), rows).reference()


def test_functions_tied_on_exclusive_time_may_be_ranked_either_way():
    """The ordering check compares weights rather than keys, so a profile with ties is
    not forced into one arbitrary order."""
    rows = dataset_rows()
    profile = [dict(f) for f in rows[0][FUNCTIONS_FIELD]]
    profile[0][HOTSPOT_TIME_FIELD] = profile[1][HOTSPOT_TIME_FIELD]
    rows[0][FUNCTIONS_FIELD] = profile
    hotspots = [dict(f) for f in profile[:HOTSPOTS]]
    hotspots[0], hotspots[1] = hotspots[1], hotspots[0]
    rows[0][HOTSPOTS_FIELD] = hotspots

    reference = with_dataset(task(), rows).reference()
    assert reference[FIRST_ID]["ranked"][:2] == [profile[1]["key"], profile[0]["key"]]


def test_a_duplicate_instance_in_the_dataset_is_rejected():
    helpers.expect_duplicate_instance_rejected(task(), dataset_rows())


def test_a_dataset_missing_a_needed_column_names_what_is_available():
    helpers.expect_missing_column_named(task(), dataset_rows(), FUNCTIONS_FIELD)


def test_a_baseline_ranks_once_and_slices_it_into_three_questions():
    rows = with_dataset(task(model="most_called")).predict(None)

    assert len(rows) == 2
    assert [len(rows[0][q]) for q in HOTSPOT_QUESTIONS] == [TOP_K[q] for q in HOTSPOT_QUESTIONS]
    assert rows[0]["exclusive_top5"] == rows[0]["exclusive_top20"][:5]  # one ranking, sliced
    # A call count says nothing about where time went; in this fixture it puts the least
    # expensive function first.
    assert rows[0]["exclusive_top1"] == [keys(FIRST_ID, PROFILED)[-1]]


def test_most_called_is_a_heuristic_and_is_scored_like_any_other():
    """This fixture makes call count run exactly against exclusive cost, so the baseline
    bottoms out at `top1` and still picks up most of `top20`: 20 of the 25 profiled
    functions are hotspots, so any ranking of the profile catches 15 of them.
    """
    hotspot = with_dataset(task(model="most_called"))
    results = Task.flatten(hotspot.score(hotspot.predict(None), REFERENCE))

    assert results["exclusive_top1_hit_rate"] == 0.0  # this fixture is adversarial to it by design
    assert results["exclusive_top20_hit_rate"] == pytest.approx(15 / 20)


def test_random_functions_is_seeded_and_draws_from_the_profile():
    def predict(seed):
        hotspot = with_dataset(task(model="random_functions"))
        hotspot.cfg.predict.params = {"seed": seed}
        return hotspot.predict(None)

    first = predict(7)
    assert [r["exclusive_top5"] for r in predict(7)] == [r["exclusive_top5"] for r in first]
    assert set(first[0]["exclusive_top20"]) <= set(keys(FIRST_ID, PROFILED))


def test_the_oracle_round_trips_through_validation_and_scoring():
    """The sanity check to run against a new config: it must come out at 1.0."""
    hotspot = with_dataset(task(model="oracle"))

    rows = hotspot.predict(None)
    reference = hotspot.reference()
    hotspot.validate_shape(rows)
    hotspot.validate(rows, reference)
    results = Task.flatten(hotspot.score(rows, reference))

    assert results["exclusive_top1_hit_rate"] == 1.0
    assert results["exclusive_top20_time_captured"] == 1.0
    assert results["exclusive_top20_somers_d"] == 1.0


def test_an_external_file_is_normalised_to_the_id_and_questions():
    external = [row(FIRST_ID, exclusive_top1=keys(FIRST_ID, 1), notes="dropped")]
    assert task().predict(external) == [{"instance_id": FIRST_ID, "exclusive_top1": keys(FIRST_ID, 1)}]


def test_an_external_column_can_be_renamed_and_questions_narrowed():
    hotspot = task()
    hotspot.cfg.predict.params = {
        "questions": ["exclusive_top1"],
        "question_fields": {"exclusive_top1": "top_function"},
    }
    rows = hotspot.predict([{"instance_id": FIRST_ID, "top_function": ["x"], "exclusive_top5": ["y"]}])

    # exclusive_top5 was not kept, so it is not scored.
    assert rows == [{"instance_id": FIRST_ID, "exclusive_top1": ["x"]}]


def test_an_external_row_answering_nothing_fails_the_run():
    with pytest.raises(PredictionFileError, match="answer none of"):
        task().predict([{"instance_id": FIRST_ID}])


def test_config_selects_the_task():
    assert isinstance(resolve(eval_config()), TimeHotspot)
