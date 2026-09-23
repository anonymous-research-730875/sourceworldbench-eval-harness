import math

import pytest

import helpers
from helpers import FakeDataset, row
from sourceworldbench_eval_harness.models.core.base import PredictionFileError
from sourceworldbench_eval_harness.tasks import resolve
from sourceworldbench_eval_harness.tasks.core.base import InvalidSubmissionError, Task
from sourceworldbench_eval_harness.tasks.time.prediction import TimePrediction

# Three instances spanning the orders of magnitude the real dataset spans, chosen so
# both classes occur for every budget question and macro F1 is defined.
CASES = {
    "matplotlib__aaaaaaaa__prediction_x": {"seconds": 2.0},
    "pandas__bbbbbbbb__prediction_y": {"seconds": 30.0},
    "astropy__cccccccc__prediction_z": {"seconds": 400.0},
}

FAST_ID, MEDIUM_ID, SLOW_ID = CASES

# The shape `reference` returns.
REFERENCE = {instance_id: {"seconds": case["seconds"]} for instance_id, case in CASES.items()}


def eval_config(name="time_prediction", **task_settings):
    return helpers.eval_config(name, **task_settings)


def task(require_full_coverage=True, model=None, **settings):
    return helpers.build_task("time_prediction", require_full_coverage=require_full_coverage, model=model, **settings)


def flat_score(rows, **settings):
    return Task.flatten(task(**settings).score(rows, REFERENCE))


def answers(seconds):
    """Every question answered from one predicted runtime, as a model would."""
    return {
        "seconds": seconds,
        "budget_5s": seconds <= 5.0,
        "budget_60s": seconds <= 60.0,
        "budget_300s": seconds <= 300.0,
    }


def oracle_rows():
    return [row(instance_id, **answers(case["seconds"])) for instance_id, case in CASES.items()]


def dataset_rows():
    return FakeDataset(
        [
            {
                "instance_id": instance_id,
                "wall_time_s": case["seconds"],
                "repo": instance_id.split("__")[0],
                "base_commit": "deadbeef",
                "patch": "--- a/x\n+++ b/x\n",
                "container": "registry/image@sha256:abc",
                "workload": "tests/test_x.py",
                "test_command": "pytest -v",
                "full_command": "pytest -v tests/test_x.py",
            }
            for instance_id, case in CASES.items()
        ]
    )


def with_dataset(task_, rows=None):
    return helpers.attach_dataset(task_, dataset_rows() if rows is None else rows)


def test_oracle_scores_perfectly_on_every_question_in_one_run():
    results = flat_score(oracle_rows())

    assert results["seconds_log10_error"] == 0.0
    assert results["seconds_within_factor_2"] == 1.0
    assert results["seconds_within_factor_1.25"] == 1.0
    assert results["seconds_log_slope"] == pytest.approx(1.0)
    assert results["seconds_log_intercept"] == pytest.approx(0.0)
    assert results["budget_5s_macro_f1"] == 1.0
    assert results["budget_60s_macro_f1"] == 1.0
    assert results["budget_300s_macro_f1"] == 1.0
    assert results["instances"] == 3
    assert results["questions_answered"] == ["seconds", "budget_5s", "budget_60s", "budget_300s"]


def test_seconds_is_scored_scale_free_so_the_slowest_instance_cannot_dominate():
    """Out by 2x costs the same wherever on the scale it happens, which an error in
    seconds does not."""
    doubled = [row(i, seconds=CASES[i]["seconds"] * 2) for i in CASES]
    halved = [row(i, seconds=CASES[i]["seconds"] / 2) for i in CASES]

    assert flat_score(doubled)["seconds_log10_error"] == pytest.approx(math.log10(2))
    assert flat_score(halved)["seconds_log10_error"] == pytest.approx(math.log10(2))
    # And no error in the target's own units is reported at all: on this spread the
    # 400s instance would set it alone, at (2 + 30 + 400) / 3.
    assert not [key for key in flat_score(doubled) if key.endswith(("absolute_error", "relative_error"))]


def test_within_factor_counts_the_boundary_and_the_looser_band_is_never_stricter():
    rows = [row(FAST_ID, seconds=CASES[FAST_ID]["seconds"] * 2)]
    results = flat_score(rows, require_full_coverage=False)

    assert results["seconds_within_factor_2"] == 1.0  # exactly 2x still counts
    assert results["seconds_within_factor_1.25"] == 0.0


def test_debiased_error_separates_a_uniform_offset_from_scatter():
    """Both models are out by 10x on average; only the first is one constant away."""
    uniformly_high = [row(i, seconds=CASES[i]["seconds"] * 10) for i in CASES]
    scattered = [
        row(FAST_ID, seconds=CASES[FAST_ID]["seconds"] * 10),
        row(MEDIUM_ID, seconds=CASES[MEDIUM_ID]["seconds"] / 10),
        row(SLOW_ID, seconds=CASES[SLOW_ID]["seconds"] * 10),
    ]

    offset = flat_score(uniformly_high)
    assert offset["seconds_log10_error"] == pytest.approx(1.0)
    assert offset["seconds_debiased_log10_error"] == pytest.approx(0.0)

    spread = flat_score(scattered)
    assert spread["seconds_log10_error"] == pytest.approx(1.0)
    # The median residual is +1, so debiasing pulls the two high instances to 0 and
    # leaves the low one out by two orders of magnitude.
    assert spread["seconds_debiased_log10_error"] == pytest.approx(2 / 3)

    assert flat_score(oracle_rows())["seconds_debiased_log10_error"] == pytest.approx(0.0)


def test_the_flat_seconds_figures_are_none_until_a_line_is_determined():
    single = flat_score([row(FAST_ID, seconds=2.0)], require_full_coverage=False)
    assert single["seconds_log_slope"] is None  # one point determines no line

    constant = flat_score([row(i, seconds=7.0) for i in CASES])
    assert constant["seconds_log10_error"] is not None  # the error is still measurable
    # And the slope says what the error alone cannot: nothing responded to the input.
    assert constant["seconds_log_slope"] == pytest.approx(0.0)
    assert constant["seconds_log_intercept"] == pytest.approx(math.log10(7))


def test_log_slope_separates_a_calibration_error_from_no_signal():
    """Both models are badly wrong; only one of them understood relative cost."""
    calibrated = [row(i, seconds=CASES[i]["seconds"] * 100) for i in CASES]  # uniformly 100x high
    constant = [row(i, seconds=42.0) for i in CASES]  # one plausible number, every time

    tracking = flat_score(calibrated)
    assert tracking["seconds_log_slope"] == pytest.approx(1.0)  # tracks the target exactly
    assert tracking["seconds_log_intercept"] == pytest.approx(2.0)  # and is out by 10**2

    flat = flat_score(constant)
    assert flat["seconds_log_slope"] == pytest.approx(0.0)
    # Yet the constant guess reports the *smaller* error of the two, which is the
    # reason the slope is worth reporting beside it.
    assert flat["seconds_log10_error"] < tracking["seconds_log10_error"]


def test_a_constant_guess_scores_zero_on_the_budget_class_it_never_predicts():
    """The floor: one guess either fits a threshold or does not, for every instance."""
    rows = [row(i, **answers(1.0)) for i in CASES]
    results = flat_score(rows)

    assert results["budget_5s_fits_f1"] == pytest.approx(0.5)  # only the fast instance fits
    assert results["budget_5s_misses_f1"] == 0.0
    assert results["budget_5s_instances"] == 3


def test_metrics_are_flat_and_prefixed_by_question():
    results = flat_score([row(FAST_ID, seconds=2.0)], require_full_coverage=False)
    scalars = {key: value for key, value in results.items() if key != "details"}

    assert all(not isinstance(value, dict) for value in scalars.values())
    assert not any(key.startswith("budget_") for key in scalars)  # unanswered, so unscored


def test_only_the_budget_questions_carry_a_confusion_table():
    """`seconds` is a regression, so it has no details."""
    assert set(flat_score(oracle_rows())["details"]) == {"budget_5s", "budget_60s", "budget_300s"}


def test_valid_submission_passes():
    task().validate_shape(oracle_rows())
    task().validate(oracle_rows(), REFERENCE)


def test_a_non_positive_or_infinite_runtime_is_rejected():
    for value in (0, -1.0, float("inf"), float("nan")):
        with pytest.raises(InvalidSubmissionError, match="positive, finite number of seconds"):
            task().validate_shape([row(FAST_ID, seconds=value)])


def test_a_runtime_that_is_not_a_number_is_rejected():
    """A bool is an int in Python, and would otherwise pass as `1` second."""
    for value in ("2.0", None, True, {"seconds": 2.0}):
        with pytest.raises(InvalidSubmissionError, match="positive, finite number of seconds"):
            task().validate_shape([row(FAST_ID, seconds=value)])


def test_a_budget_question_takes_only_true_or_false():
    task().validate_shape([row(FAST_ID, budget_5s=True), row(MEDIUM_ID, budget_60s=False)])
    with pytest.raises(InvalidSubmissionError, match="does it finish within 5s"):
        task().validate_shape([row(FAST_ID, budget_5s="yes")])
    with pytest.raises(InvalidSubmissionError, match="does it finish within 300s"):
        task().validate_shape([row(FAST_ID, budget_300s=1)])


def test_reference_keys_the_time_by_instance():
    assert with_dataset(task()).reference() == REFERENCE


def test_a_missing_or_non_positive_time_is_refused_as_an_answer_key():
    for bad in (None, 0.0):
        rows = dataset_rows()
        rows[1]["wall_time_s"] = bad
        with pytest.raises(InvalidSubmissionError, match="cannot be scored against"):
            with_dataset(task(), rows).reference()


def test_a_duplicate_instance_in_the_dataset_is_rejected():
    helpers.expect_duplicate_instance_rejected(task(), dataset_rows())


def test_a_dataset_missing_a_needed_column_names_what_is_available():
    helpers.expect_missing_column_named(task(), dataset_rows(), "wall_time_s")


def test_a_baseline_answers_every_question_from_its_one_guess():
    rows = with_dataset(task(model="constant_seconds")).predict(None)

    assert len(rows) == 3
    assert rows[0]["instance_id"] == FAST_ID
    assert rows[0]["seconds"] == 5.0  # the default constant
    assert rows[0]["budget_5s"] is True and rows[0]["budget_60s"] is True and rows[0]["budget_300s"] is True


def test_the_constant_can_be_set_and_reaches_every_derived_question():
    time = with_dataset(task(model="constant_seconds"))
    time.cfg.predict.params = {"seconds": 600.0}
    rows = time.predict(None)

    assert rows[0]["seconds"] == 600.0
    assert rows[0]["budget_5s"] is False and rows[0]["budget_300s"] is False


def test_random_seconds_is_seeded_and_stays_inside_its_range():
    time = with_dataset(task(model="random_seconds"))
    time.cfg.predict.params = {"seed": 7, "low": 1.0, "high": 600.0}
    first = [r["seconds"] for r in time.predict(None)]

    time = with_dataset(task(model="random_seconds"))
    time.cfg.predict.params = {"seed": 7, "low": 1.0, "high": 600.0}
    assert [r["seconds"] for r in time.predict(None)] == first
    assert all(1.0 <= value <= 600.0 for value in first)


def test_the_oracle_round_trips_through_validation_and_scoring():
    """The sanity check to run against a new config: it must come out at 1.0."""
    time = with_dataset(task(model="oracle"))

    rows = time.predict(None)
    reference = time.reference()
    time.validate_shape(rows)
    time.validate(rows, reference)
    results = Task.flatten(time.score(rows, reference))

    assert results["seconds_log10_error"] == 0.0
    assert results["budget_5s_macro_f1"] == 1.0


def test_an_external_file_is_normalised_to_the_id_and_questions():
    external = [row(FAST_ID, seconds=2.0, budget_5s=True, notes="dropped")]
    assert task().predict(external) == [{"instance_id": FAST_ID, "seconds": 2.0, "budget_5s": True}]


def test_an_external_column_can_be_renamed_and_questions_narrowed():
    time = task()
    time.cfg.predict.params = {"questions": ["seconds"], "question_fields": {"seconds": "predicted_seconds"}}
    rows = time.predict([{"instance_id": FAST_ID, "predicted_seconds": 9.5, "budget_5s": True}])

    assert rows == [{"instance_id": FAST_ID, "seconds": 9.5}]  # budget_5s was not kept, so it is not scored


def test_an_external_row_answering_nothing_fails_the_run():
    with pytest.raises(PredictionFileError, match="answer none of"):
        task().predict([{"instance_id": FAST_ID}])


def test_config_selects_the_task():
    assert isinstance(resolve(eval_config()), TimePrediction)
