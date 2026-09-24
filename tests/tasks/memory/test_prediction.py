import math

import pytest

import helpers
from helpers import FakeDataset, row
from sourceworldbench_eval_harness.models.core.base import PredictionFileError
from sourceworldbench_eval_harness.schemas.memory import spell_bytes
from sourceworldbench_eval_harness.tasks import resolve
from sourceworldbench_eval_harness.tasks.core.base import InvalidSubmissionError, Task
from sourceworldbench_eval_harness.tasks.memory.prediction import MemoryPrediction

MB = 10**6
GB = 10**9

# Both classes occur for every budget question, so macro F1 is defined.
CASES = {
    "matplotlib__aaaaaaaa__prediction_x": {"bytes": 2 * MB},
    "pandas__bbbbbbbb__prediction_y": {"bytes": 800 * MB},
    "astropy__cccccccc__prediction_z": {"bytes": 20 * GB},
}

SMALL_ID, MEDIUM_ID, LARGE_ID = CASES

# The shape `reference` returns.
REFERENCE = {instance_id: {"bytes": case["bytes"]} for instance_id, case in CASES.items()}


def eval_config(name="memory_prediction", **task_settings):
    return helpers.eval_config(name, **task_settings)


def task(require_full_coverage=True, model=None, **settings):
    return helpers.build_task("memory_prediction", require_full_coverage=require_full_coverage, model=model, **settings)


def flat_score(rows, **settings):
    return Task.flatten(task(**settings).score(rows, REFERENCE))


def answers(size):
    """Every question answered from one predicted footprint, as a model would."""
    return {
        "bytes": size,
        "budget_500mb": size <= 500 * MB,
        "budget_1gb": size <= 1 * GB,
        "budget_5gb": size <= 5 * GB,
    }


def oracle_rows():
    return [row(instance_id, **answers(case["bytes"])) for instance_id, case in CASES.items()]


def dataset_rows():
    return FakeDataset(
        [
            {
                "instance_id": instance_id,
                "traced_delta_bytes": case["bytes"],
                "within-budget-500MB": case["bytes"] <= 500 * MB,
                "within-budget-1GB": case["bytes"] <= 1 * GB,
                "within-budget-5GB": case["bytes"] <= 5 * GB,
                "repo": instance_id.split("__")[0],
                "base_commit": "deadbeef",
                "patch": "--- a/x\n+++ b/x\n",
                "container": "registry/image@sha256:abc",
                "workload": "tests/test_x.py",
                "test_command": "pytest -v",
                "full_command": "pytest -v tests/test_x.py",
                "scale": "medium",
            }
            for instance_id, case in CASES.items()
        ]
    )


def with_dataset(task_, rows=None):
    return helpers.attach_dataset(task_, dataset_rows() if rows is None else rows)


def test_oracle_scores_perfectly_on_every_question_in_one_run():
    results = flat_score(oracle_rows())

    assert results["bytes_log10_error"] == 0.0
    assert results["bytes_within_factor_2"] == 1.0
    assert results["bytes_within_factor_1.25"] == 1.0
    assert results["bytes_log_slope"] == pytest.approx(1.0)
    assert results["bytes_log_intercept"] == pytest.approx(0.0)
    assert results["budget_500mb_macro_f1"] == 1.0
    assert results["budget_1gb_macro_f1"] == 1.0
    assert results["budget_5gb_macro_f1"] == 1.0
    assert results["instances"] == 3
    assert results["questions_answered"] == ["bytes", "budget_500mb", "budget_1gb", "budget_5gb"]


def test_bytes_is_scored_scale_free_so_the_largest_instance_cannot_dominate():
    """Out by 2x costs the same wherever on the scale it happens, which an error in
    bytes does not."""
    doubled = [row(i, bytes=CASES[i]["bytes"] * 2) for i in CASES]
    halved = [row(i, bytes=CASES[i]["bytes"] / 2) for i in CASES]

    assert flat_score(doubled)["bytes_log10_error"] == pytest.approx(math.log10(2))
    assert flat_score(halved)["bytes_log10_error"] == pytest.approx(math.log10(2))
    # And no error in the target's own units is reported at all: on this spread the
    # 20GB instance would set it alone.
    assert not [key for key in flat_score(doubled) if key.endswith(("absolute_error", "relative_error"))]


def test_within_factor_counts_the_boundary_and_the_looser_band_is_never_stricter():
    rows = [row(SMALL_ID, bytes=CASES[SMALL_ID]["bytes"] * 2)]
    results = flat_score(rows, require_full_coverage=False)

    assert results["bytes_within_factor_2"] == 1.0  # exactly 2x still counts
    assert results["bytes_within_factor_1.25"] == 0.0


def test_debiased_error_separates_a_uniform_offset_from_scatter():
    """Both models are out by 10x on average; only the first is one constant away."""
    uniformly_high = [row(i, bytes=CASES[i]["bytes"] * 10) for i in CASES]
    scattered = [
        row(SMALL_ID, bytes=CASES[SMALL_ID]["bytes"] * 10),
        row(MEDIUM_ID, bytes=CASES[MEDIUM_ID]["bytes"] / 10),
        row(LARGE_ID, bytes=CASES[LARGE_ID]["bytes"] * 10),
    ]

    offset = flat_score(uniformly_high)
    assert offset["bytes_log10_error"] == pytest.approx(1.0)
    assert offset["bytes_debiased_log10_error"] == pytest.approx(0.0)

    spread = flat_score(scattered)
    assert spread["bytes_log10_error"] == pytest.approx(1.0)
    # The median residual is +1, so debiasing pulls the two high instances to 0 and
    # leaves the low one out by two orders of magnitude.
    assert spread["bytes_debiased_log10_error"] == pytest.approx(2 / 3)

    assert flat_score(oracle_rows())["bytes_debiased_log10_error"] == pytest.approx(0.0)


def test_calibrated_error_frees_the_scale_where_debiasing_only_shifts():
    """Predictions compressed toward the middle: no single offset fixes a dynamic range,
    so the calibrated figure is the only one that reads the ordering as intact."""
    squashed = [row(i, bytes=CASES[i]["bytes"] ** 0.5) for i in CASES]
    results = flat_score(squashed)

    assert results["bytes_log10_error"] > results["bytes_debiased_log10_error"]
    assert results["bytes_calibrated_log10_error"] == pytest.approx(0.0, abs=1e-9)

    # And a uniform offset, which debiasing already removes, leaves nothing further.
    offset = flat_score([row(i, bytes=CASES[i]["bytes"] * 10) for i in CASES])
    assert offset["bytes_debiased_log10_error"] == pytest.approx(0.0)
    assert offset["bytes_calibrated_log10_error"] == pytest.approx(0.0)

    assert flat_score(oracle_rows())["bytes_calibrated_log10_error"] == pytest.approx(0.0)


def test_log_slope_separates_a_calibration_error_from_no_signal():
    """Both models are badly wrong; only one of them understood relative cost."""
    calibrated = [row(i, bytes=CASES[i]["bytes"] * 100) for i in CASES]  # uniformly 100x high
    constant = [row(i, bytes=42.0 * MB) for i in CASES]  # one plausible number, every time

    tracking = flat_score(calibrated)
    assert tracking["bytes_log_slope"] == pytest.approx(1.0)  # tracks the target exactly
    assert tracking["bytes_log_intercept"] == pytest.approx(2.0)  # and is out by 10**2

    flat = flat_score(constant)
    assert flat["bytes_log_slope"] == pytest.approx(0.0)
    # Yet the constant guess reports the *smaller* error of the two, which is the
    # reason the slope is worth reporting beside it.
    assert flat["bytes_log10_error"] < tracking["bytes_log10_error"]


def test_the_flat_bytes_figures_are_none_until_a_line_is_determined():
    single = flat_score([row(SMALL_ID, bytes=2 * MB)], require_full_coverage=False)
    assert single["bytes_log_slope"] is None  # one point determines no line


def test_a_constant_guess_scores_zero_on_the_budget_class_it_never_predicts():
    """The floor: one guess either fits a threshold or does not, for every instance."""
    rows = [row(i, **answers(1 * MB)) for i in CASES]
    results = flat_score(rows)

    assert results["budget_500mb_fits_f1"] == pytest.approx(0.5)  # only the small instance fits
    assert results["budget_500mb_exceeds_f1"] == 0.0
    assert results["budget_500mb_instances"] == 3


def test_only_the_budget_questions_carry_a_confusion_table():
    """`bytes` is a regression, so it has no details."""
    assert set(flat_score(oracle_rows())["details"]) == {"budget_500mb", "budget_1gb", "budget_5gb"}


def test_valid_submission_passes():
    task().validate_shape(oracle_rows())
    task().validate(oracle_rows(), REFERENCE)


def test_a_non_positive_or_infinite_footprint_is_rejected():
    for value in (0, -1.0, float("inf"), float("nan")):
        with pytest.raises(InvalidSubmissionError, match="positive, finite number of bytes"):
            task().validate_shape([row(SMALL_ID, bytes=value)])


def test_a_footprint_that_is_not_a_number_is_rejected():
    """A bool is an int in Python, and would otherwise pass as `1` byte."""
    for value in ("2.0", None, True, {"bytes": 2.0}):
        with pytest.raises(InvalidSubmissionError, match="positive, finite number of bytes"):
            task().validate_shape([row(SMALL_ID, bytes=value)])


def test_a_budget_question_takes_only_true_or_false():
    task().validate_shape([row(SMALL_ID, budget_500mb=True), row(MEDIUM_ID, budget_1gb=False)])
    with pytest.raises(InvalidSubmissionError, match="at most 500MB"):
        task().validate_shape([row(SMALL_ID, budget_500mb="yes")])
    with pytest.raises(InvalidSubmissionError, match="at most 5GB"):
        task().validate_shape([row(SMALL_ID, budget_5gb=1)])


def test_budgets_are_spelled_with_their_unit():
    """The prompt writes `500MB`, `1.5GB`, `999B` — the spelling `check_shape` errors use."""
    assert spell_bytes(1.5 * GB) == "1.5GB"
    assert spell_bytes(999) == "999B"


def test_reference_keys_the_footprint_by_instance():
    assert with_dataset(task()).reference() == REFERENCE


def test_a_missing_or_non_positive_footprint_is_refused_as_an_answer_key():
    for bad in (None, 0.0):
        rows = dataset_rows()
        rows[1]["traced_delta_bytes"] = bad
        with pytest.raises(InvalidSubmissionError, match="cannot be scored against"):
            with_dataset(task(), rows).reference()


def test_a_budget_column_the_footprint_contradicts_is_rejected():
    """A revision that moved a threshold — decimal units to binary, say — would mark right
    answers wrong. The dataset's own answers are the check on that."""
    rows = dataset_rows()
    rows[0]["within-budget-500MB"] = False  # the row is 2MB, which fits comfortably

    with pytest.raises(InvalidSubmissionError, match="answer a budget question differently"):
        with_dataset(task(), rows).reference()


def test_a_dataset_without_the_budget_columns_is_still_scoreable():
    """They are a cross-check on the thresholds, not an input to scoring."""
    rows = FakeDataset(
        [{key: value for key, value in r.items() if not key.startswith("within-budget")} for r in dataset_rows()]
    )
    assert with_dataset(task(), rows).reference() == REFERENCE


def test_a_duplicate_instance_in_the_dataset_is_rejected():
    helpers.expect_duplicate_instance_rejected(task(), dataset_rows())


def test_a_dataset_missing_a_needed_column_names_what_is_available():
    helpers.expect_missing_column_named(task(), dataset_rows(), "traced_delta_bytes")


def test_a_baseline_answers_every_question_from_its_one_guess():
    rows = with_dataset(task(model="constant_bytes")).predict(None)

    assert len(rows) == 3
    assert rows[0]["instance_id"] == SMALL_ID
    assert rows[0]["bytes"] == float(GB)  # the default constant
    assert rows[0]["budget_500mb"] is False
    assert rows[0]["budget_1gb"] is True and rows[0]["budget_5gb"] is True


def test_the_constant_can_be_set_and_reaches_every_derived_question():
    memory = with_dataset(task(model="constant_bytes"))
    memory.cfg.predict.params = {"bytes": 20.0 * GB}
    rows = memory.predict(None)

    assert rows[0]["bytes"] == 20.0 * GB
    assert rows[0]["budget_500mb"] is False and rows[0]["budget_5gb"] is False


def test_random_bytes_is_seeded_and_stays_inside_its_range():
    memory = with_dataset(task(model="random_bytes"))
    memory.cfg.predict.params = {"seed": 7, "low": 1.0 * MB, "high": 10.0 * GB}
    first = [r["bytes"] for r in memory.predict(None)]

    memory = with_dataset(task(model="random_bytes"))
    memory.cfg.predict.params = {"seed": 7, "low": 1.0 * MB, "high": 10.0 * GB}
    assert [r["bytes"] for r in memory.predict(None)] == first
    assert all(1.0 * MB <= value <= 10.0 * GB for value in first)


def test_the_oracle_round_trips_through_validation_and_scoring():
    """The sanity check to run against a new config: it must come out at 1.0."""
    memory = with_dataset(task(model="oracle"))

    rows = memory.predict(None)
    reference = memory.reference()
    memory.validate_shape(rows)
    memory.validate(rows, reference)
    results = Task.flatten(memory.score(rows, reference))

    assert results["bytes_log10_error"] == 0.0
    assert results["budget_500mb_macro_f1"] == 1.0


def test_an_external_file_is_normalised_to_the_id_and_questions():
    external = [row(SMALL_ID, bytes=2.0 * MB, budget_500mb=True, notes="dropped")]
    assert task().predict(external) == [{"instance_id": SMALL_ID, "bytes": 2.0 * MB, "budget_500mb": True}]


def test_an_external_column_can_be_renamed_and_questions_narrowed():
    memory = task()
    memory.cfg.predict.params = {"questions": ["bytes"], "question_fields": {"bytes": "predicted_bytes"}}
    rows = memory.predict([{"instance_id": SMALL_ID, "predicted_bytes": 9.5 * MB, "budget_500mb": True}])

    assert rows == [{"instance_id": SMALL_ID, "bytes": 9.5 * MB}]  # budget_500mb was not kept, so it is not scored


def test_an_external_row_answering_nothing_fails_the_run():
    with pytest.raises(PredictionFileError, match="answer none of"):
        task().predict([{"instance_id": SMALL_ID}])


def test_config_selects_the_task():
    assert isinstance(resolve(eval_config()), MemoryPrediction)
