"""Question keys and answer shapes for the timing datasets."""

from sourceworldbench_eval_harness.schemas.profiling import (
    alias,
    is_positive_measurement,
    lookup_index,
    resolve,
    trim,
    triple,
)

# A baseline ignores these; a real predictor cannot answer without them.
STATE_FIELDS: tuple[str, ...] = (
    "repo",
    "base_commit",
    "patch",
    "container",
    "workload",
    "test_command",
    "full_command",
)


SECONDS = "seconds"
BUDGETS: dict[str, float] = {"budget_5s": 5.0, "budget_60s": 60.0, "budget_300s": 300.0}
PREDICTION_QUESTIONS: tuple[str, ...] = (SECONDS, *BUDGETS)

TIME_FIELD = "wall_time_s"

# Two bands, because one cannot separate a calibrated prediction from a lucky one: a model
# uniformly 1.5x out clears 2.0 and fails 1.25, so the gap between the columns is the
# diagnostic. The same pair in both tasks keeps the columns comparable.
FACTORS: tuple[float, ...] = (1.25, 2.0)

is_seconds = is_positive_measurement


HOTSPOT_AXIS = "exclusive"
HOTSPOT_TOPS: tuple[int, ...] = (1, 5, 20)

HOTSPOT_QUESTIONS: tuple[str, ...] = tuple(f"{HOTSPOT_AXIS}_top{top}" for top in HOTSPOT_TOPS)

# How many functions a question asks for, read back off the question name rather than
# maintained beside it, so another `k` cannot be added to one and forgotten in the other.
TOP_K: dict[str, int] = {f"{HOTSPOT_AXIS}_top{top}": top for top in HOTSPOT_TOPS}

FUNCTIONS_FIELD = "functions_complete_list"
HOTSPOTS_FIELD = "hotspots"
HOTSPOT_TIME_FIELD = f"{HOTSPOT_AXIS}_time_s"


__all__ = [
    "BUDGETS",
    "FACTORS",
    "FUNCTIONS_FIELD",
    "HOTSPOTS_FIELD",
    "HOTSPOT_AXIS",
    "HOTSPOT_QUESTIONS",
    "HOTSPOT_TIME_FIELD",
    "HOTSPOT_TOPS",
    "PREDICTION_QUESTIONS",
    "SECONDS",
    "STATE_FIELDS",
    "TIME_FIELD",
    "TOP_K",
    "alias",
    "is_seconds",
    "lookup_index",
    "resolve",
    "trim",
    "triple",
]
