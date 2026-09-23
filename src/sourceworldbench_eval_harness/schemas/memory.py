"""Question keys and answer shapes for the memory datasets."""

from typing import NamedTuple

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
    "scale",
)


BYTES = "bytes"

# Decimal, not binary: scoring `500MB` as mebibytes would mark correct answers wrong
# either side of every threshold.
KB = 10**3
MB = 10**6
GB = 10**9

BUDGETS: dict[str, float] = {"budget_500mb": 500 * MB, "budget_1gb": 1 * GB, "budget_5gb": 5 * GB}
PREDICTION_QUESTIONS: tuple[str, ...] = (BYTES, *BUDGETS)

BYTES_FIELD = "traced_delta_bytes"

# Cross-checked against the byte count, so a revision that redefined a budget stops the
# run instead of being scored against silently.
BUDGET_FIELDS: dict[str, str] = {
    "budget_500mb": "within-budget-500MB",
    "budget_1gb": "within-budget-1GB",
    "budget_5gb": "within-budget-5GB",
}

# Two bands, because one cannot separate a calibrated prediction from a lucky one: a model
# uniformly 1.5x out clears 2.0 and fails 1.25, so the gap between the columns is the
# diagnostic. The same pair in both tasks keeps the columns comparable.
FACTORS: tuple[float, ...] = (1.25, 2.0)

_UNITS: tuple[tuple[str, int], ...] = (("GB", GB), ("MB", MB), ("KB", KB))

is_bytes = is_positive_measurement


def spell_bytes(value: float) -> str:
    """As the prompts write it: `500KB`, `5MB`, `1GB`."""
    for suffix, size in _UNITS:
        if value >= size:
            return f"{trim(value / size)}{suffix}"
    return f"{trim(value)}B"


class HotspotAxis(NamedTuple):
    """One ranking of the profile, with the questions asked about it."""

    name: str
    hotspots_field: str
    weight_field: str
    tops: tuple[int, ...]

    def question(self, top: int) -> str:
        return f"{self.name}_top{top}"

    @property
    def questions(self) -> tuple[str, ...]:
        return tuple(self.question(top) for top in self.tops)

    @property
    def deepest(self) -> int:
        """The shortest ranking this axis can still be scored from."""
        return max(self.tops)


HOTSPOT_AXES: tuple[HotspotAxis, ...] = (
    HotspotAxis("cumulative", "hotspots_cumulative", "exclusive_alloc_delta_sum", (1, 5)),
)

HOTSPOT_QUESTIONS: tuple[str, ...] = tuple(question for axis in HOTSPOT_AXES for question in axis.questions)

# Read off the axes rather than maintained beside them, so a depth cannot be added to one
# and forgotten in the other.
AXIS_FOR: dict[str, HotspotAxis] = {question: axis for axis in HOTSPOT_AXES for question in axis.questions}
TOP_K: dict[str, int] = {axis.question(top): top for axis in HOTSPOT_AXES for top in axis.tops}

FUNCTIONS_FIELD = "functions_complete_list"
HOTSPOTS_FIELDS: tuple[str, ...] = tuple(axis.hotspots_field for axis in HOTSPOT_AXES)


__all__ = [
    "AXIS_FOR",
    "BUDGETS",
    "BUDGET_FIELDS",
    "BYTES",
    "BYTES_FIELD",
    "FACTORS",
    "FUNCTIONS_FIELD",
    "GB",
    "HOTSPOTS_FIELDS",
    "HOTSPOT_AXES",
    "HOTSPOT_QUESTIONS",
    "KB",
    "MB",
    "PREDICTION_QUESTIONS",
    "STATE_FIELDS",
    "TOP_K",
    "HotspotAxis",
    "alias",
    "is_bytes",
    "lookup_index",
    "resolve",
    "spell_bytes",
    "trim",
    "triple",
]
