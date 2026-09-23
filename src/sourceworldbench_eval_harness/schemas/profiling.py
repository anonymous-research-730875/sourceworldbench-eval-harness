"""Shared by the timing and memory datasets, which differ only in the unit they measure.

Here so the two cannot drift apart on it.
"""

import math
from collections.abc import Iterable
from typing import Any


def trim(value: float) -> str:
    """Spelled as a prompt writes a boundary: `1.0` -> `1`, `0.5` -> `0.5`."""
    return str(int(value)) if float(value).is_integer() else str(value)


def is_positive_measurement(value: Any) -> bool:
    """Rejects bools, which Python would otherwise count as ints."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value) and value > 0


def triple(filename: str, firstlineno: Any, qualname: str) -> str:
    """Spelled as the candidate list spells it, since that is what a model echoes back."""
    return f"{filename}:{firstlineno} {qualname}"


def alias(entry: Any) -> str | None:
    """Both spellings a prompt permits are accepted, so neither is penalised. `None` for
    anything else, which `check_shape` reports."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        if any(field not in entry for field in ("filename", "firstlineno", "qualname")):
            return None
        return triple(entry["filename"], entry["firstlineno"], entry["qualname"])
    return None


def resolve(entry: Any, index: dict[str, str]) -> str | None:
    """`None` covers both an unprofiled function and a non-identifier, neither scoreable."""
    key = alias(entry)
    return index.get(key) if key is not None else None


def lookup_index(functions: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Built per instance, because a `key` is only unique within one profile."""
    index: dict[str, str] = {}
    for function in functions:
        key = function["key"]
        index[key] = key
        index[triple(function["filename"], function["firstlineno"], function["qualname"])] = key
    return index


__all__ = [
    "alias",
    "is_positive_measurement",
    "lookup_index",
    "resolve",
    "trim",
    "triple",
]
