"""What every model must provide: one prediction per input row."""

import random
from abc import ABC, abstractmethod
from typing import Any

Row = dict[str, Any]

SAMPLE = 5  # how many offending rows to name in an error message


class PredictionFileError(ValueError):
    """A prediction file that cannot be read as a task's input."""


class BasePredictor(ABC):
    # Set by the per-task base class, as `Task.name` is by each task.
    task: str

    # The registry key this model was resolved under, for logs and error messages.
    # Set by `models.resolve`; empty when built directly.
    name: str = ""

    def __init__(self, **kwargs: Any) -> None:
        """The end of the cooperative `__init__` chain: anything still here is a name this
        model does not have. Dropping it silently would run a config that reads as if it were
        honoured — `predict.params: {second: 30}` quietly leaving the default in place — so
        this is the `extra="forbid"` the config blocks get, for params Pydantic cannot see."""
        if kwargs:
            raise TypeError(
                f"unknown parameter(s) {sorted(kwargs)} for model {type(self).__name__}; "
                f"check `predict.params` against the model's own arguments"
            )

    @abstractmethod
    def predict_single(self, row: Row) -> Any: ...

    def predict(self, rows: list[Row]) -> list[Any]:
        return [self.predict_single(row) for row in rows]


class SeededModel(BasePredictor, ABC):
    """A model that draws at random. One home for `seed` so every random model spells
    the parameter the same way and a run stays reproducible across tasks."""

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rng = random.Random(seed)
