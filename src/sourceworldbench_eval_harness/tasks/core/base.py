"""What every task must provide."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

from sourceworldbench_eval_harness import models
from sourceworldbench_eval_harness.config import EvalConfig, TaskConfig
from sourceworldbench_eval_harness.hub import load_split

SAMPLE = 5  # how many offending ids to name in an error message

C = TypeVar("C", bound=TaskConfig)

# What `Task.score` returns: `{group or None: {metric: value}}`.
Groups = dict[str | None, dict[str, Any]]


class InvalidSubmissionError(ValueError):
    """Raised with every problem found, so one run tells you everything to fix."""


def fail(problems: list[str]) -> None:
    if problems:
        raise InvalidSubmissionError("\n".join(f"- {problem}" for problem in problems))


class Task(ABC, Generic[C]):
    """Subclasses read their settings from the `task` block, so the surrounding
    pipeline blocks stay identical across configs."""

    name: str

    # Revalidated by `tasks.resolve`. A task with no settings of its own needs no
    # subclass.
    Config: ClassVar[type[TaskConfig]] = TaskConfig

    def __init__(self, cfg: EvalConfig, settings: C, config_path: Path | None = None) -> None:
        self.cfg = cfg
        self.settings = settings
        # Kept so a task can run an earlier pipeline step on demand — those
        # entrypoints take a config path and load it themselves.
        self.config_path = config_path

    @property
    def id_field(self) -> str | None:
        """Configurable so no task module names a dataset column in code. A task that
        requires one declares `id_field: str` on its `Config` and narrows this to `str`.
        """
        return self.settings.id_field

    @abstractmethod
    def predict(self, external: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Prediction rows to write at `predict.output`. `external` is the parsed
        `predict.external_input`, or None; what such a file *means* is the task's call.
        """

    @abstractmethod
    def reference(self) -> Any:
        """Ground truth, in whatever shape this task's scorer wants."""

    @abstractmethod
    def validate(self, rows: list[dict[str, Any]], reference: Any) -> None:
        """Only the checks needing *both* sides. Raise `InvalidSubmissionError`
        listing every problem. Shape checks belong to `predict`.
        """

    @abstractmethod
    def score(self, rows: list[dict[str, Any]], reference: Any) -> Groups:
        """Metrics grouped for reporting. `None` keys the ungrouped ones, and is where
        `details` goes, since the leaderboard turns every column into a metric.
        Emission order is preserved by `flatten` and by `evaluate`'s tables.
        """

    def validate_shape(self, rows: list[dict[str, Any]]) -> None:
        """Checks on the prediction file alone, run by `predict`: a failure here says
        "your file is malformed", one in `validate` says "it describes another dataset".
        """

    def write_extras(self, out: Path, rows: list[dict[str, Any]]) -> None:
        """Companion artifacts written beside `predict.output`."""

    @staticmethod
    def flatten(groups: Groups) -> dict[str, Any]:
        """`score`'s groups as the flat `{group}_{metric}` file, in emission order."""
        return {
            (f"{group}_{key}" if group else key): value
            for group, metrics in groups.items()
            for key, value in metrics.items()
        }

    @property
    def external_path(self) -> Path | None:
        external = self.cfg.predict.external_input
        return Path(external).expanduser() if external else None

    def configured_model(self, default: str | None = None) -> models.BasePredictor:
        name = self.cfg.predict.model or default
        if not name:
            valid = sorted(models.MODELS.get(self.name, ()))
            raise ValueError(f"{self.name} needs a `predict.model`; valid names: {valid}")
        return models.resolve(name, self.name, self.cfg.predict.params)

    def load_dataset_rows(self) -> Any:
        """Same resolution the `load` step performs, so a task can never score a
        different split or revision. The cache is reused; only `load` refetches."""
        return load_split(self.cfg.load, self.name)
