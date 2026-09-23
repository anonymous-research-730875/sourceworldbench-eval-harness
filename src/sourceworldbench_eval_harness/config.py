"""The YAML configuration every pipeline step reads."""

import re
from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, ValidationError, model_validator


class ConfigModel(BaseModel):
    """Strict base for user-authored configuration blocks."""

    model_config = ConfigDict(extra="forbid")


class SharedConfig(ConfigModel):
    output_root: str = Field(min_length=1)
    output_tag: str | None = None
    output_dir: str = ""  # always derived below, so never absent for a caller to narrow

    @model_validator(mode="after")
    def derive_output_dir(self) -> Self:
        suffix = f"_{self.output_tag}" if self.output_tag else ""
        self.output_dir = f"{self.output_root}{suffix}"
        return self


class TaskConfig(ConfigModel):
    """Extras are allowed because the settings a task adds are its own to declare:
    `tasks.resolve` revalidates this block against the named task's `Config`, which
    forbids the rest."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    id_field: str | None = None


class LoadConfig(ConfigModel):
    name: str = Field(min_length=1)
    split: str = Field(default="test", min_length=1)
    revision: str | None = None
    output: Annotated[str, Field(min_length=1)] | None = None
    download_mode: str | None = None
    # Runtime provenance added by ``hub.load_split`` and persisted in sidecars.
    resolved_sha: str | None = None


class PredictConfig(ConfigModel):
    model: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    input: str | None = None
    external_input: str | None = None
    output: str = Field(min_length=1)

    @model_validator(mode="after")
    def one_input_source(self) -> Self:
        if self.input is not None and self.external_input is not None:
            raise ValueError("only one of input and external_input can be specified")
        return self


class EvaluateConfig(ConfigModel):
    input: str = Field(min_length=1)
    output: str = Field(min_length=1)


class LeaderboardConfig(ConfigModel):
    input: str | None = None
    dataset: str = Field(min_length=1)
    model: str = Field(min_length=1)
    repo: str | None = None
    revision: str | None = None
    submitter: str | None = None


class EvalConfig(ConfigModel):
    shared: SharedConfig
    # `SerializeAsAny` because `tasks.resolve` replaces this with the named task's
    # subclass; without it pydantic would drop that task's settings from the sidecar.
    task: SerializeAsAny[TaskConfig]
    load: LoadConfig
    predict: PredictConfig
    evaluate: EvaluateConfig
    # Optional: a run that only scores locally never submits. `leaderboard.push` is
    # the sole reader and reports its absence itself.
    leaderboard: LeaderboardConfig | None = None

    @model_validator(mode="after")
    def check_pipeline(self) -> Self:
        leaderboard = self.leaderboard
        steps: list[tuple[str, str | None, str | None]] = [
            ("load", None, self.load.output),
            ("predict", self.predict.input, self.predict.output),
            ("evaluate", self.evaluate.input, self.evaluate.output),
        ]
        if leaderboard is not None:
            steps.append(("leaderboard", leaderboard.input, None))
        for (name, _, output), (next_name, input_, _) in zip(steps, steps[1:]):
            if next_name == "predict" and self.predict.external_input is not None:
                continue
            if output is not None and input_ is not None and output != input_:
                raise ValueError(f"fields {name}.output and {next_name}.input must match if both are specified")

        leaderboard_revision = leaderboard.revision if leaderboard is not None else None
        pinned = [revision for revision in (self.load.revision, leaderboard_revision) if revision]
        if len(set(pinned)) > 1:
            raise ValueError(
                f"load.revision ({self.load.revision}) and leaderboard.revision "
                f"({leaderboard_revision}) must be consistent — pin the version "
                "in one block and it is applied to both"
            )
        if pinned:
            self.load.revision = pinned[0]
            if leaderboard is not None:
                leaderboard.revision = pinned[0]
        return self


PLACEHOLDER = re.compile(r"<[^<>]+>")


def _find_placeholders(node: Any, path: str = "") -> list[tuple[str, str]]:
    """Every `<placeholder>` a copied template still carries, as (field path, value)."""
    if isinstance(node, dict):
        return [found for key, value in node.items() for found in _find_placeholders(value, f"{path}.{key}".lstrip("."))]
    if isinstance(node, list):
        return [found for i, value in enumerate(node) for found in _find_placeholders(value, f"{path}[{i}]")]
    if isinstance(node, str) and PLACEHOLDER.search(node):
        return [(path, node)]
    return []


def _require_no_placeholders(raw: dict[str, Any], path: Path) -> None:
    unreplaced = _find_placeholders(raw)
    if unreplaced:
        fields = "\n".join(f"  {field}: {value}" for field, value in unreplaced)
        raise InvalidConfigError(
            f"Config at {path} still has template placeholders — replace them before running:\n{fields}"
        )


def load_config(path: Path) -> EvalConfig:
    """`tasks.resolve` completes the `task` block."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise InvalidConfigError(f"Config at {path} must be a YAML mapping, got {type(raw).__name__}")
    _require_no_placeholders(raw, path)
    try:
        return EvalConfig.model_validate(raw)
    except ValidationError as exc:
        raise InvalidConfigError(f"Invalid config at {path}:\n{exc}") from exc


class InvalidConfigError(ValueError):
    pass
