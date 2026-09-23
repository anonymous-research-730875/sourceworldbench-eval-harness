from pathlib import Path

import pytest

from sourceworldbench_eval_harness.config import InvalidConfigError, load_config
from sourceworldbench_eval_harness.tasks import resolve


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


BASE = """
shared:
  output_root: outputs/run
  output_tag: trial
task:
  name: test_status_anchored
load:
  name: owner/dataset
  revision: v1
predict:
  model: all_passed
  output: predictions.jsonl
evaluate:
  input: predictions.jsonl
  output: metrics.json
leaderboard:
  input: metrics.json
  dataset: owner/dataset
  model: baseline
"""


def test_load_config_validates_defaults_derivations_and_revision(tmp_path: Path) -> None:
    cfg = load_config(write_config(tmp_path, BASE))

    assert cfg.shared.output_dir == "outputs/run_trial"
    assert cfg.load.split == "test"
    assert cfg.load.output is None
    assert cfg.leaderboard is not None
    assert cfg.leaderboard.revision == "v1"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("output_tag: trial", "output_tga: trial", "output_tga"),
    ],
)
def test_invalid_fields_fail_at_load(tmp_path: Path, old: str, new: str, message: str) -> None:
    text = BASE.replace(old, new)

    with pytest.raises(InvalidConfigError, match=message):
        load_config(write_config(tmp_path, text))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("  name: test_status_anchored", "  name: test_status_anchored\n  id_feild: pair_id", "id_feild"),
    ],
)
def test_task_settings_are_validated_by_the_named_task(tmp_path: Path, old: str, new: str, message: str) -> None:
    """`task` settings belong to the task, so `resolve` is where they're checked."""
    cfg = load_config(write_config(tmp_path, BASE.replace(old, new)))

    with pytest.raises(InvalidConfigError, match=message):
        resolve(cfg)


def test_resolve_fills_task_defaults(tmp_path: Path) -> None:
    cfg = load_config(write_config(tmp_path, BASE))

    task = resolve(cfg)

    assert task.settings.id_field == "instance_id"
    assert task.settings.require_full_coverage is True
    # `resolve` installs the completed block, so the recorded sidecar matches
    # what the task actually ran with.
    assert cfg.task is task.settings


def test_dumped_config_keeps_the_named_tasks_settings(tmp_path: Path) -> None:
    """The sidecar has to record the settings scored with, not just the shared ones."""
    cfg = load_config(write_config(tmp_path, BASE))
    resolve(cfg)

    dumped = cfg.model_dump(exclude_none=True)

    assert dumped["task"]["id_field"] == "instance_id"
    assert dumped["task"]["require_full_coverage"] is True


def test_predict_rejects_two_input_sources(tmp_path: Path) -> None:
    text = BASE.replace(
        "  model: all_passed",
        "  model: all_passed\n  input: loaded.jsonl\n  external_input: external.jsonl",
    )
    with pytest.raises(InvalidConfigError, match="only one of input and external_input"):
        load_config(write_config(tmp_path, text))


def test_pipeline_paths_must_connect(tmp_path: Path) -> None:
    text = BASE.replace("input: predictions.jsonl", "input: other.jsonl")
    with pytest.raises(InvalidConfigError, match="predict.output and evaluate.input"):
        load_config(write_config(tmp_path, text))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("output_tag: trial", "output_tag: <tag>", r"shared\.output_tag: <tag>"),
        ("  model: all_passed", "  model: <model>", r"predict\.model: <model>"),
        ("  model: baseline", "  model: <model>", r"leaderboard\.model: <model>"),
    ],
)
def test_unreplaced_template_placeholders_are_rejected(tmp_path: Path, old: str, new: str, message: str) -> None:
    """A copied template must not reach the leaderboard as a `<model>` submission."""
    with pytest.raises(InvalidConfigError, match=message):
        load_config(write_config(tmp_path, BASE.replace(old, new)))


def test_empty_config_is_rejected(tmp_path: Path) -> None:
    """An empty file parses to `None`, which must fail as a config rather than crash."""
    with pytest.raises(InvalidConfigError, match="shared"):
        load_config(write_config(tmp_path, ""))


def test_non_mapping_config_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidConfigError, match="must be a YAML mapping, got list"):
        load_config(write_config(tmp_path, "- shared\n- task\n"))


def drop_block(text: str, block: str) -> str:
    lines = text.splitlines()
    start = lines.index(f"{block}:")
    end = next((i for i in range(start + 1, len(lines)) if lines[i] and not lines[i].startswith(" ")), len(lines))
    return "\n".join(lines[:start] + lines[end:])


@pytest.mark.parametrize("block", ["shared", "task", "load", "predict", "evaluate"])
def test_every_pipeline_block_is_required(tmp_path: Path, block: str) -> None:
    with pytest.raises(InvalidConfigError, match=block):
        load_config(write_config(tmp_path, drop_block(BASE, block)))


def test_leaderboard_block_is_optional(tmp_path: Path) -> None:
    """Scoring predictions locally submits nothing, so it needs no `leaderboard`."""
    cfg = load_config(write_config(tmp_path, drop_block(BASE, "leaderboard")))

    assert cfg.leaderboard is None
    assert cfg.load.revision == "v1"
