"""Hugging Face Hub helpers, shared by the load, evaluate and push steps."""

import logging
import os
import threading
from pathlib import Path
from typing import Any

from sourceworldbench_eval_harness.config import LoadConfig

DEFAULT_TIMEOUT = 15.0


def load_token() -> str | None:
    """`HF_TOKEN` from `.env`, or None. Two details are centralised: `.env` is resolved from *this*
    file upwards, since bare `load_dotenv()` searches from the calling frame and finds nothing
    when run from outside the repo; and the value is stripped, since a trailing newline fails
    authentication with no visible reason.
    """
    from dotenv import load_dotenv

    load_dotenv(_dotenv_path())
    token = os.environ.get("HF_TOKEN")
    if token is None:
        return None
    token = token.strip()
    if not token:
        return None
    os.environ["HF_TOKEN"] = token  # so libraries reading the env see the clean value
    return token


def _dotenv_path() -> Path | None:
    for directory in Path(__file__).resolve().parents:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def require_token(purpose: str) -> str:
    """Like `load_token`, but fails immediately rather than letting the Hub return a
    confusing not-found for a private repo."""
    token = load_token()
    if not token:
        raise RuntimeError(
            f"HF_TOKEN is required to {purpose}. Copy .env.example to .env and set it, "
            "or export HF_TOKEN in your shell."
        )
    return token


def resolve_sha(name: str, revision: str | None, timeout: float = DEFAULT_TIMEOUT) -> str:
    """The commit `revision` — or `main` — points at, recorded in the sidecar because a tag is
    mutable.

    Best-effort: a daemon thread with a bounded wait returning `"unresolved"`, since provenance
    is not worth failing a run over and a step may proceed from cache with no network.
    """
    from huggingface_hub import HfApi

    token = load_token()
    resolved: dict[str, str] = {}
    failure: dict[str, str] = {}

    def lookup() -> None:
        try:
            resolved["sha"] = HfApi().dataset_info(name, revision=revision, token=token).sha or "unknown"
        except Exception as exc:  # provenance failures are never fatal
            failure["reason"] = f"{type(exc).__name__}: {exc}".splitlines()[0]

    thread = threading.Thread(target=lookup, daemon=True)
    thread.start()
    thread.join(timeout)

    if "sha" in resolved:
        return resolved["sha"]
    logging.warning(
        "could not resolve %s@%s (%s); continuing, but the sidecar will not pin the exact commit",
        name,
        revision or "main",
        failure.get("reason", f"no response within {timeout:.0f}s"),
    )
    return "unresolved"


def load_split(load_cfg: LoadConfig, caller: str, download_mode: str | None = None) -> Any:
    """The one place a `load` block becomes a dataset, so the `load` step and the
    tasks cannot disagree about which bytes a run reads. `download_mode` is the one
    thing callers differ on: `load` pulls fresh, everything else reuses the cache.
    """
    from datasets import load_dataset

    revision = load_cfg.revision
    load_cfg.resolved_sha = resolve_sha(load_cfg.name, revision)
    logging.info(
        "[%s] dataset %s @ %s (sha %s)",
        caller,
        load_cfg.name,
        revision or "main",
        load_cfg.resolved_sha[:8],
    )
    return load_dataset(
        load_cfg.name,
        split=load_cfg.split,
        revision=revision,
        download_mode=download_mode,
    )
