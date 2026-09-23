"""Public API for the model framework and the concrete-model registry."""

from sourceworldbench_eval_harness.models.core.base import BasePredictor
from sourceworldbench_eval_harness.models.registry import MODELS, resolve

__all__ = ["MODELS", "BasePredictor", "resolve"]
