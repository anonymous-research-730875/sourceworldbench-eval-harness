"""Public API for the task framework and the concrete-task registry."""

from sourceworldbench_eval_harness.tasks.core.base import Task
from sourceworldbench_eval_harness.tasks.registry import TASKS, resolve

__all__ = ["TASKS", "Task", "resolve"]
