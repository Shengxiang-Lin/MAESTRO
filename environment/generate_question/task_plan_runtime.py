"""Compatibility import: authoritative implementation lives in `execution.task_plan_runtime`."""
from .execution.task_plan_runtime import *  # noqa: F401,F403
from .execution import task_plan_runtime as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
