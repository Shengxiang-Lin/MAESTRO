"""Compatibility import: authoritative implementation lives in `planning.task_plan`."""
from .planning.task_plan import *  # noqa: F401,F403
from .planning import task_plan as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
