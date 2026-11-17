"""Compatibility import: authoritative implementation lives in `planning.goal_plan`."""
from .planning.goal_plan import *  # noqa: F401,F403
from .planning import goal_plan as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
