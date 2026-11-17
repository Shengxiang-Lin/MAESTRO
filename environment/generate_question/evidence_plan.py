"""Compatibility import: authoritative implementation lives in `planning.evidence_plan`."""
from .planning.evidence_plan import *  # noqa: F401,F403
from .planning import evidence_plan as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
