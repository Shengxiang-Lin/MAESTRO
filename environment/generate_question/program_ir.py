"""Compatibility import: authoritative implementation lives in `execution.program_ir`."""
from .execution.program_ir import *  # noqa: F401,F403
from .execution import program_ir as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
