"""Programmatic synthetic task construction for MAESTRO.

This package preserves the production Orchestra-O3 construction semantics while
living under a functional MAESTRO namespace.
"""

from __future__ import annotations

import sys

from .ir import ProgramIR, ProgramNode
from .world import EvidenceSource, Fact, SyntheticWorld

from .data_sources import kqa as _kqa
from .data_sources import mmqa as _mmqa
from .data_sources import taxonomy as _taxonomy
from .data_sources import yago as _yago

for _name, _module in {
    "kqa": _kqa,
    "mmqa": _mmqa,
    "taxonomy": _taxonomy,
    "yago": _yago,
}.items():
    sys.modules.setdefault(f"{__name__}.{_name}", _module)

from .quality import difficulty as _difficulty
from .quality import diversity as _diversity
from .quality import hardness as _hardness
from .quality import semantic_policy as _semantic_policy
from .utils import progress as _progress

for _name, _module in {
    "difficulty": _difficulty,
    "diversity": _diversity,
    "hardness": _hardness,
    "semantic_policy": _semantic_policy,
    "progress": _progress,
}.items():
    sys.modules.setdefault(f"{__name__}.{_name}", _module)

from .executor import ExecutionError, SymbolicExecutor
from .validation import ValidationReport, validate_task

__all__ = [
    "ExecutionError", "SymbolicExecutor", "ProgramIR", "ProgramNode",
    "ValidationReport", "validate_task", "EvidenceSource", "Fact", "SyntheticWorld",
]
