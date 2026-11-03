"""Stage-2 adapter around the pinned P1.5 question-realization pipeline.

The old ``llm_pipeline`` combined question authoring with optional P2 trajectory
synthesis.  MAESTRO deliberately forces ``--questions-only`` here so the three
stages have a clean boundary while preserving the existing P1.5 implementation.
"""
from __future__ import annotations

import sys
from typing import Sequence

from environment.upstream import run_upstream_module


def run(args: Sequence[str]) -> int:
    forwarded = list(args)
    if "--questions-only" not in forwarded:
        forwarded.append("--questions-only")
    return run_upstream_module("environment.llm_pipeline", forwarded)


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))
