"""Stage-3 adapter for trajectory planning/synthesis.

The current implementation starts from the existing P2 ``clean_trajectory``
Teacher synthesis.  This directory is the extension point for MAESTRO-specific
unreliable-feedback, recovery and counterfactual trajectory simulation.
"""
from __future__ import annotations

import sys
from typing import Sequence

from environment.upstream import run_upstream_module

_COMMANDS = {"plan", "synthesize"}


def run(args: Sequence[str]) -> int:
    forwarded = list(args)
    if not forwarded or forwarded[0] not in _COMMANDS:
        forwarded.insert(0, "synthesize")
    return run_upstream_module("environment.clean_trajectory.cli", forwarded)


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))
