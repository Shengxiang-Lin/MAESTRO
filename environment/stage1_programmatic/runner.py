"""Stage-1 adapter around the pinned Orchestra-O3 deterministic build.

The implementation is intentionally not rewritten here.  Every argument is
forwarded unchanged to ``environment.build_all`` in the pinned upstream commit.
"""
from __future__ import annotations

import sys
from typing import Sequence

from environment.upstream import run_upstream_module


def run(args: Sequence[str]) -> int:
    return run_upstream_module("environment.build_all", args)


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))
