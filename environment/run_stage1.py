"""CLI entry for Stage 1: programmatic composition and frozen task construction."""
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.stage1_programmatic.runner import main


if __name__ == "__main__":
    main()
