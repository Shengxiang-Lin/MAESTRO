"""CLI entry for Stage 3: LLM trajectory simulation."""
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.stage3_trajectory_simulation.runner import main


if __name__ == "__main__":
    main()
