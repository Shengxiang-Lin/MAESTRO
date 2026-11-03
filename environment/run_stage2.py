"""CLI entry for Stage 2: LLM question generation over a frozen Stage-1 batch."""
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.stage2_question_generation.runner import main


if __name__ == "__main__":
    main()
