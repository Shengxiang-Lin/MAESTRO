"""Construct and freeze synthetic orchestration tasks."""
# Support both python -m environment.<entry> and python environment/<entry>.py
from pathlib import Path
import sys
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.task_construction.pipeline import main

if __name__ == "__main__":
    main()
