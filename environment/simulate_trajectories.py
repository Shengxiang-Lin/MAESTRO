"""Generate LLM-simulated orchestration trajectories."""
# Support both python -m environment.<entry> and python environment/<entry>.py
from pathlib import Path
import sys
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.trajectory_simulation.cli import main

if __name__ == "__main__":
    main()
