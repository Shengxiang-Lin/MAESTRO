"""Generate validated natural-language questions from frozen tasks."""
# Support both python -m environment.<entry> and python environment/<entry>.py
from pathlib import Path
import sys
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.question_generation.pipeline import main

if __name__ == "__main__":
    main()
