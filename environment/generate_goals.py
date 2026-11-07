"""Stage 2A CLI: generate, review and persist certified Subtask Blueprints."""
from __future__ import annotations

# Support both: python -m environment.generate_goals
# and:          python environment/generate_goals.py
from pathlib import Path
import sys
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from environment.config import EnvironmentConfigError, parse_with_config
from environment.question_generation.goals_pipeline import run
from environment.question_generation.pipeline import build_parser
from environment.question_generation.streaming import QuestionCheckpointError


def main():
    parser = build_parser()
    parser.description = (
        "Stage 2A: independently generate and review each DAG Fragment Goal; "
        "save only fully certified Blueprints for Stage 2B."
    )
    parser.add_argument("--max-goal-revisions", type=int, default=2)
    parser.add_argument(
        "--reuse-goal-audit", default=None,
        help="Optional Stage2A v5 goal_review_audit.jsonl from a prior run; "
             "reuse only already passed same-contract Fragments with matching models.",
    )
    parser.add_argument("--restart-goals", action="store_true")
    try:
        args = parse_with_config(parser, "question_generation")
    except EnvironmentConfigError as exc:
        raise SystemExit(f"Environment configuration error: {exc}") from exc
    required = ("manifest", "authoring_inputs", "worlds",
                "capabilities", "output_dir")
    missing = [key for key in required if not getattr(args, key, None)]
    if missing:
        raise SystemExit("Missing Stage-2A arguments: " + ", ".join(missing))
    try:
        result = run(args)
    except (EnvironmentConfigError, QuestionCheckpointError, ValueError) as exc:
        raise SystemExit(f"Stage-2A error: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
