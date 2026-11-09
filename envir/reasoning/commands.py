"""Canonical CLI registration for MAESTRO reasoning-first dataset generation.

Modern command names are user-facing; legacy *-v2 aliases stay functional
to permit reproducible experiments and safe migration.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import build_bank, refine_bank, replay_bank, audit_bank
from .workflow import generate, quality


def _input_flags(p):
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--mmqa", type=Path,
                        help="MultiModalQA train .jsonl(.gz); uses only topic/source metadata")
    inputs.add_argument("--scenes", type=Path,
                        help="A locally existing Task-First scene bank or scene_bank.jsonl")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target", type=int, default=100)
    p.add_argument("--max-records", type=int, default=30000)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--difficulty-mix", default="20,50,30",
                   help="Percentages Easy,Medium,Hard (sum=100)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable tqdm progress bars, e.g. for batch logs")


def _scenario_flags(p):
    p.add_argument("--scenario-api-base", default=None,
                   help="Optional OpenAI-compatible endpoint for pre-World scenario authoring")
    p.add_argument("--scenario-model", default=None)
    p.add_argument("--scenario-critic-api-base", default=None)
    p.add_argument("--scenario-critic-model", default=None)
    p.add_argument("--scenario-api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--scenario-critic-api-key-env", default=None)
    p.add_argument("--scenario-workers", type=int, default=4)
    p.add_argument("--scenario-max-revisions", type=int, default=2)
    p.add_argument("--scenario-timeout", type=int, default=180)


def _question_flags(p, *, required: bool):
    p.add_argument("--question-api-base", required=required, default=None)
    p.add_argument("--question-model", required=required, default=None)
    p.add_argument("--question-critic-api-base", default=None)
    p.add_argument("--question-critic-model", default=None)
    p.add_argument("--question-api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--question-workers", type=int, default=4)
    p.add_argument("--question-timeout", type=int, default=180)
    p.add_argument("--max-revisions", type=int, default=2)


def _scenes_flags(p):
    p.add_argument("--scenes", type=Path, required=True)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")


def register_reasoning_commands(sub) -> None:
    primary = sub.add_parser(
        "generate",
        help="Canonical end-to-end: dataset -> ProblemSpec -> World -> Program "
             "-> Source-Scoped TaskGraph -> optional LLM question -> quality",
    )
    _input_flags(primary)
    _scenario_flags(primary)
    _question_flags(primary, required=False)

    quality_cmd = sub.add_parser(
        "quality", help="Structured replay, provenance and question-quality report"
    )
    _scenes_flags(quality_cmd)
    quality_cmd.add_argument("--output", type=Path, default=None)

    check = sub.add_parser(
        "validate", help="Strict reproducibility check for generated scenes"
    )
    _scenes_flags(check)

    refine = sub.add_parser(
        "refine-questions",
        help="LLM Author/Critic on a frozen source-scoped scene bank",
    )
    _scenes_flags(refine)
    refine.add_argument("--output-dir", type=Path, required=True)
    refine.add_argument("--overwrite", action="store_true")
    _question_flags(refine, required=True)

    # Compatibility with previously published commands. These use the
    # EXACT same canonical implementation; no duplicate logic.
    build_old = sub.add_parser("build-v2", help=argparse.SUPPRESS)
    _input_flags(build_old)
    _scenario_flags(build_old)
    replay_old = sub.add_parser("replay-v2", help=argparse.SUPPRESS)
    _scenes_flags(replay_old)
    audit_old = sub.add_parser("audit-v2", help=argparse.SUPPRESS)
    _scenes_flags(audit_old)
    question_old = sub.add_parser("naturalize-v2", help=argparse.SUPPRESS)
    _scenes_flags(question_old)
    question_old.add_argument("--output-dir", type=Path, required=True)
    question_old.add_argument("--overwrite", action="store_true")
    _question_flags(question_old, required=True)


def reasoning_handlers() -> dict:
    return {
        "generate": generate,
        "quality": quality,
        "validate": replay_bank,
        "refine-questions": refine_bank,
        "build-v2": build_bank,
        "replay-v2": replay_bank,
        "audit-v2": audit_bank,
        "naturalize-v2": refine_bank,
    }
