"""Build/replay coarse-grained orchestration TaskPlans, fixed concurrency 10.

This is an independent task-planning experiment. Teacher-private graph files
are not runnable Sub-Agent feedback; the public file has only user intent and
resource descriptions, not hidden dependencies or internal execution steps.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

from .planner import build_batch
from .priors import load_kqa, load_mmqa
from .validation import validate_plan


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _jsonl(path: Path, values) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for item in values:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def build(args) -> dict:
    if min(args.target, args.max_kqa_records, args.max_mmqa_records) < 1:
        raise ValueError("target and dataset limits must be positive")
    for path in (args.kqa, args.mmqa):
        if not path.is_file():
            raise FileNotFoundError(f"dataset not found: {path}")
    out = args.output_dir
    names = ("private_taskplans.jsonl", "public_questions.jsonl",
             "macro_summary.json", "prior_statistics.json")
    outputs = {name: out / name for name in names}
    if not args.overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("macro outputs already exist; use a new directory or --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    kqa, kqa_stats = load_kqa(args.kqa, limit=args.max_kqa_records)
    mmqa, mmqa_stats = load_mmqa(args.mmqa, limit=args.max_mmqa_records)
    if args.bank is not None and not args.bank.is_file():
        raise FileNotFoundError(f"YAGO source bank not found: {args.bank}")
    if args.bank:
        # Teacher-side YAGO prior query must not mutate an existing bank.
        uri = f"file:{args.bank.resolve().as_posix()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            db.row_factory = sqlite3.Row
            plans, stats = build_batch(kqa=kqa, mmqa=mmqa,
                                       target=args.target, seed=args.seed,
                                       store=SimpleNamespace(db=db))
    else:
        plans, stats = build_batch(kqa=kqa, mmqa=mmqa,
                                   target=args.target, seed=args.seed)
    question_authoring = {"mode": "deterministic", "accepted": 0,
                          "rejected": 0, "errors": 0}
    if args.question_model or args.question_api_base:
        if not args.question_model or not args.question_api_base:
            raise ValueError("--question-model and --question-api-base are both required")
        from .author import OpenAICompatibleQuestionAuthor, naturalize_questions
        author = OpenAICompatibleQuestionAuthor(
            api_base=args.question_api_base,
            model=args.question_model,
            timeout=args.question_timeout,
        )
        author_stats = naturalize_questions(
            plans, author=author, workers=args.question_workers,
        )
        question_authoring = {"mode": "optional_llm_surface", **author_stats}
        from .validation import validate_plan
        for plan in plans:
            problems = validate_plan(plan)
            if problems:
                raise ValueError("LLM surface modification invalid: " + str(problems))
    stats["unique_question_texts"] = len({p["question"] for p in plans})
    stats["question_duplicate_count"] = len(plans) - stats["unique_question_texts"]
    stats["question_authoring"] = question_authoring
    _jsonl(outputs["private_taskplans.jsonl"], plans)
    _jsonl(outputs["public_questions.jsonl"], [p["public"] for p in plans])
    priors = {
        "KQA": kqa_stats,
        "MultiModalQA": mmqa_stats,
        "YAGO_bank": str(args.bank) if args.bank else None,
        "policy": "source_patterns_are_priors_not_grounded_answers",
    }
    _json(outputs["prior_statistics.json"], priors)
    summary = {
        "status": "done", "requested": args.target,
        "statistics": stats, "priors": priors,
        "files": {key: str(value) for key, value in outputs.items()},
        "certification_scope": "typed macro DAG, nonleaking question and coarsening only",
        "pending": ["world_grounding", "reliable_subagent_feedback",
                    "evidence_replay", "natural_question_semantic_review"],
    }
    _json(outputs["macro_summary.json"], summary)
    return summary


def replay(args) -> dict:
    checked = 0
    failures = []
    questions = []
    with args.gold.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            checked += 1
            plan = json.loads(line)
            problems = validate_plan(plan)
            if problems:
                failures.append({"task_id": plan.get("task_id"),
                                 "issues": problems})
            questions.append(plan.get("question"))
    return {
        "status": "valid" if not failures and checked else "invalid",
        "checked": checked,
        "issues": failures[:20],
        "failure_count": len(failures),
        "unique_question_texts": len(set(questions)),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MAESTRO coarse-grained multi-subtask TaskPlan construction"
    )
    sub = p.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build", help="Build macro TaskPlans from original dataset priors")
    b.add_argument("--kqa", type=Path,
                   default=Path("data/world-source/KQA-Pro/train.json"))
    b.add_argument("--mmqa", type=Path,
                   default=Path("data/world-source/MultiModalQA/MMQA_train.jsonl.gz"))
    b.add_argument("--bank", type=Path, default=None,
                   help="Optional read-only YAGO bank from envir.cli prepare")
    b.add_argument("--output-dir", type=Path, default=Path("envir/data/macro_v1"))
    b.add_argument("--target", type=int, default=100)
    b.add_argument("--max-kqa-records", type=int, default=10000)
    b.add_argument("--max-mmqa-records", type=int, default=3000)
    b.add_argument("--seed", type=int, default=17)
    b.add_argument("--overwrite", action="store_true")
    b.add_argument("--question-api-base", default=None,
                   help="Optional OpenAI-compatible vLLM base, e.g. http://localhost:8000/v1")
    b.add_argument("--question-model", default=None,
                   help="Optional local model for public question naturalization only")
    b.add_argument("--question-workers", type=int, default=4)
    b.add_argument("--question-timeout", type=int, default=90)
    r = sub.add_parser("replay", help="Recheck macro DAG contracts from saved teacher plans")
    r.add_argument("--plans", "--gold", dest="gold", type=Path, required=True)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    result = {"build": build, "replay": replay}[args.action](args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"done", "valid"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
