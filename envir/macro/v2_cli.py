"""MAESTRO macro v2: evidence-need-driven work composition, no runtime profiles.

Run: python -m envir.macro.v2_cli build ...
     python -m envir.macro.v2_cli replay --plans ...
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

from .priors import load_kqa, load_mmqa
from .v2_omni import load_omni_styles
from .v2_planner import build_v2_batch
from .v2_validation import validate_v2


def _dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(args) -> dict:
    if min(args.target, args.max_kqa_records, args.max_mmqa_records) < 1:
        raise ValueError("target and source record limits must be positive")
    for p in (args.kqa, args.mmqa):
        if not p.is_file():
            raise FileNotFoundError(p)
    if args.omnigaia and not args.omnigaia.is_file():
        raise FileNotFoundError(args.omnigaia)
    if args.bank and not args.bank.is_file():
        raise FileNotFoundError(args.bank)
    out = args.output_dir
    targets = {
        "private": out / "private_taskplans_v2.jsonl",
        "public": out / "public_questions_v2.jsonl",
        "summary": out / "macro_v2_summary.json",
        "priors": out / "macro_v2_source_priors.json",
    }
    if any(p.exists() for p in targets.values()) and not args.overwrite:
        raise FileExistsError("macro v2 output exists; change directory or pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    kqa, kqa_stats = load_kqa(args.kqa, limit=args.max_kqa_records)
    mmqa, mmqa_stats = load_mmqa(args.mmqa, limit=args.max_mmqa_records)
    omni_styles, omni_stats = None, None
    if args.omnigaia:
        omni_styles, omni_stats = load_omni_styles(args.omnigaia)
    if args.bank:
        uri = "file:" + args.bank.resolve().as_posix() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            db.row_factory = sqlite3.Row
            plans, stats = build_v2_batch(
                kqa=kqa, mmqa=mmqa, target=args.target, seed=args.seed,
                store=SimpleNamespace(db=db), omni_styles=omni_styles)
    else:
        plans, stats = build_v2_batch(
            kqa=kqa, mmqa=mmqa, target=args.target, seed=args.seed,
            omni_styles=omni_styles)
    source_priors = {
        "KQA": kqa_stats, "MultiModalQA": mmqa_stats,
        "OmniGAIA": omni_stats,
        "YAGO_bank": str(args.bank) if args.bank else None,
        "policy": "original_question_shapes_and_knowledge_schema_as_priors_only",
    }
    _jsonl(targets["private"], plans)
    _jsonl(targets["public"], [p["public"] for p in plans])
    _dump(targets["priors"], source_priors)
    summary = {
        "status": "done",
        "generation": "macro_v2_backward_work_product_planner",
        "requested": args.target,
        "statistics": stats,
        "source_priors": source_priors,
        "files": {k: str(path) for k, path in targets.items()},
        "verification_scope": "structural and named semantic handoff contract only",
        "pending": [
            "instantiate_simulated_evidence_and_world",
            "run_reliable_scoped_subagent_feedback",
            "prove_fact_support_and_question_equivalence",
        ],
    }
    _dump(targets["summary"], summary)
    return summary


def replay(args) -> dict:
    count, errors, questions = 0, [], []
    with args.plans.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            item = json.loads(line)
            faults = validate_v2(item)
            if faults:
                errors.append({"task_id": item.get("task_id"), "issues": faults})
            questions.append(item["question"])
    return {
        "status": "valid" if count and not errors else "invalid",
        "checked": count,
        "failure_count": len(errors),
        "failures": errors[:20],
        "unique_question_texts": len(set(questions)),
    }


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Macro TaskPlan v2: semantic work products, typed dependencies, max concurrency 10")
    subs = ap.add_subparsers(dest="command", required=True)
    b = subs.add_parser("build", help="Create ungrounded macro TaskPlan candidates")
    b.add_argument("--kqa", type=Path,
                   default=Path("data/world-source/KQA-Pro/train.json"))
    b.add_argument("--mmqa", type=Path,
                   default=Path("data/world-source/MultiModalQA/MMQA_train.jsonl.gz"))
    b.add_argument("--omnigaia", type=Path, default=None,
                   help="Optional official OmniGAIA raw JSON/JSONL; use question STYLE only")
    b.add_argument("--bank", type=Path, default=None,
                   help="Optional existing read-only YAGO EvidenceStore SQLite")
    b.add_argument("--output-dir", type=Path, default=Path("envir/data/macro_v2"))
    b.add_argument("--target", type=int, default=100)
    b.add_argument("--max-kqa-records", type=int, default=10000)
    b.add_argument("--max-mmqa-records", type=int, default=3000)
    b.add_argument("--seed", type=int, default=17)
    b.add_argument("--overwrite", action="store_true")
    r = subs.add_parser("replay", help="Validate previously generated private macro plans")
    r.add_argument("--plans", type=Path, required=True)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    output = {"build": build, "replay": replay}[args.command](args)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["status"] in {"done", "valid"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
