"""Task-first clean evidence pilot.

python -m envir.macro.v4_cli build --kqa ... --mmqa ... --target 100
python -m envir.macro.v4_cli replay --scenes .../private_clean_scenes_v4.jsonl

No runtime variants, recovery, evidence corruption or final Gold answers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .priors import load_kqa, load_mmqa
from .v4_builder import build_batch
from .v4_validation import validate_clean_scene


def _json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonl(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as writer:
        for entry in entries:
            writer.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build(args):
    if min(args.target, args.max_kqa_records, args.max_mmqa_records) < 1:
        raise ValueError("target and data limits must be positive")
    for path in (args.kqa, args.mmqa):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_dir
    paths = {
        "private": output / "private_clean_scenes_v4.jsonl",
        "public": output / "public_questions_v4.jsonl",
        "summary": output / "macro_v4_summary.json",
    }
    if any(path.exists() for path in paths.values()) and not args.overwrite:
        raise FileExistsError("V4 output exists; choose another dir or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    kqa, ks = load_kqa(args.kqa, limit=args.max_kqa_records)
    mmqa, ms = load_mmqa(args.mmqa, limit=args.max_mmqa_records)
    scenes, stats = build_batch(
        kqa=kqa, mmqa=mmqa, target=args.target, seed=args.seed)
    _jsonl(paths["private"], scenes)
    _jsonl(paths["public"], [s["public"] for s in scenes])
    summary = {
        "status": "done",
        "requested": args.target,
        "statistics": stats,
        "original_pattern_coverage": {"KQA": ks, "MultiModalQA": ms},
        "files": {key: str(path) for key, path in paths.items()},
        "stage_order": [
            "compose_macro_tasks",
            "derive_public_question_from_macro_work",
            "construct_per_task_synthetic_evidence",
            "derive_clean_feedback_and_summary_trace",
            "verify_terminal_evidence_coverage",
        ],
        "validation_scope": (
            "typed_macro_handoffs_and_deterministic_clean_evidence_replay"
        ),
        "limitations": [
            "textual_simulated_source_surrogates_not_actual_media",
            "no_external_world_fact_verification",
            "no_final_gold_answer_or_answer_supervision",
            "no_conflicts_missing_evidence_or_task_recovery",
            "question_naturalness_not_human_certified",
        ],
    }
    _json(paths["summary"], summary)
    return summary


def replay(args):
    tested, issues = 0, []
    with args.scenes.open("r", encoding="utf-8") as reader:
        for raw in reader:
            if not raw.strip():
                continue
            tested += 1
            scene = json.loads(raw)
            problems = validate_clean_scene(scene)
            if problems:
                issues.append({"task_id": scene.get("task_id"),
                               "issues": problems})
    return {
        "status": "valid" if tested and not issues else "invalid",
        "checked": tested, "failure_count": len(issues),
        "failures": issues[:20],
    }


def parser():
    p = argparse.ArgumentParser(
        description="MAESTRO V4 clean Task-first / Evidence-within-Task pilot"
    )
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--kqa", type=Path,
                   default=Path("data/world-source/KQA-Pro/train.json"))
    b.add_argument("--mmqa", type=Path,
                   default=Path("data/world-source/MultiModalQA/MMQA_train.jsonl.gz"))
    b.add_argument("--output-dir", type=Path, default=Path("envir/data/macro_v4"))
    b.add_argument("--target", type=int, default=100)
    b.add_argument("--seed", type=int, default=17)
    b.add_argument("--max-kqa-records", type=int, default=10000)
    b.add_argument("--max-mmqa-records", type=int, default=3000)
    b.add_argument("--overwrite", action="store_true")
    r = sub.add_parser("replay")
    r.add_argument("--scenes", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    result = {"build": build, "replay": replay}[args.command](args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"done", "valid"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
