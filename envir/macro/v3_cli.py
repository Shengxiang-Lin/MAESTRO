"""Macro V3 concrete question + source-grounded reliable feedback pilot.

  python -m envir.macro.v3_cli build --kqa ... --mmqa ... --target 100
  python -m envir.macro.v3_cli replay --scenes ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .priors import load_kqa, load_mmqa
from .v2_omni import load_omni_styles
from .v3_builder import build_many, counterfactual_sensitivity
from .v3_runtime import validate_grounded


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build(args):
    for p in (args.kqa, args.mmqa):
        if not p.is_file():
            raise FileNotFoundError(p)
    if min(args.target, args.max_kqa_records, args.max_mmqa_records) < 1:
        raise ValueError("target and source limits must be positive")
    out = args.output_dir
    paths = {
        "teacher": out / "private_grounded_scenes_v3.jsonl",
        "public": out / "public_questions_v3.jsonl",
        "summary": out / "macro_v3_summary.json",
    }
    if not args.overwrite and any(p.exists() for p in paths.values()):
        raise FileExistsError("v3 output already exists: use new directory or --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    kqa, kqa_stats = load_kqa(args.kqa, limit=args.max_kqa_records)
    mmqa, mmqa_stats = load_mmqa(args.mmqa, limit=args.max_mmqa_records)
    omni_styles, omni_stats = None, None
    if args.omnigaia:
        if not args.omnigaia.is_file():
            raise FileNotFoundError(args.omnigaia)
        omni_styles, omni_stats = load_omni_styles(args.omnigaia)
    scenes, stats = build_many(kqa=kqa, mmqa=mmqa, target=args.target,
                               seed=args.seed, omni_styles=omni_styles)
    authoring = {"mode": "deterministic", "accepted": 0,
                 "rejected": 0, "backend_error": 0}
    if args.question_model or args.question_api_base:
        if not args.question_model or not args.question_api_base:
            raise ValueError("both --question-model and --question-api-base are required")
        from .author import OpenAICompatibleQuestionAuthor
        from .v3_author import rewrite_public_questions
        agent = OpenAICompatibleQuestionAuthor(
            api_base=args.question_api_base, model=args.question_model,
            timeout=args.question_timeout,
        )
        authoring = {"mode": "optional_local_llm_surface",
                     **rewrite_public_questions(scenes, author=agent,
                                                workers=args.question_workers)}
    stats["unique_question_texts"] = len({s["question"] for s in scenes})
    stats["question_authoring"] = authoring
    _write_jsonl(paths["teacher"], scenes)
    _write_jsonl(paths["public"], [s["public"] for s in scenes])
    summary = {
        "status": "done", "requested": args.target,
        "statistics": stats,
        "prior_coverage": {
            "KQA": kqa_stats, "MultiModalQA": mmqa_stats, "OmniGAIA": omni_stats
        },
        "files": {k: str(v) for k, v in paths.items()},
        "certification": {
            "graph_contract": "typed_source_scoped_handoffs",
            "world": "deterministically_synthetic_not_external_truth",
            "feedback": "all_subtasks_source_replay_verified",
            "evidence": "all_required_sources_ablated_and_one_value_counterfactual",
            "media": "textual_simulated_observations_only",
            "max_concurrent_subagents": 10,
        },
        "known_limitations": [
            "questions_are_deterministic_templated_not_human_rated",
            "not_real_media_perception",
            "original_KQA_MMQA_patterns_are_abstracted_not_original_semantic_facts",
            "single_fact_counterfactual_is_not_full_causal_identification",
        ],
    }
    _dump(paths["summary"], summary)
    return summary


def replay(args):
    errors, tested = [], 0
    with args.scenes.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            tested += 1
            scene = json.loads(line)
            issues = validate_grounded(scene)
            if not issues:
                actual = counterfactual_sensitivity(scene)
                saved = scene["verification"].get("counterfactual_sensitivity")
                if actual != saved or not actual["answer_changed"]:
                    issues.append("counterfactual_witness_drift")
            if issues:
                errors.append({"task_id": scene.get("task_id"),
                               "issues": issues})
    return {"status": "valid" if tested and not errors else "invalid",
            "checked": tested, "failure_count": len(errors),
            "failures": errors[:20]}


def parser():
    ap = argparse.ArgumentParser(
        description="MAESTRO Macro V3: concrete information needs and verified clean feedback")
    sub = ap.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--kqa", type=Path,
                              default=Path("data/world-source/KQA-Pro/train.json"))
    build_parser.add_argument("--mmqa", type=Path,
                              default=Path("data/world-source/MultiModalQA/MMQA_train.jsonl.gz"))
    build_parser.add_argument("--omnigaia", type=Path, default=None,
                              help="Optional OmniGAIA JSON/JSONL for question style only")
    build_parser.add_argument("--output-dir", type=Path,
                              default=Path("envir/data/macro_v3"))
    build_parser.add_argument("--target", type=int, default=100)
    build_parser.add_argument("--max-kqa-records", type=int, default=10000)
    build_parser.add_argument("--max-mmqa-records", type=int, default=3000)
    build_parser.add_argument("--seed", type=int, default=17)
    build_parser.add_argument("--overwrite", action="store_true")
    build_parser.add_argument("--question-api-base", default=None,
                              help="Optional local OpenAI-compatible endpoint for Question text only")
    build_parser.add_argument("--question-model", default=None)
    build_parser.add_argument("--question-workers", type=int, default=4)
    build_parser.add_argument("--question-timeout", type=int, default=90)
    check = sub.add_parser("replay")
    check.add_argument("--scenes", type=Path, required=True)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    result = {"build": build, "replay": replay}[args.command](args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"done", "valid"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
