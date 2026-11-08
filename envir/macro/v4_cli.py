"""V4 clean Task-first Question generation and optional LLM review.

Deterministic pilot:
  python -m envir.macro.v4_cli build --kqa ... --mmqa ... --target 100

Naturalize an EXISTING immutable Task/Evidence bank (no KQA/MMQA reload):
  python -m envir.macro.v4_cli naturalize \
    --scenes envir/data/macro_v4/private_clean_scenes_v4.jsonl \
    --output-dir envir/data/macro_v4_naturalized \
    --question-api-base http://127.0.0.1:8807/v1 \
    --question-model local-qwen3.5-9b --max-revisions 2

Or add --question-api-base and --question-model to 'build'.
Always verify after either path with 'replay'. Failed author/critic
candidates revert to the deterministic task-derived reference question.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .priors import load_kqa, load_mmqa
from .v4_builder import build_batch
from .v4_question_review import naturalize_scenes
from .v4_validation import validate_clean_scene


def _json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonl(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as writer:
        for entry in entries:
            writer.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _paths(output: Path, *, include_audit: bool):
    paths = {
        "private": output / "private_clean_scenes_v4.jsonl",
        "public": output / "public_questions_v4.jsonl",
        "summary": output / "macro_v4_summary.json",
    }
    if include_audit:
        paths["question_review_audit"] = output / "question_review_audit_v4.jsonl"
    return paths


def _check_output(paths: dict, overwrite: bool):
    if not overwrite and any(p.exists() for p in paths.values()):
        raise FileExistsError("V4 output exists; choose a new directory or --overwrite")


def _write_outputs(paths: dict, scenes: list[dict], summary: dict):
    for scene in scenes:
        errors = validate_clean_scene(scene)
        if errors:
            raise ValueError(
                f"Invalid scene {scene.get('task_id')}: {errors}"
            )
    _jsonl(paths["private"], scenes)
    _jsonl(paths["public"], [scene["public"] for scene in scenes])
    if "question_review_audit" in paths:
        _jsonl(paths["question_review_audit"], [
            {
                "task_id": scene["task_id"],
                "reference_question": scene["question_authoring"]["reference_question"],
                "question": scene["question"],
                "review": scene["question_authoring"],
            }
            for scene in scenes
        ])
    _json(paths["summary"], summary)


def _llm_requested(args) -> bool:
    return bool(
        args.question_model or args.question_api_base
        or args.question_critic_model or args.question_critic_api_base
    )


def _naturalize_if_requested(scenes, args):
    if not _llm_requested(args):
        if args.command == "naturalize":
            raise ValueError("naturalize requires --question-model and --question-api-base")
        return {
            "mode": "deterministic",
            "accepted": 0, "fallback": 0, "author_calls": 0,
            "critic_calls": 0, "accepted_after_revision": 0,
            "model_review_not_formal_proof": True,
        }
    if not args.question_model or not args.question_api_base:
        raise ValueError("both --question-model and --question-api-base are required")
    if not 0 <= args.max_revisions <= 8 or args.question_workers < 1:
        raise ValueError("--max-revisions must be 0..8; --question-workers >= 1")
    from environment.question_generation.backend import OpenAICompatibleBackend
    key = os.environ.get(args.question_api_key_env) if args.question_api_key_env else None
    critic_key = (
        os.environ.get(args.question_critic_api_key_env)
        if args.question_critic_api_key_env else key
    )
    author = OpenAICompatibleBackend(
        base_url=args.question_api_base,
        model=args.question_model, api_key=key,
        temperature=args.question_temperature,
        max_tokens=args.question_max_tokens,
        timeout=args.question_timeout,
        json_mode=not args.question_no_json_mode,
    )
    critic = OpenAICompatibleBackend(
        base_url=args.question_critic_api_base or args.question_api_base,
        model=args.question_critic_model or args.question_model,
        api_key=critic_key,
        temperature=args.question_critic_temperature,
        max_tokens=args.question_critic_max_tokens,
        timeout=args.question_timeout,
        json_mode=not args.question_no_json_mode,
    )
    stats = naturalize_scenes(
        scenes, author=author, critic=critic,
        max_revisions=args.max_revisions,
        workers=args.question_workers,
    )
    stats["author_model"] = args.question_model
    stats["critic_model"] = args.question_critic_model or args.question_model
    return stats


def build(args):
    if min(args.target, args.max_kqa_records, args.max_mmqa_records) < 1:
        raise ValueError("target and data limits must be positive")
    for path in (args.kqa, args.mmqa):
        if not path.is_file():
            raise FileNotFoundError(path)
    paths = _paths(args.output_dir, include_audit=_llm_requested(args))
    _check_output(paths, args.overwrite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kqa, ks = load_kqa(args.kqa, limit=args.max_kqa_records)
    mmqa, ms = load_mmqa(args.mmqa, limit=args.max_mmqa_records)
    scenes, stats = build_batch(
        kqa=kqa, mmqa=mmqa, target=args.target, seed=args.seed,
    )
    review_stats = _naturalize_if_requested(scenes, args)
    stats["unique_question_texts"] = len({scene["question"] for scene in scenes})
    stats["question_authoring"] = review_stats
    summary = {
        "status": "done",
        "requested": args.target,
        "statistics": stats,
        "original_pattern_coverage": {"KQA": ks, "MultiModalQA": ms},
        "files": {key: str(path) for key, path in paths.items()},
        "stage_order": [
            "compose_macro_tasks",
            "derive_reference_question_from_macro_work",
            "optional_question_author_critic_revision",
            "construct_per_task_synthetic_evidence",
            "derive_clean_feedback_and_summary_trace",
            "verify_terminal_evidence_coverage",
        ],
        "question_review_scope": (
            "independent_llm_critic_plus_high_precision_surface_checks_"
            "and_frozen_macro_graph_fingerprints_not_formal_equivalence"
        ),
        "validation_scope": (
            "typed_macro_handoffs_and_deterministic_clean_evidence_replay"
        ),
        "limitations": [
            "textual_simulated_source_surrogates_not_actual_media",
            "no_external_world_fact_verification",
            "no_final_gold_answer_or_answer_supervision",
            "no_conflicts_missing_evidence_or_task_recovery",
            "LLM_semantic_review_is_a_judgment_not_formal_proof",
        ],
    }
    _write_outputs(paths, scenes, summary)
    return summary


def naturalize(args):
    """Read already compiled scenes; NEVER rebuild or modify their DAG/Evidence."""
    if not args.scenes.is_file():
        raise FileNotFoundError(args.scenes)
    if args.max_records is not None and args.max_records < 1:
        raise ValueError("--max-records must be positive")
    if args.output_dir.resolve() == args.scenes.parent.resolve():
        raise ValueError(
            "naturalize output directory must differ from input directory"
        )
    paths = _paths(args.output_dir, include_audit=True)
    _check_output(paths, args.overwrite)
    scenes = []
    with args.scenes.open("r", encoding="utf-8") as reader:
        for raw in reader:
            if not raw.strip():
                continue
            scene = json.loads(raw)
            issues = validate_clean_scene(scene)
            if issues:
                raise ValueError(
                    f"Invalid input scene {scene.get('task_id')}: {issues}"
                )
            if "question_authoring" in scene:
                raise ValueError(
                    "Input already naturalized. Use the original V4 bank, "
                    "not an accepted/rejected LLM revision bank."
                )
            scenes.append(scene)
            if args.max_records is not None and len(scenes) >= args.max_records:
                break
    if not scenes:
        raise ValueError("no valid scenes loaded")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    question_stats = _naturalize_if_requested(scenes, args)
    summary = {
        "status": "done",
        "pipeline": "v4_question_only_author_critic_revision",
        "source_scenes": str(args.scenes),
        "requested": len(scenes),
        "statistics": {
            "generated": len(scenes),
            "unique_question_texts": len({s["question"] for s in scenes}),
            "question_authoring": question_stats,
            "macro_graphs_preserved": True,
            "evidence_graphs_preserved": True,
            "feedback_bank_preserved": True,
            "fixed_max_concurrency": 10,
        },
        "files": {key: str(path) for key, path in paths.items()},
        "validation_scope": (
            "same_immutable_task_graph_and_clean_evidence_replay_"
            "with_independently_reviewed_question_surface"
        ),
        "limitations": [
            "critic_pass_is_not_formal_semantic_equivalence",
            "no_real_media_perception",
            "no_evidence_conflicts_or_replanning",
        ],
    }
    _write_outputs(paths, scenes, summary)
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


def _question_flags(parser):
    parser.add_argument("--question-api-base", default=None,
                        help="Local OpenAI-compatible endpoint, e.g. http://127.0.0.1:8807/v1")
    parser.add_argument("--question-model", default=None,
                        help="LLM served model name (default: off/deterministic)")
    parser.add_argument("--question-critic-api-base", default=None,
                        help="Optional second endpoint; defaults to the author endpoint")
    parser.add_argument("--question-critic-model", default=None,
                        help="Optional Critic model; defaults to author model")
    parser.add_argument("--question-api-key-env", default="OPENAI_API_KEY",
                        help="Environment variable name, not the key itself")
    parser.add_argument("--question-critic-api-key-env", default=None)
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--question-workers", type=int, default=4)
    parser.add_argument("--question-temperature", type=float, default=0.55)
    parser.add_argument("--question-critic-temperature", type=float, default=0.0)
    parser.add_argument("--question-max-tokens", type=int, default=1400)
    parser.add_argument("--question-critic-max-tokens", type=int, default=2000)
    parser.add_argument("--question-timeout", type=int, default=180)
    parser.add_argument("--question-no-json-mode", action="store_true")


def parser():
    p = argparse.ArgumentParser(
        description="MAESTRO V4 Clean Task-first with optional Author/Critic Question naturalization"
    )
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--kqa", type=Path,
                   default=Path("data/world-source/KQA-Pro/train.json"))
    b.add_argument("--mmqa", type=Path,
                   default=Path("data/world-source/MultiModalQA/MMQA_train.jsonl.gz"))
    b.add_argument("--output-dir", type=Path,
                   default=Path("envir/data/macro_v4"))
    b.add_argument("--target", type=int, default=100)
    b.add_argument("--seed", type=int, default=17)
    b.add_argument("--max-kqa-records", type=int, default=10000)
    b.add_argument("--max-mmqa-records", type=int, default=3000)
    b.add_argument("--overwrite", action="store_true")
    _question_flags(b)
    n = sub.add_parser(
        "naturalize",
        help="Refine existing V4 questions only; NEVER regenerate TaskGraph or Evidence",
    )
    n.add_argument("--scenes", type=Path, required=True)
    n.add_argument("--output-dir", type=Path, required=True)
    n.add_argument("--max-records", type=int, default=None)
    n.add_argument("--overwrite", action="store_true")
    _question_flags(n)
    r = sub.add_parser("replay")
    r.add_argument("--scenes", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    result = {
        "build": build,
        "naturalize": naturalize,
        "replay": replay,
    }[args.command](args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"done", "valid"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
