"""MAESTRO Task-First dataset generation and optional LLM Question review.

Deterministic pilot:
  python -m envir.macro.cli build --kqa ... --mmqa ... --target 100

Naturalize an EXISTING immutable Task/Evidence bank (no KQA/MMQA reload):
  python -m envir.macro.cli naturalize \
    --scenes envir/data/macro_v4/scenes.jsonl \
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

from .priors import load_kqa
from .scene_catalog import load_scenario_seeds, enrich_with_yago
from .source_contexts import load_supporting_contexts
from .goal_grammar import goal_bank_manifest
from .recipes import recipe_manifest, RECIPES_BY_OUTPUT
from .builder import build_batch
from .question_streaming import stream_naturalize_scenes
from .validation import validate_clean_scene
from .question_audit import audit_file


def _json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonl(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as writer:
        for entry in entries:
            writer.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _paths(output: Path, *, include_audit: bool):
    paths = {
        "private": output / "scenes.jsonl",
        "public": output / "questions.jsonl",
        "summary": output / "summary.json",
    }
    if include_audit:
        paths["question_review_audit"] = output / "question_audit.jsonl"
    return paths


def _check_output(paths: dict, overwrite: bool):
    if not overwrite and any(p.exists() for p in paths.values()):
        raise FileExistsError("Task-First output exists; use a new directory or --overwrite")


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


def _author_critic_clients(args):
    if not args.question_model or not args.question_api_base:
        raise ValueError("both --question-model and --question-api-base are required")
    if (not 0 <= args.max_revisions <= 8 or args.question_workers < 1
            or args.flush_every < 1):
        raise ValueError(
            "--max-revisions must be 0..8; --question-workers and --flush-every must be positive"
        )
    from environment.question_generation.backend import OpenAICompatibleBackend
    key = (
        os.environ.get(args.question_api_key_env)
        if args.question_api_key_env else None
    )
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
    return author, critic


def _stream_questions(scenes, args):
    author, critic = _author_critic_clients(args)
    return stream_naturalize_scenes(
        scenes, args=args, output_dir=args.output_dir,
        author=author, critic=critic, overwrite=args.overwrite,
    )


def build(args):
    if min(args.target, args.max_kqa_records, args.max_mmqa_records) < 1:
        raise ValueError("target and data limits must be positive")
    for path in (args.kqa, args.mmqa):
        if not path.is_file():
            raise FileNotFoundError(path)
    llm_mode = _llm_requested(args)
    paths = _paths(args.output_dir, include_audit=llm_mode)
    if not llm_mode:
        _check_output(paths, args.overwrite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kqa, ks = load_kqa(args.kqa, limit=args.max_kqa_records)
    scenario_seeds, ms = load_scenario_seeds(
        args.mmqa, limit=args.max_mmqa_records)
    yago_stats = enrich_with_yago(
        scenario_seeds, args.yago, max_triples=args.max_yago_triples)
    # Offline preparation: materialize an independently reusable Scene Bank
    # and a curated Goal/Pattern Bank BEFORE selecting any question.
    # Original MMQA questions are intentionally excluded from the scene bank.
    scene_bank_file = args.output_dir / "scene_bank.jsonl"
    pattern_bank_file = args.output_dir / "goal_bank.json"
    recipe_bank_file = args.output_dir / "recipe_bank.json"
    _jsonl(scene_bank_file, [{
        "scene_id": row["record_id"],
        "topic": row["topic"],
        "topic_provenance": row.get("scene_topic_policy"),
        "domain_hint": row["domain_hint"],
        "entities": row["entities"],
        "source_modalities": row["source_modalities"],
        "supporting_sources": row["supporting_sources"],
        "source_structure_hint": row["composition"],
        "source_structure_type": row["question_type"],
        "yago_matches": row.get("yago_matches", []),
    } for row in scenario_seeds])
    pattern_bank = goal_bank_manifest()
    _json(pattern_bank_file, {
        "policy": "independent_curated_workproduct_grammar",
        "patterns": pattern_bank,
        "total_patterns": len(pattern_bank),
        "combination_policy": "seeded_random_scene_pattern_pairing_with_source_compatibility",
    })
    _json(recipe_bank_file, {
        "policy": "recursive_backwards_typed_workproduct_grammar",
        "recipes": recipe_manifest(),
        "producer_alternatives": {
            key: [recipe.id for recipe in alternatives]
            for key, alternatives in RECIPES_BY_OUTPUT.items()
        },
    })
    context_dir = args.mmqa_context_dir or args.mmqa.parent
    contexts, context_stats = (
        ({}, {"policy": "disabled", "available_context_excerpts": 0})
        if args.no_mmqa_context else
        load_supporting_contexts(
            scenario_seeds, context_dir, excerpt_chars=args.mmqa_excerpt_chars)
    )
    scenes, stats = build_batch(
        kqa=kqa, scenario_seeds=scenario_seeds,
        target=args.target, seed=args.seed, contexts=contexts,
    )
    # Only selected, successfully compiled source records are exported.
    manifest = args.output_dir / "source_manifest.jsonl"
    _jsonl(manifest, [{
        "task_id": sc["task_id"],
        "mmqa_record_id": sc["source_priors"]["MultiModalQA"]["record_id"],
        "question": sc["question"],
        "original_mmqa_question_audit_only": sc["source_priors"]["MultiModalQA"]["original_question"],
        "domain_hint": sc["scenario"]["domain"],
        "selected_goal_pattern": sc["goal"]["intent"],
        "composition_type": sc["source_priors"]["MultiModalQA"]["official_question_type"],
        "supporting_sources": sc["source_priors"]["MultiModalQA"]["supporting_sources"],
        "yago_matches": sc["source_priors"]["YAGO"]["matches"],
    } for sc in scenes])
    if llm_mode:
        # Writes private/public/audit one record at a time, with a checkpoint.
        review_stats, unique_questions = _stream_questions(scenes, args)
        stats["unique_question_texts"] = unique_questions
    else:
        review_stats = {
            "mode": "deterministic", "accepted": 0, "fallback": 0,
            "accepted_after_revision": 0, "author_calls": 0,
            "critic_calls": 0, "model_review_not_formal_proof": True,
        }
    stats["question_authoring"] = review_stats
    summary = {
        "status": "done",
        "requested": args.target,
        "statistics": stats,
        "original_pattern_coverage": {"KQA": ks, "MultiModalQA": ms},
        "yago_enrichment": yago_stats,
        "mmqa_context_enrichment": context_stats,
        "scenario_seed_manifest": str(manifest),
        "scene_bank": str(scene_bank_file),
        "goal_pattern_bank": str(pattern_bank_file),
        "recipe_bank": str(recipe_bank_file),
        "scene_pattern_bank": {
            "scene_count": len(scenario_seeds),
            "pattern_count": len(pattern_bank),
            "cross_product_candidate_count": len(scenario_seeds) * len(pattern_bank),
            "policy": "random_independent_pairing_then_macro_graph_construction",
        },
        "files": {key: str(path) for key, path in paths.items()},
        "stage_order": [
            "extract_scene_bank_from_mmqa_metadata",
            "enrich_exact_yago_entities_when_available",
            "prepare_independent_goal_and_problem_pattern_bank",
            "prepare_typed_workproduct_recipe_bank",
            "randomly_combine_compatible_scenes_with_work_goals",
            "backchain_recursive_workproduct_recipes_from_final_goal",
            "bind_all_supporting_sources_and_optional_original_context",
            "derive_new_question_from_terminal_workproduct_blueprint",
            "optional_question_author_critic_revision",
            "execute_per_task_synthetic_typed_operations",
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
    if llm_mode:
        # The streamed JSONLs are already committed; finish with summary.
        _json(paths["summary"], summary)
    else:
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
    if not _llm_requested(args):
        raise ValueError("naturalize requires --question-model and --question-api-base")
    paths = _paths(args.output_dir, include_audit=True)
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
    question_stats, unique_questions = _stream_questions(scenes, args)
    summary = {
        "status": "done",
        "pipeline": "taskgraph_question_only_author_critic_revision",
        "source_scenes": str(args.scenes),
        "requested": len(scenes),
        "statistics": {
            "generated": len(scenes),
            "unique_question_texts": unique_questions,
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


def audit_questions(args):
    """Read-only diagnostics; no LLM calls or scene modifications."""
    summary = audit_file(
        args.scenes, max_records=args.max_records,
        max_examples=args.max_examples,
    )
    if args.output is not None:
        _json(args.output, summary)
    return {"status": "done", **summary}


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
    parser.add_argument("--flush-every", type=int, default=1,
                        help="Checkpoint after each N questions (default: 1)")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable tqdm progress display")


def parser():
    p = argparse.ArgumentParser(
        description="MAESTRO recursive Work-Product Task-First generator"
    )
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--kqa", type=Path,
                   default=Path("data/world-source/KQA-Pro/train.json"))
    b.add_argument("--mmqa", type=Path,
                   default=Path("data/world-source/MultiModalQA/MMQA_train.jsonl.gz"))
    b.add_argument("--output-dir", type=Path,
                   default=Path("envir/data/taskfirst"))
    b.add_argument("--target", type=int, default=100)
    b.add_argument("--yago", type=Path, default=None,
                   help="Optional YAGO-4.6 TTL for bounded exact entity enrichment")
    b.add_argument("--max-yago-triples", type=int, default=500000)
    b.add_argument("--seed", type=int, default=17)
    b.add_argument("--max-kqa-records", type=int, default=10000)
    b.add_argument("--max-mmqa-records", type=int, default=30000)
    b.add_argument("--mmqa-context-dir", type=Path, default=None,
                   help="Existing MMQA bank directory (defaults to --mmqa file directory)")
    b.add_argument("--no-mmqa-context", action="store_true",
                   help="Skip optional lookup of original MMQA text/table/caption banks")
    b.add_argument("--mmqa-excerpt-chars", type=int, default=1200,
                   help="Maximum excerpt characters per original supporting source")
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
    a = sub.add_parser("audit-questions",
        help="Read-only fidelity/style audit of existing naturalized V4 Questions")
    a.add_argument("--scenes", type=Path, required=True)
    a.add_argument("--max-records", type=int, default=None)
    a.add_argument("--max-examples", type=int, default=12)
    a.add_argument("--output", type=Path, default=None)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    result = {
        "build": build,
        "naturalize": naturalize,
        "replay": replay,
        "audit-questions": audit_questions,
    }[args.command](args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"done", "valid"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
