"""CLI: compile existing MacroPlans into semantic-first synthetic Worlds.

The input scenes are only a portable way of loading an already compiled
TaskGraph and deterministic seed: old World facts/feedback never enter
Semantic World LLM inputs and are not reused for accepted scenes.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from .semantic_world import (
    fingerprint, realize_spec_one, compile_semantic_world_scene,
)
from .validation import (
    validate_clean_scene, validate_original_macroplan_input,
)
from .evidence import run_clean_tasks


def _write_json(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_rows(path: Path) -> list[dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for i, row in enumerate(f, start=1):
            if row.strip():
                try:
                    data.append(json.loads(row))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL {path}:{i}; use new directory or --overwrite"
                    ) from exc
    return data


def _run_signature(args, scenes: list[dict]) -> str:
    from . import (
        semantic_world, world, evidence, relational_world,
        validation, goal_invariants, semantic_world_cli,
    )
    return fingerprint({
        "input_sha256": hashlib.sha256(args.scenes.read_bytes()).hexdigest(),
        "selected_tasks": [s["task_id"] for s in scenes],
        "author": [
            args.grounding_api_base, args.grounding_model,
            args.grounding_temperature, args.grounding_max_tokens,
        ],
        "critic": [
            args.grounding_critic_api_base or args.grounding_api_base,
            args.grounding_critic_model or args.grounding_model,
            args.grounding_critic_temperature, args.grounding_critic_max_tokens,
        ],
        "max_revisions": args.max_revisions,
        "implementation": {
            "semantic_world": hashlib.sha256(
                Path(semantic_world.__file__).read_bytes()).hexdigest(),
            "world": hashlib.sha256(
                Path(world.__file__).read_bytes()).hexdigest(),
            "evidence": hashlib.sha256(
                Path(evidence.__file__).read_bytes()).hexdigest(),
            "relational_world": hashlib.sha256(
                Path(relational_world.__file__).read_bytes()).hexdigest(),
            "validation": hashlib.sha256(
                Path(validation.__file__).read_bytes()).hexdigest(),
            "goal_invariants": hashlib.sha256(
                Path(goal_invariants.__file__).read_bytes()).hexdigest(),
            "cli": hashlib.sha256(
                Path(semantic_world_cli.__file__).read_bytes()).hexdigest(),
        },
    })


def _clients(args):
    """Two independent OpenAI-compatible schema-generation clients."""
    from environment.question_generation.backend import OpenAICompatibleBackend
    author_key = os.environ.get(args.grounding_api_key_env)
    critic_key = (
        os.environ.get(args.grounding_critic_api_key_env)
        if args.grounding_critic_api_key_env else author_key
    )
    def build(url, model, temperature, tokens, key):
        return OpenAICompatibleBackend(
            base_url=url, model=model, api_key=key,
            temperature=temperature, max_tokens=tokens,
            timeout=args.grounding_timeout,
            json_mode=not args.grounding_no_json_mode,
        )
    return (
        build(args.grounding_api_base, args.grounding_model,
              args.grounding_temperature, args.grounding_max_tokens,
              author_key),
        build(args.grounding_critic_api_base or args.grounding_api_base,
              args.grounding_critic_model or args.grounding_model,
              args.grounding_critic_temperature,
              args.grounding_critic_max_tokens, critic_key),
    )


def realize_world(args):
    if not args.scenes.is_file():
        raise FileNotFoundError(args.scenes)
    if not args.grounding_api_base or not args.grounding_model:
        raise ValueError("--world-api-base and --world-model are required")
    if args.output_dir.resolve() == args.scenes.parent.resolve():
        raise ValueError("semantic-world output must differ from input directory")
    if (args.max_records is not None and args.max_records < 1
            or not 0 <= args.max_revisions <= 8
            or args.grounding_workers < 1 or args.flush_every < 1):
        raise ValueError("invalid record count, workers or revision parameters")
    input_scenes = []
    with args.scenes.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            scene = json.loads(line)
            if any(key in scene for key in (
                    "scene_grounding", "semantic_world_spec",
                    "question_authoring")):
                raise ValueError(
                    "realize-world expects original pre-grounding TaskGraph "
                    "scenes; do not input a naturalized/grounded bank")
            # Historical pre-semantic plans carry evidence payloads
            # generated by the pre-relational executor. Those exact
            # DERIVED payloads are not the source of truth for a new run.
            # Preserve structural/source/closure checks, but re-execute
            # Feedback and EvidenceGraphs using the current interpreter.
            errors = validate_original_macroplan_input(scene)
            if errors:
                raise ValueError(
                    f"invalid input {scene.get('task_id')}: {errors}")
            input_scenes.append(scene)
            if (args.max_records is not None
                    and len(input_scenes) >= args.max_records):
                break
    if not input_scenes:
        raise ValueError("no original compiled MacroPlans in input")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    private = output / "scenes.jsonl"
    public = output / "questions.jsonl"
    audit = output / "semantic_world_audit.jsonl"
    accepted_file = output / "accepted_scenes.jsonl"
    checkpoint = output / "semantic_world_checkpoint.json"
    summary_path = output / "summary.json"
    paths = (private, public, audit, accepted_file, checkpoint, summary_path)
    if args.overwrite:
        for path in paths:
            if path.exists():
                path.unlink()
    signature = _run_signature(args, input_scenes)
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise ValueError(
                "semantic-world checkpoint mismatch; use new directory")
        if not all(p.exists() for p in (private, public, audit)):
            raise ValueError("missing persisted output during semantic-world resume")
        prior = _read_rows(private)
        prior_public = _read_rows(public)
        prior_audit = _read_rows(audit)
        if not (len(prior) == len(prior_public) == len(prior_audit)
                == state.get("completed")):
            raise ValueError("semantic-world checkpoint/output prefix mismatch")
        for index, scene in enumerate(prior):
            if scene["task_id"] != input_scenes[index]["task_id"]:
                raise ValueError("semantic-world resume task-order mismatch")
            if scene["public"] != prior_public[index]:
                raise ValueError("semantic-world public projection drift")
            if validate_clean_scene(scene):
                raise ValueError("invalid committed semantic-world scene")
        start = len(prior)
    else:
        if any(p.exists() for p in paths):
            raise FileExistsError(
                "semantic-world output exists; use --overwrite or new directory")
        for p in (private, public, audit):
            p.touch()
        state = {
            "schema": "maestro.semantic_world_stream.v1",
            "signature": signature, "completed": 0,
        }
        _write_json(checkpoint, state)
        start, prior_audit = 0, []
    counts = Counter()
    for row in prior_audit:
        counts[row["status"]] += 1
        counts["author_calls"] += row["author_calls"]
        counts["critic_calls"] += row["critic_calls"]
        counts["revised"] += int(
            row["status"] == "accepted" and row["revision_rounds"] > 0)
        if row["status"] != "accepted":
            counts["reason:" + row["reason"]] += 1
        for attempt in row["attempts"]:
            for problem in set(attempt.get("issues", [])):
                counts["issue:" + problem] += 1
    author, critic = _clients(args)

    def _reexecute_original_world(original):
        """Normalize a fallback against the current EvidenceGraph schema.

        The MacroPlan, synthetic source facts, source grants and Question
        remain frozen; old derived feedback is discarded rather than
        silently treated as if it passed the new operation interpreter.
        """
        return {
            **original,
            **run_clean_tasks(original, original["world"]),
        }

    def process(original):
        spec, review = realize_spec_one(
            original, author=author, critic=critic,
            max_revisions=args.max_revisions)
        if spec is not None:
            try:
                result = compile_semantic_world_scene(original, spec)
            except (ValueError, KeyError, TypeError) as exc:
                review = {
                    **review,
                    "status": "rejected",
                    "reason": "semantic_world_reinstantiation_failed",
                    "recompile_error": str(exc)[:400],
                }
                result = _reexecute_original_world(original)
        else:
            result = _reexecute_original_world(original)
        errors = validate_clean_scene(result)
        if errors:
            raise ValueError(
                f"invalid resulting scene {original['task_id']}: {errors}")
        return result, review

    with (
        private.open("a", encoding="utf-8") as p,
        public.open("a", encoding="utf-8") as q,
        audit.open("a", encoding="utf-8") as a,
        ThreadPoolExecutor(max_workers=args.grounding_workers) as executor,
    ):
        progress = tqdm(total=len(input_scenes), initial=start,
                        desc="Semantic-first World", unit="scene",
                        disable=args.no_progress, dynamic_ncols=True)
        try:
            for index, (scene, review) in enumerate(
                executor.map(process, input_scenes[start:]), start
            ):
                row = {
                    "task_id": scene["task_id"],
                    "status": review["status"],
                    "reason": review["reason"],
                    "revision_rounds": review["revision_rounds"],
                    "author_calls": review["author_calls"],
                    "critic_calls": review["critic_calls"],
                    "attempts": review["attempts"],
                    **({"recompile_error": review["recompile_error"]}
                       if review.get("recompile_error") else {}),
                }
                for f, data in (
                    (p, scene), (q, scene["public"]), (a, row)
                ):
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                counts[review["status"]] += 1
                counts["author_calls"] += review["author_calls"]
                counts["critic_calls"] += review["critic_calls"]
                counts["revised"] += int(
                    review["status"] == "accepted" and
                    review["revision_rounds"] > 0)
                if review["status"] != "accepted":
                    counts["reason:" + review["reason"]] += 1
                for attempt in review["attempts"]:
                    for problem in set(attempt.get("issues", [])):
                        counts["issue:" + problem] += 1
                progress.update(1)
                if ((index+1) % args.flush_every == 0
                        or index+1 == len(input_scenes)):
                    for f in (p, q, a):
                        f.flush()
                        os.fsync(f.fileno())
                    state["completed"] = index + 1
                    _write_json(checkpoint, state)
        finally:
            progress.close()
    committed = _read_rows(private)
    # Machine-friendly subset for later downstream task/trace training.
    with accepted_file.open("w", encoding="utf-8") as f:
        for scene in committed:
            if "semantic_world_spec" in scene:
                f.write(json.dumps(scene, ensure_ascii=False) + "\n")
    summary = {
        "status": "done",
        "pipeline": "semantic_spec_before_deterministic_world_and_evidence_v1",
        "source_scenes": str(args.scenes),
        "requested": len(input_scenes),
        "statistics": {
            "generated": len(committed),
            "accepted": counts["accepted"],
            "rejected": counts["rejected"],
            "not_realizable": counts["not_realizable"],
            "revised": counts["revised"],
            "author_calls": counts["author_calls"],
            "critic_calls": counts["critic_calls"],
            "rejection_reasons": {
                k[7:]: v for k, v in sorted(counts.items())
                if k.startswith("reason:")
            },
            "validation_issue_counts": {
                k[6:]: v for k, v in sorted(counts.items())
                if k.startswith("issue:")
            },
            "macro_graphs_preserved": True,
            "accepted_worlds_and_evidence_rebuilt": True,
            "old_world_facts_not_used_for_new_values": True,
            "legacy_input_validation": (
                "structural_source_provenance_and_evidence_closure_"
                "excluding_old_derived_executor_format"),
            "fallback_world_facts_unchanged_feedback_reexecuted": True,
            "original_mmqa_question_or_excerpts_given_to_llm": False,
            "real_media_perception_performed": False,
        },
        "files": {
            "private": str(private), "public": str(public),
            "accepted_only": str(accepted_file), "audit": str(audit),
            "checkpoint": str(checkpoint), "summary": str(summary_path),
        },
        "next_stage": (
            "python -m envir naturalize --scenes " + str(private) +
            " --output-dir NEW_DIRECTORY ..."
        ),
        "limitations": [
            "fictional_semantic_world_not_original_mmqa_source_truth",
            "llm_critic_not_formal_semantic_proof",
            "fallback_scenes_keep_original_world_facts_but_reexecute_evidence_with_current_schema",
            "no_new_macro_recipe_or_world_operation_semantics",
            "no_conflicts_missing_evidence_or_recovery",
        ],
    }
    _write_json(summary_path, summary)
    return summary
