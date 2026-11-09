"""CLI runner for post-TaskGraph World semantic concretization.

Writes separate scenes.jsonl and grounding_audit.jsonl with deterministic,
ordered, resumable commits. The existing World/Feedback/EvidenceGraph remain
byte-for-byte unchanged. Accepted overlays are validated on replay.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from .scene_grounding import fingerprint
from .scenario_contract import ground_one_contract
from .validation import validate_clean_scene


def _write_json(path: Path, value: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for number, raw in enumerate(stream, 1):
            if raw.strip():
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid interrupted JSONL record {path}:{number}; "
                        "rerun with --overwrite if safe"
                    ) from exc
    return rows


def _clients(args):
    from environment.question_generation.backend import OpenAICompatibleBackend
    key = os.environ.get(args.grounding_api_key_env)
    critic_key = (os.environ.get(args.grounding_critic_api_key_env)
                  if args.grounding_critic_api_key_env else key)
    def build(base, model, token, temperature, api_key):
        return OpenAICompatibleBackend(
            base_url=base, model=model, api_key=api_key,
            temperature=temperature, max_tokens=token,
            timeout=args.grounding_timeout,
            json_mode=not args.grounding_no_json_mode,
        )
    return (
        build(args.grounding_api_base, args.grounding_model,
              args.grounding_max_tokens, args.grounding_temperature, key),
        build(args.grounding_critic_api_base or args.grounding_api_base,
              args.grounding_critic_model or args.grounding_model,
              args.grounding_critic_max_tokens,
              args.grounding_critic_temperature, critic_key),
    )


def _stamp(args, source: Path, task_ids: list[str]) -> str:
    from . import scene_grounding as sg
    from . import scenario_contract as sc
    from . import scene_grounding_cli as sgcli
    return fingerprint({
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "task_ids": task_ids,
        "constructor": [
            args.grounding_api_base, args.grounding_model,
            args.grounding_temperature, args.grounding_max_tokens,
        ],
        "critic": [
            args.grounding_critic_api_base or args.grounding_api_base,
            args.grounding_critic_model or args.grounding_model,
            args.grounding_critic_temperature, args.grounding_critic_max_tokens,
        ],
        "max_revisions": args.max_revisions,
        "code": {
            "scene_grounding": hashlib.sha256(
                Path(sg.__file__).read_bytes()).hexdigest(),
            "scenario_contract": hashlib.sha256(
                Path(sc.__file__).read_bytes()).hexdigest(),
            "runner": hashlib.sha256(
                Path(sgcli.__file__).read_bytes()).hexdigest(),
        },
    })


def ground_scenes(args):
    if not args.scenes.is_file():
        raise FileNotFoundError(args.scenes)
    if not args.grounding_api_base or not args.grounding_model:
        raise ValueError("ground-scenes needs --grounding-api-base and --grounding-model")
    if args.output_dir.resolve() == args.scenes.parent.resolve():
        raise ValueError("ground output directory must differ from input directory")
    if (args.max_records is not None and args.max_records < 1
            or not 0 <= args.max_revisions <= 8
            or args.grounding_workers < 1
            or args.flush_every < 1):
        raise ValueError("invalid grounding count, revision, worker or flush configuration")
    scenes = []
    with args.scenes.open("r", encoding="utf-8") as src:
        for raw in src:
            if not raw.strip():
                continue
            scene = json.loads(raw)
            if "scene_grounding" in scene or "question_authoring" in scene:
                raise ValueError(
                    "ground-scenes expects original compiled scenes, "
                    "not already grounded/naturalized scenes"
                )
            errors = validate_clean_scene(scene)
            if errors:
                raise ValueError("invalid input scene: " + "; ".join(errors))
            scenes.append(scene)
            if args.max_records and len(scenes) >= args.max_records:
                break
    if not scenes:
        raise ValueError("no input scenes")
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    private = folder / "scenes.jsonl"
    public = folder / "questions.jsonl"
    audit = folder / "grounding_audit.jsonl"
    checkpoint = folder / "grounding_checkpoint.json"
    summary_file = folder / "summary.json"
    outputs = [private, public, audit, checkpoint, summary_file]
    if args.overwrite:
        for path in outputs:
            if path.exists():
                path.unlink()
    signature = _stamp(args, args.scenes, [row["task_id"] for row in scenes])
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise ValueError("grounding checkpoint signature mismatch; use new output dir")
        if not all(path.exists() for path in (private, public, audit)):
            raise ValueError("incomplete grounding checkpoint")
        done = _read_jsonl(private)
        previous_audit = _read_jsonl(audit)
        previous_public = _read_jsonl(public)
        if len(done) != len(previous_audit) or len(done) != len(previous_public):
            raise ValueError("grounding output files have different committed prefixes")
        if len(done) != state.get("completed", -1):
            raise ValueError("grounding checkpoint does not match output prefix")
        for index, item in enumerate(done):
            if item["task_id"] != scenes[index]["task_id"]:
                raise ValueError("grounding resume task order mismatch")
            issues = validate_clean_scene(item)
            if issues:
                raise ValueError("invalid committed grounding: " + ",".join(issues))
        start = len(done)
    else:
        if any(path.exists() for path in outputs):
            raise FileExistsError("grounding output exists; use --overwrite or new directory")
        for path in (private, public, audit):
            path.touch()
        state = {"signature": signature, "completed": 0,
                 "schema": "maestro.world_concretization.v1"}
        _write_json(checkpoint, state)
        start = 0
    constructor, critic = _clients(args)
    counts = Counter()
    # Counts are derived from accepted prefix, not mutable partial summary.
    for item in _read_jsonl(audit):
        counts[item["status"]] += 1
        for attempt in item.get("attempts", []):
            for issue in set(attempt.get("validation_errors", [])):
                counts["validation_error:" + issue] += 1
        counts["author_calls"] += item["author_calls"]
        counts["critic_calls"] += item["critic_calls"]
        counts["revised"] += int(
            item["status"] == "accepted" and item["revision_rounds"] > 0)
        if item["status"] != "accepted":
            counts["reason:" + item["reason"]] += 1
    for committed in _read_jsonl(private):
        overlay = committed.get("scene_grounding")
        if overlay is not None:
            ablation = overlay["validation"]["goal_level_source_ablation"]
            ignored = len(ablation["non_influential_sources"])
            counts["goal_proxy_non_influential_sources"] += ignored
            counts["scenes_with_goal_proxy_non_influential_sources"] += int(
                ignored > 0
            )

    def one(scene):
        overlay, review = ground_one_contract(
            scene, constructor=constructor, critic=critic,
            max_revisions=args.max_revisions)
        result = dict(scene)
        if overlay is not None:
            result["scene_grounding"] = overlay
        issues = validate_clean_scene(result)
        if issues:
            raise ValueError("grounded scene changed replay: " + "; ".join(issues))
        return result, review

    with (
        private.open("a", encoding="utf-8") as p,
        public.open("a", encoding="utf-8") as q,
        audit.open("a", encoding="utf-8") as a,
        ThreadPoolExecutor(max_workers=args.grounding_workers) as pool,
    ):
        progress = tqdm(
            total=len(scenes), initial=start, unit="scene",
            desc="World concretization", dynamic_ncols=True,
            disable=args.no_progress,
        )
        try:
            # map yields ordered results although model requests run in parallel.
            for index, (result, review) in enumerate(
                pool.map(one, scenes[start:]), start
            ):
                row = {
                    "task_id": result["task_id"], "status": review["status"],
                    "reason": review["reason"], "revision_rounds":
                    review["revision_rounds"],
                    "author_calls": review["author_calls"],
                    "critic_calls": review["critic_calls"],
                    "attempts": review["attempts"],
                    "last_validation_errors": review.get(
                        "last_validation_errors", []),
                }
                for handle, data in (
                    (p, result), (q, result["public"]), (a, row)
                ):
                    handle.write(json.dumps(data, ensure_ascii=False) + "\n")
                counts[review["status"]] += 1
                for attempt in review.get("attempts", []):
                    for issue in set(attempt.get("validation_errors", [])):
                        counts["validation_error:" + issue] += 1
                counts["author_calls"] += review["author_calls"]
                counts["critic_calls"] += review["critic_calls"]
                counts["revised"] += int(
                    review["status"] == "accepted" and
                    review["revision_rounds"] > 0)
                if review["status"] != "accepted":
                    counts["reason:" + review["reason"]] += 1
                else:
                    ablation = result["scene_grounding"]["validation"][
                        "goal_level_source_ablation"]
                    ignored = len(ablation["non_influential_sources"])
                    counts["goal_proxy_non_influential_sources"] += ignored
                    counts["scenes_with_goal_proxy_non_influential_sources"] += int(
                        ignored > 0
                    )
                progress.update(1)
                # Flush and fsync BEFORE publishing checkpoint, so a resumed
                # process never assumes uncommitted output bytes are durable.
                if ((index + 1) % args.flush_every == 0
                        or index + 1 == len(scenes)):
                    for handle in (p, q, a):
                        handle.flush()
                        os.fsync(handle.fileno())
                    state["completed"] = index + 1
                    _write_json(checkpoint, state)
        finally:
            progress.close()
    result = {
        "status": "done",
        "pipeline": "post_macro_taskgraph_semantic_scenario_contract_v1",
        "requested": len(scenes),
        "statistics": {
            "accepted": counts["accepted"],
            "rejected": counts["rejected"],
            "not_realizable": counts["not_realizable"],
            "revised": counts["revised"],
            "author_calls": counts["author_calls"],
            "critic_calls": counts["critic_calls"],
            "rejection_reasons": {
                key[7:]: count for key, count in sorted(counts.items())
                if key.startswith("reason:")
            },
            "validation_error_counts": {
                key[len("validation_error:"):]: count
                for key, count in sorted(counts.items())
                if key.startswith("validation_error:")
            },
            "frozen_world_and_graph_preserved": True,
            "original_mmqa_question_in_constructor": False,
            "scenario_contract_version": "semantic_scenario_contract_v1",
            "source_fact_binding": "exact_source_id_fact_id_values",
            "critic_not_formal_semantic_proof": True,
            "goal_proxy_scenes_with_non_influential_sources": counts[
                "scenes_with_goal_proxy_non_influential_sources"],
            "goal_proxy_total_non_influential_sources": counts[
                "goal_proxy_non_influential_sources"],
        },
        "files": {
            "private": str(private), "public": str(public),
            "grounding_audit": str(audit), "checkpoint": str(checkpoint),
            "summary": str(summary_file),
        },
        "next_stage": (
            "python -m envir naturalize --scenes "
            + str(private) + " --output-dir NEW_DIRECTORY ..."
        ),
        "limitations": [
            "hypothetical_synthetic_world_not_original_mmqa_ground_truth",
            "not_realizable_scenes_fallback_to_abstract_original",
            "original_media_not_perceived",
            "does_not_repair_missing_operation_semantics",
            "critic_judgment_not_formal_proof",
        ],
    }
    _write_json(summary_file, result)
    return result
