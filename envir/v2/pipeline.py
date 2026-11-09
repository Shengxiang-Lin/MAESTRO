"""End-to-end MAESTRO Reasoning-First V2 generation, audit and replay CLI.

Maintains a separate schema/output tree; legacy envir build/realize-world,
naturalize and replay remain byte-for-byte backward compatible.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from envir.schema import normalize_label
from .spec import difficulty_sequence, make_spec
from .world import initial_public_view
from .question import (
    guard_question, naturalize_one, reference_question,
)
from .quality import compile_and_validate, validate_replay


def _json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    temp.replace(path)


def _jsonl(path: Path, rows) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as writer:
        for value in rows:
            writer.write(json.dumps(value, ensure_ascii=False) + "\n")
    temp.replace(path)


def read_jsonl(path: Path, *, limit: int | None = None) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    entries = []
    with path.open("r", encoding="utf-8") as reader:
        for lineno, line in enumerate(reader, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL at {path}:{lineno}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"non_object_JSONL at {path}:{lineno}")
                entries.append(row)
                if limit is not None and len(entries) >= limit:
                    break
    return entries


def _seeds_from_scenes(path: Path, max_records: int) -> list[dict]:
    extracted = []
    for row in read_jsonl(path, limit=max_records):
        if "problem_spec" in row:
            raise ValueError(
                "build-v2 needs original TaskGraph/Scene Bank, not an existing V2 bank"
            )
        scenario = row.get("scenario", {})
        priors = row.get("source_priors", {}).get("MultiModalQA", {})
        topic = (scenario.get("topic") or row.get("topic"))
        if not topic:
            continue
        extracted.append({
            "record_id": (priors.get("record_id") or row.get("scene_id")
                          or row.get("task_id") or str(len(extracted))),
            "topic": topic,
            "domain_hint": scenario.get("domain", row.get("domain_hint", "general")),
            "source_modalities": priors.get(
                "source_modalities", row.get("source_modalities", [])),
            "supporting_sources": priors.get(
                "supporting_sources", row.get("supporting_sources", [])),
        })
    return extracted


def _parse_mix(text: str) -> tuple[int, int, int]:
    try:
        pieces = tuple(int(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise ValueError("difficulty mix must be easy,medium,hard percentages") from exc
    if len(pieces) != 3 or any(x < 0 for x in pieces) or sum(pieces) != 100:
        raise ValueError("difficulty mix must be three percentages totalling 100")
    return pieces


def _out_files(output_dir: Path) -> dict:
    return {
        "private": output_dir / "scenes.jsonl",
        "public": output_dir / "questions.jsonl",
        "audit": output_dir / "difficulty_audit.jsonl",
        "summary": output_dir / "summary.json",
    }


def build_v2(args) -> dict:
    if args.target < 1 or args.max_records < 1:
        raise ValueError("target and max-records must be positive")
    if bool(args.scenes) == bool(args.mmqa):
        raise ValueError("supply exactly one of --scenes or --mmqa")
    if args.scenes:
        seeds = _seeds_from_scenes(args.scenes, args.max_records)
        seed_source = str(args.scenes)
    else:
        from envir.macro.scene_catalog import load_scenario_seeds
        seeds, _ = load_scenario_seeds(args.mmqa, limit=args.max_records)
        seed_source = str(args.mmqa)
    if len(seeds) < args.target:
        raise ValueError(
            f"not enough scene/topic seeds: {len(seeds)} for target {args.target}"
        )
    files = _out_files(args.output_dir)
    if not args.overwrite and any(p.exists() for p in files.values()):
        raise FileExistsError("V2 destination already exists; use --overwrite")
    sequence = difficulty_sequence(
        target=args.target, mix=_parse_mix(args.difficulty_mix),
        run_seed=args.seed,
    )
    order = list(seeds)
    random.Random(args.seed * 123457 + 137).shuffle(order)
    generated, rejected, seen_questions = [], Counter(), set()
    for index, seed_record in enumerate(order):
        if len(generated) >= args.target:
            break
        try:
            ordinal = len(generated)
            difficulty = sequence[ordinal]
            spec = make_spec(
                seed_record=seed_record, ordinal=ordinal,
                run_seed=args.seed, difficulty=difficulty,
            )
            question = reference_question(spec)
            scene = compile_and_validate(spec, question)
            normalized = normalize_label(scene["question"])
            if normalized in seen_questions:
                rejected["duplicate_question"] += 1
                continue
            seen_questions.add(normalized)
            generated.append(scene)
        except (ValueError, TypeError, KeyError) as exc:
            rejected[type(exc).__name__ + ":" + str(exc)[:140]] += 1
    if len(generated) != args.target:
        raise ValueError(
            f"only {len(generated)}/{args.target} V2 scenes passed validation; "
            f"rejections={dict(rejected)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _jsonl(files["private"], generated)
    _jsonl(files["public"], [scene["public"] for scene in generated])
    _jsonl(files["audit"], [{
        "task_id": scene["task_id"],
        **scene["difficulty_audit"],
    } for scene in generated])
    summary = {
        "status": "done",
        "pipeline": "reasoning_program_before_world_before_macrograph_before_question_v2",
        "seed_source": seed_source,
        "target": args.target,
        "generated": len(generated),
        "rejected": dict(rejected),
        "difficulty_target_distribution": dict(
            Counter(s["problem_spec"]["difficulty_target"] for s in generated)
        ),
        "goal_family_distribution": dict(
            Counter(s["problem_spec"]["family"] for s in generated)
        ),
        "modality_roles": dict(Counter(
            source["modality"]
            for scene in generated
            for source in scene["problem_spec"]["sources"]
        )),
        "mean_reasoning_operator_depth": round(sum(
            s["difficulty_audit"]["reasoning"]["necessary_structural_operator_depth"]
            for s in generated
        ) / len(generated), 3),
        "mean_macro_task_count": round(sum(
            s["macro_graph"]["macro_task_count"] for s in generated
        ) / len(generated), 3),
        "counterfactual_full_pass": sum(
            all(x["changing_source_changes_answer"]
                for x in scene["difficulty_audit"]["counterfactual_source_tests"].values())
            for scene in generated
        ),
        "one_source_shortcut_successes": sum(
            scene["difficulty_audit"]["one_source_baseline"]["exact_match"]
            for scene in generated
        ),
        "source_observation_policy": (
            "fictional_synthetic_media_annotations_not_original_perception"
        ),
        "llm_question_review": "off_use_naturalize-v2_if_needed",
        "files": {key: str(value) for key, value in files.items()},
    }
    _json(files["summary"], summary)
    return summary


def replay_v2(args) -> dict:
    rows = read_jsonl(args.scenes, limit=getattr(args, "max_records", None))
    failures = []
    for row in rows:
        result = validate_replay(row)
        if result["status"] != "valid":
            failures.append(result)
    return {
        "status": "valid" if rows and not failures else "invalid",
        "checked": len(rows), "failure_count": len(failures),
        "failures": failures[:30],
    }


def audit_v2(args) -> dict:
    rows = read_jsonl(args.scenes, limit=getattr(args, "max_records", None))
    reports = [validate_replay(row) for row in rows]
    passed = [row for row, report in zip(rows, reports)
              if report["status"] == "valid"]
    invalid = [report for report in reports if report["status"] != "valid"]
    return {
        "status": "valid" if rows and not invalid else "invalid",
        "checked": len(rows),
        "replay_pass": len(passed),
        "replay_fail": len(invalid),
        "difficulty_distribution": dict(
            Counter(row["problem_spec"]["difficulty_target"] for row in passed)
        ),
        "goal_families": dict(Counter(
            row["problem_spec"]["family"] for row in passed
        )),
        "mean_reasoning_depth": (
            round(sum(
                row["difficulty_audit"]["reasoning"]["necessary_structural_operator_depth"]
                for row in passed
            ) / len(passed), 3) if passed else None
        ),
        "mean_macro_depth": (
            round(sum(row["macro_graph"]["critical_path_depth"] for row in passed)
                  / len(passed), 3) if passed else None
        ),
        "one_source_shortcut_successes": sum(
            row["difficulty_audit"]["one_source_baseline"]["exact_match"]
            for row in passed
        ),
        "counterfactual_full_pass": sum(
            all(item["changing_source_changes_answer"] for item in
                row["difficulty_audit"]["counterfactual_source_tests"].values())
            for row in passed
        ),
        "first_failures": invalid[:20],
    }


def _llm_clients(args):
    from environment.question_generation.backend import OpenAICompatibleBackend
    if not args.question_api_base or not args.question_model:
        raise ValueError("both question endpoint and model are required")
    key = os.environ.get(args.question_api_key_env)
    return (
        OpenAICompatibleBackend(
            base_url=args.question_api_base, model=args.question_model,
            api_key=key, temperature=0.55, max_tokens=1800,
            timeout=args.question_timeout,
        ),
        OpenAICompatibleBackend(
            base_url=args.question_critic_api_base or args.question_api_base,
            model=args.question_critic_model or args.question_model,
            api_key=key, temperature=0.0, max_tokens=1400,
            timeout=args.question_timeout,
        ),
    )


def _naturalize_entry(scene: dict, *, clients, max_revisions: int) -> dict:
    question, review = naturalize_one(
        scene["problem_spec"], author=clients[0], critic=clients[1],
        max_revisions=max_revisions,
    )
    updated = copy.deepcopy(scene)
    updated["question"] = question
    updated["question_origin"] = (
        "llm_accepted_frozen_spec" if review["status"] == "accepted"
        else "deterministic_frozen_problem_spec"
    )
    updated["question_authoring"] = review
    updated["public"]["question"] = question
    validation = validate_replay(updated)
    if validation["status"] != "valid":
        raise ValueError(
            "question update unexpectedly changed executable semantics: "
            + str(validation["issues"])
        )
    return updated


def naturalize_v2(args) -> dict:
    if args.output_dir.resolve() == args.scenes.parent.resolve():
        raise ValueError("naturalize-v2 requires a separate output directory")
    if args.question_workers < 1 or args.max_revisions < 0 or args.max_revisions > 8:
        raise ValueError("invalid worker/revision settings")
    rows = read_jsonl(args.scenes, limit=args.max_records)
    if not rows:
        raise ValueError("no V2 scenes loaded")
    for scene in rows:
        result = validate_replay(scene)
        if result["status"] != "valid":
            raise ValueError("invalid incoming V2 scene:" + str(result))
        if scene.get("question_authoring") is not None:
            raise ValueError("input already naturalized")
    files = _out_files(args.output_dir)
    if not args.overwrite and any(path.exists() for path in files.values()):
        raise FileExistsError("naturalize-v2 output exists")
    clients = _llm_clients(args)
    def one(scene):
        return _naturalize_entry(
            scene, clients=clients, max_revisions=args.max_revisions
        )
    with ThreadPoolExecutor(max_workers=args.question_workers) as pool:
        updated = list(pool.map(one, rows))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _jsonl(files["private"], updated)
    _jsonl(files["public"], [s["public"] for s in updated])
    _jsonl(files["audit"], [{
        "task_id": s["task_id"],
        "difficulty": s["difficulty_audit"],
        "question_authoring": s["question_authoring"],
    } for s in updated])
    statuses = Counter(s["question_authoring"]["status"] for s in updated)
    summary = {
        "status": "done",
        "pipeline": "frozen_reasoning_program_question_author_critic_v2",
        "source_scenes": str(args.scenes),
        "generated": len(updated),
        "question_results": dict(statuses),
        "world_program_macrograph_unmodified": True,
        "accepted_question_equivalence": "critic_judgment_not_formal_proof",
        "files": {key: str(path) for key, path in files.items()},
    }
    _json(files["summary"], summary)
    return summary
