"""Source-scoped reasoning bank construction, audit and replay implementation.

Maintains a separate schema/output tree; legacy envir build/realize-world,
naturalize and replay remain byte-for-byte backward compatible.
"""
from __future__ import annotations

import copy
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from envir.schema import normalize_label
from .spec import difficulty_sequence, make_spec
from .world import initial_public_view
from .question import (
    guard_question, naturalize_one, reference_question,
)
from .quality import compile_and_validate, validate_replay


def _parallel_ordered(items, job, *, workers: int, desc: str,
                      no_progress: bool = False):
    """Show *completed* futures immediately, but keep deterministic row order."""
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, item): index
                   for index, item in enumerate(items)}
        with tqdm(total=len(items), desc=desc, unit="task",
                  disable=no_progress) as bar:
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                bar.update(1)
    return results


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
        raise FileNotFoundError(
            f"Input file does not exist: {path}. "
            "Check the directory of this command's --scenes argument. "
            "For generation, you can instead pass "
            "'--mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz' "
            "if the file exists, or locate a scene bank using "
            "'find envir environment -name scenes.jsonl -type f'."
        )
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
                "generation needs original TaskGraph/Scene Bank, not an already "
                "generated reasoning scene bank"
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


def build_bank(args) -> dict:
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
        raise FileExistsError("Destination already exists; use --overwrite")
    sequence = difficulty_sequence(
        target=args.target, mix=_parse_mix(args.difficulty_mix),
        run_seed=args.seed,
    )
    order = list(seeds)
    random.Random(args.seed * 123457 + 137).shuffle(order)
    chosen, rejected, seen_questions = [], Counter(), set()
    # Freeze Goal, structure, source permissions and all substantive
    # constraints before calling any optional LLM or sampling World values.
    for seed_record in order:
        if len(chosen) >= args.target:
            break
        try:
            ordinal = len(chosen)
            spec = make_spec(
                seed_record=seed_record, ordinal=ordinal,
                run_seed=args.seed, difficulty=sequence[ordinal],
            )
            q = reference_question(spec)
            normalized = normalize_label(q)
            if normalized in seen_questions:
                rejected["duplicate_question"] += 1
                continue
            seen_questions.add(normalized)
            chosen.append(spec)
        except (ValueError, TypeError, KeyError) as exc:
            rejected[type(exc).__name__ + ":" + str(exc)[:140]] += 1
    if len(chosen) != args.target:
        raise ValueError(
            f"only {len(chosen)}/{args.target} V2 ProblemSpecs passed "
            f"pre-World checks; rejected={dict(rejected)}"
        )

    scenario_model = getattr(args, "scenario_model", None)
    scenario_api_base = getattr(args, "scenario_api_base", None)
    if bool(scenario_model) != bool(scenario_api_base):
        raise ValueError("provide both --scenario-model and --scenario-api-base")
    if scenario_model:
        from .scenario import enrich_one
        if args.scenario_workers < 1:
            raise ValueError("--scenario-workers must be positive")
        clients = _scenario_clients(args)
        def enrich(spec):
            return enrich_one(
                spec, author=clients[0], critic=clients[1],
                max_revisions=args.scenario_max_revisions,
            )[0]
        selected = _parallel_ordered(
            chosen, enrich, workers=args.scenario_workers,
            desc="Scenario Author/Critic",
            no_progress=getattr(args, "no_progress", False),
        )
    else:
        selected = chosen

    generated = []
    for spec in tqdm(selected, desc="Compile & difficulty audit", unit="task",
                     disable=getattr(args, "no_progress", False)):
        question = reference_question(spec)
        try:
            scene = compile_and_validate(spec, question)
            generated.append(scene)
        except (ValueError, TypeError, KeyError) as exc:
            rejected[type(exc).__name__ + ":" + str(exc)[:140]] += 1
    if len(generated) != args.target:
        raise ValueError(
            f"only {len(generated)}/{args.target} V2 scenes passed "
            f"full execution and difficulty checks; rejected={dict(rejected)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _jsonl(files["private"], generated)
    _jsonl(files["public"], [scene["public"] for scene in generated])
    _jsonl(files["audit"], [{
        "task_id": scene["task_id"],
        "scenario_authoring": scene["problem_spec"].get("scenario_authoring"),
        **scene["difficulty_audit"],
    } for scene in generated])
    summary = {
        "status": "done",
        "pipeline": "maestro_reasoning_program_world_taskgraph_question",
        "seed_source": seed_source,
        "target": args.target,
        "generated": len(generated),
        "rejected": dict(rejected),
        "scenario_authoring": dict(Counter(
            s["problem_spec"].get("scenario_authoring", {}).get("status", "off")
            for s in generated
        )),
        "difficulty_target_distribution": dict(Counter(
            s["problem_spec"]["difficulty_target"] for s in generated
        )),
        "goal_family_distribution": dict(Counter(
            s["problem_spec"]["family"] for s in generated
        )),
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
        "counterfactual_full_pass": sum(all(
            x["changing_source_changes_answer"] for x in
            scene["difficulty_audit"]["counterfactual_source_tests"].values()
        ) for scene in generated),
        "one_source_shortcut_successes": sum(
            scene["difficulty_audit"]["one_source_baseline"]["exact_match"]
            for scene in generated
        ),
        "source_observation_policy": (
            "fictional_synthetic_media_annotations_not_original_perception"
        ),
        "llm_question_review": "off_use_refine-questions_or_generate_with_question_model",
        "files": {key: str(value) for key, value in files.items()},
    }
    _json(files["summary"], summary)
    return summary


def replay_bank(args) -> dict:
    rows = read_jsonl(args.scenes, limit=getattr(args, "max_records", None))
    failures = []
    for row in tqdm(rows, desc="Strict replay", unit="task",
                    disable=getattr(args, "no_progress", False)):
        result = validate_replay(row)
        if result["status"] != "valid":
            failures.append(result)
    return {
        "status": "valid" if rows and not failures else "invalid",
        "checked": len(rows), "failure_count": len(failures),
        "failures": failures[:30],
    }


def audit_bank(args) -> dict:
    rows = read_jsonl(args.scenes, limit=getattr(args, "max_records", None))
    reports = [validate_replay(row) for row in tqdm(
        rows, desc="Quality audit", unit="task",
        disable=getattr(args, "no_progress", False))]
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


def _scenario_clients(args):
    from environment.question_generation.backend import OpenAICompatibleBackend
    author_key = os.environ.get(args.scenario_api_key_env)
    critic_key = (os.environ.get(args.scenario_critic_api_key_env)
                  if args.scenario_critic_api_key_env else author_key)
    return (
        OpenAICompatibleBackend(
            base_url=args.scenario_api_base, model=args.scenario_model,
            api_key=author_key, temperature=0.55, max_tokens=1500,
            timeout=args.scenario_timeout,
        ),
        OpenAICompatibleBackend(
            base_url=args.scenario_critic_api_base or args.scenario_api_base,
            model=args.scenario_critic_model or args.scenario_model,
            api_key=critic_key, temperature=0.0, max_tokens=1000,
            timeout=args.scenario_timeout,
        ),
    )


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


def refine_bank(args) -> dict:
    if args.output_dir.resolve() == args.scenes.parent.resolve():
        raise ValueError("naturalize-v2 requires a separate output directory")
    if args.question_workers < 1 or args.max_revisions < 0 or args.max_revisions > 8:
        raise ValueError("invalid worker/revision settings")
    rows = read_jsonl(args.scenes, limit=args.max_records)
    if not rows:
        raise ValueError("no V2 scenes loaded")
    for scene in tqdm(rows, desc="Validate frozen scenes", unit="task",
                      disable=getattr(args, "no_progress", False)):
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
    updated = _parallel_ordered(
        rows, one, workers=args.question_workers,
        desc="Question Author/Critic",
        no_progress=getattr(args, "no_progress", False),
    )
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
        "pipeline": "maestro_frozen_program_question_author_critic",
        "source_scenes": str(args.scenes),
        "generated": len(updated),
        "question_results": dict(statuses),
        "world_program_macrograph_unmodified": True,
        "accepted_question_equivalence": "critic_judgment_not_formal_proof",
        "files": {key: str(path) for key, path in files.items()},
    }
    _json(files["summary"], summary)
    return summary


# Historic Python callers can keep the old symbol names. The new canonical
# application imports build_bank/replay_bank/audit_bank/refine_bank directly.
build_v2 = build_bank
replay_v2 = replay_bank
audit_v2 = audit_bank
naturalize_v2 = refine_bank
