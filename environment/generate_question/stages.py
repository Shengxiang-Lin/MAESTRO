"""Stage-1 scene construction and independent LLM Question review."""
from __future__ import annotations

import copy
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from environment.schema import normalize_label
from .spec import difficulty_sequence, make_spec
from .question import naturalize_one, reference_question
from .quality import compile_and_validate, validate_replay
from environment.streaming import (
    OrderedJournal, StreamingError, file_digest,
    output_lock, run_ordered, stable_signature,
)


def _knowledge_input_identity(args) -> dict | None:
    """Cheap corpus identity for resumability; avoids hashing multi-GB ZIPs."""
    location = getattr(args, "knowledge_yago", None)
    if not location:
        return None
    from environment.macro_v3.yago import _files
    candidates = _files(Path(location))
    if not candidates:
        raise FileNotFoundError("knowledge_yago_has_no_supported_facts_file")
    source = candidates[0]
    meta = source.stat()
    return {
        "file": str(source.resolve()),
        "size_bytes": meta.st_size,
        "mtime_ns": meta.st_mtime_ns,
    }


def _user_task_identity(spec: dict, question: str) -> tuple:
    """Deduplicate the entire visible task, not its generic question alone.

    Different provided reference materials make two tasks genuinely distinct.
    Ignore opaque reference IDs and ordinal seeds: they must never be used
    to disguise equivalent question/material content as a new task.
    Old scenes without separate user material keep question-only identity.
    """
    title = normalize_label(question)
    card = spec.get("task_input")
    if card is None:
        return (title,)
    if card.get("version") != "macro_goal_reference_card_v2":
        raise ValueError("unsupported_task_input_version")
    observations = tuple(sorted(
        (
            normalize_label(str(item["modality"])),
            normalize_label(str(item["content"])),
        ) for item in card["observations"]
    ))
    return (title, observations)


def _construction_identity(args) -> str:
    """Resume identity of the only supported Stage 1 generation path."""
    return stable_signature({
        "stage": "output_first",
        "target": args.target,
        "max_records": args.max_records,
        "seed": args.seed,
        "difficulty_mix": args.difficulty_mix,
        "knowledge_source_identity": _knowledge_input_identity(args),
        "knowledge_limits": [
            getattr(args, key) for key in (
                "knowledge_max_triples", "knowledge_max_edges",
                "knowledge_min_hops", "knowledge_max_hops",
            )
        ],
        "generation_code": {
            name: file_digest(
                Path(__file__).parent / (
                    "planning" if name in {
                        "task_plan.py", "goal_plan.py", "evidence_plan.py"
                    } else "execution" if name in {
                        "task_plan_runtime.py", "program_ir.py"
                    } else "authoring" if name == "surface_quality.py"
                    else ""
                ) / name
            )
            for name in (
                "task_plan.py", "task_plan_runtime.py", "goal_plan.py",
                "evidence_plan.py", "world_motifs.py", "knowledge_macro.py",
                "question.py", "intent.py", "surface_quality.py",
                "program_ir.py", "quality.py", "stages.py",
            )
        },
        "scenario_model": args.scenario_model,
        "scenario_api_base": args.scenario_api_base,
        "scenario_critic_model": args.scenario_critic_model,
        "scenario_critic_api_base": args.scenario_critic_api_base,
        "scenario_max_revisions": args.scenario_max_revisions,
    })


def _question_signature(args, rows: list[dict]) -> str:
    """Exact resume identity shared by LLM reviewer and root pipeline."""
    return stable_signature({
        "stage": "question_author_critic.v1",
        "ids": [scene["task_id"] for scene in rows],
        "scene_bank_hash": file_digest(args.scenes),
        "model": args.question_model,
        "api_base": args.question_api_base,
        "critic_model": args.question_critic_model,
        "critic_api_base": args.question_critic_api_base,
        "author_temperature": getattr(args, "question_author_temperature", None),
        "author_top_p": getattr(args, "question_author_top_p", None),
        "critic_temperature": getattr(args, "question_critic_temperature", None),
        "critic_top_p": getattr(args, "question_critic_top_p", None),
        "max_revisions": args.max_revisions,
        "question_code": file_digest(Path(__file__).with_name("question.py")),
        "semantic_evidence_code": file_digest(
            Path(__file__).with_name("semantic_evidence.py")),
        "question_fidelity_code": file_digest(
            Path(__file__).with_name("authoring") / "question_fidelity.py"),
        "question_semantic_lock_code": file_digest(
            Path(__file__).with_name("authoring") / "question_semantic_lock.py"),
        "intent_code": file_digest(Path(__file__).with_name("intent.py")),
        "surface_quality_code": file_digest(
            Path(__file__).with_name("authoring") / "surface_quality.py"),
        "prompt_templates_code": file_digest(
            Path(__file__).with_name("authoring") / "prompt_templates.py"),
        "stage_code": file_digest(Path(__file__)),
        "backend_code": file_digest(Path(__file__).with_name("backend.py")),
    })


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
            "'find environment environment -name scenes.jsonl -type f'."
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
        "failed": output_dir / "failed.jsonl",
    }


def construct_scenes(args) -> dict:
    if args.target < 1 or args.max_records < 1:
        raise ValueError("target and max-records must be positive")
    if not getattr(args, "knowledge_yago", None):
        raise ValueError("knowledge_yago_required_for_stage1")
    from environment.macro_v3.yago import load_yago
    from .knowledge_macro import YagoKnowledgeIndex
    knowledge_graph = load_yago(
        args.knowledge_yago,
        max_triples=args.knowledge_max_triples,
        max_edges=args.knowledge_max_edges,
    )
    knowledge_index = YagoKnowledgeIndex(knowledge_graph)
    candidate_budget = max(
        args.target, min(args.max_records, args.target * 4)
    )
    seeds = [
        {"topic": "archival reference collection",
         "record_id": f"knowledge-{i:06d}"}
        for i in range(candidate_budget)
    ]
    seed_source = knowledge_graph.input_file
    if len(seeds) < args.target:
        raise ValueError(
            f"not enough scene/topic seeds: {len(seeds)} for target {args.target}"
        )
    files = _out_files(args.output_dir)
    checkpoint = args.output_dir / "scenario_authoring_progress.json"
    if not args.overwrite and any(p.exists() for p in files.values()):
        if files["summary"].exists() or not checkpoint.exists():
            raise FileExistsError(
                "Destination already contains a completed or legacy scene bank; "
                "choose a new directory or enable overwrite"
            )
    sequence = difficulty_sequence(
        target=args.target, mix=_parse_mix(args.difficulty_mix),
        run_seed=args.seed,
    )
    order = list(seeds)
    random.Random(args.seed * 123457 + 137).shuffle(order)
    chosen, rejected, seen_tasks = [], Counter(), set()
    # Freeze Goal, structure, source permissions and all substantive
    # constraints before calling any optional LLM or sampling World values.
    for candidate_ordinal, seed_record in enumerate(order):
        if len(chosen) >= args.target:
            break
        try:
            ordinal = candidate_ordinal
            from .task_plan import make_spec
            spec = make_spec(
                knowledge_graph, seed=args.seed, ordinal=ordinal,
                difficulty=sequence[len(chosen)],
                min_hops=args.knowledge_min_hops,
                max_hops=args.knowledge_max_hops,
                index=knowledge_index,
            )
            q = reference_question(spec)
            identity = _user_task_identity(spec, q)
            if identity in seen_tasks:
                rejected[
                    "duplicate_user_task" if "task_input" in spec
                    else "duplicate_question"
                ] += 1
                continue
            seen_tasks.add(identity)
            chosen.append(spec)
        except (ValueError, TypeError, KeyError) as exc:
            rejected[type(exc).__name__ + ":" + str(exc)[:140]] += 1
    if len(chosen) != args.target:
        raise ValueError(
            f"only {len(chosen)}/{args.target} ProblemSpecs passed "
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
        signature = stable_signature({
            "stage": "scenario_author_critic.v1",
            "specs": chosen,
            "model": args.scenario_model,
            "api_base": args.scenario_api_base,
            "critic_model": args.scenario_critic_model,
            "critic_api_base": args.scenario_critic_api_base,
            "max_revisions": args.scenario_max_revisions,
            "scenario_code": file_digest(Path(__file__).with_name("scenario.py")),
            "semantic_evidence_code": file_digest(
                Path(__file__).with_name("semantic_evidence.py")),
            "intent_code": file_digest(Path(__file__).with_name("intent.py")),
            "prompt_templates_code": file_digest(
                Path(__file__).with_name("authoring") / "prompt_templates.py"),
            "stage_code": file_digest(Path(__file__)),
            "backend_code": file_digest(Path(__file__).with_name("backend.py")),
        })
        task_ids = [spec["task_id"] for spec in chosen]
        with output_lock(args.output_dir, "scenario_authoring"):
            with OrderedJournal(
                args.output_dir, "scenario_authoring", signature, task_ids,
                flush_every=getattr(args, "flush_every", 1),
                restart=args.overwrite,
            ) as journal:
                run_ordered(
                    chosen, enrich,
                    lambda spec: (
                        spec.get("scenario_authoring", {}).get("status") == "accepted",
                        spec,
                        {"status": spec.get("scenario_authoring", {}).get("status",
                                                                         "failed")},
                    ),
                    journal=journal,
                    workers=args.scenario_workers,
                    desc="Scenario Author/Critic",
                    no_progress=getattr(args, "no_progress", False),
                    on_exception=lambda idx, exc: (
                        False,
                        {
                            **chosen[idx],
                            "scenario_authoring": {
                                "status": "failed",
                                "failures": [{
                                    "stage": "backend",
                                    "error_type": type(exc).__name__,
                                    "error": str(exc)[:220],
                                }],
                            },
                        },
                        {"status": "failed", "error": str(exc)[:220]},
                    ),
                )
                selected = [payload for _, payload, _ in journal.replay()]
    else:
        selected = chosen

    failed_cases = []
    if scenario_model:
        valid_specs = []
        for spec in selected:
            review = spec.get("scenario_authoring", {})
            if review.get("status") == "accepted":
                valid_specs.append(spec)
            else:
                rejected["scenario_fallback"] += 1
                failed_cases.append({
                    "stage": "scenario", "task_id": spec["task_id"],
                    "review": review, "problem_spec": spec,
                })
        selected_to_compile = valid_specs
    else:
        selected_to_compile = selected
    # Compilation is fast/deterministic, so rerun it from the committed
    # Scenario journal on resume; expensive LLM Scenario calls are reused.
    # Unlike before, even compiled scenes and failures are written immediately.
    generated = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (files["private"].open("w", encoding="utf-8") as private,
          files["public"].open("w", encoding="utf-8") as public,
          files["audit"].open("w", encoding="utf-8") as audit,
          files["failed"].open("w", encoding="utf-8") as failed):
        for record in failed_cases:
            failed.write(json.dumps(record, ensure_ascii=False) + "\n")
        with tqdm(selected_to_compile, desc="Compile & difficulty audit",
                  unit="task", disable=getattr(args, "no_progress", False),
                  dynamic_ncols=True) as bar:
            for spec in bar:
                question = reference_question(spec)
                try:
                    scene = compile_and_validate(spec, question)
                    generated.append(scene)
                    private.write(json.dumps(scene, ensure_ascii=False) + "\n")
                    public.write(json.dumps(scene["public"], ensure_ascii=False) + "\n")
                    audit.write(json.dumps({
                        "task_id": scene["task_id"],
                        "scenario_authoring": (
                            scene["problem_spec"].get("scenario_authoring")
                        ),
                        **scene["difficulty_audit"],
                    }, ensure_ascii=False) + "\n")
                except (ValueError, TypeError, KeyError) as exc:
                    rejected[type(exc).__name__ + ":" + str(exc)[:140]] += 1
                    case = {
                        "stage": "execution", "task_id": spec["task_id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:400],
                        "problem_spec": spec,
                    }
                    failed_cases.append(case)
                    failed.write(json.dumps(case, ensure_ascii=False) + "\n")
                for writer in (private, public, audit, failed):
                    writer.flush()
                bar.set_postfix({
                    "accepted": len(generated),
                    "rejected": len(failed_cases),
                    "written": len(generated) + len(failed_cases),
                }, refresh=False)

    summary = {
        "status": (
            "done" if len(generated) == args.target else
            "partial" if generated else "failed"
        ),
        "pipeline": "maestro_reasoning_program_world_taskgraph_question",
        "run_identity": _construction_identity(args),
        "seed_source": seed_source,
        "target": args.target,
        "attempted": len(chosen),
        "generated": len(generated),
        "failed": len(failed_cases),
        "failed_file": str(files["failed"]),
        "rejected": dict(rejected),
        "scenario_authoring": dict(Counter(
            s.get("scenario_authoring", {}).get("status", "off")
            for s in selected
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
        ) / len(generated), 3) if generated else None,
        "operator_grammar_scenes": sum(
            bool(s["problem_spec"].get("operator_grammar_version"))
            for s in generated
        ),
        "mean_macro_task_count": round(sum(
            s["macro_graph"]["macro_task_count"] for s in generated
        ) / len(generated), 3) if generated else None,
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


def _scenario_clients(args):
    from .backend import OpenAICompatibleBackend
    author_key = (os.environ.get(args.scenario_api_key_env)
                  if args.scenario_api_key_env else None)
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
    from .backend import OpenAICompatibleBackend
    if not args.question_api_base or not args.question_model:
        raise ValueError("both question endpoint and model are required")
    key = (os.environ.get(args.question_api_key_env)
           if args.question_api_key_env else None)
    critic_key = (
        os.environ.get(args.question_critic_api_key_env)
        if args.question_critic_api_key_env else key
    )
    if args.question_api_key_env and not key:
        raise ValueError(
            "Stage-1 Question Author API key missing: export "
            + args.question_api_key_env
        )
    if args.question_critic_api_key_env and not critic_key:
        raise ValueError(
            "Stage-1 Question Critic API key missing: export "
            + args.question_critic_api_key_env
        )
    author_temperature = getattr(args, "question_author_temperature", None)
    author_top_p = getattr(args, "question_author_top_p", None)
    critic_temperature = getattr(args, "question_critic_temperature", None)
    critic_top_p = getattr(args, "question_critic_top_p", None)
    return (
        OpenAICompatibleBackend(
            base_url=args.question_api_base, model=args.question_model,
            api_key=key,
            temperature=(author_temperature if author_temperature is not None
                         else 0.55),
            top_p=author_top_p,
            max_tokens=1800, timeout=args.question_timeout,
        ),
        OpenAICompatibleBackend(
            base_url=args.question_critic_api_base or args.question_api_base,
            model=args.question_critic_model or args.question_model,
            api_key=critic_key,
            temperature=(critic_temperature if critic_temperature is not None
                         else 0.0),
            top_p=critic_top_p,
            max_tokens=1400, timeout=args.question_timeout,
        ),
    )


def _naturalize_entry(scene: dict, *, clients, max_revisions: int) -> dict:
    question, review = naturalize_one(
        scene["problem_spec"], author=clients[0], critic=clients[1],
        max_revisions=max_revisions, macro_graph=scene["macro_graph"],
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


def review_questions(args) -> dict:
    """Review frozen scenes with immediate per-Question durable streaming.

    Accepted private/public Question files are updated as decisions are
    committed; fallback/rejected decisions appear in failed.jsonl.
    An interruption can resume safely from the ordered Question journal.
    """
    if args.output_dir.resolve() == args.scenes.parent.resolve():
        raise ValueError("Question review requires a separate output directory")
    if args.question_workers < 1 or args.max_revisions < 0 or args.max_revisions > 8:
        raise ValueError("invalid worker/revision settings")
    rows = read_jsonl(args.scenes, limit=args.max_records)
    if not rows:
        raise ValueError("no scenes loaded")
    # This cheap deterministic preflight is not a separate user-facing
    # generation step. Keep it visible only for explicit verbose diagnostics.
    with tqdm(rows, desc="Validate frozen scenes", unit="task",
              disable=(getattr(args, "no_progress", False)
                       or not getattr(args, "verbose_progress", False)),
              dynamic_ncols=True) as bar:
        for ix, scene in enumerate(bar, 1):
            result = validate_replay(scene)
            if result["status"] != "valid":
                raise ValueError("invalid incoming scene:" + str(result))
            if scene.get("question_authoring") is not None:
                raise ValueError("input already naturalized")
            bar.set_postfix({"accepted": ix, "rejected": 0}, refresh=False)
    files = _out_files(args.output_dir)
    progress_path = args.output_dir / "question_authoring_progress.json"
    if not args.overwrite and any(path.exists() for path in files.values()):
        running_summary = (
            json.loads(files["summary"].read_text(encoding="utf-8"))
            if files["summary"].is_file() else {}
        )
        if (not progress_path.exists()
                or running_summary.get("status") not in ({}, "running")):
            raise FileExistsError(
                "Question review output exists without resumable progress; "
                "choose a new directory or enable overwrite"
            )
    clients = _llm_clients(args)

    def one(scene):
        return _naturalize_entry(
            scene, clients=clients, max_revisions=args.max_revisions
        )

    def classify(scene):
        review = scene.get("question_authoring", {})
        return (
            review.get("status") == "accepted",
            scene,
            {"status": review.get("status", "failed")},
        )

    def on_exception(index, exc):
        failed_scene = copy.deepcopy(rows[index])
        failed_scene["question_authoring"] = {
            "status": "failed", "attempts": 0,
            "failures": [{
                "stage": "backend",
                "error_type": type(exc).__name__,
                "error": str(exc)[:220],
            }],
        }
        return False, failed_scene, {
            "status": "failed", "error": str(exc)[:220],
        }

    ids = [scene["task_id"] for scene in rows]
    signature = _question_signature(args, rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_lock(args.output_dir, "question_authoring"):
        with OrderedJournal(
            args.output_dir, "question_authoring", signature, ids,
            flush_every=getattr(args, "flush_every", 1),
            restart=args.overwrite,
        ) as journal:
            statuses = Counter()
            accepted_count = rejected_count = 0

            def summary() -> dict:
                finished = journal.completed == len(rows)
                return {
                    "status": (
                        "running" if not finished else
                        "done" if accepted_count == len(rows) else
                        "partial" if accepted_count else "failed"
                    ),
                    "pipeline": "maestro_frozen_program_question_author_critic",
                    "source_scenes": str(args.scenes),
                    "attempted": len(rows),
                    "completed": journal.completed,
                    "generated": accepted_count,
                    "failed": rejected_count,
                    "question_results": dict(statuses),
                    "world_program_macrograph_unmodified": True,
                    "accepted_question_equivalence": (
                        "critic_judgment_not_formal_proof"
                    ),
                    "streaming": {
                        "ordered_writes": True,
                        "flush_every": getattr(args, "flush_every", 1),
                        "resumable": True,
                        "checkpoint": str(journal.state_path),
                        "resumed_prefix": journal.completed,
                    },
                    "files": {key: str(path) for key, path in files.items()},
                }

            with (files["private"].open("w", encoding="utf-8") as private,
                  files["public"].open("w", encoding="utf-8") as public,
                  files["audit"].open("w", encoding="utf-8") as audit,
                  files["failed"].open("w", encoding="utf-8") as failed):
                def publish(index, good, scene, meta, *, snapshot=True):
                    nonlocal accepted_count, rejected_count
                    review = scene.get("question_authoring", {})
                    statuses[review.get("status", "failed")] += 1
                    if good:
                        accepted_count += 1
                        private.write(json.dumps(scene, ensure_ascii=False) + "\n")
                        public.write(json.dumps(scene["public"], ensure_ascii=False) + "\n")
                        audit.write(json.dumps({
                            "task_id": scene["task_id"],
                            "difficulty": scene["difficulty_audit"],
                            "question_authoring": review,
                        }, ensure_ascii=False) + "\n")
                    else:
                        rejected_count += 1
                        failed.write(json.dumps({
                            "task_id": scene["task_id"], "stage": "question",
                            "review": review,
                            "reference_question": reference_question(
                                scene["problem_spec"]
                            ),
                            "problem_spec": scene["problem_spec"],
                        }, ensure_ascii=False) + "\n")
                    # Make user-visible output available immediately, while
                    # the journal checkpoint guarantees crash-safe replay.
                    for stream in (private, public, audit, failed):
                        stream.flush()
                    if snapshot:
                        _json(files["summary"], summary())

                try:
                    # Rebuild public files solely from committed checkpoints.
                    # All half-written tails from a crash are discarded.
                    for idx, (good, payload, meta) in enumerate(journal.replay()):
                        publish(idx, good, payload, meta, snapshot=False)
                    _json(files["summary"], summary())
                    run_ordered(
                        rows, one, classify, journal=journal,
                        workers=args.question_workers,
                        desc="Question Author/Critic",
                        no_progress=getattr(args, "no_progress", False),
                        on_commit=publish, on_exception=on_exception,
                    )
                    result = summary()
                finally:
                    _json(files["summary"], summary())
                return result
