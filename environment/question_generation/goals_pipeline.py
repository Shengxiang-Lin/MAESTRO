"""Stage 2A: independent, reviewed Subtask Goals, persisted before questions.

Only certified Blueprint rows are eligible for Stage 2B. Stage 2A never
generates Main questions, Sub-Agent answers, or run-time delegations.
"""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from time import monotonic

from tqdm.auto import tqdm

from environment.config import EnvironmentConfigError
from environment.delegation_blueprint import (
    VERSION, compile_delegation_outline, naturalize_fragment_goal,
)
from environment.trajectory_simulation.compiler import (
    GoldCompilationError, compile_gold_solution,
)
from environment.trajectory_simulation.eligibility import (
    controlled_question_reference, digest,
)
from .boundary import BoundaryError, validate_frozen_inputs
from .goal_review import (
    certify_goal_record, review_fragment_goal,
)
from .pipeline import (
    _rows, _world_by_task, _capability_by_task, _backends,
)
from .scoped_surface import build_scoped_plan, render_scoped
from .streaming import (
    GoalProgress, QuestionCheckpointError, question_output_lock,
    signature_for_question_run,
)
from .validators import postflight


def _reject(payload, reason, audit, exc=None):
    return False, {
        "task_id": payload.get("task_id"), "reason": reason,
        "detail": str(exc)[:240] if exc is not None else "",
    }, audit


def generate_goal_record(payload, world, capability, author, critic, *,
                         max_goal_revisions=2, blueprint_workers=2):
    """Compile public DAG once; review one Goal at a time, fail closed."""
    from .backend import BackendError
    canonical = payload.get("draft_question", "")
    rendered = world.get("_rendered")
    audit = []
    if not canonical or getattr(rendered, "question", None) != canonical:
        return _reject(payload, "rejected_reference", audit,
                       "frozen_canonical_draft_mismatch")
    try:
        gold = compile_gold_solution({
            "task_id": payload["task_id"], "question": canonical,
            "capability_world": capability,
        }, world, rendered, schedule="compact")
        controlled = controlled_question_reference(gold, world, rendered)
        if controlled is None or controlled.get("question") != canonical:
            raise ValueError("unverified_controlled_reference")
        plan = build_scoped_plan(world, canonical, gold, controlled)
        scoped = render_scoped(plan, [0] * len(plan))
        preflight = postflight(
            payload, scoped, oracle_answer=world.get("_display_answer"),
            check_contract=False,
        )
        if not preflight.accepted:
            raise ValueError("scoped_reference_invalid:" +
                             ",".join(preflight.issues))
        outline = compile_delegation_outline(
            gold, world, canonical, plan, payload.get("resources", ()),
        )
    except (GoldCompilationError, ValueError, KeyError, TypeError, IndexError) as exc:
        return _reject(payload, "rejected_reference", audit, exc)
    audit.append({"stage": "reference", "outline_sha256": digest(outline)})
    if (not callable(getattr(author, "complete_json", None))
            or not callable(getattr(critic, "complete_json", None))):
        return _reject(payload, "rejected_validation_error", audit,
                       "goal_author_and_critic_required")

    fragments = outline["tasks"]
    fragment_audits = [{
        "stage": "fragment_goal", "fragment_id": fragment["fragment_id"],
        "author_calls": 0, "critic_calls": 0,
        "attempts": [], "protocol_failures": [],
    } for fragment in fragments]
    audit.extend(fragment_audits)

    def one(pair):
        index, fragment = pair
        log = fragment_audits[index]
        feedback = None
        last_reason = "rejected_goal_semantic"
        for attempt_index in range(max_goal_revisions + 1):
            entry = {"attempt": attempt_index + 1}
            log["attempts"].append(entry)
            before = {"calls": 0}
            log["author_calls"] += 1
            try:
                item = naturalize_fragment_goal(
                    author, outline, fragment, scoped,
                    payload=payload, oracle_answer=world.get("_display_answer"),
                    max_attempts=1, call_log=before, feedback=feedback,
                )
            except (BackendError, ValueError, KeyError, TypeError) as exc:
                entry["failure"] = "goal_writer_error"
                entry["detail"] = str(exc)[:180]
                last_reason = "rejected_goal_protocol"
                continue
            goal = item["goal"]
            entry["goal"] = goal
            review = review_fragment_goal(
                critic, outline, fragment, scoped, goal, audit=log,
            )
            entry["review"] = review
            if review["verdict"] == "pass":
                log["accepted"] = True
                return item, {"fragment_id": fragment["fragment_id"],
                              "review": review}, None
            if review["verdict"] == "uncertain":
                last_reason = "rejected_goal_uncertain"
            else:
                last_reason = "rejected_goal_semantic"
            feedback = {"previous_goal": goal,
                        "issues": review["issues"] or [review["summary"]]}
        log["accepted"] = False
        log["failure"] = last_reason
        return None, None, last_reason

    try:
        with ThreadPoolExecutor(
            max_workers=min(blueprint_workers, max(1, len(fragments))),
            thread_name_prefix="stage2a-goal",
        ) as pool:
            results = list(pool.map(one, enumerate(fragments)))
    except Exception as exc:
        return _reject(payload, "rejected_goal_protocol", audit, exc)
    failures = [
        {"fragment_id": fragment["fragment_id"], "reason": triple[2]}
        for fragment, triple in zip(fragments, results) if triple[2]
    ]
    if failures:
        return _reject(payload, "rejected_goal_fidelity", audit, failures)

    goals = [triple[0] for triple in results]
    reviews = [triple[1] for triple in results]
    blueprint = {
        "version": VERSION,
        "outline": outline, "goals": goals,
    }
    try:
        certified = certify_goal_record(
            payload, world, canonical, outline, blueprint, reviews,
            oracle_answer=world.get("_display_answer"),
        )
    except (ValueError, KeyError, TypeError) as exc:
        return _reject(payload, "rejected_goal_certificate", audit, exc)
    audit.append({"stage": "certificate", "approved": True,
                  "certificate_sha256": digest(certified["goal_certificate"])})
    return True, certified, audit


def _safe_result(future, index, authoring):
    try:
        return future.result()
    except Exception as exc:
        return _reject(authoring[index], "unhandled_exception", [], exc)


def _generate_parallel(authoring, worlds, capabilities, author, critic,
                       args, progress):
    total = len(authoring)
    next_submit = progress.completed
    pending = {}
    with ThreadPoolExecutor(max_workers=args.question_workers,
                            thread_name_prefix="stage2a-task") as pool:
        active = {}
        bar = tqdm(total=total, initial=progress.completed, desc="Stage 2A Goals",
                   unit="task", disable=args.no_progress, dynamic_ncols=True)
        try:
            def submit():
                nonlocal next_submit
                while len(active) < 2 * args.question_workers and next_submit < total:
                    index = next_submit
                    payload = authoring[index]
                    future = pool.submit(
                        generate_goal_record, payload,
                        worlds.get(str(payload.get("task_id")), {}),
                        capabilities.get(str(payload.get("task_id")), {}),
                        author, critic,
                        max_goal_revisions=args.max_goal_revisions,
                        blueprint_workers=args.blueprint_workers,
                    )
                    active[future] = index
                    next_submit += 1
            submit()
            while active:
                ready, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
                for future in ready:
                    index = active.pop(future)
                    pending[index] = _safe_result(future, index, authoring)
                    bar.update(1)
                while progress.completed in pending:
                    index = progress.completed
                    progress.append(index, *pending.pop(index))
                bar.set_postfix(written=progress.completed,
                                certified=progress.accepted)
                submit()
        finally:
            bar.close()


def run(args):
    started = monotonic()
    try:
        frozen = validate_frozen_inputs(
            args.manifest, args.authoring_inputs, args.capabilities,
            worlds_path=args.worlds,
        )
    except (BoundaryError, FileNotFoundError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Frozen-input validation failed: {exc}") from exc
    if args.start_record < 0 or args.question_workers < 1 or args.blueprint_workers < 1:
        raise ValueError("invalid_goal_workers_or_range")
    if not 0 <= args.max_goal_revisions <= 4:
        raise ValueError("max_goal_revisions_must_be_0_to_4")
    authoring = _rows(args.authoring_inputs)[args.start_record:]
    if args.max_records is not None:
        authoring = authoring[:args.max_records]
    worlds = _world_by_task(args.worlds, seed=args.seed)
    capabilities = _capability_by_task(args.capabilities)
    if any(str(row.get("task_id")) not in worlds for row in authoring):
        raise SystemExit("authoring tasks are missing executable worlds")
    author, critic = _backends(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = [str(row["task_id"]) for row in authoring]
    sources = [
        Path(args.manifest), Path(args.authoring_inputs),
        Path(args.worlds), Path(args.capabilities),
        Path(__file__), Path(__file__).with_name("goal_review.py"),
        Path(__file__).with_name("realization.py"),
        Path(__file__).with_name("scoped_surface.py"),
        Path(__file__).with_name("public_semantics.py"),
        Path(__file__).with_name("streaming.py"),
        Path(__file__).parents[1] / "delegation_blueprint.py",
        Path(__file__).parents[1] / "trajectory_simulation/compiler.py",
        Path(__file__).parents[1] / "trajectory_simulation/skeleton.py",
    ]
    signature = signature_for_question_run(
        frozen, args, ids, source_files=sources,
        extra_settings={
            "phase": "stage2a", "max_goal_revisions": args.max_goal_revisions,
            "blueprint_workers": args.blueprint_workers,
        },
    )
    with question_output_lock(output_dir):
        with GoalProgress(
            output_dir, signature, ids, flush_every=args.flush_every,
            restart=args.restart_goals,
        ) as progress:
            resumed = progress.completed
            _generate_parallel(
                authoring, worlds, capabilities, author, critic, args, progress,
            )
            if progress.completed != len(authoring):
                raise QuestionCheckpointError("incomplete_stage2a_goal_journal")
            accepted, rejected = progress.accepted, progress.rejected

        reasons = Counter()
        for row in _rows(output_dir / "rejected_blueprints.jsonl"):
            reasons[row["reason"]] += 1
        author_calls = critic_calls = 0
        fragment_count = passed_fragments = 0
        for entry in _rows(output_dir / "goal_review_audit.jsonl"):
            for step in entry.get("audit", []):
                if step.get("stage") != "fragment_goal":
                    continue
                fragment_count += 1
                passed_fragments += bool(step.get("accepted"))
                author_calls += step.get("author_calls", 0)
                critic_calls += step.get("critic_calls", 0)
        summary = {
            "pipeline": "stage2a_independent_reviewed_fragment_goals",
            "frozen_batch_validation": frozen,
            "model_roles": args._resolved_models,
            "requested": len(authoring), "certified": accepted,
            "rejected": rejected, "certification_rate":
                accepted / len(authoring) if authoring else 0.0,
            "rejection_reasons": dict(reasons),
            "review_statistics": {
                "fragments_attempted": fragment_count,
                "fragments_passed": passed_fragments,
                "goal_author_calls": author_calls,
                "goal_critic_calls": critic_calls,
                "total_model_calls": author_calls + critic_calls,
            },
            "frozen_task_range": {
                "start_record": args.start_record,
                "end_record_exclusive": args.start_record + len(authoring),
            },
            "current_invocation_elapsed_seconds": round(monotonic() - started, 3),
            "streaming": {
                "enabled": True, "resumed_records": resumed,
                "question_workers": args.question_workers,
                "blueprint_workers_per_task": args.blueprint_workers,
                "checkpoint": str(output_dir / "goal_progress_state.json"),
            },
            "artifacts": {
                "certified": str(output_dir / "certified_blueprints.jsonl"),
                "rejected": str(output_dir / "rejected_blueprints.jsonl"),
                "audit": str(output_dir / "goal_review_audit.jsonl"),
                "summary": str(output_dir / "goal_generation_summary.json"),
            },
        }
        destination = output_dir / "goal_generation_summary.json"
        tmp = destination.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(destination)
    return summary


__all__ = ["generate_goal_record", "run"]
