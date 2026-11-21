"""Event-driven untrusted O1 Main decision generation.

Every prior feedback round is a legal single-atom scenario. Opt-in event
scenarios additionally schedule an executable later fault and/or defer a
not-yet-returned independent evidence task. No historical Main action is
retroactively rewritten; all events are replayed in causal order.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import os
import random
import shutil
from threading import Lock

from tqdm.auto import tqdm

from environment.untrusted_orchestration.runtime.faults import FaultUnsupported
from environment.untrusted_orchestration.runtime.atomic_faults import ATOM_TO_MODE
from .scenario.scenario_generator import make_scenario
from .trajectory_builder import build_branch
from environment.base_orchestration.trajectory_builder import (
    _strict_reports, clean_reference_identity,
)
from .offline_pipeline import author_branch_reports, author_branch_main
from .trusted_prefix import compile_trusted_prefixes
from .publication import _publish
from .inspection.publication_guard import review_turn_surface
from .storage import _atomic_json, _atomic_jsonl
from .scenario.candidate_exploration import expand_all_rounds
from .scenario.single_turn_selection import select_single_turn_candidates
from .scenario.round_sampling import select_feedback_rounds
from environment.untrusted_orchestration.inspection.evidence_affordances import analyze_scene_evidence_affordances


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines()
            if row.strip()]


# Stage 4A is CPU-bound; threads do not parallelize Python replay under
# the GIL. Worker processes receive stable, prevalidated scene contracts
# ONCE via initializer, then only small candidate plans per invocation.
_EVIDENCE_CONTEXT: dict | None = None


def _init_evidence_worker(context: dict) -> None:
    global _EVIDENCE_CONTEXT
    _EVIDENCE_CONTEXT = context


def _execute_evidence_candidate(plan: dict) -> str:
    """4A-only worker: write one resumable journal and return its status."""
    if _EVIDENCE_CONTEXT is None:
        raise RuntimeError("stage4_evidence_worker_not_initialized")
    ctx = _EVIDENCE_CONTEXT
    scene = ctx["scenes"][plan["scene_index"]]
    result = {
        "task_id": scene["task_id"],
        "round_sample_id": plan["sample_id"],
        "fork_round": plan["branch_round"],
        "episode": None,
        "turns": [],
        "private_audit": None,
    }
    if plan["skip_reason"]:
        result.update(status="skipped_unsupported", error=plan["skip_reason"])
    else:
        try:
            episode, _, audit = build_branch(
                scene, ctx["reports"][scene["task_id"]],
                ctx["base"][scene["task_id"]],
                branch_round=plan["branch_round"], faults=plan["faults"],
                scenario_plan=plan["scenario_plan"],
                validated_reference=ctx["validated"][scene["task_id"]],
                **ctx["build_options"],
            )
            result.update(
                status=audit["execution_outcome"],
                episode=episode, private_audit=audit,
            )
        except (FaultUnsupported, ValueError, KeyError, TypeError) as exc:
            result.update(
                status=("skipped_unsupported" if isinstance(
                    exc, FaultUnsupported) else "rejected_validation"),
                error=str(exc)[:300],
            )
    _atomic_json(
        Path(ctx["journal_dir"]) / f"{plan['index']:06d}.json", result,
    )
    return result["status"]


def _hash_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _signature(inputs: tuple[Path, Path], settings: dict) -> dict:
    directory = Path(__file__).resolve().parent
    code_files = (
        "runtime/faults.py", "runtime/atomic_faults.py", "reporting/faults.py",
        "scenario/scenario_generator.py", "scenario/round_sampling.py",
        "scenario/candidate_exploration.py",
        "scenario/observable_scenarios.py",
        "scenario/single_turn_selection.py",
        "inspection/evidence_affordances.py", "orchestration/reasoning_fidelity.py",
        "inspection/history_reports.py", "inspection/visible_history.py", "inspection/visible_claim_state.py",
        "inspection/visible_claim_extraction.py",
        "inspection/claim_comparison.py",
        "orchestration/main_decision_contract.py",
        "reporting/contract.py", "reporting/surface.py",
        "orchestration/action_binding.py",
        "runtime/evidence_planning.py",
        "rewards/decision_reward_state.py",
        "rewards/orchestration_reward.py",
        "reporting/validation.py",
        "reporting/work_evidence.py",
        "inspection/publication_guard.py",
        "state_machine.py", "reporting/author.py",
        "trusted_prefix.py",
        "trajectory_builder.py", "generate_turns.py",
        "publication.py", "storage.py",
        "offline_pipeline.py", "decision_reward_replay.py",
        "runtime/evidence_operators.py",
        "../generate_question/execution/compiler.py",
        "../generate_question/execution/program_ir.py",
        "../base_orchestration/o1_compat.py",
        "../base_orchestration/trajectory_builder.py",
        "../base_orchestration/action_author.py",
        "../base_orchestration/generation_meta.py",
        "../implement_subtask/reporting.py",
        "../implement_subtask/visible_evidence.py",
        "config.py",
        "../../orchestra/base_main_agent.py",
    )
    return {
        "version": "maestro.untrusted_orchestration.single_turn_v1",
        "input_sha256": [_hash_file(path) for path in inputs],
        "source_code_sha256": {
            name: _hash_file(directory / name) for name in code_files
        },
        "settings": settings,
    }



def _plan_branches(
    scenes: list[dict], base: dict[str, dict], *,
    branches_per_scene: int | None = None, seed: int = 17,
    round_sampling_strategy: str = "chronological",
) -> tuple[list[dict], int, int]:
    """Extract genuine returned Stage-3 batches without selecting old faults."""
    if branches_per_scene is not None and (
        type(branches_per_scene) is not int or branches_per_scene < 1
    ):
        raise ValueError("invalid_debug_round_cap")
    planned, no_base, no_rounds = [], 0, 0
    for scene_index, scene in enumerate(scenes):
        episode = base.get(scene["task_id"])
        if episode is None:
            no_base += 1
            continue
        delegates = [
            row for row in episode["trajectory"]
            if row["action"] == "delegate_task"
        ]
        waves = scene["macro_graph"]["reference_waves"]
        if len(delegates) > len(waves):
            raise ValueError("trusted_delegate_exceeds_reference_waves")
        rounds = []
        for index, delegate in enumerate(delegates):
            if delegate.get("attempt") != index + 1:
                raise ValueError("trusted_delegate_round_alignment_invalid")
            results = (delegate.get("result") or {}).get("subtask_results")
            if not isinstance(results, list) or len(results) != len(waves[index]):
                raise ValueError("trusted_returned_batch_alignment_invalid")
            rounds.append((index + 2, delegate))
        rounds = select_feedback_rounds(
            rounds, limit=branches_per_scene,
            strategy=round_sampling_strategy,
            seed=seed + scene_index * 1009, scene=scene,
        )
        if not rounds:
            no_rounds += 1
        for decision_round, previous_delegate in rounds:
            plan = {
                "index": len(planned),
                "unit": "main_feedback_round",
                "sample_id": (
                    scene["task_id"] + f"-fork-round-{decision_round:02d}"
                ),
                "scene_index": scene_index,
                "task_id": scene["task_id"],
                "branch_round": decision_round,
                "received_from_delegate_round": decision_round - 1,
                "incoming_task_ids": list(waves[decision_round - 2]),
                "incoming_subtask_count": len(
                    previous_delegate["result"]["subtask_results"]
                ),
                "fault_mode": None,
                "faults": [],
                "skip_reason": None,
            }
            plan["scenario_plan"] = make_scenario(scene, plan)
            planned.append(plan)
    return planned, no_base, no_rounds


def generate_turns(
    *, scenes_path: Path, reports_path: Path, output_dir: Path,
    max_scenes: int | None = 30, branches_per_scene: int | None = None,
    decision_unit: str = "single_turn",
    round_sampling_strategy: str = "chronological",
    seed: int = 17,
    max_rounds: int = 16, max_cost: int = 80, max_retries: int = 1,
    max_parallel_tasks: int = 4, workers: int = 4,
    atomic_fault_kinds: list[str] | None = None,
    max_candidates_per_round: int | None = None,
    max_distractors_per_fault: int | None = None,
    max_selected_per_round: int | None = None,
    target_main_turns: int | None = None,
    max_revisions: int = 2,
    report_max_revisions: int | None = None,
    main_max_revisions: int | None = None,
    sub_model: str = "qwen3-8b",
    model_alias: str = "model_1", overwrite: bool = False,
    no_progress: bool = False,
    author=None, critic=None, report_author=None, report_critic=None,
    author_model: str | None = None, critic_model: str | None = None,
    report_author_model: str | None = None,
    report_critic_model: str | None = None,
    clean_episodes_path: Path | None = None,
    main_mode: str | None = None,
    report_mode: str | None = None,
) -> dict:
    paths = tuple(map(Path, (scenes_path, reports_path)))
    if clean_episodes_path is not None:
        paths += (Path(clean_episodes_path),)
    output = Path(output_dir)
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("stage4_requires_stage1_scenes_and_stage2_reports")
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("untrusted_workers_invalid")
    if branches_per_scene is not None and (
        type(branches_per_scene) is not int or branches_per_scene < 1
    ):
        raise ValueError("untrusted_invalid_round_debug_cap")
    if round_sampling_strategy not in {
        "chronological", "stratified", "evidence_aware",
    }:
        raise ValueError("invalid_feedback_round_sampling_strategy")
    if bool(author) != bool(critic) or bool(report_author) != bool(report_critic):
        raise ValueError("untrusted_author_and_critic_must_be_paired")
    if main_mode is None:
        main_mode = "llm" if author is not None else "template"
    if report_mode is None:
        report_mode = "llm" if report_author is not None else "template"
    if (main_mode not in {"template", "llm"}
            or report_mode not in {"template", "llm"}):
        raise ValueError("untrusted_invalid_stage_llm_mode")
    if ((main_mode == "llm") != (author is not None)
            or (report_mode == "llm") != (report_author is not None)):
        raise ValueError("untrusted_stage_mode_requires_matching_clients")
    if report_max_revisions is None:
        report_max_revisions = max_revisions
    if main_max_revisions is None:
        main_max_revisions = max_revisions
    if output.resolve() in {
        path.resolve() for path in paths
    } or any(output.resolve() == p.parent.resolve() for p in paths):
        raise ValueError("untrusted_output_overwrites_input")

    if decision_unit != "single_turn":
        raise ValueError("untrusted_invalid_decision_unit")
    for name, value, maximum in (
        ("max_candidates_per_round", max_candidates_per_round, 128),
        ("max_distractors_per_fault", max_distractors_per_fault, 10),
        ("max_selected_per_round", max_selected_per_round, 8),
    ):
        if value is not None and (
            type(value) is not int or not 1 <= value <= maximum
        ):
            raise ValueError("invalid_" + name)
    if target_main_turns is not None and (
        type(target_main_turns) is not int or target_main_turns < 1
    ):
        raise ValueError("invalid_global_main_turn_budget")
    kinds = list(atomic_fault_kinds or [
        "no_result", "incomplete_retrieval", "done_but_incomplete",
        "wrong_retrieval", "wrong_entity_binding", "computation_error",
        "report_missing",
        "result_summary_conflict", "fabricated_claim",
    ])
    if (not kinds or any(kind not in ATOM_TO_MODE for kind in kinds)
            or len(kinds) != len(set(kinds))):
        raise ValueError("invalid_atomic_fault_kinds")
    settings = {
        "decision_unit": decision_unit,
        "max_scenes": max_scenes,
        "branches_per_scene": branches_per_scene,
        "round_sampling_strategy": round_sampling_strategy,
        "seed": seed,
        "scenario_mode": "event_driven",
        "atomic_fault_kinds": kinds,
        "candidate_strategy": "enumerate",
        "max_candidates_per_round": max_candidates_per_round,
        "max_distractors_per_fault": max_distractors_per_fault,
        "max_selected_per_round": max_selected_per_round,
        "target_main_turns": target_main_turns,
        "max_rounds": max_rounds, "max_cost": max_cost,
        "max_retries": max_retries, "max_parallel_tasks": max_parallel_tasks,
        "max_revisions": max_revisions,
        "report_max_revisions": report_max_revisions,
        "main_max_revisions": main_max_revisions,
        "sub_model": sub_model,
        "model_alias": model_alias, "author_model": author_model,
        "critic_model": critic_model,
        "report_author_model": report_author_model,
        "report_critic_model": report_critic_model,
        "main_mode": main_mode,
        "report_mode": report_mode,
        "clean_reference_mode": (
            "reviewed_stage3" if clean_episodes_path is not None
            else "deterministic_stage4"
        ),
    }
    signature = _signature(paths, settings)
    scenes = _jsonl(paths[0])[:max_scenes]
    stage2_rows = _jsonl(paths[1])
    rows_by_scene = defaultdict(list)
    for row in stage2_rows:
        rows_by_scene[row["scene_task_id"]].append(row)
    # The deterministic bootstrap checks every candidate against the
    # frozen Stage-1/2 inputs. When explicitly supplied, use the actual
    # reviewed Stage-3 history ONLY after validating identical task waves
    # and delivered Finish Results.
    external_clean = None
    if clean_episodes_path is not None:
        external_rows = _jsonl(paths[2])
        external_clean = {row["task_id"]: row for row in external_rows}
        if len(external_clean) != len(external_rows):
            raise ValueError("stage4_duplicate_external_clean_reference")
    base, prefix_reviews = compile_trusted_prefixes(
        scenes, stage2_rows, max_rounds=max_rounds,
        sub_model=sub_model, model_alias=model_alias,
        clean_episodes=external_clean,
    )
    planned, no_base, no_eligible = _plan_branches(
        scenes, base, branches_per_scene=branches_per_scene,
        round_sampling_strategy=round_sampling_strategy, seed=seed,
    )
    base_feedback_round_count = len(planned)
    planned = expand_all_rounds(
        scenes, planned,
        atomic_fault_kinds=kinds,
        max_candidates_per_round=max_candidates_per_round,
        max_distractors_per_fault=max_distractors_per_fault,
        seed=seed, base=base,
    )
    # Only current feedback can be altered; all earlier reports, dispatches,
    # and source provenance are frozen.
    if any(
        item["scenario_plan"].get("same_source_recheck_plan")
        or item["scenario_plan"].get("history_first_plan")
        or item["scenario_plan"].get("evidence_events")
        for item in planned
    ):
        raise ValueError("single_turn_must_preserve_frozen_stage3_history")
    if overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding="utf-8")) != signature:
            raise ValueError("untrusted_resume_input_or_code_signature_mismatch")
    elif any(output.iterdir()):
        raise ValueError("untrusted_existing_output_without_manifest")
    else:
        _atomic_json(manifest, signature)
    # Baseline Macro/Source verification affordances are diagnostic only.
    # Do not reclassify complementary evidence as an independent verifier
    # or read hidden World rows into Main decisions.
    affordance_rows = [
        analyze_scene_evidence_affordances(scene)
        for scene in scenes if scene["task_id"] in base
    ]
    _atomic_jsonl(
        output / "private" / "scene_evidence_affordances.jsonl",
        affordance_rows,
    )
    _atomic_jsonl(
        output / "private" / "trusted_prefix_checks.jsonl",
        prefix_reviews,
    )
    _atomic_jsonl(
        output / "private" / "clean_references.jsonl",
        [base[scene["task_id"]] for scene in scenes
         if scene["task_id"] in base],
    )
    # Explicit, inspectable round index; unsupported feedback rounds are
    # retained instead of disappearing silently from the total.
    # Two explicit levels of indexing: baseline feedback rounds and
    # the expanded one-fault trial ledger. Do not call 2,000 trials
    # "2,000 Main rounds" in a downstream data analysis.
    feedback_groups = {}
    for plan in planned:
        key = (plan["task_id"], plan["branch_round"])
        if key not in feedback_groups:
            feedback_groups[key] = {
                "question_id": plan["task_id"],
                "main_decision_round": plan["branch_round"],
                "received_from_delegate_round":
                    plan["received_from_delegate_round"],
                "latest_subagent_task_ids": plan["incoming_task_ids"],
                "candidate_count": 0,
                "candidate_trial_ids": [],
                "candidates_pruned_by_cap":
                    plan.get("candidate_count_pruned", 0),
            }
        feedback_groups[key]["candidate_count"] += 1
        feedback_groups[key]["candidate_trial_ids"].append(
            plan["sample_id"]
        )
    _atomic_jsonl(
        output / "feedback_round_index.jsonl",
        list(feedback_groups.values()),
    )
    _atomic_jsonl(output / "round_index.jsonl", [{
        "round_sample_id": plan["sample_id"],
        "question_id": plan["task_id"],
        "main_decision_round": plan["branch_round"],
        "received_from_delegate_round":
            plan["received_from_delegate_round"],
        "latest_subagent_task_ids": plan["incoming_task_ids"],
        "latest_subagent_count": plan["incoming_subtask_count"],
        "fault_eligible": not bool(plan["skip_reason"]),
        "skip_reason": plan["skip_reason"],
        "scenario_mode": plan["scenario_plan"]["mode"],
        "scenario_blueprint": plan["scenario_plan"]["blueprint"],
        "candidate_variant_index": plan.get("candidate_variant_index"),
        "candidate_count_total": plan.get("candidate_count_total"),
        "candidate_count_pruned": plan.get("candidate_count_pruned"),
        "atomic_fault_kinds": [
            a["kind"] for a in plan["scenario_plan"]["fault_atoms"]
        ],
        "scheduled_evidence_events": len(
            plan["scenario_plan"]["evidence_events"]
        ),
    } for plan in planned])

    # Three genuine phases with separate signed, resumable artifacts.
    # 4A: zero model calls; Global execution + Main-visible constrained
    #     evidence requests and ordered task attempts.
    # 4B: only Sub-Agent Author/Guard/Critic calls.
    # 4C: only Main Author/Guard/Critic calls, based on FINAL 4B reports.
    stage_a = output / "stage4a" / "journal"
    stage_b = output / "stage4b" / "journal"
    journal = output / "journal"
    for directory in (stage_a, stage_b, journal):
        directory.mkdir(parents=True, exist_ok=True)

    def saved(directory: Path, plan: dict) -> dict | None:
        path = directory / f"{plan['index']:06d}.json"
        return (json.loads(path.read_text(encoding="utf-8"))
                if path.is_file() else None)

    def write(directory: Path, plan: dict, data: dict) -> None:
        _atomic_json(directory / f"{plan['index']:06d}.json", data)

    def checkpoint_directory(directory: Path, plan: dict) -> Path:
        return directory / f"{plan['index']:06d}.rounds"

    def load_checkpoints(directory: Path, plan: dict,
                         phase: str) -> list[dict]:
        folder = checkpoint_directory(directory, plan)
        if not folder.exists():
            return []
        if not folder.is_dir():
            raise RuntimeError("stage4_checkpoint_path_not_directory")
        checkpoints = []
        for ordinal, file in enumerate(sorted(folder.glob("round-*.json")), 1):
            if file.name != f"round-{ordinal:04d}.json":
                raise RuntimeError("stage4_checkpoint_noncontiguous_sequence")
            item = json.loads(file.read_text(encoding="utf-8"))
            if (item.get("ordinal") != ordinal
                    or item.get("round_sample_id") != plan["sample_id"]
                    or item.get("phase") != phase):
                raise RuntimeError("stage4_checkpoint_candidate_mismatch")
            checkpoints.append(item)
        return checkpoints

    def save_checkpoint(directory: Path, plan: dict,
                        phase: str, checkpoint: dict) -> None:
        folder = checkpoint_directory(directory, plan)
        ordinal = checkpoint["ordinal"]
        file = folder / f"round-{ordinal:04d}.json"
        if file.exists():
            raise RuntimeError("stage4_checkpoint_would_overwrite")
        if ordinal > 1 and not (
            folder / f"round-{ordinal-1:04d}.json"
        ).exists():
            raise RuntimeError("stage4_checkpoint_noncontiguous_write")
        _atomic_json(file, {
            **checkpoint, "round_sample_id": plan["sample_id"],
            "phase": phase,
        })

    def as_failure(scene: dict, exc: Exception, *,
                   source: dict | None = None,
                   phase: str | None = None) -> dict:
        status = ("skipped_unsupported" if isinstance(exc, FaultUnsupported)
                  else "rejected_validation")
        base = source or {"task_id": scene["task_id"]}
        # Stage 4C exceptions must NEVER erase the already-replayed 4A
        # outcome or the actual 4B report reviews. Only the publication
        # attempt failed. A failed 4B attempt also retains 4A, but must
        # not present deterministic 4A previews as LLM-reviewed reports.
        audit = deepcopy(base.get("private_audit"))
        if isinstance(audit, dict):
            audit["language_exception"] = {
                "phase": phase, "kind": type(exc).__name__,
                "message": str(exc)[:300],
            }
            audit["outcome"] = status
            audit["language_review_accepted"] = False
            audit["accepted_turns"] = 0
            audit["candidate_local_turns"] = []
            audit["main_action_reviews"] = []
            if phase == "stage4b":
                audit["subagent_report_reviews"] = []
        return {
            **base, "task_id": scene["task_id"],
            "status": status, "error": str(exc)[:300],
            "episode": None, "turns": [],
            "private_audit": audit,
        }

    def _progress_bucket(data: dict) -> str:
        """Classify only the current candidate's next-decision viability."""
        status = data.get("status", "rejected_validation")
        if status == "single_turn_planned":
            return "actionable"
        if status == "no_current_visible_issue":
            return "no_public_witness"
        if status in {"skipped_unsupported", "rejected_validation"}:
            return "skipped"
        return "other"

        def _review_progress(counts: Counter, result: dict) -> None:
        status = result.get("status", "failed")
        if status == "accepted":
            counts["accepted"] += 1
        elif status == "fallback":
            counts["fallback"] += 1
        else:
            counts["failed"] += 1
        # A completion fallback can carry status=accepted, unlike report
        # fallback. Keep it visible as a separate publication-quality flag.
        if ("fallback" in str(result.get("mode", ""))
                and status != "fallback"):
            counts["fallback"] += 1

    def run_parallel(phase: str, directory: Path, unit: str,
                     total: int, initial: int, callback_worker,
                     *, stage: str, subset: list[dict] | None = None):
        """Concurrent independent round samples, truthful live counters.

        Per-review callbacks update the *completed-call* bar immediately.
        'written' changes only after a full round journal is atomically
        committed; it is not a misleading per-review persistence claim.
        Resume reconstructs both kinds of counts from existing journals.
        """
        if stage not in {"simulation", "reports", "main"}:
            raise ValueError("unknown_stage4_progress_kind")
        review_key = {
            "reports": "subagent_report_reviews",
            "main": "main_action_reviews",
        }.get(stage)
        units = planned if subset is None else subset
        pending = [p for p in units if saved(directory, p) is None]
        counts = Counter()
        for plan in units:
            existing = saved(directory, plan)
            if existing is None:
                continue
            counts["saved_rounds"] += 1
            if stage == "simulation":
                counts[_progress_bucket(existing)] += 1
                counts["written"] += 1
            else:
                reviews = ((existing.get("private_audit") or {})
                           .get(review_key, []))
                for review in reviews:
                    _review_progress(counts, review.get("authoring") or {})
                counts["written"] += len(reviews)
                if stage == "main":
                    counts["processed"] += len(reviews)
                    remaining = max(0, action_count(saved(stage_b, plan))
                                    - len(reviews))
                    counts["skipped_downstream"] += remaining
        with tqdm(
            total=max(total, initial), initial=initial,
            desc=phase, unit=unit, dynamic_ncols=True,
            disable=no_progress, leave=True,
        ) as bar:
            lock = Lock()

            def refresh() -> None:
                if stage == "simulation":
                    display = {
                        "actionable": counts["actionable"],
                        "no_public_witness": counts["no_public_witness"],
                        "other": counts["other"],
                        "skipped": counts["skipped"],
                        "written": counts["written"],
                    }
                elif stage == "reports":
                    display = {
                        "accepted": counts["accepted"],
                        "fallback": counts["fallback"],
                        "failed": counts["failed"],
                        "written": counts["written"],
                    }
                else:
                    display = {
                        "processed": counts["processed"],
                        "planned": total,
                        "accepted": counts["accepted"],
                        "failed": counts["failed"],
                        "skipped_downstream":
                            counts["skipped_downstream"],
                        "written": counts["written"],
                    }
                bar.set_postfix(display, refresh=False)

            def tick(status: dict | None = None) -> None:
                with lock:
                    if stage in {"reports", "main"} and status is not None:
                        _review_progress(counts, status)
                        if stage == "main":
                            counts["processed"] += 1
                    if bar.total is not None and bar.n + 1 > bar.total:
                        bar.total = bar.n + 1
                    refresh()
                    bar.update(1)

            def commit_round(plan: dict, *, process_status: str | None = None) -> None:
                # 4A processes return a committed status; avoid rereading
                # large private audit JSON just to update a progress bar.
                current = ({"status": process_status}
                           if process_status is not None
                           else saved(directory, plan))
                if current is None:
                    raise RuntimeError("stage4_worker_did_not_write_journal")
                with lock:
                    counts["saved_rounds"] += 1
                    if stage == "simulation":
                        counts[_progress_bucket(current)] += 1
                        counts["written"] += 1
                    else:
                        review_count = len(
                            ((current.get("private_audit") or {})
                             .get(review_key, []))
                        )
                        counts["written"] += review_count
                        if stage == "main":
                            remaining = max(
                                0, action_count(saved(stage_b, plan))
                                - review_count
                            )
                            counts["skipped_downstream"] += remaining
                            # Mark planned but not attempted actions as
                            # accounted-for after the round was persisted.
                            # This is *not* a reviewed/accepted action.
                            if remaining:
                                bar.update(remaining)
                    refresh()
                    bar.refresh()

            def run_one(plan: dict) -> None:
                callback_worker(plan, tick)
                commit_round(plan)

            refresh()
            if stage == "simulation":
                # Bound workers to actual available CPU quota (Py3.13
                # process_cpu_count is cgroup-aware). Many more *threads*
                # did not help because 4A replay is Python-CPU-bound.
                cpu_count = getattr(
                    os, "process_cpu_count", os.cpu_count
                )() or 1
                process_count = min(workers, cpu_count, max(1, len(pending)))
                if pending:
                    with ProcessPoolExecutor(
                        max_workers=process_count,
                        initializer=_init_evidence_worker,
                        initargs=(evidence_context,),
                    ) as pool:
                        futures = {
                            pool.submit(_execute_evidence_candidate, plan): plan
                            for plan in pending
                        }
                        try:
                            for future in as_completed(futures):
                                commit_round(
                                    futures[future],
                                    process_status=future.result(),
                                )
                                tick()
                        except BaseException:
                            # Fail fast if an unexpected programming error
                            # reaches the coordinator. Journals already
                            # atomically written remain resumable; cancel
                            # tasks still queued rather than replaying them.
                            for future in futures:
                                future.cancel()
                            raise
            else:
                with ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="stage4-" + phase[:2],
                ) as pool:
                    futures = [
                        pool.submit(run_one, plan) for plan in pending
                    ]
                    for future in as_completed(futures):
                        future.result()
            # Main progress includes explicitly skipped downstream
            # actions, so its denominator stays at the 4A plan count.
            # Report counts can vary after a failed sample; allow that
            # bar to settle on the actual reviewed-report total.
            with lock:
                if stage != "main":
                    bar.total = bar.n
                refresh()
                bar.refresh()

    # Strict Stage-2 scene validation used to run once PER candidate.
    # It is invariant under atomic report faults and is now computed once
    # for each accepted scene. Stage 4B/4C remain unchanged.
    validated_references = {}
    for scene in scenes:
        tid = scene["task_id"]
        if tid not in base:
            continue
        reviewed = _strict_reports(scene, rows_by_scene[tid])
        validated_references[tid] = (
            reviewed, clean_reference_identity(scene, reviewed),
        )
    evidence_context = {
        "scenes": scenes,
        "reports": dict(rows_by_scene),
        "base": base,
        "validated": validated_references,
        "journal_dir": str(stage_a),
        "build_options": {
            "max_rounds": max_rounds,
            "max_cost": max_cost,
            "max_retries": max_retries,
            "max_parallel_tasks": max_parallel_tasks,
            "sub_model": sub_model, "model_alias": model_alias,
            "author": None, "critic": None,
            "report_author": None, "report_critic": None,
        },
    }
    run_parallel(
        "Evidence simulation", stage_a, "round",
        len(planned), sum(saved(stage_a, p) is not None for p in planned),
        None, stage="simulation",
    )

    # The 4A search is intentionally wider than the expensive 4B/4C
    # publication candidates. Keep ALL executions in stage4a/journal
    # and an explicit private selection ledger, including failures.
    selected_indices, selection_rows, selection_stats = (
        select_single_turn_candidates(
            planned, [saved(stage_a, plan) for plan in planned],
            target_main_turns=target_main_turns,
            max_selected_per_round=max_selected_per_round,
            seed=seed,
        )
    )
    def report_count(data: dict) -> int:
        audit = (data or {}).get("private_audit") or {}
        return len(audit.get("subagent_report_reviews", []))

    def action_count(data: dict) -> int:
        if data is None or data.get("episode") is None:
            return 0
        audit = data["private_audit"]
        fork = audit["branch_round"]
        return sum(
            row["attempt"] >= fork
            and row["action"] in {"delegate_task", "complete"}
            for row in data["episode"]["trajectory"]
        )

    # Keep viable 4A candidates as reserves if a 4B/4C review fails.
    # Backfilling never executes a future Sub-Agent or replays a fault.
    by_index = {plan["index"]: plan for plan in planned}
    def make_reports(plan: dict, tick) -> None:
        source = saved(stage_a, plan)
        scene = scenes[plan["scene_index"]]
        if source["episode"] is None:
            write(stage_b, plan, source)
            return
        try:
            existing = load_checkpoints(stage_b, plan, "stage4b")
            episode, audit = author_branch_reports(
                scene, rows_by_scene[scene["task_id"]],
                source["episode"], source["private_audit"],
                author=report_author, critic=report_critic,
                max_revisions=report_max_revisions,
                resume_reports=existing,
                on_report_checkpoint=lambda row: save_checkpoint(
                    stage_b, plan, "stage4b", row,
                ),
                on_report=lambda status: tick(status),
            )
            data = {
                **source, "episode": episode, "private_audit": audit,
            }
        except (ValueError, KeyError, TypeError) as exc:
            data = as_failure(
                scene, exc, source=source, phase="stage4b",
            )
        write(stage_b, plan, data)

    def make_main(plan: dict, tick) -> None:
        source = saved(stage_b, plan)
        scene = scenes[plan["scene_index"]]
        if source["episode"] is None:
            write(journal, plan, source)
            return
        try:
            existing = load_checkpoints(journal, plan, "stage4c")
            episode, turns, audit = author_branch_main(
                scene, source["episode"], source["private_audit"],
                author=author, critic=critic,
                max_revisions=main_max_revisions,
                max_rounds=max_rounds, sub_model=sub_model,
                model_alias=model_alias, resume_actions=existing,
                on_main_checkpoint=lambda row: save_checkpoint(
                    journal, plan, "stage4c", row,
                ),
                on_main=lambda status: tick(status),
            )
            data = {
                **source, "status": audit["outcome"],
                "episode": episode, "turns": turns,
                "private_audit": audit,
            }
        except (ValueError, KeyError, TypeError) as exc:
            data = as_failure(
                scene, exc, source=source, phase="stage4c",
            )
        write(journal, plan, data)

    def published_decision(index: int) -> bool:
        current = saved(journal, by_index[index])
        return (
            current is not None
            and current.get("status") == "single_turn_accepted"
            and len(current.get("turns", [])) == 1
            and not any(review_turn_surface(turn) for turn in current["turns"])
        )

    # Review selected single-turn candidates and backfill failed reviews
    # with distinct, already executed 4A public-witness candidates.
    active = set(selected_indices)
    attempted = set()
    initial_selection = set(active)
    backfilled_groups = set()
    total_actions = 0
    while active:
        batch_plans = [by_index[i] for i in sorted(active)]
        total_reports = sum(
            report_count(saved(stage_a, plan)) for plan in batch_plans
        )
        initial_reports = sum(
            report_count(saved(stage_b, plan))
            for plan in batch_plans if saved(stage_b, plan) is not None
        )
        run_parallel(
            "Sub-Agent feedback", stage_b, "report",
            total_reports, initial_reports, make_reports,
            stage="reports", subset=batch_plans,
        )
        wave_actions = sum(
            action_count(saved(stage_b, plan)) for plan in batch_plans
        )
        total_actions += wave_actions
        initial_actions = sum(
            action_count(saved(stage_b, plan))
            for plan in batch_plans if saved(journal, plan) is not None
        )
        run_parallel(
            "Main orchestration", journal, "action",
            wave_actions, initial_actions, make_main,
            stage="main", subset=batch_plans,
        )
        attempted.update(active)
        next_active = set()
        successful = sum(
            published_decision(i) for i in attempted
        )
        remaining = (max(0, target_main_turns - successful)
                     if target_main_turns is not None else 0)
        if remaining <= 0:
            active = set()
            continue
        # Backfill within the same observable scenario and policy first;
        # fault mechanisms are implementation details, not scene classes.
        failed_keys = {
            (row["question_id"], row["feedback_round"],
             row["realized_scenario"], row["coordination_pattern"])
            for row in selection_rows
            if row["candidate_index"] in attempted
            and row.get("selection_purpose") == "positive"
            and not published_decision(row["candidate_index"])
        }
        pools = defaultdict(list)
        for row in selection_rows:
            if (row.get("recovery_rank") is not None
                    and row["candidate_index"] not in attempted):
                key = (
                    row["question_id"], row["feedback_round"],
                    row["realized_scenario"], row["coordination_pattern"],
                )
                pools[key].append(row)
        for pool in pools.values():
            pool.sort(key=lambda row: row["recovery_rank"])
        accepted_by_round = Counter(
            (row["question_id"], row["feedback_round"])
            for row in selection_rows
            if row["candidate_index"] in attempted
            and published_decision(row["candidate_index"])
        )
        keys = sorted(pools, key=lambda key: (
            key not in failed_keys, accepted_by_round[key[:2]], key
        ))
        next_active = set()
        while remaining and any(pools.values()):
            made_progress = False
            for key in keys:
                if remaining == 0:
                    break
                if (max_selected_per_round is not None
                        and accepted_by_round[key[:2]] >= max_selected_per_round):
                    continue
                if not pools[key]:
                    continue
                row = pools[key].pop(0)
                next_active.add(row["candidate_index"])
                accepted_by_round[key[:2]] += 1
                backfilled_groups.add(key[:2])
                remaining -= 1
                made_progress = True
            if not made_progress:
                break
        active = next_active
    
    # Preserve one journal slot per 4A candidate for resumable alignment,
    # while keeping skipped candidates private and LLM-free.
    for plan in planned:
        if plan["index"] in attempted:
            continue
        source = saved(stage_a, plan)
        placeholder = {
            **source, "status": "not_selected_for_language_review",
            "error": "4a_candidate_screened_out",
            "episode": None, "turns": [],
        }
        write(stage_b, plan, placeholder)
        write(journal, plan, placeholder)

    for row in selection_rows:
        index = row["candidate_index"]
        row["selected_for_language_review"] = index in attempted
        # Preserve 4A screening separately from 4B/4C review selection.
        row["language_review_selection_reason"] = (
            "language_review_primary" if index in initial_selection
            else "language_review_backfill_after_rejection"
            if index in attempted
            else "not_selected_for_language_review"
        )
        if index in attempted:
            row["language_review_outcome"] = (
                saved(journal, by_index[index]) or {}
            ).get("status")
    selection_stats["candidate_screening_outcomes"] = dict(sorted(
        Counter(
            "selected" if row["selected_for_language_review"]
            else "no_public_witness" if row["realized_scenario"] is None
            else "target_mismatch"
            if row["selection_reason"] == "target_scenario_not_publicly_realized"
            else "no_legal_action"
            if row["selection_reason"] == "no_legal_current_decision_or_no_visible_issue"
            else "duplicate"
            if row["selection_reason"] == "duplicate_public_issue_and_action"
            else "budget_reserve" if row["recovery_rank"] is not None
            else "other_unselected"
            for row in selection_rows
        ).items()
    ))
    selection_stats["target_realization_distribution"] = dict(sorted(
        Counter(
            f"{row.get('target_scenario') or 'none'}->"
            f"{row.get('realized_scenario') or 'none'}"
            for row in selection_rows
        ).items()
    ))
    selection_stats["no_public_witness_by_fault_kind"] = dict(sorted(
        Counter(
            row["atom_kind"] for row in selection_rows
            if row["realized_scenario"] is None
        ).items()
    ))
    selection_stats["budget_reserve_by_scenario"] = dict(sorted(
        Counter(
            row["realized_scenario"] for row in selection_rows
            if not row["selected_for_language_review"]
            and row["recovery_rank"] is not None
        ).items()
    ))
    selection_stats["candidates_selected_for_language_review"] = len(attempted)
    selection_stats["candidates_filtered_before_llm"] = (
        len(planned) - len(attempted)
    )
    selection_stats["selected_behavior_distribution"] = dict(Counter(
        row["category"] for row in selection_rows
        if row["selected_for_language_review"]
    ))
    selection_stats["feedback_rounds_with_selected_candidates"] = len({
        (row["question_id"], row["feedback_round"])
        for row in selection_rows if row["selected_for_language_review"]
    })
    selection_stats["candidate_language_backfill_attempts"] = len(
        attempted - initial_selection
    )
    selection_stats["feedback_rounds_backfilled"] = len(backfilled_groups)
    _atomic_jsonl(
        output / "private" / "candidate_selection.jsonl",
        selection_rows,
    )
    counts_summary = _publish(output, planned, base)
    published_rows = [
        row for row in selection_rows
        if row.get("candidate_index") in attempted
        and published_decision(row["candidate_index"])
    ]
    selection_stats["published_answer_impact_distribution"] = dict(sorted(
        Counter(row.get("answer_impact") or "not_evaluated"
                for row in published_rows).items()
    ))
    selection_stats["published_recovery_effect_distribution"] = dict(sorted(
        Counter(row.get("recovery_effect") or "unknown"
                for row in published_rows).items()
    ))
    selection_stats["published_by_fault_kind"] = dict(sorted(
        Counter(row.get("atom_kind") or "unknown"
                for row in published_rows).items()
    ))
    selection_stats["unfilled_main_turn_budget"] = (
        max(0, target_main_turns - counts_summary["main_training_turns"])
        if target_main_turns is not None else None
    )
    processed_actions = sum(
        len(((saved(journal, by_index[i]) or {}).get("private_audit")
             or {}).get("main_action_reviews", []))
        for i in attempted
    )
    summary = {
        # "status" describes whether all requested round units were
        # processed. Training-label acceptance is a separate measurement:
        # an incorrect/unresolved fault scenario is still a FINISHED
        # simulation, not an unfinished generator job.
        "status": (
            "done" if counts_summary["completed_branches"] == len(planned)
            else "partial"
        ),
        "publication_status": "filtered_by_visibility_or_language_review",
        "decision_unit": decision_unit,
        "pipeline": "maestro_untrusted_orchestration",
        "scenario_mode": "event_driven",
        "candidate_strategy": "enumerate",
        "selection_strategy": "scenario_first",
        "round_sampling_strategy": round_sampling_strategy,
        "max_candidates_per_feedback_round": max_candidates_per_round,
        "max_distractors_per_fault": max_distractors_per_fault,
        "max_selected_per_feedback_round": max_selected_per_round,
        "candidate_selection_diagnostics": str(
            output / "private" / "candidate_selection.jsonl"
        ),
        "evidence_affordance_diagnostics": str(
            output / "private" / "scene_evidence_affordances.jsonl"
        ),
        "potential_cross_check_macro_pairs": sum(
            x["potential_cross_check_pairs"] for x in affordance_rows
        ),
        "complementary_fusion_macro_pairs": sum(
            x["complementary_fusion_pairs"] for x in affordance_rows
        ),
        "internal_multi_source_macros": sum(
            x["internal_multi_source_macros"] for x in affordance_rows
        ),
        "internal_verification_macros": sum(
            x["internal_verification_macros"] for x in affordance_rows
        ),
        "internal_multi_source_supported_outputs": sum(
            x["internal_multi_source_supported_outputs"]
            for x in affordance_rows
        ),
        "provisional_downstream_late_witness_pairs": sum(
            x["provisional_downstream_before_late_witness_pairs"]
            for x in affordance_rows
        ),
        **selection_stats,
        "scenario_blueprints": dict(Counter(
            p["scenario_plan"]["blueprint"] for p in planned
        )),
        "atomic_fault_kinds_planned": dict(Counter(
            atom["kind"] for p in planned
            for atom in p["scenario_plan"]["fault_atoms"]
        )),
        "evidence_events_scheduled": sum(
            len(p["scenario_plan"]["evidence_events"])
            for p in planned
        ),
        "mode": ("llm" if main_mode == "llm" or report_mode == "llm"
                 else "template"),
        "subagent_report_mode": report_mode,
        "main_action_mode": main_mode,
        "main_actions_planned": total_actions,
        "global_target_main_turns": target_main_turns,
        "global_published_main_turn_gap": (
            max(0, target_main_turns - counts_summary["main_training_turns"])
            if target_main_turns is not None else None
        ),
        "main_actions_processed": processed_actions,
        "main_actions_skipped_downstream": max(
            0, total_actions - processed_actions
        ),
        "input_scenes": len(scenes),
        "trusted_prefixes_compiled": len(base),
        "trusted_prefixes_rejected": len(prefix_reviews) - len(base),
        "trusted_prefix_checks": str(output / "private" / "trusted_prefix_checks.jsonl"),
        "scenes_without_accepted_base": no_base,
        "round_sampling_unit": "question_x_feedback_round_x_atomic_instance",
        "rounds_extracted_from_base": base_feedback_round_count,
        "candidate_trials": len(planned),
        "rounds_with_executable_fault": len({
            (plan["task_id"], plan["branch_round"])
            for plan in planned if not plan["skip_reason"]
        }),
        "rounds_without_executable_fault": sum(
            plan["skip_reason"] == "no_executable_fault_on_returned_batch"
            for plan in planned
        ),
        "questions_with_extracted_rounds": len({
            plan["task_id"] for plan in planned
        }),
        "rounds_per_question": dict(Counter(
            task_id for task_id, _ in sorted({
                (plan["task_id"], plan["branch_round"])
                for plan in planned
            })
        )),
        "round_index": str(output / "round_index.jsonl"),
        "feedback_round_index": str(
            output / "feedback_round_index.jsonl"
        ),
        "scenes_without_eligible_fault": no_eligible,
        **counts_summary,
        "stage4a_evidence_plans": str(output / "stage4a" / "journal"),
        "stage4b_reviewed_reports": str(output / "stage4b" / "journal"),
        "public_turns": str(output / "main_turns.jsonl"),
        "public_episodes": str(output / "episodes.jsonl"),
        "private_fault_alignment": str(
            output / "private" / "fault_alignment.jsonl"
        ),
    }
    _atomic_json(output / "summary.json", summary)
    if not no_progress:
        tqdm.write(
            "Evidence candidate trials: "
            f"{summary['completed_branches']}/{summary['candidate_trials']} "
            f"| feedback rounds={summary['rounds_extracted_from_base']} "
            f"| selected={summary['candidates_selected_for_language_review']} "
            f"| 4A actionable={summary['single_turn_actionable']} "
            f"| accepted decisions={summary['single_turn_accepted']} "
            f"| reviewed turns={summary['main_training_turns']}"
        )
    return summary
