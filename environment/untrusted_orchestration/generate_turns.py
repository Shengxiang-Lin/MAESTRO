"""Event-driven untrusted O1 Main decision generation.

Every prior feedback round is a legal single-atom scenario. Opt-in event
scenarios additionally schedule an executable later fault and/or defer a
not-yet-returned independent evidence task. No historical Main action is
retroactively rewritten; all events are replayed in causal order.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import os
import random
import shutil
from threading import Lock

from tqdm.auto import tqdm

from .faults import (
    Fault, FaultUnsupported, choose_targets, probe_eligible_fault_modes,
)
from .atomic_faults import AtomicFault, probe_atoms, ATOM_TO_MODE
from .scenario_generator import make_scenario
from .trajectory_builder import build_branch
from .offline_pipeline import author_branch_reports, author_branch_main
from .trusted_prefix import compile_trusted_prefixes
from .publication_guard import review_turn_surface
from .candidate_exploration import expand_all_rounds
from .candidate_selection import select_recovered_candidates
from .round_sampling import select_feedback_rounds
from .evidence_affordances import analyze_scene_evidence_affordances


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines()
            if row.strip()]


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)
                         + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _hash_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _signature(inputs: tuple[Path, Path], settings: dict) -> dict:
    directory = Path(__file__).resolve().parent
    code_files = (
        "faults.py", "atomic_faults.py", "report_faults.py",
        "event_scheduler.py", "scenario_generator.py",
        "candidate_exploration.py", "candidate_selection.py",
        "evidence_affordances.py", "reasoning_fidelity.py",
        "visible_history.py", "visible_claim_state.py",
        "visible_claim_extraction.py",
        "main_decision_contract.py",
        "report_contract.py", "report_evidence.py",
        "report_validation.py", "public_language.py",
        "publication_guard.py",
        "state_machine.py", "report_author.py",
        "trusted_prefix.py",
        "trajectory_builder.py", "generate_turns.py",
        "evidence_planning.py", "offline_pipeline.py",
        "evidence_operators.py",
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
        "version": "maestro.untrusted_orchestration.event_engine_v1",
        "input_sha256": [_hash_file(path) for path in inputs],
        "source_code_sha256": {
            name: _hash_file(directory / name) for name in code_files
        },
        "settings": settings,
    }


def _plan_branches(
    scenes: list[dict], base: dict[str, dict],
    *, branches_per_scene: int | None = None, seed: int,
    fault_modes: list[str], persistence: str,
    round_sampling_strategy: str = "chronological",
    multi_target_probability: float,
    max_rounds: int,
    scenario_mode: str = "round_fork",
    atomic_fault_kinds: list[str] | None = None,
    max_fault_atoms: int = 1,
    delayed_evidence_probability: float = 0.0,
) -> tuple[list[dict], int, int]:
    """Expand EVERY accepted Main feedback round into one fault scenario.

    The unit is (question, Main decision round), *not* a randomly sampled
    number of branches per Question. A decision round t receives the latest
    Sub-Agent batch returned from delegate round t-1. Every such available
    round is retained in the plan, even when no admissible fault exists;
    those rounds produce explicit skipped_unsupported journals.

    `branches_per_scene` is an optional *debugging cap* for compatibility
    with prior callers. None means no cap; the production YAML uses None.
    The max_rounds budget also limits the global round positions.
    """
    if branches_per_scene is not None and (
        type(branches_per_scene) is not int or branches_per_scene < 1
    ):
        raise ValueError("invalid_debug_round_cap")
    planned, no_base, no_eligible = [], 0, 0
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
        # Check every trusted returned batch against the actual global DAG
        # rather than silently producing incorrect unit/position labels.
        if len(delegates) > len(waves):
            raise ValueError("trusted_delegate_exceeds_reference_waves")
        rounds = []
        for index, delegate in enumerate(delegates):
            expected = index + 1
            if delegate.get("attempt") != expected:
                raise ValueError("trusted_delegate_round_alignment_invalid")
            results = (delegate.get("result") or {}).get("subtask_results")
            if not isinstance(results, list) or len(results) != len(waves[index]):
                raise ValueError("trusted_returned_batch_alignment_invalid")
            decision_round = expected + 1
            # Preserve all accepted base feedback rounds, including those
            # exceeding this run's configured recovery horizon. Such a
            # round is explicitly accounted for as budget-skipped below.
            rounds.append((decision_round, delegate))
        rounds = select_feedback_rounds(
            rounds, limit=branches_per_scene,
            strategy=round_sampling_strategy,
            seed=seed + scene_index * 1009,
        )
        if not rounds:
            no_eligible += 1
            continue

        # The ordinary latest-feedback fork is expressed as one
        # ScenarioPlan rather than maintained as a separate executor.
        if scenario_mode == "event_driven":
            eligible = probe_atoms(
                scene, atomic_fault_kinds or [
                    "no_result", "incomplete_retrieval",
                    "wrong_retrieval", "wrong_entity_binding",
                    "report_missing", "result_summary_conflict",
                    "fabricated_claim",
                ],
            )
        else:
            eligible = probe_eligible_fault_modes(scene, fault_modes)
        tasks = {task["task_id"]: task
                 for task in scene["macro_graph"]["tasks"]}
        eligible_count = 0
        for decision_round, previous_delegate in rounds:
            wave = waves[decision_round - 2]
            available_modes = [
                kind for kind in (
                    atomic_fault_kinds if scenario_mode == "event_driven"
                    and atomic_fault_kinds is not None else
                    list(ATOM_TO_MODE)[:7] if scenario_mode == "event_driven"
                    else fault_modes
                )
                if any(kind in eligible.get(tid, []) for tid in wave)
            ]
            # Independent deterministic randomness per (question, round)
            # means adding/removing another round cannot perturb this one.
            round_seed = seed + scene_index * 1009 + decision_round * 92821
            rng = random.Random(round_seed)
            faults = []
            skip_reason = (
                "insufficient_recovery_round_budget"
                if decision_round > max_rounds else None
            )
            chosen_mode = None
            if available_modes and skip_reason is None:
                chosen_mode = rng.choice(available_modes)
                supported = [
                    tid for tid in wave if chosen_mode in eligible.get(tid, [])
                ]
                multiple = (
                    rng.random() < multi_target_probability
                    and len(supported) > 1
                    and (scenario_mode != "event_driven"
                         or max_fault_atoms >= 2)
                )
                chosen_targets = choose_targets(
                    supported, tasks, seed=round_seed, multiple=multiple,
                )
                if scenario_mode == "event_driven":
                    faults = [
                        AtomicFault(
                            chosen_mode, tid, decision_round - 1,
                            seed=round_seed,
                            persistence=persistence,
                        ).as_legacy_fault()
                        for tid in chosen_targets
                    ]
                else:
                    faults = [
                        Fault(task_id=tid, mode=chosen_mode,
                              seed=round_seed, persistence=persistence)
                        for tid in chosen_targets
                    ]
                eligible_count += 1
            elif skip_reason is None:
                skip_reason = "no_executable_fault_on_returned_batch"

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
                "incoming_task_ids": list(wave),
                "incoming_subtask_count": len(
                    previous_delegate["result"]["subtask_results"]
                ),
                "fault_mode": chosen_mode,
                "faults": faults,
                "skip_reason": skip_reason,
            }
            plan["scenario_plan"] = make_scenario(
                scene, plan,
                mode=scenario_mode, eligible_atoms=(
                    eligible if scenario_mode == "event_driven" else None
                ),
                max_fault_atoms=max_fault_atoms,
                delayed_evidence_probability=delayed_evidence_probability,
            )
            planned.append(plan)
        if not eligible_count:
            no_eligible += 1
    return planned, no_base, no_eligible

def _publish(output: Path, planned: list[dict],
             base: dict[str, dict] | None = None) -> dict:
    base = base or {}
    journals = output / "journal"
    episodes, positive_turns, audits, failures = [], [], [], []
    counters = Counter()
    mode_outcomes: dict[str, Counter] = defaultdict(Counter)
    report_modes = Counter()
    main_modes = Counter()
    report_statuses = Counter()
    main_statuses = Counter()
    execution_outcomes = Counter()
    report_guard_failures = Counter()
    report_review_failures = Counter()
    executor_normalized_results = 0
    contract_reports_reviewed = 0
    contract_reports_accepted = 0
    contract_results_compiled = 0
    contract_mandatory_facts = 0
    contract_optional_summary_omitted = 0
    witnessed_report_reviews = 0
    main_decision_contract_reviews = 0
    main_decision_contract_accepted = 0
    main_confirmed_visible_conflicts = 0
    main_conflict_critic_rereviews = 0
    witnessed_report_fallbacks = 0
    witness_critic_rereviews = 0
    retained_language_exceptions = 0
    local_candidates, local_qualified, local_rejections = [], [], []
    claim_state_supervision = []
    # Observable evidence labels also exist in genuine non-recovered runs.
    # They are never positive Main-action supervision.
    reviewed_observable_states = []
    decision_reward_states = []
    decision_reward_missing = 0
    claim_state_diagnostics = []
    paired_trajectories = []
    claim_state_statuses = Counter()
    wrong_visibility_rows = []
    wrong_visibility_counts = Counter()
    wrong_visibility_outcomes: dict[str, Counter] = defaultdict(Counter)
    publication_guard_rows = []
    event_outcomes: dict[str, Counter] = defaultdict(Counter)
    atom_outcomes: dict[str, Counter] = defaultdict(Counter)
    event_timeline_rows = []
    invalidations_total = 0
    revalidations_total = 0
    late_conflicts_total = 0
    recovered_after_actual_invalidation = 0
    published_with_downstream_invalidation = 0
    published_with_invalidation_and_visible_conflict = 0
    for plan in planned:
        path = journals / f"{plan['index']:06d}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["task_id"] != plan["task_id"]:
            raise RuntimeError("untrusted_journal_scene_mismatch")
        status = data["status"]
        leaks = [
            {"sample_id": turn.get("sample_id"),
             "issues": review_turn_surface(turn)}
            for turn in data.get("turns", [])
        ]
        leaks = [row for row in leaks if row["issues"]]
        if status == "recovered" and leaks:
            status = "publication_generation_process_leak"
            publication_guard_rows.append({
                "branch_index": plan["index"],
                "round_sample_id": plan["sample_id"],
                "rejected_turns": leaks,
            })
        audit_pair = data.get("private_audit") or {}
        before_pair = audit_pair.get("counterfactual_unrepaired") or {}
        reference = base.get(plan["task_id"], {}).get("clean_reference_id")
        recorded = (audit_pair.get("clean_reference_identity") or {}).get(
            "clean_reference_id"
        )
        if base and recorded is not None and recorded != reference:
            raise RuntimeError("stage4_paired_clean_reference_drift")
        paired_trajectories.append({
            "scene_task_id": plan["task_id"],
            "round_sample_id": plan["sample_id"],
            "clean_reference_id": reference,
            "fork_round": plan["branch_round"],
            "fault_kinds_private": [
                atom["kind"] for atom in plan["scenario_plan"]["fault_atoms"]
            ],
            "naive_execution_status": before_pair.get("status"),
            "naive_answer_correct": before_pair.get("correct"),
            "robust_publication_status": status,
            "private_only": True,
        })
        counters[status] += 1
        for mode in {fault.mode for fault in plan["faults"]}:
            mode_outcomes[mode][status] += 1
        event_outcomes[plan["scenario_plan"]["blueprint"]][status] += 1
        for atom in plan["scenario_plan"]["fault_atoms"]:
            atom_outcomes[atom["kind"]][status] += 1
        if status == "recovered" and (
            not isinstance(data.get("private_audit"), dict)
            or data["private_audit"].get("parallel_batches_verified") is not True
        ):
            raise RuntimeError(
                "untrusted_unsafe_parallel_batch_cannot_publish_positive_sft"
            )
        # A private branch journal may retain an Episode even after its
        # final language/action review failed. Such Episodes are never
        # positive training data. Apply the SAME acceptance predicate to
        # public episodes and public Main turns.
        if status == "recovered":
            if not isinstance(data.get("episode"), dict):
                raise RuntimeError(
                    "untrusted_recovered_branch_missing_public_episode"
                )
            public_episode = deepcopy(data["episode"])
            # Truth labels belong in private audit, never public reviewed-turn.
            public_episode.pop("outcome", None)
            public_episode.pop("recovered", None)
            # Stable *candidate identity*, not a fault label. Stage 4 may
            # publish multiple distinct fault candidates for one fork round.
            # It is dataset metadata only, never included in the Main prompt.
            public_episode["round_sample_id"] = plan["sample_id"]
            episodes.append(public_episode)
            private_states = (
                data["private_audit"].get("decision_reward_states") or []
            )
            state_by_round = {
                row["decision_round"]: row for row in private_states
            }
            if len(state_by_round) != len(private_states):
                raise RuntimeError("duplicate private decision reward round")
            for turn in data["turns"]:
                positive_turns.append({
                    **turn, "round_sample_id": plan["sample_id"]
                })
                snapshot = state_by_round.get(turn["round"])
                if snapshot is None:
                    # Legacy journals have no pre-action snapshot: never
                    # invent a choice set from the teacher action alone.
                    decision_reward_missing += 1
                else:
                    decision_reward_states.append({
                        "task_id": data["task_id"],
                        "round_sample_id": plan["sample_id"],
                        "decision_round": turn["round"],
                        "state": deepcopy(snapshot),
                        "private_only": True,
                    })
        elif status != "not_selected_for_language_review":
            failures.append({
                "task_id": data["task_id"],
                "branch_index": plan["index"],
                "branch_round": plan["branch_round"],
                "status": status,
                "error": data.get("error") or (
                    "generation_process_leak" if leaks else None
                ),
            })
        if data.get("private_audit") is not None:
            audit = data["private_audit"]
            audits.append(audit)
            # Keep every observational state in private diagnostics, but
            # expose the standalone training-side state labels ONLY when
            # this exact Episode passed global recovery and review.
            claim_audit = audit.get("visible_claim_state") or {}
            for snapshot in claim_audit.get("rounds", []):
                state_label = {
                    "round_sample_id": plan["sample_id"],
                    "task_id": data["task_id"],
                    "main_can_act_from_round":
                        snapshot["main_can_act_from_round"],
                    "schema": claim_audit.get("schema"),
                    "schema_version": claim_audit.get("schema_version"),
                    "claim_nodes": snapshot.get("claim_nodes", []),
                    "claim_states": snapshot.get("claim_states", {}),
                    "claim_relations": snapshot.get("claim_relations", []),
                    "claim_dependency_edges":
                        snapshot.get("claim_dependency_edges", []),
                    "task_dependency_edges":
                        snapshot.get("task_dependency_edges", []),
                    "report_events":
                        snapshot.get("returned_source_reports", []),
                    "transitions": snapshot.get("state_changes", []),
                    "source": "main_visible_received_reports_only",
                    "private_only": True,
                }
                # Nonselected 4A candidates have only deterministic
                # preview text, not finalized 4B/4C reports. Keep their
                # exploratory audit elsewhere; never present those state
                # snapshots as reviewed language diagnostics.
                if status != "not_selected_for_language_review":
                    claim_state_diagnostics.append({
                        **state_label, "global_publication_status": status,
                    })
                if (status == "recovered" or (
                    status != "not_selected_for_language_review"
                    and audit.get("language_review_accepted") is True
                )):
                    reviewed_observable_states.append({
                        **state_label,
                        "global_outcome_private": status,
                        "positive_main_action_label": status == "recovered",
                        "label_scope": "observable_claim_state_only",
                    })
                if status == "recovered":
                    claim_state_supervision.append(state_label)
                    claim_state_statuses.update(
                        node["status"] for node in
                        state_label["claim_states"].values()
                    )
            retained_language_exceptions += bool(
                audit.get("language_exception")
            )
            execution_outcomes[audit.get("execution_outcome", audit["outcome"])] += 1
            runtime = audit.get("runtime") or {}
            invalidations = runtime.get("invalidations", [])
            conflicts = runtime.get("detected_conflicts", [])
            invalidations_total += len(invalidations)
            if status == "recovered" and invalidations:
                # Count only globally recovered and language-reviewed
                # published branches, never 4A-only simulated candidates.
                published_with_downstream_invalidation += 1
                published_with_invalidation_and_visible_conflict += bool(
                    conflicts
                )
            revalidations_total += len(runtime.get("revalidations", []))
            late_conflicts_total += len(conflicts)
            recovered_after_actual_invalidation += (
                bool(invalidations) and bool(conflicts)
                and audit.get("execution_outcome") == "recovered"
            )
            event_timeline_rows.append({
                "round_sample_id": plan["sample_id"],
                "scenario_blueprint": plan["scenario_plan"]["blueprint"],
                "fault_events_applied": audit.get(
                    "fault_events_applied", []
                ),
                "evidence_events": audit.get("event_timeline", []),
                "observable_late_conflicts": conflicts,
                "invalidations": invalidations,
                "revalidations": runtime.get("revalidations", []),
                "execution_outcome": audit.get("execution_outcome"),
            })
            if any(f.mode == "wrong" for f in plan["faults"]):
                # Diagnostic classes refer ONLY to whether the evidence
                # planner emitted a Main-visible conflict verification.
                # Lack of a VERIFY request does not prove that the wrong
                # fact was inherently undetectable.
                verifies = [
                    (step["round"], request)
                    for step in audit.get("evidence_plan", [])
                    for request in step.get("evidence_requests", [])
                    if request["operation"] == "verify_conflicting_evidence"
                ]
                at_fork = [
                    (rnd, req) for rnd, req in verifies
                    if rnd == plan["branch_round"]
                ]
                if at_fork:
                    classification = "verify_planned_at_fork"
                elif verifies:
                    classification = "verify_planned_later"
                else:
                    classification = "no_explicit_verify_planned"
                wrong_visibility_counts[classification] += 1
                execution_status = audit.get(
                    "execution_outcome", audit["outcome"]
                )
                wrong_visibility_outcomes[classification][execution_status] += 1
                wrong_visibility_rows.append({
                    "round_sample_id": plan["sample_id"],
                    "question_id": plan["task_id"],
                    "fork_round": plan["branch_round"],
                    "classification": classification,
                    "visible_verify_requests": [{
                        "round": rnd,
                        "target_fields": req["target_fields"],
                        "main_visible_justification":
                            req["main_visible_justification"],
                    } for rnd, req in verifies],
                    "execution_outcome": execution_status,
                    "final_publication_status": status,
                    "interpretation": (
                        "Evidence-plan observability only; absence of "
                        "an explicit VERIFY is not proof that the error "
                        "is intrinsically undetectable."
                    ),
                })
            # For screened-out trials, the inherited 4A audit contains
            # *template* reviews, not actual 4B/4C model invocations.
            # Preserve execution metrics above, but never count those
            # previews as accepted language review or local SFT turns.
            if status == "not_selected_for_language_review":
                continue
            for review in audit.get("subagent_report_reviews", []):
                result = review["authoring"]
                report_modes[result["mode"]] += 1
                report_statuses[result.get("status", "unknown")] += 1
                executor_normalized_results += bool(
                    result.get("executor_result_normalized")
                )
                is_contract = result.get("report_contract_version") == "v10_3"
                contract_reports_reviewed += bool(is_contract)
                contract_reports_accepted += (
                    bool(is_contract) and result.get("status") == "accepted"
                )
                contract_results_compiled += bool(
                    is_contract and result.get(
                        "result_compiled_after_author_omission"
                    )
                )
                contract_mandatory_facts += (
                    result.get("mandatory_fact_clauses", 0)
                    if is_contract else 0
                )
                contract_optional_summary_omitted += bool(
                    result.get("optional_summary_omitted_due_to_length")
                )
                witnessed = bool(
                    result.get("observed_handoff_discrepancy_count", 0)
                )
                witnessed_report_reviews += witnessed
                witnessed_report_fallbacks += (
                    witnessed and result.get("status") == "fallback"
                )
                witness_critic_rereviews += bool(
                    result.get("critic_witness_rereviewed")
                )
                for failure in result.get("failures", []):
                    if failure.get("stage") == "guard":
                        report_guard_failures.update(
                            failure.get("errors", [])
                        )
                    else:
                        report_review_failures.update(
                            [failure.get("stage", "unknown")]
                        )
            for review in audit.get("main_action_reviews", []):
                result = review["authoring"]
                main_modes[result["mode"]] += 1
                main_statuses[result.get("status", "unknown")] += 1
                contracted = result.get("decision_contract_version") == "v10_4_3"
                main_decision_contract_reviews += bool(contracted)
                main_decision_contract_accepted += (
                    bool(contracted) and result.get("status") == "accepted"
                )
                main_confirmed_visible_conflicts += (
                    result.get("confirmed_visible_conflicts", 0)
                    if contracted else 0
                )
                main_conflict_critic_rereviews += bool(
                    contracted and result.get("critic_conflict_rereviewed")
                )
            reviewed_main = {
                item["round"]: item["authoring"]
                for item in audit.get("main_action_reviews", [])
            }
            for turn in audit.get("candidate_local_turns", []):
                # This is a *diagnostic* per-decision label, not a
                # counterfactual guarantee about the final global answer.
                # Only currently visible previous reports can influence the
                # decision's local review eligibility.
                history_reports = [
                    item["authoring"] for item in
                    audit.get("subagent_report_reviews", [])
                    if item["round"] < turn["round"]
                ]
                report_accepted = all(
                    item.get("status") == "accepted"
                    and item.get("mode") != "grounded_template_fallback"
                    for item in history_reports
                )
                main_review = reviewed_main.get(turn["round"], {})
                main_accepted = (
                    main_review.get("status") == "accepted"
                    and "fallback" not in main_review.get("mode", "")
                )
                dependency_safe = (
                    audit.get("parallel_batches_verified") is True
                )
                locally_qualified = (
                    report_accepted and main_accepted and dependency_safe
                )
                info = {
                    "turn": deepcopy(turn),
                    "round_sample_id": plan.get("sample_id"),
                    "branch_index": plan["index"],
                    "fork_round": plan["branch_round"],
                    "local_report_reviews_passed": report_accepted,
                    "local_main_action_review_passed": main_accepted,
                    "dependency_safe": dependency_safe,
                    "locally_qualified": locally_qualified,
                    "globally_recovered": status == "recovered",
                    "terminal_execution_outcome": audit.get(
                        "execution_outcome"
                    ),
                    "release_policy": (
                        "diagnostic_only_never_auto_promote_to_reviewed_turns"
                    ),
                }
                local_candidates.append(info)
                if locally_qualified:
                    local_qualified.append(info)
                else:
                    local_rejections.append(info)
    _atomic_jsonl(
        output / "private" / "paired_trajectories.jsonl",
        paired_trajectories,
    )
    _atomic_jsonl(output / "episodes.jsonl", episodes)
    _atomic_jsonl(output / "main_turns.jsonl", positive_turns)
    _atomic_jsonl(output / "failed.jsonl", failures)
    _atomic_jsonl(output / "private" / "fault_alignment.jsonl", audits)
    _atomic_jsonl(
        output / "private" / "decision_reward_states.jsonl",
        decision_reward_states,
    )
    _atomic_jsonl(
        output / "private" / "claim_state_supervision.jsonl",
        claim_state_supervision,
    )
    _atomic_jsonl(
        output / "private" / "claim_state_diagnostics.jsonl",
        claim_state_diagnostics,
    )
    _atomic_jsonl(
        output / "private" / "reviewed_observable_states.jsonl",
        reviewed_observable_states,
    )
    _atomic_jsonl(output / "private" / "publication_guard.jsonl",
                  publication_guard_rows)
    _atomic_jsonl(output / "private" / "event_timeline.jsonl",
                  event_timeline_rows)
    _atomic_jsonl(output / "private" / "wrong_visibility.jsonl",
                  wrong_visibility_rows)
    _atomic_jsonl(output / "private" / "main_turns_local_review.jsonl",
                  local_candidates)
    _atomic_jsonl(output / "private" / "main_turns_local_qualified.jsonl",
                  local_qualified)
    return {
        "completed_branches": sum(counters.values()),
        "recovered": counters["recovered"],
        "incorrect": counters["incorrect_result_after_actual_propagation"],
        "unresolved": sum(count for status, count in counters.items()
                          if status not in {
                              "recovered", "skipped_unsupported",
                              "rejected_validation",
                              "incorrect_result_after_actual_propagation",
                              "not_selected_for_language_review",
                          }),
        "screened_out_before_llm": counters[
            "not_selected_for_language_review"
        ],
        "skipped_or_rejected": (
            counters["skipped_unsupported"] +
            counters["rejected_validation"]
        ),
        "published_episodes": len(episodes),
        "paired_trajectory_index_rows": len(paired_trajectories),
        "main_training_turns": len(positive_turns),
        "reviewed_branches": counters["recovered"],
        "reviewed_turns": len(positive_turns),
        "decision_reward_states": len(decision_reward_states),
        "decision_reward_missing": decision_reward_missing,
        "decision_reward_replay_failures": sum(
            bool((audit or {}).get("decision_reward_replay_error"))
            for audit in audits
        ),
        "decision_reward_unscorable": sum(
            not row["state"]["scorable"] for row in decision_reward_states
        ),
        "local_decisions_reviewed": len(local_candidates),
        "local_decisions_qualified_diagnostic": len(local_qualified),
        "local_decisions_rejected_diagnostic": len(local_rejections),
        "local_qualified_from_non_recovered_branches": sum(
            item["locally_qualified"] and not item["globally_recovered"]
            for item in local_candidates
        ),
        "local_decision_review_file": str(
            output / "private" / "main_turns_local_review.jsonl"
        ),
        "execution_recovered_branches": execution_outcomes["recovered"],
        "execution_incorrect_branches":
            execution_outcomes["incorrect_result_after_actual_propagation"],
        "execution_unresolved_branches": sum(
            count for outcome, count in execution_outcomes.items()
            if outcome not in {
                "recovered", "incorrect_result_after_actual_propagation",
            }
        ),
        "publication_rejected_by_language_or_action": (
            counters["language_review_unaccepted"]
            + counters["main_action_not_accepted"]
        ),
        "publication_language_review_rejected":
            counters["language_review_unaccepted"],
        "publication_main_action_failed":
            counters["main_action_not_accepted"],
        "execution_outcomes": dict(execution_outcomes),
        "event_blueprint_outcomes": {
            name: dict(counts)
            for name, counts in sorted(event_outcomes.items())
        },
        "atomic_fault_outcomes": {
            name: dict(counts)
            for name, counts in sorted(atom_outcomes.items())
        },
        "actual_downstream_invalidations": invalidations_total,
        "verified_without_downstream_invalidation": revalidations_total,
        "visible_late_conflicts_detected": late_conflicts_total,
        "execution_recovered_after_invalidation":
            recovered_after_actual_invalidation,
        "published_with_downstream_invalidation":
            published_with_downstream_invalidation,
        "published_with_invalidation_and_visible_conflict":
            published_with_invalidation_and_visible_conflict,
        "event_timeline_diagnostics": str(
            output / "private" / "event_timeline.jsonl"
        ),
        "publication_generation_process_leaks": len(
            publication_guard_rows
        ),
        "publication_surface_leaks": len(publication_guard_rows),
        "language_exceptions_with_retained_audit":
            retained_language_exceptions,
        "wrong_visibility_by_explicit_verify": dict(wrong_visibility_counts),
        "wrong_visibility_execution_outcomes": {
            name: dict(counters)
            for name, counters in sorted(wrong_visibility_outcomes.items())
        },
        "wrong_visibility_diagnostics": str(
            output / "private" / "wrong_visibility.jsonl"
        ),
        "language_review_rejected_after_recovery":
            sum(1 for audit in audits if audit.get("execution_recovered")
                and not audit.get("language_review_accepted", True)),
        "outcomes": dict(counters),
        "fault_mode_outcomes": {
            mode: dict(counts)
            for mode, counts in sorted(mode_outcomes.items())
        },
        "subagent_report_generation_modes": dict(report_modes),
        "subagent_report_review_statuses": dict(report_statuses),
        "subagent_executor_result_normalized": executor_normalized_results,
        "main_decision_contract_reviews": main_decision_contract_reviews,
        "main_decision_contract_accepted": main_decision_contract_accepted,
        "main_confirmed_visible_conflicts": main_confirmed_visible_conflicts,
        "main_conflict_critic_rereviews": main_conflict_critic_rereviews,
        "subagent_fact_contract_reports": contract_reports_reviewed,
        "subagent_fact_contract_accepted": contract_reports_accepted,
        "subagent_contract_results_compiled": contract_results_compiled,
        "subagent_fact_clauses_compiled": contract_mandatory_facts,
        "subagent_optional_summary_omitted":
            contract_optional_summary_omitted,
        "subagent_reports_with_complete_step_discrepancy":
            witnessed_report_reviews,
        "subagent_complete_step_discrepancy_fallbacks":
            witnessed_report_fallbacks,
        "subagent_witness_critic_rereviews": witness_critic_rereviews,
        "subagent_guard_failure_causes": dict(report_guard_failures),
        "subagent_review_failure_stages": dict(report_review_failures),
        "claim_state_supervision_rounds": len(claim_state_supervision),
        "reviewed_observable_state_rounds": len(reviewed_observable_states),
        "reviewed_nonrecovered_state_rounds": sum(
            not row["positive_main_action_label"]
            for row in reviewed_observable_states
        ),
        "reviewed_observable_states": str(
            output / "private" / "reviewed_observable_states.jsonl"
        ),
        "claim_state_diagnostic_rounds": len(claim_state_diagnostics),
        "claim_state_status_distribution": dict(claim_state_statuses),
        "claim_state_supervision": str(
            output / "private" / "claim_state_supervision.jsonl"
        ),
        "claim_state_diagnostics": str(
            output / "private" / "claim_state_diagnostics.jsonl"
        ),
        "main_action_generation_modes": dict(main_modes),
        "main_action_review_statuses": dict(main_statuses),
    }


def generate_turns(
    *, scenes_path: Path, reports_path: Path, output_dir: Path,
    max_scenes: int | None = 30, branches_per_scene: int | None = None,
    round_sampling_strategy: str = "chronological",
    max_post_fork_rounds: int | None = None,
    seed: int = 17, fault_modes: list[str] | None = None,
    persistence: str = "once", multi_target_probability: float = 0.0,
    max_rounds: int = 16, max_cost: int = 80, max_retries: int = 1,
    max_parallel_tasks: int = 4, workers: int = 4,
    scenario_mode: str = "round_fork",
    atomic_fault_kinds: list[str] | None = None,
    max_fault_atoms: int = 1,
    delayed_evidence_probability: float = 0.0,
    candidate_strategy: str = "sample_one",
    max_candidates_per_round: int = 24,
    max_distractors_per_fault: int = 2,
    max_selected_per_round: int = 1,
    selection_strategy: str = "causal_rank",
    max_diagnostics_per_scene: int = 0,
    max_language_attempts_per_round: int = 1,
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
    if round_sampling_strategy not in {"chronological", "stratified"}:
        raise ValueError("invalid_feedback_round_sampling_strategy")
    if max_post_fork_rounds is not None and (
        type(max_post_fork_rounds) is not int or max_post_fork_rounds < 1
    ):
        raise ValueError("invalid_max_post_fork_rounds")
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

    if candidate_strategy not in {"sample_one", "enumerate"}:
        raise ValueError("invalid_candidate_strategy")
    if candidate_strategy == "enumerate" and scenario_mode != "event_driven":
        raise ValueError("enumeration_requires_event_driven_scenario")
    for name, value, maximum in (
        ("max_candidates_per_round", max_candidates_per_round, 128),
        ("max_distractors_per_fault", max_distractors_per_fault, 10),
        ("max_selected_per_round", max_selected_per_round, 8),
        ("max_language_attempts_per_round", max_language_attempts_per_round, 8),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("invalid_" + name)
    if selection_strategy not in {"causal_rank", "stratified"}:
        raise ValueError("invalid_candidate_selection_strategy")
    if type(max_diagnostics_per_scene) is not int or not (
        0 <= max_diagnostics_per_scene <= 8
    ):
        raise ValueError("invalid_max_diagnostics_per_scene")
    if (candidate_strategy == "enumerate"
            and max_language_attempts_per_round < max_selected_per_round):
        raise ValueError("language_budget_smaller_than_selection")
    modes = list(fault_modes or ["missing", "partial", "wrong"])
    if scenario_mode not in {"round_fork", "event_driven"}:
        raise ValueError("invalid_scenario_mode")
    kinds = list(atomic_fault_kinds or [
        "no_result", "incomplete_retrieval", "wrong_retrieval",
        "wrong_entity_binding", "report_missing",
        "result_summary_conflict", "fabricated_claim",
    ])
    if (not kinds or any(kind not in ATOM_TO_MODE for kind in kinds)
            or len(kinds) != len(set(kinds))):
        raise ValueError("invalid_atomic_fault_kinds")
    if type(max_fault_atoms) is not int or not 1 <= max_fault_atoms <= 2:
        raise ValueError("invalid_max_atomic_faults")
    if type(delayed_evidence_probability) not in (int, float) or (
        not 0 <= delayed_evidence_probability <= 1
    ):
        raise ValueError("invalid_delayed_evidence_probability")
    settings = {
        "max_scenes": max_scenes,
        "branches_per_scene": branches_per_scene,
        "round_sampling_strategy": round_sampling_strategy,
        "max_post_fork_rounds": max_post_fork_rounds,
        "seed": seed,
        "fault_modes": modes, "persistence": persistence,
        "scenario_mode": scenario_mode,
        "atomic_fault_kinds": kinds,
        "max_fault_atoms": max_fault_atoms,
        "delayed_evidence_probability": delayed_evidence_probability,
        "candidate_strategy": candidate_strategy,
        "max_candidates_per_round": max_candidates_per_round,
        "max_distractors_per_fault": max_distractors_per_fault,
        "max_selected_per_round": max_selected_per_round,
        "selection_strategy": selection_strategy,
        "max_diagnostics_per_scene": max_diagnostics_per_scene,
        "max_language_attempts_per_round": max_language_attempts_per_round,
        "multi_target_probability": multi_target_probability,
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
        round_sampling_strategy=round_sampling_strategy,
        seed=seed, fault_modes=modes, persistence=persistence,
        multi_target_probability=multi_target_probability,
        max_rounds=max_rounds,
        scenario_mode=scenario_mode, atomic_fault_kinds=kinds,
        max_fault_atoms=max_fault_atoms,
        delayed_evidence_probability=delayed_evidence_probability,
    )
    base_feedback_round_count = len(planned)
    if candidate_strategy == "enumerate":
        # Source-true variants are tested by the executable 4A preflight,
        # not ranked by LLM preference or arbitrary hallucinated facts.
        planned = expand_all_rounds(
            scenes, planned,
            atomic_fault_kinds=kinds,
            max_candidates_per_round=max_candidates_per_round,
            max_distractors_per_fault=max_distractors_per_fault,
            max_fault_atoms=max_fault_atoms,
            delayed_evidence_probability=delayed_evidence_probability,
            seed=seed,
        )
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
        """Execution outcomes are independent of reviewed-turn publication review."""
        status = data.get("status", "rejected_validation")
        if status == "recovered":
            return "recovered"
        if status == "incorrect_result_after_actual_propagation":
            return "incorrect"
        if status in {"skipped_unsupported", "rejected_validation"}:
            return "skipped"
        return "unresolved"

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
                        "recovered": counts["recovered"],
                        "incorrect": counts["incorrect"],
                        "unresolved": counts["unresolved"],
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

            def commit_round(plan: dict) -> None:
                # callback_worker has *already* written the round journal.
                # Keep progress and durable persistence counters distinct.
                current = saved(directory, plan)
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

    def simulate(plan: dict, tick) -> None:
        scene = scenes[plan["scene_index"]]
        if plan["skip_reason"]:
            write(stage_a, plan, {
                "task_id": scene["task_id"],
                "round_sample_id": plan["sample_id"],
                "fork_round": plan["branch_round"],
                "status": "skipped_unsupported",
                "error": plan["skip_reason"],
                "episode": None, "turns": [], "private_audit": None,
            })
            tick()
            return
        try:
            episode, _, audit = build_branch(
                scene, rows_by_scene[scene["task_id"]],
                base[scene["task_id"]],
                branch_round=plan["branch_round"], faults=plan["faults"],
                scenario_plan=plan["scenario_plan"],
                max_rounds=max_rounds, max_cost=max_cost,
                max_post_fork_rounds=max_post_fork_rounds,
                max_retries=max_retries,
                max_parallel_tasks=max_parallel_tasks,
                sub_model=sub_model, model_alias=model_alias,
                # Stage 4A must never call the language models.
                author=None, critic=None, report_author=None,
                report_critic=None,
            )
            data = {
                "task_id": scene["task_id"],
                "round_sample_id": plan["sample_id"],
                "fork_round": plan["branch_round"],
                "status": audit["execution_outcome"],
                "episode": episode, "turns": [],
                "private_audit": audit,
            }
        except (FaultUnsupported, ValueError, KeyError, TypeError) as exc:
            data = as_failure(scene, exc)
        write(stage_a, plan, data)
        tick()

    run_parallel(
        "Evidence simulation", stage_a, "round",
        len(planned), sum(saved(stage_a, p) is not None for p in planned),
        simulate, stage="simulation",
    )

    # The 4A search is intentionally wider than the expensive 4B/4C
    # publication candidates. Keep ALL executions in stage4a/journal
    # and an explicit private selection ledger, including failures.
    if candidate_strategy == "enumerate":
        selected_indices, selection_rows, selection_stats = (
            select_recovered_candidates(
                planned, [saved(stage_a, plan) for plan in planned],
                max_selected_per_round=max_selected_per_round,
                strategy=selection_strategy,
                max_diagnostics_per_scene=max_diagnostics_per_scene,
                seed=seed,
            )
        )
    else:
        selected_indices = {plan["index"] for plan in planned}
        selection_rows = [{
            "candidate_index": plan["index"],
            "round_sample_id": plan["sample_id"],
            "selected_for_language_review": True,
            "selection_reason": "legacy_sample_one_all_candidates",
        } for plan in planned]
        selection_stats = {
            "candidates_simulated": len(planned),
            "candidates_selected_for_language_review": len(planned),
            "candidates_filtered_before_llm": 0,
            "candidate_behavior_distribution": {},
            "selected_behavior_distribution": {},
        }
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

    # One deterministic rank per distinct *recovered execution behavior*.
    # Reserve candidates remain in the 4A journal until an earlier language
    # candidate fails; there is never an extra execution replay or LLM call
    # just for constructing the reserve pool.
    by_index = {plan["index"]: plan for plan in planned}
    ranked = defaultdict(list)
    if candidate_strategy == "enumerate":
        for row in selection_rows:
            if row.get("recovery_rank") is not None:
                ranked[(row["question_id"], row["feedback_round"])].append(row)
        for key in ranked:
            ranked[key].sort(key=lambda row: row["recovery_rank"])

    def make_reports(plan: dict, tick) -> None:
        source = saved(stage_a, plan)
        scene = scenes[plan["scene_index"]]
        if source["episode"] is None:
            write(stage_b, plan, source)
            return
        try:
            episode, audit = author_branch_reports(
                scene, rows_by_scene[scene["task_id"]],
                source["episode"], source["private_audit"],
                author=report_author, critic=report_critic,
                max_revisions=report_max_revisions,
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
            episode, turns, audit = author_branch_main(
                scene, source["episode"], source["private_audit"],
                author=author, critic=critic,
                max_revisions=main_max_revisions,
                max_rounds=max_rounds, sub_model=sub_model,
                model_alias=model_alias,
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

    def published_recovery(index: int) -> bool:
        current = saved(journal, by_index[index])
        return (
            current is not None and current.get("status") == "recovered"
            and not any(review_turn_surface(turn) for turn in current["turns"])
        )

    # Each round starts with max_selected_per_round candidates. On review
    # failure, try another *distinct* successful 4A replay from the same
    # question/feedback round. Never spend language budget on a recovered
    # branch once the round already has its required accepted output(s).
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
        if candidate_strategy != "enumerate":
            break
        next_active = set()
        for key, candidates in ranked.items():
            past = [r for r in candidates if r["candidate_index"] in attempted]
            succeeded = sum(
                published_recovery(r["candidate_index"]) for r in past
            )
            room = max(0, max_selected_per_round - succeeded)
            budget = max(0, max_language_attempts_per_round - len(past))
            if room == 0 or budget == 0:
                continue
            for candidate in candidates:
                if room == 0 or budget == 0:
                    break
                index = candidate["candidate_index"]
                if index in attempted:
                    continue
                next_active.add(index)
                room -= 1
                budget -= 1
                backfilled_groups.add(key)
        active = next_active

    # Retain compatibility with the V10 full candidate journal while keeping
    # all rejected/unused candidates private and LLM-free.
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

    if candidate_strategy == "enumerate":
        for row in selection_rows:
            index = row["candidate_index"]
            row["selected_for_language_review"] = index in attempted
            row["selection_reason"] = (
                "language_review_diagnostic"
                if index in attempted and row.get("selection_purpose") == "diagnostic"
                else "language_review_primary" if index in initial_selection
                else "language_review_backfill_after_rejection"
                if index in attempted
                else "not_execution_recovered"
                if row["execution_outcome"] != "recovered"
                else "not_needed_or_language_budget_exhausted"
            )
            if index in attempted:
                row["language_review_outcome"] = (
                    saved(journal, by_index[index]) or {}
                ).get("status")
        selection_stats["candidates_selected_for_language_review"] = len(
            attempted
        )
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
    summary = {
        # "status" describes whether all requested round units were
        # processed. Training-label acceptance is a separate measurement:
        # an incorrect/unresolved fault scenario is still a FINISHED
        # simulation, not an unfinished generator job.
        "status": (
            "done" if counts_summary["completed_branches"] == len(planned)
            else "partial"
        ),
        "publication_status": (
            "all_rounds_accepted"
            if counts_summary["recovered"] == len(planned)
            else "filtered_by_recovery_or_language_review"
        ),
        "pipeline": "maestro_untrusted_orchestration",
        "scenario_mode": scenario_mode,
        "candidate_strategy": candidate_strategy,
        "selection_strategy": selection_strategy,
        "max_diagnostics_per_scene": max_diagnostics_per_scene,
        "round_sampling_strategy": round_sampling_strategy,
        "max_post_fork_rounds": max_post_fork_rounds,
        "max_candidates_per_feedback_round": max_candidates_per_round,
        "max_distractors_per_fault": max_distractors_per_fault,
        "max_selected_per_feedback_round": max_selected_per_round,
        "max_language_attempts_per_feedback_round":
            max_language_attempts_per_round,
        "candidate_selection_diagnostics": str(
            output / "private" / "candidate_selection.jsonl"
        ),
        "candidate_backfill_diversity": (
            "causal_recovery_rank_distinct_behavior"
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
        "delayed_requested_but_not_executable": sum(
            p["scenario_plan"]["delayed_requested_but_unsupported"]
            for p in planned
        ),
        "mode": ("llm" if main_mode == "llm" or report_mode == "llm"
                 else "template"),
        "subagent_report_mode": report_mode,
        "main_action_mode": main_mode,
        "main_actions_planned": total_actions,
        "main_actions_processed": sum(
            len(((saved(journal, by_index[i]) or {}).get("private_audit")
                 or {}).get("main_action_reviews", []))
            for i in attempted
        ),
        "main_actions_skipped_downstream": max(0,
            total_actions - sum(
                len(((saved(journal, by_index[i]) or {}).get("private_audit")
                     or {}).get("main_action_reviews", []))
                for i in attempted
            )
        ),
        "input_scenes": len(scenes),
        "trusted_prefixes_compiled": len(base),
        "trusted_prefixes_rejected": len(prefix_reviews) - len(base),
        "trusted_prefix_checks": str(output / "private" / "trusted_prefix_checks.jsonl"),
        "scenes_without_accepted_base": no_base,
        "round_sampling_unit": (
            "question_x_feedback_round_x_atomic_instance"
            if candidate_strategy == "enumerate"
            else "question_x_main_feedback_round"
        ),
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
        "rounds_skipped_by_recovery_budget": sum(
            plan["skip_reason"] == "insufficient_recovery_round_budget"
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
        "requested_branches": len(planned),
        "requested_feedback_rounds": base_feedback_round_count,
        **counts_summary,
        "stage4a_evidence_plans": str(output / "stage4a" / "journal"),
        "stage4b_reviewed_reports": str(output / "stage4b" / "journal"),
        "public_turns": str(output / "main_turns.jsonl"),
        "public_episodes": str(output / "episodes.jsonl"),
        "private_fault_alignment": str(
            output / "private" / "fault_alignment.jsonl"
        ),
        "limitations": [
            "Candidate enumeration and ranking are deterministic 4A-only; successful candidates are author-reviewed in ranked order only if earlier reviewed candidates fail, within a per-round budget.",
            "The default 4A search only retries the frozen executable Macro path; new independent verification operators are not yet synthesized.",
            "One atomic fault per candidate is sufficient to expose several multi-round outcomes; trial count is not the number of distinct questions.",
            "Every accepted base Main decision round receiving a prior Sub-Agent batch is indexed; only an explicit debug cap limits extraction.",
            "Feedback rounds after the configured max_rounds budget are indexed but transparently skipped, never silently omitted.",
            "The initial fault injection window is the latest completed Sub-Agent batch. Optional event-scheduled future atoms are separately audited.",
            "Only preflight-executable fault/macro pairs are sampled.",
            "Only publicly observable discrepancies trigger targeted repair labels.",
            "Clean World/Gold never change; uncorrected counterfactuals are private, not reviewed turns.",
            "Stage 4A determines evidence requests and an executable global schedule using only Main-visible report justifications.", 
            "4B materializes every actual executed Sub-Agent Attempt before any Stage 4C Main LLM is called.",
            "4C naturalizes 4A's fixed legal scheduling contracts against the finalized Main-visible 4B history.",
            "Template actions are deterministic teacher demonstrations, not online Main-Agent rollouts.",
            "Sub-Agent Author/Critic write full prose for this attempt; status and actual WorkProduct remain code-owned.",
            "Main-Agent Author/Critic only naturalize the program-selected O1 tasks; private state does not enter their prompt.",
            "LLM-mode fallback reports are diagnostic only, never silently counted as reviewed positive turns.",
            "4C action task_instruction and context are fixed against the current 4A/4B executable evidence contract; LLM edits only decision reasoning.",
            "Per-turn local eligibility is diagnostic in private/main_turns_local_review.jsonl; it is never silently merged into globally verified reviewed turns.",
            "Claim–Source–Dependency supervision is private and derived from actually returned report text only; unknown text remains unknown and no future Gold is materialized.",
        ],
    }
    _atomic_json(output / "summary.json", summary)
    if not no_progress:
        tqdm.write(
            "Evidence candidate trials: "
            f"{summary['completed_branches']}/{summary['candidate_trials']} "
            f"| feedback rounds={summary['requested_feedback_rounds']} "
            f"| selected={summary['candidates_selected_for_language_review']} "
            f"| execution recovered={summary['execution_recovered_branches']} "
            f"| incorrect={summary['incorrect']} "
            f"| reviewed branches={summary['reviewed_branches']} "
            f"| reviewed turns={summary['reviewed_turns']}"
        )
    return summary
