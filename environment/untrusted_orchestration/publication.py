"""Stage 4 private audit publication and canonical summary aggregation.

Read completed journal entries once, validate each public surface and emit
positive Episodes/decisions separately from private diagnostic data.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import json

from .inspection.publication_guard import review_turn_surface
from .scenario.single_turn_selection import (
    coordination_pattern, current_public_witness, evidence_relation,
)
from .scenario.observable_scenarios import classify_observable_scenario
from .storage import _atomic_jsonl

def _published_result_formats(episodes: list[dict]) -> dict:
    """Audit O1 Result surface by phase, without rewriting clean evidence.

    Stage 2 clean Answer prose and Stage 4 provenance-bearing typed Results
    currently differ. Expose the shift before using these rows for training;
    changing trusted Stage-3 prefix presentation would break exact pairing.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for episode in episodes:
        fault_round = episode.get("branch_round")
        for step in episode.get("trajectory", []):
            if step.get("action") != "delegate_task":
                continue
            number = step.get("attempt")
            if not isinstance(number, int) or not isinstance(fault_round, int):
                continue
            # No post-fork Sub-Agent has executed in a single-turn episode.
            if number >= fault_round:
                continue
            phase = ("clean_prefix" if number < fault_round - 1
                     else "faulted_latest_batch")
            for report in (step.get("result") or {}).get("subtask_results", []):
                finish = report.get("finish_result") or report
                value = finish.get("result") if isinstance(finish, dict) else None
                if not isinstance(value, str) or not value.strip():
                    label = "missing"
                else:
                    try:
                        decoded = json.loads(value)
                    except (ValueError, TypeError):
                        decoded = None
                    from .reporting.surface import parse_records
                    if parse_records(value) is not None:
                        label = "cited_record_prose_v1"
                    elif (isinstance(decoded, list)
                          and all(isinstance(item, dict)
                                  and isinstance(item.get("values"), dict)
                                  and isinstance(item.get("source_fact_refs"), list)
                                  for item in decoded)):
                        label = "typed_evidence_json"
                    else:
                        label = "legacy_prose"
                counts[phase][label] += 1
    return {phase: dict(sorted(values.items()))
            for phase, values in sorted(counts.items())}


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
    published_single_turn_relations = Counter()
    published_scenario_families = Counter()
    published_scenario_types = Counter()
    published_single_turn_patterns = Counter()
    published_single_turn_parallel = Counter()
    published_single_turn_conflict_kinds = Counter()
    published_single_turn_prefix_preserved = Counter()
    witnessed_report_fallbacks = 0
    witness_critic_rereviews = 0
    retained_language_exceptions = 0
    local_candidates, local_qualified, local_rejections = [], [], []
    claim_state_supervision = []
    # Observable evidence labels for rejected candidates are diagnostics,
    # never positive supervised actions.
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
        if status == "single_turn_accepted" and leaks:
            status = "publication_generation_process_leak"
            publication_guard_rows.append({
                "branch_index": plan["index"],
                "round_sample_id": plan["sample_id"],
                "rejected_turns": leaks,
            })
        audit_pair = data.get("private_audit") or {}
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
            "robust_publication_status": status,
            "private_only": True,
        })
        counters[status] += 1
        for mode in {fault.mode for fault in plan["faults"]}:
            mode_outcomes[mode][status] += 1
        event_outcomes[plan["scenario_plan"]["blueprint"]][status] += 1
        for atom in plan["scenario_plan"]["fault_atoms"]:
            atom_outcomes[atom["kind"]][status] += 1
        if status == "single_turn_accepted" and (
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
        if status == "single_turn_accepted":
            private = data["private_audit"]
            issues, conflicts = current_public_witness(
                private, plan["branch_round"]
            )
            if not issues and not conflicts:
                raise RuntimeError(
                    "single_turn_accepted_without_current_public_witness"
                )
            relation = evidence_relation(issues, conflicts)
            realized = classify_observable_scenario(issues, conflicts)
            target = plan.get("target_scenario")
            if not realized["scenario_id"] or (
                target is not None and target != realized["scenario_id"]
            ):
                raise RuntimeError(
                    "stage4_published_target_scenario_lacks_public_witness"
                )
            published_scenario_families[realized["scenario_family"]] += 1
            published_scenario_types[realized["scenario_id"]] += 1
            pattern, operations, parallel_tasks = coordination_pattern(
                private.get("evidence_plan") or []
            )
            published_single_turn_relations[relation] += 1
            published_single_turn_patterns[pattern] += 1
            published_single_turn_parallel[
                "parallel" if parallel_tasks > 1 else "single"
            ] += 1
            published_single_turn_prefix_preserved[
                "preserved" if private.get(
                    "stage3_prior_history_preserved") is True else
                "unverified_legacy"
            ] += 1
            published_single_turn_conflict_kinds.update(
                event.get("kind", "unknown") for event in conflicts
            )
            if not isinstance(data.get("episode"), dict):
                raise RuntimeError(
                    "untrusted_recovered_branch_missing_public_episode"
                )
            public_episode = deepcopy(data["episode"])
            # Truth labels belong in private audit, never public reviewed-turn.
            public_episode.pop("outcome", None)
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
                    # A missing decision-time snapshot is not reconstructed
                    # from an accepted action alone.
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
            # release training-side state labels only after this current
            # next-decision episode passed contract and language review.
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
                if (status == "single_turn_accepted" or (
                    status != "not_selected_for_language_review"
                    and audit.get("language_review_accepted") is True
                )):
                    reviewed_observable_states.append({
                        **state_label,
                        "global_outcome_private": status,
                        "positive_main_action_label": status == "single_turn_accepted",
                        "label_scope": "observable_claim_state_only",
                    })
                if status == "single_turn_accepted":
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
            revalidations_total += len(runtime.get("revalidations", []))
            late_conflicts_total += len(conflicts)
            event_timeline_rows.append({
                "round_sample_id": plan["sample_id"],
                "scenario_blueprint": plan["scenario_plan"]["blueprint"],
                "fault_events_applied": audit.get(
                    "fault_events_applied", []
                ),
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
                # Diagnostic per-decision label, not a future outcome guarantee.
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
                    "decision_accepted": status == "single_turn_accepted",
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
        "single_turn_accepted": counters["single_turn_accepted"],
        "skipped_or_rejected": (
            counters["skipped_unsupported"] +
            counters["rejected_validation"]
        ),
        "published_episodes": len(episodes),
        "published_single_turn_evidence_relations_4a": dict(
            published_single_turn_relations
        ),
        "published_scenario_families": dict(published_scenario_families),
        "published_scenario_types": dict(published_scenario_types),
        "published_single_turn_coordination_patterns": dict(
            published_single_turn_patterns
        ),
        "published_single_turn_parallelism": dict(
            published_single_turn_parallel
        ),
        "published_single_turn_detected_conflicts_4a": dict(
            published_single_turn_conflict_kinds
        ),
        "published_single_turn_stage3_prior_history": dict(
            published_single_turn_prefix_preserved
        ),
        "published_result_format_by_phase": _published_result_formats(episodes),
        "paired_trajectory_index_rows": len(paired_trajectories),
        "main_training_turns": len(positive_turns),
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
        "local_decision_review_file": str(
            output / "private" / "main_turns_local_review.jsonl"
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
        "event_timeline_diagnostics": str(
            output / "private" / "event_timeline.jsonl"
        ),
        "publication_generation_process_leaks": len(
            publication_guard_rows
        ),
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
        "reviewed_nonaccepted_state_rounds": sum(
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


