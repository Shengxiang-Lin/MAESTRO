"""4A-only causal outcome classification, deduplication and LLM budget gate.

This reads private simulation audits ONLY AFTER deterministic replay.
It does not alter a Main-visible state or use Gold to choose a Main action.
Gold may determine whether the completed teacher episode is publishable,
which is a data-curation decision, not an action-policy decision.

Rejected/unselected trials remain in Stage-4A journals and an explicit
selection manifest. They must never consume Stage-4B/4C LLM calls.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json


_PRIORITY = {
    "recovered_with_downstream_invalidation": 0,
    "recovered_after_visible_verification": 1,
    "recovered_with_revalidation": 2,
    "recovered_unreliable_evidence": 3,
    "recovered_partial_evidence": 4,
    "recovered_missing_evidence": 5,
    "recovered_other": 6,
}


_FAULT_FAMILIES = {
    "no_result": "missing",
    "report_missing": "missing",
    "incomplete_retrieval": "partial",
    "done_but_incomplete": "partial",
    "wrong_retrieval": "unreliable",
    "computation_error": "unreliable",
    "wrong_entity_binding": "unreliable",
    "legacy_wrong_evidence": "unreliable",
    "fabricated_claim": "unreliable",
    "result_summary_conflict": "unreliable",
}


def _visible_at_injection(plan: dict, journal: dict) -> str:
    """What the Main actually received, NOT the private fault family."""
    target = (plan.get("faults") or [None])[0]
    incoming = plan.get("incoming_task_ids") or []
    if target is None or target.task_id not in incoming:
        return "unknown"
    slot = incoming.index(target.task_id)
    for step in (journal.get("episode") or {}).get("trajectory", []):
        if (step.get("attempt") != plan["branch_round"] - 1
                or step.get("action") != "delegate_task"):
            continue
        results = step.get("result", {}).get("subtask_results") or []
        if slot >= len(results):
            return "unknown"
        finish = results[slot].get("finish_result")
        if not isinstance(finish, dict):
            return "missing"
        status = str(finish.get("status", "unknown"))
        if status == "partial":
            return "partial"
        if status == "missing":
            return "missing"
        if status == "done":
            return "apparently_complete"
        return status
    return "unknown"


def classify_trial(plan: dict, journal: dict) -> dict:
    audit = journal.get("private_audit") or {}
    runtime = audit.get("runtime") or {}
    applied = audit.get("fault_events_applied") or []
    invalidations = runtime.get("invalidations") or []
    revalidations = runtime.get("revalidations") or []
    conflicts = runtime.get("detected_conflicts") or []
    # Historical conflicts include ordinary prior-vs-latest reports and
    # independently cited reports that arrived in different feedback waves.
    # Keep the actual public report rounds; do not infer history from the
    # private fault label or from a fixed N-1 window.
    history_events = [
        event for event in conflicts
        if event.get("kind") == "cross_round_report_disagreement"
        or (event.get("kind") == "explicit_independent_claim_disagreement"
            and event.get("conflict_timing") == "delayed")
    ]
    history_round_pairs = sorted({
        (event["older_round"], event["latest_round"])
        for event in history_events
        if type(event.get("older_round")) is int
        and type(event.get("latest_round")) is int
        and 0 < event["older_round"] < event["latest_round"]
    })
    older_rounds = sorted({old for old, _ in history_round_pairs})
    round_gaps = sorted({new - old for old, new in history_round_pairs})
    requests = [
        request for entry in audit.get("evidence_plan", [])
        for request in entry.get("evidence_requests", [])
    ]
    operations = Counter(row.get("operation", "unknown") for row in requests)
    # A fault's private kind is not an observation available to Main.
    # Label detectability only from 4A snapshots of already returned
    # Sub-Agent reports at each decision, never from the hidden Gold.
    snapshots = audit.get("observable_issue_snapshots") or []
    fork_round = plan["branch_round"]
    fault_targets = {
        fault.task_id for fault in plan.get("faults", [])
    }
    detected = [
        (snap["decision_round"], issue)
        for snap in snapshots
        for issue in snap.get("issues", [])
        if (issue.get("task_id") in fault_targets
            or issue.get("kind") in {
                "cross_source_conflict", "historical_report_conflict",
            }
            or (snap["decision_round"] > fork_round and (
                issue.get("reported_round", 0) >= fork_round
                or issue.get("kind") == "stale_downstream_result"
            )))
    ]
    immediate = any(rnd == fork_round for rnd, _ in detected)
    later = any(rnd > fork_round for rnd, _ in detected)
    observability = (
        "immediately_observable" if immediate
        else "eventually_observable" if later
        else "not_observed_in_replay"
    )
    first_visible_issue_round = (
        min(rnd for rnd, _ in detected) if detected else None
    )
    visible_issue_types = sorted({
        issue["kind"] for _, issue in detected
    })
    observable_family = (
        "inter_report_conflict"
        if ({"cross_source_conflict", "historical_report_conflict"}
            & set(visible_issue_types))
        else "intra_report_issue" if visible_issue_types
        else "not_observed"
    )
    unrepaired = audit.get("counterfactual_unrepaired") or {}
    answer_impact = (
        "answer_changing" if unrepaired.get("correct") is False and unrepaired.get("status") == "completed"
        else "consistency_only" if unrepaired.get("correct") is True
        else "completion_blocked" if unrepaired.get("status") and unrepaired.get("status") != "completed"
        else "not_evaluated"
    )
    replay_outcome = audit.get("execution_outcome") or journal.get("status")
    if replay_outcome == "recovered":
        if (unrepaired.get("status") == "completed"
                and unrepaired.get("correct") is False):
            recovery_effect = "incorrect_result_corrected"
        elif invalidations:
            recovery_effect = "downstream_recomputed"
        elif unrepaired.get("status") != "completed":
            recovery_effect = "execution_failure_recovered"
        else:
            recovery_effect = "evidence_inconsistency_resolved"
    else:
        recovery_effect = "not_recovered"
    kinds = [
        a["kind"] for a in plan["scenario_plan"]["fault_atoms"]
    ]
    outcome = journal.get("status")
    if outcome != "recovered":
        category = "non_recovered"
    elif invalidations:
        category = "recovered_with_downstream_invalidation"
    elif conflicts or operations["verify_conflicting_evidence"]:
        category = "recovered_after_visible_verification"
    elif revalidations:
        category = "recovered_with_revalidation"
    elif any(kind in {
        "wrong_retrieval", "wrong_entity_binding",
        "computation_error", "legacy_wrong_evidence", "fabricated_claim",
    } for kind in kinds):
        category = "recovered_unreliable_evidence"
    elif "incomplete_retrieval" in kinds or "done_but_incomplete" in kinds:
        category = "recovered_partial_evidence"
    elif "no_result" in kinds or "report_missing" in kinds:
        category = "recovered_missing_evidence"
    else:
        category = "recovered_other"

    # A source-free normalization of the actual *visible-oriented*
    # evidence requests; useful for suppressing repeated boilerplate
    # teacher decisions in the same Question/feedback round.
    behavior = [
        (
            request.get("operation"),
            tuple(request.get("target_fields") or []),
            tuple(request.get("authorized_source_ids") or []),
        )
        for request in requests
    ]
    history_context = [{
        "comparison_kind": event.get("comparison_kind", "same_claim"),
        "tasks": event.get("compared_parent_tasks", []),
        "claim": event.get("claim"),
        "values": event.get("observed_values", []),
        "rounds": [event.get("older_round"), event.get("latest_round")],
    } for event in history_events]
    # Policy diversity deliberately ignores concrete task IDs, values and
    # historical placement. Case diversity still preserves all of them.
    policy_signature = sha256(json.dumps(
        {"category": category, "operations": [
            request[0] for request in behavior
        ]}, sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()[:16]
    signature = sha256(json.dumps(
        {"category": category, "requests": behavior,
         "history_round_pairs": history_round_pairs,
         "history_context": history_context},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()[:16]
    return {
        "round_sample_id": plan["sample_id"],
        "candidate_index": plan["index"],
        "private_fault_primitive_kind": (
            ((audit.get("faults") or [{}])[0].get("execution") or {})
            .get("kind")
        ),
        "private_computation_operation": (
            ((audit.get("faults") or [{}])[0].get("execution") or {})
            .get("calculation_operation")
        ),
        "observed_issue_types": visible_issue_types,
        "question_id": plan["task_id"],
        "feedback_round": plan["branch_round"],
        "atom_kind": kinds[0] if kinds else None,
        "private_fault_family": (
            _FAULT_FAMILIES.get(kinds[0], "other") if kinds else None
        ),
        "main_visible_delivery_state": _visible_at_injection(
            plan, journal
        ),
        "fault_target": (
            plan["faults"][0].task_id if plan["faults"] else None
        ),
        "variant_index": plan.get("candidate_variant_index"),
        "fault_subtype_private": (
            "wrong_scope" if (
                kinds and kinds[0] == "wrong_retrieval"
                and "scope_constraint_mismatch" in visible_issue_types
            ) else None
        ),
        "execution_outcome": (
            audit.get("execution_outcome") or outcome
        ),
        "category": category,
        "behavior_signature": signature,
        "policy_signature": policy_signature,
        "scenario_blueprint": plan["scenario_plan"].get("blueprint"),
        "delayed_causal_audit": audit.get("delayed_causal_audit"),
        "fault_events_applied": len(applied),
        "observability": observability,
        "observable_issue_family": observable_family,
        "historical_anchor_status": (
            (audit.get("history_conflict_anchor") or {}).get("status")
        ),
        "historical_anchor_reason": (
            (audit.get("history_conflict_anchor") or {}).get("reason")
        ),
        "history_conflicts_detected": len(history_events),
        "history_conflict_older_rounds": older_rounds,
        "history_conflict_round_gaps": round_gaps,
        "recovery_effect": recovery_effect,
        "answer_impact": answer_impact,
        "first_visible_issue_round": first_visible_issue_round,
        "visible_issue_types": visible_issue_types,
        "visible_conflicts": len(conflicts),
        "actual_invalidations": len(invalidations),
        "revalidations": len(revalidations),
        "evidence_operations": dict(operations),
        "decision_rounds_post_fork": len([
            step for step in (journal.get("episode") or {}).get(
                "trajectory", []
            )
            if step["attempt"] >= plan["branch_round"]
        ]),
    }


def select_recovered_candidates(
    planned: list[dict], journals: list[dict], *,
    max_selected_per_round: int | None,
    strategy: str = "causal_rank",
    max_diagnostics_per_scene: int = 0,
    target_main_turns: int | None = None,
    seed: int = 17,
) -> tuple[set[int], list[dict], dict]:
    """Select distinct recoveries plus a bounded sample of real failures.

    The optional stratified strategy balances *observed replay behaviors*
    across feedback rounds. A non-recovered diagnostic can be language
    reviewed for evidence-state supervision, but is NEVER positive SFT.
    """
    if max_selected_per_round is not None and (
        type(max_selected_per_round) is not int
        or not 1 <= max_selected_per_round <= 8
    ):
        raise ValueError("invalid_selection_cap")
    if strategy not in {"causal_rank", "stratified", "global_balanced"}:
        raise ValueError("invalid_candidate_selection_strategy")
    if type(max_diagnostics_per_scene) is not int or not (
        0 <= max_diagnostics_per_scene <= 8
    ):
        raise ValueError("invalid_diagnostic_selection_cap")
    if target_main_turns is not None and (
        type(target_main_turns) is not int or target_main_turns < 1
    ):
        raise ValueError("invalid_global_main_turn_budget")
    if len(planned) != len(journals):
        raise ValueError("candidate_selection_journal_alignment_invalid")

    rows = [
        classify_trial(plan, journal)
        for plan, journal in zip(planned, journals)
    ]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["question_id"], row["feedback_round"])].append(row)
        row["recovery_rank"] = None
        row["selection_purpose"] = "unselected"

    selected = set()
    category_usage = Counter()
    def tie(row):
        return sha256((
            str(seed) + ":" + row["round_sample_id"]
        ).encode("utf-8")).hexdigest()

    # Iterate in stable Scene/round order, independent of worker completion.
    for key in sorted(groups):
        eligible = [
            row for row in groups[key]
            if row["execution_outcome"] == "recovered"
        ]
        if strategy in {"stratified", "global_balanced"}:
            eligible.sort(key=lambda row: (
                # Scarce, successfully recovered historical disagreements
                # outrank routine missing/partial report recoveries.
                -bool(row["history_conflicts_detected"]),
                -len(row["history_conflict_older_rounds"]),
                category_usage[row["category"]],
                _PRIORITY.get(row["category"], 99),
                -row["actual_invalidations"],
                -row["visible_conflicts"],
                tie(row),
            ))
        else:
            eligible.sort(key=lambda row: (
                -bool(row["history_conflicts_detected"]),
                -len(row["history_conflict_older_rounds"]),
                _PRIORITY.get(row["category"], 99),
                -row["actual_invalidations"],
                -row["visible_conflicts"],
                -row["decision_rounds_post_fork"],
                row["candidate_index"],
            ))
        seen = set()
        for candidate in eligible:
            signature = candidate["behavior_signature"]
            if signature in seen:
                continue
            seen.add(signature)
            candidate["recovery_rank"] = len(seen)
            if max_selected_per_round is None or len(seen) <= max_selected_per_round:
                selected.add(candidate["candidate_index"])
                candidate["selection_purpose"] = "positive"
                category_usage[candidate["category"]] += 1

    # The global budget counts complete, recoverable MAIN decision turns,
    # not 4A trial count or number of Episodes. Each Episode stays intact.
    # Simulated failures never become positive recovery SFT data. Scarce
    # source-backed kinds get priority instead of hallucinated balancing.
    available_by_kind = Counter(
        rows[i]["atom_kind"] for i in selected
    )
    if target_main_turns is not None:
        from collections import deque
        candidates = defaultdict(list)
        for i in selected:
            candidates[rows[i]["atom_kind"]].append(rows[i])
        # Prefer rare recovery EFFECTS and policies within each kind.
        for kind, choices in candidates.items():
            effects = Counter(row["recovery_effect"] for row in choices)
            policies = Counter(row["policy_signature"] for row in choices)
            choices.sort(key=lambda row: (
                row.get("answer_impact") == "consistency_only",
                effects[row["recovery_effect"]],
                policies[row["policy_signature"]],
                -row["actual_invalidations"],
                -len(row["history_conflict_round_gaps"]),
                tie(row),
            ))
        queues = {kind: deque(items) for kind, items in candidates.items()}
        chosen = set()
        spent = Counter()
        remaining = target_main_turns
        while remaining > 0:
            feasible = []
            for kind, queue in queues.items():
                # Skip large Episodes only when the remaining capacity
                # cannot fit them. They remain private 4A alternatives.
                while queue and queue[0]["decision_rounds_post_fork"] > remaining:
                    queue.popleft()
                if queue:
                    feasible.append(kind)
            if not feasible:
                break
            kind = min(feasible, key=lambda name: (spent[name], name))
            pick = queues[kind].popleft()
            chosen.add(pick["candidate_index"])
            turns = pick["decision_rounds_post_fork"]
            remaining -= turns
            spent[kind] += turns
        for index in selected - chosen:
            rows[index]["selection_purpose"] = "budget_reserve"
        selected = chosen

    # Keep some *genuine* non-recovered branches for reliable observable
    # claim-state diagnostics. They must have a replayed episode; do not
    # invent failures or turn them into supervised recovery decisions.
    diagnostics = set()
    if max_diagnostics_per_scene:
        by_scene = defaultdict(list)
        for row in rows:
            if row["execution_outcome"] == "recovered":
                continue
            journal = journals[row["candidate_index"]]
            if not isinstance(journal.get("episode"), dict):
                continue
            if not isinstance(journal.get("private_audit"), dict):
                continue
            by_scene[row["question_id"]].append(row)
        outcome_priority = {
            "incorrect_result_after_actual_propagation": 0,
            "unverified_report_claim": 1,
            "short_replay_horizon": 2,
            "unresolved_dependencies": 3,
            "max_decision_rounds": 4,
        }
        diagnostic_counts = Counter()
        for scene_id in sorted(by_scene):
            choices = sorted(by_scene[scene_id], key=lambda row: (
                outcome_priority.get(row["execution_outcome"], 9),
                tie(row),
            ))
            seen_behaviors = set()
            for candidate in choices:
                identity = (
                    candidate["execution_outcome"],
                    candidate["private_fault_family"],
                    candidate["main_visible_delivery_state"],
                )
                if identity in seen_behaviors:
                    continue
                seen_behaviors.add(identity)
                candidate["selection_purpose"] = "diagnostic"
                diagnostics.add(candidate["candidate_index"])
                diagnostic_counts[scene_id] += 1
                if diagnostic_counts[scene_id] >= max_diagnostics_per_scene:
                    break

    chosen = selected | diagnostics
    for row in rows:
        row["selected_for_language_review"] = (
            row["candidate_index"] in chosen
        )
        row["selection_reason"] = (
            "causally_distinct_recovered_behavior"
            if row["selection_purpose"] == "positive"
            else "nonrecovered_observation_diagnostic"
            if row["selection_purpose"] == "diagnostic"
            else "global_turn_budget_reserve"
            if row["selection_purpose"] == "budget_reserve"
            else "not_execution_recovered"
            if row["execution_outcome"] != "recovered"
            else "deduplicated_or_lower_priority_within_feedback_round"
        )
    counts = Counter(row["category"] for row in rows)
    chosen_counts = Counter(
        row["category"] for row in rows
        if row["selected_for_language_review"]
    )
    fault_families = Counter(
        row["private_fault_family"] for row in rows
        if row["private_fault_family"]
    )
    visible_states = Counter(
        row["main_visible_delivery_state"] for row in rows
    )
    observability_distribution = Counter(
        row["observability"] for row in rows
    )
    issue_family_distribution = Counter(
        row["observable_issue_family"] for row in rows
    )
    recovery_effect_distribution = Counter(
        row["recovery_effect"] for row in rows
    )
    stats = {
        "target_main_turns": target_main_turns,
        "selected_answer_impact_distribution": dict(Counter(
            row["answer_impact"] for row in rows if row["candidate_index"] in selected
        )),
        "fault_kind_execution_outcomes": {
            kind: dict(sorted(Counter(
                row["execution_outcome"] for row in rows
                if row["atom_kind"] == kind
            ).items()))
            for kind in sorted({row["atom_kind"] for row in rows
                                if row["atom_kind"]})
        },
        "fault_kind_observability": {
            kind: dict(sorted(Counter(
                row["observability"] for row in rows
                if row["atom_kind"] == kind
            ).items()))
            for kind in sorted({row["atom_kind"] for row in rows
                                if row["atom_kind"]})
        },
        "computation_fault_operators": dict(sorted(Counter(
            row["private_computation_operation"] or "unknown"
            for row in rows if row["atom_kind"] == "computation_error"
        ).items())),
        "estimated_selected_main_turns": sum(
            rows[index]["decision_rounds_post_fork"] for index in selected
        ),
        "available_recovered_by_fault_kind": dict(
            sorted(available_by_kind.items())
        ),
        "selected_recovered_by_fault_kind": dict(sorted(Counter(
            rows[index]["atom_kind"] for index in selected
        ).items())),
        "historical_anchor_status_distribution": dict(sorted(Counter(
            row["historical_anchor_status"] for row in rows
            if row["historical_anchor_status"] not in (None, "not_applicable")
        ).items())),
        "historical_anchor_reason_distribution": dict(sorted(Counter(
            row["historical_anchor_reason"] for row in rows
            if row["historical_anchor_reason"]
        ).items())),
        "cross_round_report_conflicts_detected": sum(
            row["history_conflicts_detected"] for row in rows
        ),
        "historical_conflict_older_round_distribution": dict(sorted(
            Counter(old for row in rows
                    for old in row["history_conflict_older_rounds"]).items()
        )),
        "historical_conflict_round_gap_distribution": dict(sorted(
            Counter(gap for row in rows
                    for gap in row["history_conflict_round_gaps"]).items()
        )),
        "selected_recovered_historical_conflicts": sum(
            row["selection_purpose"] == "positive"
            and row["history_conflicts_detected"] > 0 for row in rows
        ),
        "selected_historical_conflict_round_gap_distribution": dict(sorted(
            Counter(gap for row in rows
                    if row["selection_purpose"] == "positive"
                    for gap in row["history_conflict_round_gaps"]).items()
        )),
        "candidate_delayed_propagation_trials": sum(
            row["scenario_blueprint"] == "delayed_consumed_error_recovery"
            for row in rows
        ),
        "execution_recovered_delayed_propagation": sum(
            row["scenario_blueprint"] == "delayed_consumed_error_recovery"
            and row["execution_outcome"] == "recovered" for row in rows
        ),
        "selected_recovered_delayed_propagation": sum(
            row["scenario_blueprint"] == "delayed_consumed_error_recovery"
            and row["selection_purpose"] == "positive" for row in rows
        ),
        "distinct_selected_policy_signatures": len({
            row["policy_signature"] for row in rows
            if row["selection_purpose"] == "positive"
        }),
        "distinct_selected_case_signatures": len({
            (row["question_id"], row["feedback_round"],
             row["behavior_signature"]) for row in rows
            if row["selection_purpose"] == "positive"
        }),
        "candidates_simulated": len(rows),
        "candidates_execution_recovered": sum(
            row["execution_outcome"] == "recovered" for row in rows
        ),
        "candidates_selected_for_language_review": len(chosen),
        "diagnostics_selected_for_language_review": len(diagnostics),
        "candidates_filtered_before_llm": len(rows) - len(chosen),
        "candidate_behavior_distribution": dict(sorted(counts.items())),
        "private_fault_family_distribution": dict(
            sorted(fault_families.items())
        ),
        "main_visible_delivery_state_distribution": dict(
            sorted(visible_states.items())
        ),
        "observability_distribution": dict(
            sorted(observability_distribution.items())
        ),
        "observable_issue_family_distribution": dict(
            sorted(issue_family_distribution.items())
        ),
        "recovery_effect_distribution": dict(
            sorted(recovery_effect_distribution.items())
        ),
        "selected_observability_distribution": dict(sorted(Counter(
            row["observability"] for row in rows
            if row["selected_for_language_review"]
        ).items())),
        "selected_behavior_distribution": dict(sorted(
            chosen_counts.items()
        )),
        "feedback_rounds_with_selected_candidates": len({
            (row["question_id"], row["feedback_round"])
            for row in rows if row["selected_for_language_review"]
        }),
    }
    return chosen, rows, stats
