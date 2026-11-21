"""Choose independent, visibly actionable Main-turn counterfactuals.

A Stage-3 feedback round can generate many one-fault variants. We keep only
variants whose *current received reports* establish a public problem and for
which the frozen dependency-safe planner has emitted one legal next action.
No future Sub-Agent response or globally successful recovery is required.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from hashlib import sha256
import json


def _rank(row: dict, seed: int) -> str:
    key = f"{seed}:{row['question_id']}:{row['feedback_round']}:{row['candidate_index']}"
    return sha256(key.encode("utf-8")).hexdigest()



def evidence_relation(issues: list[dict], conflicts: list[dict]) -> str:
    """Categorize only an observable, current-feedback evidence relationship.

    Same-source repeated-query inconsistencies are NOT independent evidence.
    An independent claim disagreement needs its own explicit source-origin
    comparison in ExecutionState, never a shared field-name heuristic.
    """
    kinds = {event.get("kind") for event in conflicts}
    if "explicit_independent_claim_disagreement" in kinds:
        return "independent_source_claim_conflict"
    if "cross_round_report_disagreement" in kinds:
        kinds = {event.get("comparison_kind") for event in conflicts}
        if "same_query_result_set" in kinds:
            return "same_source_repeated_query_conflict"
        if "earlier_keyed_handoff" in kinds:
            return "historical_keyed_handoff_conflict"
        return "historical_same_claim_conflict"
    issue_kinds = {issue.get("kind") for issue in issues}
    for kind in (
        "historical_report_conflict", "cross_source_conflict",
        "intra_report_conflict", "entity_association_inconsistency",
        "arithmetic_inconsistency", "lookup_key_disagreement",
        "derived_identifier_disagreement", "done_but_incomplete",
        "partial_delivery", "missing_delivery",
    ):
        if kind in issue_kinds:
            return kind
    return "other_visible_issue" if issues else "no_visible_issue"


def coordination_pattern(evidence_plan: list[dict]) -> tuple[str, tuple[str, ...], int]:
    """Describe a dispatch's actual choices, not a synthetic fault name."""
    current = evidence_plan[-1] if evidence_plan else {}
    operations = tuple(sorted({
        item.get("operation", "unknown")
        for item in current.get("evidence_requests", [])
    }))
    dispatch_count = len(current.get("dispatch", []))
    if dispatch_count <= 1:
        return "single_task", operations, dispatch_count
    verification = "verify_conflicting_evidence" in operations
    recovery = any(kind in operations for kind in (
        "retry_missing_evidence", "complete_partial_evidence",
        "refresh_dependent_evidence", "recheck_prior_evidence",
    ))
    advance = any(kind in operations for kind in (
        "retrieve_required_evidence", "derive_from_returned_evidence",
    ))
    if verification and advance:
        label = "verify_and_advance"
    elif recovery and advance:
        label = "repair_and_advance"
    elif verification:
        label = "parallel_verification"
    elif recovery:
        label = "parallel_repair"
    else:
        label = "parallel_independent_progress"
    return label, operations, dispatch_count


def current_public_witness(audit: dict, round_number: int) -> tuple[list, list]:
    """Only reports received just before this Main turn may trigger a sample."""
    latest = round_number - 1
    issues = [
        issue for snap in audit.get("observable_issue_snapshots", [])
        if snap.get("decision_round") == round_number
        for issue in snap.get("issues", [])
        if issue.get("reported_round") == latest
    ]
    conflicts = [
        event for event in (audit.get("runtime") or {}).get("detected_conflicts", [])
        if event.get("latest_round") == latest
    ]
    return issues, conflicts


def _claim_focus(audit: dict) -> tuple:
    focus = []
    for step in audit.get("evidence_plan", []):
        for request in step.get("evidence_requests", []):
            claim = request.get("verification_focus") or {}
            if claim:
                focus.append((
                    claim.get("basis"), claim.get("subject"),
                    claim.get("attribute"),
                ))
    return tuple(sorted(set(focus)))


def select_single_turn_candidates(
    planned: list[dict], journals: list[dict], *,
    target_main_turns: int | None, max_selected_per_round: int | None,
    seed: int,
) -> tuple[set[int], list[dict], dict]:
    """Select one-step labels, balancing independent feedback rounds and faults.

    Candidates have one Main action only. Original unrecovered scenarios are
    *not* used as a teacher signal; local visibility and causal dispatch
    legality are the necessary conditions, not eventual Gold equality.
    """
    rows = []
    unique = set()
    pools: dict[tuple, list[dict]] = defaultdict(list)
    for plan, data in zip(planned, journals):
        audit = (data or {}).get("private_audit") or {}
        issues, conflicts = current_public_witness(audit, plan["branch_round"])
        relation = evidence_relation(issues, conflicts)
        pattern, operations, dispatch_size = coordination_pattern(
            audit.get("evidence_plan", [])
        )
        focus = _claim_focus(audit)
        planned_actions = [
            step for step in (data or {}).get("episode", {}).get("trajectory", [])
            if step.get("attempt") == plan["branch_round"]
        ] if isinstance((data or {}).get("episode"), dict) else []
        valid = (
            (data or {}).get("status") == "single_turn_planned"
            and len(planned_actions) == 1
            and planned_actions[0].get("action") == "delegate_task"
            and bool(issues or conflicts)
            and audit.get("parallel_batches_verified") is True
            and len(audit.get("evidence_plan") or []) == 1
        )
        atom = (plan.get("scenario_plan") or {}).get("fault_atoms") or []
        kind = atom[0].get("kind", "unknown") if atom else "unknown"
        family = tuple(sorted({i["kind"] for i in issues})) or ("cross_report_conflict",)
        evidence = tuple(sorted((
            i["kind"], i.get("task_id"), i.get("evidence")
        ) for i in issues))
        # Distinguish genuine public claim variants, not simulator variant IDs.
        signature = (
            plan["task_id"], plan["branch_round"], kind, family,
            evidence,
            tuple((e.get("kind"), e.get("older_round"), e.get("latest_round"))
                  for e in conflicts),
            tuple((d.get("operation"), tuple(d.get("target_fields", [])))
                  for step in audit.get("evidence_plan", [])
                  for d in step.get("evidence_requests", [])),
            focus, pattern, dispatch_size,
        )
        novel = valid and signature not in unique
        if novel:
            unique.add(signature)
        row = {
            "candidate_index": plan["index"],
            "question_id": plan["task_id"],
            "feedback_round": plan["branch_round"],
            "atom_kind": kind,
            "category": "+".join(family),
            "evidence_relation": relation,
            "coordination_pattern": pattern,
            "dispatch_operations": list(operations),
            "parallel_task_count": dispatch_size,
            "claim_focus": [list(item) for item in focus],
            "stage3_prior_history_preserved": audit.get(
                "stage3_prior_history_preserved"
            ),
            "observable_issue_kinds": list(family) if issues or conflicts else [],
            "main_visible_justification": [i["evidence"] for i in issues],
            "execution_outcome": (data or {}).get("status"),
            "decision_rounds_post_fork": 1 if valid else 0,
            "answer_impact": "not_evaluated",
            "recovery_effect": "not_evaluated_single_turn",
            "actual_invalidations": 0,
            "history_conflicts_detected": len(conflicts),
            "recovery_rank": None,
            "selection_purpose": "not_selected",
            "selected_for_language_review": False,
            "selection_reason": (
                "eligible_visible_current_feedback" if novel else
                "no_legal_current_decision_or_no_visible_issue" if not valid else
                "duplicate_public_issue_and_action"
            ),
        }
        rows.append(row)
        if novel:
            # A distinct claim or action policy gets a fair first chance.
            # A flood of surface-text variants from one fault must not
            # monopolize the language budget.
            pools[(
                row["question_id"], row["feedback_round"], kind,
                relation, pattern, operations, focus,
            )].append(row)

    for pool in pools.values():
        pool.sort(key=lambda row: _rank(row, seed))
        for offset, row in enumerate(pool):
            row["recovery_rank"] = offset

    chosen: set[int] = set()
    per_round = Counter()
    # Fairly cover distinct original feedback rounds and fault types before
    # repeatedly sampling many variants of one surface.
    queues = {key: list(value) for key, value in pools.items()}
    relation_keys = defaultdict(list)
    for key in queues:
        relation_keys[key[3]].append(key)
    # First balance genuinely different *evidence relationships*, then
    # round/fault/action combinations within each relationship. Once a rare
    # relationship is exhausted, the remaining budget may be filled from
    # other valid cases; this is not a fabricated quota.
    relation_order = sorted(relation_keys, key=lambda label: (
        label not in {
            "independent_source_claim_conflict", "historical_same_claim_conflict",
            "historical_keyed_handoff_conflict",
        }, label
    ))
    groups = {
        relation: deque(sorted(keys, key=lambda key: sha256(
            f"{seed}:{key}".encode("utf-8")).hexdigest()))
        for relation, keys in relation_keys.items()
    }
    remaining = (target_main_turns if target_main_turns is not None else
                 sum(len(queue) for queue in queues.values()))
    while remaining > 0:
        changed = False
        for relation in relation_order:
            keys = groups[relation]
            while keys:
                key = keys.popleft()
                pool = queues[key]
                feedback = key[:2]
                if (not pool or (max_selected_per_round is not None and
                                 per_round[feedback] >= max_selected_per_round)):
                    continue
                row = pool.pop(0)
                if pool:
                    keys.append(key)
                row["selection_purpose"] = "positive"
                row["selected_for_language_review"] = True
                row["selection_reason"] = "balanced_public_relation_and_action"
                chosen.add(row["candidate_index"])
                per_round[feedback] += 1
                remaining -= 1
                changed = True
                break
            if remaining == 0:
                break
        if not changed:
            break
    for row in rows:
        if row["recovery_rank"] is not None and not row["selected_for_language_review"]:
            row["selection_purpose"] = "budget_reserve"
            row["selection_reason"] = "single_turn_budget_reserve"

    available = [row for row in rows if row["recovery_rank"] is not None]
    selected = [row for row in rows if row["selected_for_language_review"]]
    stats = {
        "candidates_simulated": len(planned),
        "candidates_execution_recovered": 0,
        "single_turn_actionable": len(available),
        "single_turn_deduplicated": sum(
            row["selection_reason"] == "duplicate_public_issue_and_action"
            for row in rows
        ),
        "candidates_selected_for_language_review": len(chosen),
        "diagnostics_selected_for_language_review": 0,
        "candidates_filtered_before_llm": len(rows) - len(chosen),
        "estimated_selected_main_turns": len(chosen),
        "candidate_behavior_distribution": dict(Counter(
            row["category"] for row in available)),
        "available_evidence_relations": dict(Counter(
            row["evidence_relation"] for row in available)),
        "selected_evidence_relations": dict(Counter(
            row["evidence_relation"] for row in selected)),
        "selected_coordination_patterns": dict(Counter(
            row["coordination_pattern"] for row in selected)),
        "selected_behavior_distribution": dict(Counter(
            row["category"] for row in selected)),
        "selected_recovered_by_fault_kind": {},
        "available_recovered_by_fault_kind": {},
        "selected_by_fault_kind": dict(Counter(
            row["atom_kind"] for row in selected)),
        "selected_observability_distribution": {
            "immediately_observable": len(chosen),
        },
        "selected_answer_impact_distribution": {"not_evaluated": len(chosen)},
        "distinct_selected_case_signatures": len(chosen),
        "distinct_selected_policy_signatures": len({
            (
                row["evidence_relation"], row["coordination_pattern"],
                tuple(row["dispatch_operations"]),
            ) for row in selected
        }),
        "feedback_rounds_with_selected_candidates": len(per_round),
    }
    return chosen, rows, stats
