"""Choose independent, visibly actionable Main-turn counterfactuals.

A Stage-3 feedback round can generate many one-fault variants. We keep only
variants whose *current received reports* establish a public problem and for
which the frozen dependency-safe planner has emitted one legal next action.
No future Sub-Agent response or globally successful recovery is required.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json


def _rank(row: dict, seed: int) -> str:
    key = f"{seed}:{row['question_id']}:{row['feedback_round']}:{row['candidate_index']}"
    return sha256(key.encode("utf-8")).hexdigest()


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
        snapshots = audit.get("observable_issue_snapshots") or []
        latest = plan["branch_round"] - 1
        issues = [
            issue for snapshot in snapshots
            if snapshot.get("decision_round") == plan["branch_round"]
            for issue in snapshot.get("issues", [])
            if issue.get("reported_round") == latest
        ]
        # A cross-report disagreement may be anchored on a later report
        # even if the generic per-Report issue collector has no field row.
        conflicts = [
            item for item in (audit.get("runtime") or {}).get("detected_conflicts", [])
            if item.get("latest_round") == latest
        ]
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
            pools[(row["question_id"], row["feedback_round"], kind)].append(row)

    for pool in pools.values():
        pool.sort(key=lambda row: _rank(row, seed))
        for offset, row in enumerate(pool):
            row["recovery_rank"] = offset

    chosen: set[int] = set()
    per_round = Counter()
    # Fairly cover distinct original feedback rounds and fault types before
    # repeatedly sampling many variants of one surface.
    queues = {key: list(value) for key, value in pools.items()}
    keys = sorted(queues, key=lambda key: sha256(
        f"{seed}:{key}".encode("utf-8")).hexdigest())
    remaining = target_main_turns if target_main_turns is not None else sum(
        len(queue) for queue in queues.values()
    )
    while remaining > 0:
        changed = False
        for key in keys:
            if remaining == 0:
                break
            pool = queues[key]
            feedback = key[:2]
            if (not pool or (max_selected_per_round is not None
                             and per_round[feedback] >= max_selected_per_round)):
                continue
            row = pool.pop(0)
            row["selection_purpose"] = "positive"
            row["selected_for_language_review"] = True
            row["selection_reason"] = "single_turn_public_issue"
            chosen.add(row["candidate_index"])
            per_round[feedback] += 1
            remaining -= 1
            changed = True
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
            (row["category"], row["atom_kind"]) for row in selected
        }),
        "feedback_rounds_with_selected_candidates": len(per_round),
    }
    return chosen, rows, stats
