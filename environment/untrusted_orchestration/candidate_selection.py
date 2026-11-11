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


def classify_trial(plan: dict, journal: dict) -> dict:
    audit = journal.get("private_audit") or {}
    runtime = audit.get("runtime") or {}
    applied = audit.get("fault_events_applied") or []
    invalidations = runtime.get("invalidations") or []
    revalidations = runtime.get("revalidations") or []
    conflicts = runtime.get("detected_conflicts") or []
    requests = [
        request for entry in audit.get("evidence_plan", [])
        for request in entry.get("evidence_requests", [])
    ]
    operations = Counter(row.get("operation", "unknown") for row in requests)
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
        "legacy_wrong_evidence", "fabricated_claim",
    } for kind in kinds):
        category = "recovered_unreliable_evidence"
    elif "incomplete_retrieval" in kinds:
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
    signature = sha256(json.dumps(
        {"category": category, "requests": behavior},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()[:16]
    return {
        "round_sample_id": plan["sample_id"],
        "candidate_index": plan["index"],
        "question_id": plan["task_id"],
        "feedback_round": plan["branch_round"],
        "atom_kind": kinds[0] if kinds else None,
        "fault_target": (
            plan["faults"][0].task_id if plan["faults"] else None
        ),
        "variant_index": plan.get("candidate_variant_index"),
        "execution_outcome": (
            audit.get("execution_outcome") or outcome
        ),
        "category": category,
        "behavior_signature": signature,
        "fault_events_applied": len(applied),
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
    max_selected_per_round: int,
) -> tuple[set[int], list[dict], dict]:
    """Rank by ACTUAL causal behavior, not by the name of the fault."""
    if type(max_selected_per_round) is not int or not (
        1 <= max_selected_per_round <= 8
    ):
        raise ValueError("invalid_selection_cap")
    if len(planned) != len(journals):
        raise ValueError("candidate_selection_journal_alignment_invalid")
    rows = [
        classify_trial(plan, journal)
        for plan, journal in zip(planned, journals)
    ]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["question_id"], row["feedback_round"])].append(row)
    chosen = set()
    for group in groups.values():
        eligible = [
            row for row in group
            if row["execution_outcome"] == "recovered"
        ]
        eligible.sort(key=lambda row: (
            _PRIORITY.get(row["category"], 99),
            -row["actual_invalidations"],
            -row["visible_conflicts"],
            -row["decision_rounds_post_fork"],
            row["candidate_index"],
        ))
        seen = set()
        for candidate in eligible:
            if len([x for x in group if x["candidate_index"] in chosen]) >= (
                max_selected_per_round
            ):
                break
            if candidate["behavior_signature"] in seen:
                continue
            chosen.add(candidate["candidate_index"])
            seen.add(candidate["behavior_signature"])
    for row in rows:
        row["selected_for_language_review"] = (
            row["candidate_index"] in chosen
        )
        row["selection_reason"] = (
            "causal_recovered_distinct_behavior"
            if row["selected_for_language_review"]
            else "not_execution_recovered"
            if row["execution_outcome"] != "recovered"
            else "deduplicated_or_lower_priority_within_feedback_round"
        )
    counts = Counter(row["category"] for row in rows)
    chosen_counts = Counter(
        row["category"] for row in rows
        if row["selected_for_language_review"]
    )
    stats = {
        "candidates_simulated": len(rows),
        "candidates_execution_recovered": sum(
            row["execution_outcome"] == "recovered" for row in rows
        ),
        "candidates_selected_for_language_review": len(chosen),
        "candidates_filtered_before_llm": len(rows) - len(chosen),
        "candidate_behavior_distribution": dict(sorted(counts.items())),
        "selected_behavior_distribution": dict(sorted(
            chosen_counts.items()
        )),
        "feedback_rounds_with_selected_candidates": len({
            (row["question_id"], row["feedback_round"])
            for row in rows if row["selected_for_language_review"]
        }),
    }
    return chosen, rows, stats
