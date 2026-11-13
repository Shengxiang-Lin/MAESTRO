"""Claim trust-state audit based ONLY on actually Main-visible Sub-Agent text.

Private teacher-side diagnostics, NOT a new field in the O1 Main prompt.
No unseen World rows, clean Gold values, or planned future reports may be
materialized as received evidence. Evidence origins are provenance metadata,
not independently inferred from model prose.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from environment.generate_question.evidence_operators import check_warrant
from environment.implement_subtask.visible_evidence import visible_year_claims

VERSION = "maestro.visible_claim_state.v1"
REPORT_SOURCES = frozenset({"report_a", "report_b"})


def audit_visible_claim_states(
    scene: dict, trajectory: list[dict], *,
    task_ids_by_round: dict[int, list[str]],
) -> dict:
    """Replay actual *delivered* result/summary, including faulty replacements.

    Each dispatch's results become visible at the NEXT Main decision round.
    Repeated dispatch of a task replaces its earlier public assertions; a
    retry that returns nothing cannot silently inherit its clean old rows.
    """
    plan = scene.get("problem_spec", {}).get("macro_plan", {})
    if (not scene.get("problem_spec", {}).get("evidence_trust_version")
            or plan.get("pattern") not in {
                "disputed_reports", "corroborated_dispute",
            }):
        return {
            "version": VERSION, "applicable": False,
            "rounds": [], "summary": {},
        }
    tasks = {
        item["task_id"]: item
        for item in scene["macro_graph"]["tasks"]
    }
    origins = {
        sid: scene["world"]["sources"][sid].get("evidence_origin_id")
        for sid in REPORT_SOURCES
    }
    if not all(origins.values()):
        raise ValueError("visible_claim_state_missing_independence_origin")
    supporting_subjects = {
        row["subject"]
        for row in scene.get("evidence_ledger", {}).get(
            "reference_wave_trust", {}
        ).get("transitions", [])
        if row.get("answer_supporting_claim_pair")
    }
    latest_by_task = {}
    snapshots, old_states = [], {}
    counts = Counter()
    last_dispatch_round = 0
    for step in trajectory:
        if step.get("action") != "delegate_task":
            continue
        round_number = step.get("attempt")
        if (type(round_number) is not int or
                round_number <= last_dispatch_round):
            raise ValueError("visible_claim_invalid_dispatch_round")
        last_dispatch_round = round_number
        tids = task_ids_by_round.get(round_number)
        reports = (step.get("result") or {}).get("subtask_results")
        if (not isinstance(tids, list) or not isinstance(reports, list)
                or len(tids) != len(reports)):
            raise ValueError("visible_claim_dispatch_alignment_invalid")
        events = []
        for tid, report in zip(tids, reports):
            if tid not in tasks:
                raise ValueError("visible_claim_unknown_macro")
            scope = tasks[tid]["source_ids"]
            if len(scope) != 1 or scope[0] not in REPORT_SOURCES:
                continue
            source_id = scope[0]
            observed = visible_year_claims(report, source_id)
            # A source report is replaced after a retry; never add both
            # old and new assertions as independent witnesses.
            previous = latest_by_task.get(tid, [])
            latest_by_task[tid] = observed
            counts["literal_claims_extracted"] += len(observed)
            if previous:
                counts["source_report_replacements"] += 1
            if not observed and (report.get("finish_result") or {}).get(
                    "status") == "done":
                counts["unparsed_completed_report_slots"] += 1
            events.append({
                "macro_task_id": tid, "source_id": source_id,
                "visible_claim_count": len(observed),
                "replaced_earlier_report": bool(previous),
                "status": (report.get("finish_result") or {}).get(
                    "status", "missing",
                ),
            })
        visible = defaultdict(list)
        for tid, assertions in latest_by_task.items():
            for item in assertions:
                visible[item["subject"]].append({
                    "claim_id": (
                        tid + ":" + item["subject"] + ":" + str(item["value"])
                    ),
                    "subject": item["subject"],
                    "attribute": "reported_event_year",
                    "valid_scope": "same_reported_event",
                    "value": item["value"], "source_id": item["source_id"],
                    "evidence_origin_id": origins[item["source_id"]],
                    "revision_seq": 1, "revision_of": None,
                    "authoritative": False, "fact_refs": [],
                })
        current = {}
        for subject, claims in sorted(visible.items()):
            # Conflicting statements within a single report cannot be
            # treated as TWO independent confirmations or resolved as truth.
            source_values = defaultdict(set)
            for claim in claims:
                source_values[claim["source_id"]].add(claim["value"])
            internally_conflicted = any(
                len(values) > 1 for values in source_values.values()
            )
            warrant = check_warrant(claims)
            if internally_conflicted:
                warrant = {
                    **warrant,
                    "status": "intra_report_conflict",
                    "sufficient_to_accept": False,
                    "winning_claim_id": None,
                    "missing_evidence": ["reconcile_inconsistent_report"],
                }
            current[subject] = {
                "status": warrant["status"],
                "sufficient_to_accept": warrant["sufficient_to_accept"],
                "independent_origins": warrant["independent_origins"],
                "report_sources": sorted(source_values),
                "internal_disagreement": internally_conflicted,
                "answer_supporting_subject_private": subject in supporting_subjects,
            }
        transitions = []
        for subject in sorted(set(old_states) | set(current)):
            before = old_states.get(subject, {}).get("status", "missing")
            after = current.get(subject, {}).get("status", "missing")
            if before != after:
                transitions.append({
                    "subject": subject,
                    "before": before, "after": after,
                    "next_main_decision_round": round_number + 1,
                    "answer_supporting_subject_private":
                        subject in supporting_subjects,
                })
                counts["visible_status_transitions"] += 1
                if before == "unverified" and after == "disputed":
                    counts["late_visible_disputes"] += 1
                if before == "unverified" and after == "corroborated":
                    counts["late_visible_corroborations"] += 1
        snapshots.append({
            "delegation_round": round_number,
            "main_can_act_from_round": round_number + 1,
            "returned_source_reports": events,
            "states": current,
            "state_changes": transitions,
        })
        old_states = current
    return {
        "version": VERSION, "applicable": True,
        "observation_scope": (
            "only_literal_public_result_and_summary;not_private_world;"
            "state_after_completed_delegate_batch"
        ),
        "rounds": snapshots,
        "summary": dict(counts),
        "private_only": True,
    }
