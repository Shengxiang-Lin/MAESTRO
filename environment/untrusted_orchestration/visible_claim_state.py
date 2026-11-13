"""Private, round-wise Claim–Source–Dependency supervision.

The ONLY value-bearing evidence source is Sub-Agent Result/Summary text
actually returned before the next Main decision. The synthetic World is
consulted solely for DECLARED source provenance (not facts or true answers).
Teacher task dependencies are recorded in the private audit, never passed
as completed observations to Main or mixed into its public history.

One synchronous returned batch forms one snapshot; one atomic fault is
supported by the Stage-4 runner. Failed retry reports revoke earlier
claims from that same task instead of silently preserving clean values.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json

from environment.generate_question.evidence_operators import (
    check_warrant, relate_claims,
)
from .visible_claim_extraction import extract_visible_claims

SCHEMA = "claim_source_dependency"
SCHEMA_VERSION = 1


def _claim_id(task_id: str, round_number: int, claim: dict) -> str:
    payload = [task_id, round_number, claim["subject"], claim["attribute"],
               claim["valid_scope"], claim["surface"], claim["value"]]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return "claim:" + sha256(raw.encode("utf-8")).hexdigest()[:20]


def _state_for(claims: list[dict]) -> dict:
    """A claim status is an evidential warrant, never private correctness."""
    source_values, report_values = defaultdict(set), defaultdict(set)
    for claim in claims:
        source_values[claim["source_id"] or "unspecified"].add(
            (type(claim["value"]).__name__, repr(claim["value"]))
        )
        report_values[claim["report_task_id"]].add(
            (type(claim["value"]).__name__, repr(claim["value"]))
        )
    local_conflict = any(len(v) > 1 for v in report_values.values())
    partial_observed = any(
        claim.get("delivery_status") == "partial" for claim in claims
    )
    # Source-free derived reports may repeat previously returned facts.
    # They must neither manufacture an independent witness nor erase
    # two genuinely independent primary-source confirmations.
    primary = [claim for claim in claims if claim["evidence_origin_id"]]
    all_values = {(type(c["value"]).__name__, repr(c["value"]))
                  for c in claims}
    if local_conflict:
        warrant = {
            "status": "intra_report_conflict",
            "sufficient_to_accept": False,
            "independent_origins": len({
                claim["evidence_origin_id"] for claim in claims
                if claim["evidence_origin_id"]
            }),
            "missing_evidence": ["reconcile_inconsistent_report"],
        }
    elif len(all_values) > 1:
        warrant = {
            "status": "disputed",
            "sufficient_to_accept": False,
            "independent_origins": len({
                c["evidence_origin_id"] for c in primary
            }),
            "missing_evidence": ["resolve_disagreement"],
        }
    elif primary:
        warrant = check_warrant(primary)
    else:
        # Unknown provenance never counts as an independent root.
        warrant = {
            "status": "unverified",
            "sufficient_to_accept": False,
            "independent_origins": len({
                c["evidence_origin_id"] for c in claims
                if c["evidence_origin_id"]
            }),
            "missing_evidence": ["clarify_provenance"],
        }
    # A partial handoff can still expose a real contradiction, but it
    # cannot independently establish complete evidence for acceptance.
    if partial_observed and warrant["status"] not in {
        "disputed", "intra_report_conflict",
    }:
        warrant = {
            **warrant, "status": "partial",
            "sufficient_to_accept": False,
            "missing_evidence": ["complete_partial_delivery"],
        }
    return {
        "status": warrant["status"],
        "sufficient_to_accept": warrant["sufficient_to_accept"],
        "independent_origins": warrant["independent_origins"],
        "report_sources": sorted(
            x for x in source_values if x != "unspecified"
        ),
        "internal_disagreement": local_conflict,
        "evidence_count": len(claims),
        "supporting_claim_ids": sorted(c["claim_id"] for c in claims),
        "missing_evidence": list(warrant.get("missing_evidence", [])),
        "evidence_posture": (
            "eligible_to_use_with_provenance"
            if warrant["sufficient_to_accept"] else
            "reconcile" if warrant["status"] in {
                "disputed", "intra_report_conflict"
            } else "complete_partial_delivery"
            if warrant["status"] == "partial" else
            "obtain_independent_evidence"
        ),
    }


def _claim_relations(group: list[dict]) -> list[dict]:
    edges = []
    for a, b in combinations(group, 2):
        # A single report's two surfaces are not independent witnesses.
        if a["report_task_id"] == b["report_task_id"]:
            relation = ("same_report_observation"
                        if a["value"] == b["value"]
                        else "same_report_conflict")
        elif a["evidence_origin_id"] and b["evidence_origin_id"]:
            relation = relate_claims(a, b)["relation"]
        else:
            relation = ("unverified_same_value"
                        if a["value"] == b["value"]
                        else "visible_disagreement")
        edges.append({
            "from": a["claim_id"], "to": b["claim_id"],
            "relation": relation,
        })
    return edges


def audit_visible_claim_states(scene: dict, trajectory: list[dict], *,
                               task_ids_by_round: dict[int, list[str]]) -> dict:
    """Replay the received history without ever reading hidden answer rows.

    Includes all typed, explicit values that can be parsed from public
    feedback, regardless of a particular Stage-1 problem pattern.
    An unparseable completed report is UNKNOWN, not evidence of truth.
    """
    tasks = {
        row["task_id"]: row for row in scene["macro_graph"]["tasks"]
    }
    source_info = scene.get("world", {}).get("sources", {})
    # This teacher metadata is *only* used to annotate source independence;
    # no rows from source_info are interpreted or surfaced to Main.
    origin_by_source = {
        sid: source.get("evidence_origin_id")
        for sid, source in source_info.items()
        if isinstance(source, dict)
    }
    supporting_subjects = {
        item["subject"]
        for item in scene.get("evidence_ledger", {}).get(
            "reference_wave_trust", {}
        ).get("transitions", [])
        if item.get("answer_supporting_claim_pair")
    }
    latest_by_task = {}
    snapshots, prior_states = [], {}
    counters = Counter()
    last_round = 0
    for action in trajectory:
        if action.get("action") != "delegate_task":
            continue
        round_number = action.get("attempt")
        if type(round_number) is not int or round_number <= last_round:
            raise ValueError("visible_claim_invalid_dispatch_round")
        last_round = round_number
        tids = task_ids_by_round.get(round_number)
        reports = (action.get("result") or {}).get("subtask_results")
        if (not isinstance(tids, list) or not isinstance(reports, list)
                or len(tids) != len(reports)):
            raise ValueError("visible_claim_dispatch_alignment_invalid")
        events, dependencies = [], []
        for tid, report in zip(tids, reports):
            if tid not in tasks:
                raise ValueError("visible_claim_unknown_macro")
            if not isinstance(report, dict):
                raise ValueError("visible_claim_invalid_report")
            task = tasks[tid]
            scopes = list(task.get("source_ids", []))
            source_id = scopes[0] if len(scopes) == 1 else None
            fields = task.get("outputs", {}).get("typed_relation", [])
            found, parse = extract_visible_claims(
                report, output_fields=fields, source_id=source_id,
            )
            replaced = tid in latest_by_task
            if replaced:
                counters["source_report_replacements"] += 1
            # The returned content of a retry replaces, not accumulates,
            # that task's previous observations, even if missing/partial.
            latest_by_task[tid] = [{
                **c, "claim_id": _claim_id(tid, round_number, c),
                "report_task_id": tid, "observed_in_round": round_number,
                "delivery_status": parse["status"],
                "evidence_origin_id": (
                    origin_by_source.get(source_id)
                    if source_id is not None else None
                ),
                "revision_seq": 1, "revision_of": None,
                "authoritative": False,
            } for c in found]
            counters["literal_claims_extracted"] += len(found)
            if parse["status"] in {"done", "partial"} and not found:
                counters["unparsed_completed_report_slots"] += 1
            if parse["status"] != "done":
                counters["noncompleted_report_slots"] += 1
            dependencies.extend({
                "parent_task_id": parent,
                "child_task_id": tid,
                "parent_report_observed": parent in latest_by_task,
                "relation": "executed_task_depends_on",
            } for parent in task.get("depends_on", []))
            events.append({
                "macro_task_id": tid, "source_id": source_id,
                "status": parse["status"],
                "parse_mode": parse["parse_mode"],
                "visible_claim_count": len(found),
                "replaced_earlier_report": replaced,
            })

        # Claim-level lineage only links actual returned child and parent
        # assertions of the same entity. It is a dependency indicator,
        # NOT a claim that the parent or child value is factually correct.
        claim_dependency_edges = []
        for tid, claims in latest_by_task.items():
            for parent in tasks[tid].get("depends_on", []):
                for child_claim in claims:
                    for parent_claim in latest_by_task.get(parent, []):
                        if child_claim["subject"] == parent_claim["subject"]:
                            claim_dependency_edges.append({
                                "parent_claim_id": parent_claim["claim_id"],
                                "child_claim_id": child_claim["claim_id"],
                                "relation": "same_subject_task_dependency",
                            })
        grouped = defaultdict(list)
        active_claims = [
            c for claims in latest_by_task.values() for c in claims
        ]
        for claim in active_claims:
            key = (
                claim["subject"], claim["attribute"],
                claim["valid_scope"], type(claim["value"]).__name__,
            )
            grouped[key].append(claim)
        current = {}
        relations = []
        for key, claims in sorted(grouped.items()):
            group_key = "|".join(key)
            label = _state_for(claims)
            label["subject"], label["attribute"] = key[0], key[1]
            label["valid_scope"], label["value_type"] = key[2], key[3]
            label["answer_supporting_subject_private"] = (
                key[0] in supporting_subjects
            )
            current[group_key] = label
            relations.extend(_claim_relations(claims))
        transitions = []
        for key in sorted(set(prior_states) | set(current)):
            before = prior_states.get(key, {}).get("status", "missing")
            after = current.get(key, {}).get("status", "missing")
            if before == after:
                continue
            subject = (current.get(key) or prior_states[key])["subject"]
            transitions.append({
                "claim_key": key, "subject": subject,
                "before": before, "after": after,
                "next_main_decision_round": round_number + 1,
                "answer_supporting_subject_private":
                    subject in supporting_subjects,
            })
            counters["visible_status_transitions"] += 1
            if before == "unverified" and after == "disputed":
                counters["late_visible_disputes"] += 1
            if before == "unverified" and after == "corroborated":
                counters["late_visible_corroborations"] += 1

        snapshots.append({
            "delegation_round": round_number,
            "main_can_act_from_round": round_number + 1,
            "returned_source_reports": events,
            "claim_nodes": active_claims,
            "claim_states": current,
            "claim_relations": relations,
            "claim_dependency_edges": claim_dependency_edges,
            "task_dependency_edges": [{
                "parent_task_id": parent,
                "child_task_id": tid,
                "parent_report_observed": parent in latest_by_task,
                "relation": "executed_task_depends_on",
            } for tid in latest_by_task
                for parent in tasks[tid].get("depends_on", [])],
            "state_changes": transitions,
        })
        prior_states = current

    counters["rounds_with_explicit_claims"] = sum(
        bool(row["claim_nodes"]) for row in snapshots
    )
    counters["distinct_claim_keys_observed"] = len({
        key for row in snapshots for key in row["claim_states"]
    })
    counters["rounds_with_disputed_claims"] = sum(
        any(label["status"] in {"disputed", "intra_report_conflict"}
            for label in row["claim_states"].values())
        for row in snapshots
    )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "applicable": bool(snapshots),
        "observation_scope": (
            "only_main_visible_result_and_summary;"
            "source_lineage_from_declared_provenance;"
            "no_oracle_values_or_future_reports;"
            "state_after_completed_delegate_batch"
        ),
        "rounds": snapshots,
        "summary": dict(counters),
        "private_only": True,
    }
