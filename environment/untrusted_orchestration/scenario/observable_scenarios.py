"""Stage-4 scenario targets: Main-visible insufficiency and inconsistency.

This is the ONLY source of scenario labels and operator-to-target mapping.
Targets are private generation plans; realized classes are independently
derived from claims and reports actually received by Main. Neither a Gold
value nor an injected Fault name is allowed to establish observability.
"""
from __future__ import annotations

from collections import defaultdict

from ..inspection.visible_claim_extraction import extract_visible_claims

# The four observable targets, not four new fault implementations.
TARGETS = {
    "A": ("evidence_insufficiency", None, (
        "no_result", "report_missing", "incomplete_retrieval",
        "done_but_incomplete",
    )),
    "B1": ("evidence_inconsistency", "internal", (
        "result_summary_conflict", "fabricated_claim",
    )),
    "B2": ("evidence_inconsistency", "constraint", (
        "wrong_retrieval", "wrong_entity_binding", "computation_error",
        "legacy_wrong_evidence",
    )),
    "B3": ("evidence_inconsistency", "inter_report", (
        "wrong_retrieval", "wrong_entity_binding", "fabricated_claim",
        "legacy_wrong_evidence",
    )),
}
# Rare, genuinely comparable cross-report claims get first opportunity.
CONSTRUCTION_ORDER = ("B3", "B2", "A", "B1")

_INSUFFICIENT = {
    "missing_delivery", "partial_delivery", "done_but_incomplete",
    "stale_downstream_result",
}
_INTERNAL = {"intra_report_conflict"}
_CONSTRAINT = {
    "entity_association_inconsistency", "arithmetic_inconsistency",
    "lookup_key_disagreement", "derived_identifier_disagreement",
    "scope_constraint_mismatch",
}
_INTER_REPORT = {"historical_report_conflict", "cross_source_conflict"}
_CROSS_REPORT_EVENTS = {
    "cross_round_report_disagreement",
    "explicit_independent_claim_disagreement",
}
# An overlap between candidate IDs returned by separate clue searches is
# not a repeated measurement of one attribute under the SAME query scope.
_UNSCOPED_IDENTIFIERS = {
    "case_id", "candidate_id", "marker_code", "entry_label",
    "accession_id", "record_key_1", "record_key_2",
}


def classify_observable_scenario(issues: list[dict], conflicts: list[dict]) -> dict:
    """Classify what Main could observe, not why a private executor failed.

    Preserve all applicable groups. Only a deterministic primary label is
    used for balancing; comparison timing, modality and provenance remain
    independent evidence attributes rather than new fault categories.
    """
    kinds = {row.get("kind") for row in issues}
    event_kinds = {row.get("kind") for row in conflicts}
    matched = set()
    if kinds & _INSUFFICIENT:
        matched.add("A")
    if kinds & _INTERNAL:
        matched.add("B1")
    if kinds & _CONSTRAINT:
        matched.add("B2")
    if kinds & _INTER_REPORT or event_kinds & _CROSS_REPORT_EVENTS:
        matched.add("B3")
    primary = next((name for name in CONSTRUCTION_ORDER if name in matched), None)
    family, comparison = TARGETS[primary][:2] if primary else (None, None)
    spans = {
        "cross_round" if (
            event.get("kind") == "cross_round_report_disagreement"
            or event.get("conflict_timing") == "delayed"
            or (isinstance(event.get("latest_round"), int)
                and isinstance(event.get("older_round"), int)
                and event["latest_round"] > event["older_round"])
        ) else "same_round"
        for event in conflicts
    }
    origins = {
        "independent" if event.get("kind")
        == "explicit_independent_claim_disagreement" else "not_established"
        for event in conflicts
    }
    return {
        "scenario_id": primary,
        "scenario_family": family,
        "comparison_type": comparison,
        "matched_scenarios": [name for name in CONSTRUCTION_ORDER if name in matched],
        "observable_issue_kinds": sorted(kind for kind in kinds if kind),
        "temporal_relations": sorted(spans),
        "source_relations": sorted(origins),
        "observability": "explicit_issue" if primary else "no_observable_issue",
    }


def _public_claims(task: dict, report: dict) -> set[tuple]:
    """Only statements already included in the delivered Result are comparable."""
    scopes = task.get("source_ids", [])
    claims, _ = extract_visible_claims(
        report,
        output_fields=task["outputs"]["typed_relation"],
        source_id=scopes[0] if len(scopes) == 1 else None,
    )
    return {
        (claim["subject"], claim["attribute"], claim["valid_scope"])
        for claim in claims
        if claim.get("surface") == "result"
        and claim.get("declared_output_field")
        and claim.get("subject")
        and not claim["subject"].startswith("source:")
        # Different clue queries may return overlapping candidate lists.
        # Their membership is not a rival assertion about a fixed fact.
        and claim.get("attribute") != "identifier_present"
        and claim.get("attribute") not in _UNSCOPED_IDENTIFIERS
    }


def _ancestors(tasks: dict[str, dict], task_id: str) -> set[str]:
    pending = list(tasks[task_id].get("depends_on", []))
    seen = set()
    while pending:
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            pending.extend(tasks[current].get("depends_on", []))
    return seen


def comparable_incoming_tasks(
    scene: dict, base_episode: dict | None, round_plan: dict,
) -> set[str]:
    """Find latest-batch tasks with an *actual* comparable public peer.

    Historical evidence is read exclusively from frozen, already delivered
    Stage-3 reports. No World values, hidden Gold, or synthetic rechecks.
    A derived parent-child result is not an independent rival assertion.
    """
    if not base_episode:
        return set()
    tasks = {task["task_id"]: task
             for task in scene["macro_graph"]["tasks"]}
    waves = scene["macro_graph"]["reference_waves"]
    incoming = set(round_plan["incoming_task_ids"])
    claims_by_task: dict[str, set[tuple]] = defaultdict(set)
    for action in base_episode.get("trajectory", []):
        if (action.get("action") != "delegate_task"
                or not isinstance(action.get("attempt"), int)
                or action["attempt"] >= round_plan["branch_round"]):
            continue
        pos = action["attempt"] - 1
        if pos >= len(waves):
            continue
        delivered = (action.get("result") or {}).get("subtask_results") or []
        if len(delivered) != len(waves[pos]):
            continue
        for tid, report in zip(waves[pos], delivered):
            if isinstance(report, dict) and tid in tasks:
                claims_by_task[tid].update(_public_claims(tasks[tid], report))
    comparable = set()
    for target in incoming:
        if not claims_by_task.get(target):
            continue
        ancestors = _ancestors(tasks, target)
        for peer, claims in claims_by_task.items():
            if peer == target or peer in ancestors or target in _ancestors(tasks, peer):
                continue
            if claims_by_task[target] & claims:
                comparable.add(target)
                break
    return comparable


def feasible_fault_targets(
    scene: dict, round_plan: dict, *,
    kinds: list[str], base_episode: dict | None,
) -> list[tuple[str, str, str]]:
    """Enumerate feasible (scenario, incoming task, atom) *in scenario order*.

    The feasibility gate is conservative but not an oracle: an executable
    operator is only a candidate. Stage 4A must independently establish the
    requested observable scenario after applying a real fault.
    """
    allowed = set(kinds)
    tasks = {task["task_id"]: task for task in scene["macro_graph"]["tasks"]}
    comparable = comparable_incoming_tasks(scene, base_episode, round_plan)
    targets = []
    for scenario in CONSTRUCTION_ORDER:
        for tid in round_plan["incoming_task_ids"]:
            task = tasks[tid]
            operations = {op["op"] for op in task.get("operator_chain", [])}
            if scenario == "B3" and tid not in comparable:
                continue
            for atom in TARGETS[scenario][2]:
                if atom not in allowed:
                    continue
                if scenario == "B2":
                    if atom == "wrong_entity_binding" and "join" not in operations:
                        continue
                    if atom == "computation_error" and not operations & {
                        "count", "group_count", "derive_percentage",
                        "derive_arithmetic", "derive_date_gap", "goal_compute",
                    }:
                        continue
                    if atom == "wrong_retrieval" and not operations & {
                        "scan", "lookup_by_key", "filter_eq", "filter_gte",
                        "take", "project", "sort", "unique",
                    }:
                        continue
                targets.append((scenario, tid, atom))
    return targets
