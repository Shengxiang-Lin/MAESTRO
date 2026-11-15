"""Canonical Stage-1 quality checks for role/port-based Macro plans.

The original Stage-1 `quality.audit_scene`, `macro_contract.validate_handoffs`,
`information_necessity.audit_candidate_information` and strict frozen
Replay remain the authorities for legacy scenes. This module extends their
*checks*, rather than replacing them with a weaker DAG-only audit.

A typed-scene adapter supplies an executor with optional input overrides;
the checks are intentionally independent of the typed world's source format.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

VERSION = "maestro.stage1.typed_quality.v1"
RESULT_ROLES = {
    "investigation": ("finding", "FactBundle"),
    "verification": ("verified", "VerifiedClaim"),
    "comparison": ("comparison", "ComparisonResult"),
}


def audit_terminal_goal_coverage(scene: dict) -> dict:
    """Check every requested investigative conclusion appears in FinalResult.

    For current typed task drafts, Question requests linked findings and,
    conditionally, verification / comparison. A typed input merely being
    *used somewhere* is not enough: the actual terminal response must retain
    its requested work product and precise producer/port lineage.
    """
    nodes = scene["macro_plan"]["nodes"]
    if not nodes or nodes[-1]["role"] != "synthesis":
        raise ValueError("typed_goal_missing_terminal_synthesis")
    terminal = nodes[-1]
    links = terminal["input_ports"]
    final = scene["gold_answer"][0]
    if set(final) != {
        "identified_case", "identified_entity_uri", "findings",
        "original_witness_refs",
    }:
        raise ValueError("typed_goal_terminal_shape_mismatch")
    question_contract = scene["question_contract"]
    required = [
        node for node in nodes if node["role"] in RESULT_ROLES
    ]
    coverage = []
    delivered = {
        (entry.get("type"), str(entry.get("port")), entry.get("result") and
         repr(entry["result"]))
        for entry in final["findings"]
    }
    # All investigation, verification, and comparison outputs explicitly
    # requested by the Question must survive as typed FinalResult fields.
    for node in required:
        out_port, out_type = RESULT_ROLES[node["role"]]
        hits = [
            (name, link) for name, link in links.items()
            if name.startswith("result_")
            and link.get("from_macro") == node["macro_id"]
            and link.get("from_port") == out_port
            and link.get("type") == out_type
        ]
        if len(hits) != 1:
            raise ValueError("typed_goal_missing_result_handoff:" + node["macro_id"])
        name, link = hits[0]
        report = next(
            row for row in scene["task_feedback"]
            if row["macro_id"] == node["macro_id"]
        )
        expected_result = report["workproduct"][out_port]
        if not any(
            entry["port"] == name and entry["type"] == out_type
            and entry["result"] == expected_result
            for entry in final["findings"]
        ):
            raise ValueError("typed_goal_workproduct_not_delivered:" + node["macro_id"])
        coverage.append({
            "macro_id": node["macro_id"],
            "role": node["role"],
            "terminal_port": name,
            "type": out_type,
            "delivered": True,
        })
    if len(final["findings"]) != len(required):
        raise ValueError("typed_goal_extra_or_duplicate_terminal_results")
    expected_slots = sorted(
        port for port in links if port.startswith("result_")
    )
    if question_contract.get("result_slots") != expected_slots:
        raise ValueError("typed_goal_question_slots_mismatch")
    required_roles = question_contract.get("required_roles", [])
    if Counter(required_roles) != Counter(n["role"] for n in nodes):
        raise ValueError("typed_goal_question_role_mismatch")
    if question_contract.get("original_witness_required"):
        if not final["original_witness_refs"]:
            raise ValueError("typed_goal_missing_original_evidence")
    if question_contract.get("terminal_port") != "result":
        raise ValueError("typed_goal_wrong_terminal_port")
    return {
        "version": VERSION,
        "passed": True,
        "expected_results": len(required),
        "delivered_results": len(coverage),
        "coverage": coverage,
    }


def _counterfactual_value(type_name: str, original):
    """Create a deterministic, intentionally wrong but schema-shaped value."""
    value = deepcopy(original)
    if type_name == "CandidateSet":
        return []
    if type_name == "EntityRef":
        value["entity_uri"] = "urn:maestro:counterfactual:entity"
        return value
    if type_name == "FactBundle":
        value["linked_entity_uri"] = "urn:maestro:counterfactual:entity"
        value["linked_entity"] = "counterfactual-subject"
        return value
    if type_name == "VerifiedClaim":
        value["review_status"] = (
            "confirmed" if value["review_status"] != "confirmed"
            else "requires_followup"
        )
        return value
    if type_name == "ComparisonResult":
        value["same_entity"] = not value["same_entity"]
        return value
    if type_name == "EvidenceRefs":
        # EvidenceRefs are keyed per candidate; removal changes the required
        # final witness, not the identified subject.
        return {key: [] for key in value}
    raise ValueError("unsupported_counterfactual_port_type:" + type_name)


def audit_typed_handoffs(scene: dict, executor) -> dict:
    """Controlled one-port perturbation of *actual* task workproducts.

    A result-changing execution is semantic answer sensitivity. A failed
    execution under a schema-shaped perturbation is only operational
    blocking; report it separately instead of calling it proof of semantic
    necessity. This distinction matters for Verification and Comparison.
    """
    nodes = scene["macro_plan"]["nodes"]
    reports = {row["macro_id"]: row for row in scene["task_feedback"]}
    expected = scene["gold_answer"][0]
    records = []
    for child in nodes:
        for port, link in child["input_ports"].items():
            src = link["from_macro"]
            old = reports[src]["workproduct"][link["from_port"]]
            mutated = _counterfactual_value(link["type"], old)
            if mutated == old:
                raise ValueError("counterfactual_mutation_was_no_op")
            try:
                _, answer = executor(
                    scene, input_overrides={(child["macro_id"], port): mutated}
                )
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                changed, blocked = False, True
                error = f"{type(exc).__name__}:{str(exc)[:95]}"
            else:
                changed, blocked = answer != expected, False
                error = None
            records.append({
                "from_macro": src,
                "from_port": link["from_port"],
                "to_macro": child["macro_id"],
                "to_port": port,
                "type": link["type"],
                "answer_sensitive": changed,
                "execution_blocked": blocked,
                "error": error,
            })
    if not records:
        raise ValueError("typed_quality_missing_handoffs")
    # No result must silently survive a controlled change to its required
    # workproduct. This is an operational necessary/answer-sensitive gate,
    # not a proof of minimal search or universal causal necessity.
    ineffective = [
        row for row in records if not row["answer_sensitive"]
        and not row["execution_blocked"]
    ]
    return {
        "version": VERSION,
        "passed": not ineffective,
        "total_edges": len(records),
        "answer_sensitive": sum(r["answer_sensitive"] for r in records),
        "execution_blocked": sum(r["execution_blocked"] for r in records),
        "ineffective": ineffective,
        "edges": records,
    }


def audit_typed_scene(scene: dict, executor) -> dict:
    """Run both goal/causality gates for opt-in typed scenes."""
    coverage = audit_terminal_goal_coverage(scene)
    handoffs = audit_typed_handoffs(scene, executor)
    if not handoffs["passed"]:
        missing = ",".join(
            f"{x['from_macro']}:{x['from_port']}->{x['to_macro']}:{x['to_port']}"
            for x in handoffs["ineffective"][:4]
        )
        raise ValueError("typed_quality_ineffective_handoffs:" + missing)
    return {
        "version": VERSION,
        "passed": True,
        "goal_coverage": coverage,
        "handoffs": handoffs,
    }
