"""Read-only Stage-1 evidence-affordance analysis for Stage-4 dataset design.

Different Macro tasks may be *execution-independent* but their outputs
may merely be complementary, not capable of verifying one another.
This analysis uses task contracts, available source schemas and graph
edges. It never invents a witness, asserts the truth of a claim, or
exposes an unseen source to a Main-Agent prompt.

Use this diagnostic to determine whether V10 needs a more expressive
dynamic verification operator or additional Stage-1 source relationships.
"""
from __future__ import annotations

from itertools import combinations


_ID_FIELDS = {
    "candidate_id", "case_id", "marker_code", "entry_label",
    "record_key_1", "record_key_2", "accession_id",
}


def _intra_macro_evidence(scene: dict, tasks: dict[str, dict]) -> list[dict]:
    """Audit frozen Macro-local source usage, without inventing a witness.

    Multiple sources in a JOIN normally supply complementary constraints.
    This PRIVATE diagnostic is not proof of comparable claims or a reason
    for Main to verify an unseen fact.
    """
    world = scene.get("world", {}).get("sources", {})
    feedback = {
        row.get("task_id"): row
        for row in scene.get("task_feedback", [])
        if isinstance(row, dict)
    }
    rows = []
    for task in tasks.values():
        scopes = list(dict.fromkeys(task.get("source_ids", [])))
        chain = task.get("operator_chain", [])
        role = task.get("macro_role")
        if len(scopes) < 2 and role != "verification":
            continue
        has_join = any(step.get("op") == "join" for step in chain)
        observed = feedback.get(task["task_id"], {})
        supported = []
        for item in observed.get("workproduct", []):
            refs = {
                ref.split(":", 1)[0]
                for ref in item.get("source_fact_refs", [])
                if isinstance(ref, str) and ":" in ref
            }
            supported.append(refs & set(scopes))
        cited_sources = sorted(set().union(*supported)) if supported else []
        roots_by_source = {
            sid: {
                obs["evidence_origin_id"]
                for obs in world.get(sid, {}).get("observations", [])
                if isinstance(obs, dict)
                and isinstance(obs.get("evidence_origin_id"), str)
            } for sid in scopes
        }
        known = all(roots_by_source.values())
        disjoint = known and all(
            not (roots_by_source[left] & roots_by_source[right])
            for left, right in combinations(scopes, 2)
        )
        rows.append({
            "macro_task_id": task["task_id"],
            "macro_role": role,
            "direct_source_ids": scopes,
            "acquisition_operations": [
                step["op"] for step in chain
                if step.get("op") in {"scan", "lookup_by_key"}
            ],
            "uses_join": has_join,
            "relationship": (
                "frozen_verification_gate" if role == "verification"
                else "multi_source_complementary_join" if has_join
                else "multi_source_acquisition"
            ),
            "declared_origin_distinctness": (
                "distinct_observation_origins" if disjoint
                else "overlapping_observation_origins" if known
                else "unknown"
            ),
            "executed_cited_sources": cited_sources,
            "executed_row_has_multiple_direct_sources": any(
                len(part) >= 2 for part in supported
            ),
            "claim_comparison_status": "not_established_by_macro_structure",
        })
    return rows


def analyze_scene_evidence_affordances(scene: dict) -> dict:
    graph = scene["macro_graph"]
    tasks = {row["task_id"]: row for row in graph["tasks"]}
    waves = {
        tid: i + 1
        for i, wave in enumerate(graph["reference_waves"])
        for tid in wave
    }
    memo = {}
    def ancestors(tid: str) -> set[str]:
        if tid in memo:
            return memo[tid]
        parents = set(tasks[tid]["depends_on"])
        result = parents | {
            earlier for parent in parents for earlier in ancestors(parent)
        }
        memo[tid] = result
        return result

    def sources(tid: str) -> set[str]:
        return set(tasks[tid]["source_ids"]) | {
            source for parent in ancestors(tid)
            for source in tasks[parent]["source_ids"]
        }

    def is_common_join_consumer(left: str, right: str) -> bool:
        return any(
            left in ancestors(tid) and right in ancestors(tid)
            and any(op["op"] == "join" for op in task["operator_chain"])
            for tid, task in tasks.items()
        )

    rows = []
    sourced = [
        task for task in tasks.values() if task["source_ids"]
    ]
    for a, b in combinations(sourced, 2):
        left, right = a["task_id"], b["task_id"]
        if left in ancestors(right) or right in ancestors(left):
            continue
        left_sources, right_sources = sources(left), sources(right)
        if left_sources & right_sources:
            continue
        common = sorted(
            set(a["outputs"]["typed_relation"])
            & set(b["outputs"]["typed_relation"])
        )
        if not common:
            continue
        fusion = is_common_join_consumer(left, right)
        # The two independently source-scoped directory Macros each
        # deliver a candidate's route_key -> entity_0 mapping. They are
        # comparable even when a *later* verifier joins their rows.
        # This is structural potential only: actual subject/scope equality
        # must still be witnessed in delivered Stage-2/3 claims.
        sibling_reference_claims = (
            {"route_key", "entity_0"} <= set(common)
            and all(len(task["source_ids"]) == 1
                    and task.get("macro_role") == "association"
                    for task in (a, b))
        )
        first, later = sorted((left, right), key=lambda tid: waves[tid])
        # This is a *structural potential*, not a claim that the later
        # source actually contradicts the earlier report.
        used_before_later = any(
            first in ancestors(tid)
            and tid != later and waves[tid] < waves[later]
            for tid in tasks
        )
        rows.append({
            "source_tasks": [left, right],
            "independent_source_scopes": [
                sorted(left_sources), sorted(right_sources)
            ],
            "shared_output_fields": common,
            "shared_identifier_fields": sorted(set(common) & _ID_FIELDS),
            "relationship": (
                "potential_same_field_cross_check"
                if sibling_reference_claims or not fusion
                else "complementary_fusion"
            ),
            "reported_in_same_reference_wave": (
                waves[left] == waves[right]
            ),
            "late_source_wave": (
                waves[later] if waves[later] > waves[first] else None
            ),
            "earlier_claim_consumed_before_later_report": (
                waves[later] > waves[first] and used_before_later
            ),
            "verification_status": (
                "schema_only_not_proven"
            ),
        })
    internal = _intra_macro_evidence(scene, tasks)
    return {
        "question_id": scene["task_id"],
        "macro_task_count": len(tasks),
        "source_scoped_macro_task_count": len(sourced),
        "independent_comparable_pairs": len(rows),
        "complementary_fusion_pairs": sum(
            r["relationship"] == "complementary_fusion" for r in rows
        ),
        "potential_cross_check_pairs": sum(
            r["relationship"] == "potential_same_field_cross_check"
            for r in rows
        ),
        "same_wave_comparable_pairs": sum(
            r["reported_in_same_reference_wave"] for r in rows
        ),
        "structurally_late_comparable_pairs": sum(
            r["late_source_wave"] is not None for r in rows
        ),
        "provisional_downstream_before_late_witness_pairs": sum(
            r["earlier_claim_consumed_before_later_report"] for r in rows
        ),
        "candidate_relationships": rows,
        "internal_macro_evidence": internal,
        "internal_multi_source_macros": sum(
            len(item["direct_source_ids"]) > 1 for item in internal
        ),
        "internal_verification_macros": sum(
            item["macro_role"] == "verification" for item in internal
        ),
        "internal_multi_source_supported_outputs": sum(
            item["executed_row_has_multiple_direct_sources"]
            for item in internal
        ),
        "interpretation": (
            "Potential same-field relation only. Exact entity/metric scope "
            "and independent support are not verified. Shared fields or a "
            "DAG join do not prove cross-validation. Internal verification "
            "and multi-source execution are audited separately, without "
            "asserting comparable claims. No synthetic evidence "
            "or new tool access was created."
        ),
    }
