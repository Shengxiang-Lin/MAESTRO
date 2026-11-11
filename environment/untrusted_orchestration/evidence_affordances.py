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
                "complementary_fusion" if fusion
                else "potential_same_field_cross_check"
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
        "interpretation": (
            "Potential same-field relation only. Exact entity/metric scope "
            "and independent support are not verified. Shared fields or a "
            "DAG join do not prove cross-validation. No synthetic evidence "
            "or new tool access was created."
        ),
    }
