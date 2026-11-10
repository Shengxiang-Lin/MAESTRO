"""Small, executable information-necessity audit for Macro-first.

This tests candidate-set discrimination under the *actual* materialized
workproducts. It does NOT prove optimal agent search cost or that an
independent live Sub-Agent would always find the same witnesses.
"""
from __future__ import annotations


POLICY = "two_nonunique_candidate_sets_with_single_intersection_v1"


def audit_candidate_information(scene: dict) -> dict:
    """Require both independent branches to remove real alternatives."""
    spec = scene["problem_spec"]
    plan = spec["macro_plan"]
    if plan.get("information_necessity_policy") != POLICY:
        raise ValueError("information_necessity_policy_not_applicable")
    feedback = {r["task_id"]: r for r in scene["task_feedback"]}
    for mid in ("work_M1", "work_M2", "work_M3"):
        if mid not in feedback:
            raise ValueError("information_necessity_missing_macro:" + mid)
    left = {
        row["values"]["candidate_id"]
        for row in feedback["work_M1"]["workproduct"]
    }
    right = {
        row["values"]["candidate_id"]
        for row in feedback["work_M2"]["workproduct"]
    }
    overlap = left & right
    result = {
        "left_candidate_count": len(left),
        "right_candidate_count": len(right),
        "intersection_count": len(overlap),
        "left_unique_candidates": len(left - right),
        "right_unique_candidates": len(right - left),
        "left_alone_insufficient": len(left) >= 2,
        "right_alone_insufficient": len(right) >= 2,
        "jointly_unique": len(overlap) == 1,
        "both_branches_discriminate": bool(left - right) and bool(right - left),
        "fused_marker_count": feedback["work_M3"]["result_rows"],
        "interpretation": (
            "Candidate-set discrimination given the frozen working World; "
            "not a search-policy optimality proof."
        ),
    }
    result["passed"] = (
        result["left_alone_insufficient"]
        and result["right_alone_insufficient"]
        and result["jointly_unique"]
        and result["both_branches_discriminate"]
        and result["fused_marker_count"] == 1
    )
    return result
