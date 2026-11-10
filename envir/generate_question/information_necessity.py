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

    # The competing identities must also have distinguishable answers,
    # otherwise bypassing one branch could still yield the same scalar.
    sources = scene["world"]["sources"]
    markers = {
        r["values"]["candidate_id"]: r["values"]["marker_code"]
        for r in sources["footage"]["observations"]
    }
    key_to_goal = {}
    goal_field = plan["goal"]["answer_field"]
    for candidate_id in left | right:
        key = markers[candidate_id]
        for i in range(1, plan["index_hops"] + 1):
            key_field = "marker_code" if i == 1 else f"record_key_{i-1}"
            next_field = f"record_key_{i}"
            matches = [
                row["values"][next_field]
                for row in sources[f"index_{i}"]["observations"]
                if row["values"][key_field] == key
            ]
            if len(matches) != 1:
                raise ValueError("information_necessity_index_ambiguity")
            key = matches[0]
        final_key = (
            f"record_key_{plan['index_hops']}"
            if plan["index_hops"] else "marker_code"
        )
        matches = [
            row["values"][goal_field]
            for row in sources["archive"]["observations"]
            if row["values"][final_key] == key
        ]
        if len(matches) != 1:
            raise ValueError("information_necessity_archive_ambiguity")
        key_to_goal[candidate_id] = matches[0]
    competing_goals_distinct = (
        len(set(map(str, key_to_goal.values()))) == len(key_to_goal)
    )
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
        "competing_goals_distinct": competing_goals_distinct,
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
        and result["competing_goals_distinct"]
        and result["fused_marker_count"] == 1
    )
    return result
