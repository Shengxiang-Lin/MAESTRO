"""Small, executable information-necessity audit for Macro-first.

This tests candidate-set discrimination under the *actual* materialized
workproducts. It does NOT prove optimal agent search cost or that an
independent live Sub-Agent would always find the same witnesses.
"""
from __future__ import annotations


POLICY = "two_nonunique_candidate_sets_with_single_intersection_v1"

KNOWLEDGE_REPLANNING_POLICY = "knowledge_candidate_replanning_v2"


def _audit_knowledge_replanning(scene: dict) -> dict:
    """Test actual candidate-set uniqueness and executable bypass attempts.

    Unlike v1's 'mutate a cited correct row' check, these interventions
    *relax filters* while leaving all valid competing YAGO paths in place.
    A complete competing answer must remain feasible if any source or
    Discovery Macro is skipped. A blocked/empty execution does not count
    as evidence of information necessity.
    """
    import copy
    from .program_ir import evaluate_program, answer_values

    spec = scene["problem_spec"]
    plan, world = spec["macro_plan"], scene["world"]["sources"]
    target = plan["target_case"]
    clue_sets = {}
    for sid, cue in plan["clues"].items():
        clue_sets[sid] = {
            record["values"]["case_id"]
            for record in world[sid]["observations"]
            if record["values"]["cue"] == cue["value"]
        }
    if len(clue_sets) < 4:
        raise ValueError("knowledge_audit_too_few_independent_clues")
    full = set.intersection(*clue_sets.values())
    feedback = {row["macro_id"]: row for row in scene["task_feedback"]
                if "macro_id" in row}
    identity = next(t for t in plan["macro_tasks"]
                    if t["role"] == "identification")
    identity_feedback = next(row for row in scene["task_feedback"]
                             if row["task_id"] == "work_" + identity["macro_id"])
    mid_count = identity_feedback["result_rows"]
    # Need both a true candidate intersection and the actual intermediate
    # WorkProduct to agree; later relation filters cannot hide ambiguity.
    jointly_unique = (
        full == {target} and mid_count == 1
        and identity_feedback["workproduct"][0]["values"]["entity_0"]
        == plan["entity_anchor"]
    )

    program = scene["gold_program"]
    by_id = {node["id"]: node for node in program["nodes"]}
    clue_filters = {}
    for node in program["nodes"]:
        if node["op"] != "filter_eq" or node.get("field") != "cue":
            continue
        source = by_id[node["inputs"][0]]
        if source["op"] != "scan":
            raise ValueError("knowledge_filter_not_source_scoped")
        sid = source["source_id"]
        if sid in clue_filters:
            raise ValueError("knowledge_duplicate_clue_filter")
        clue_filters[sid] = node["id"]
    if set(clue_filters) != set(clue_sets):
        raise ValueError("knowledge_missing_compiled_clue_filters")

    expected = scene["gold_answer"]
    def _run_relaxed(sids: set[str]) -> list[dict]:
        altered = copy.deepcopy(program)
        filters = {clue_filters[sid] for sid in sids}
        for node in altered["nodes"]:
            if node["id"] in filters:
                nid, parents = node["id"], list(node["inputs"])
                node.clear()
                node.update({
                    "id": nid, "op": "project", "inputs": parents,
                    "fields": ["case_id", "cue"],
                })
        return answer_values(evaluate_program(altered, sources=world))

    def _ambiguous(results: list[dict]) -> bool:
        outputs = {entry["answer"] for entry in results}
        return (len(outputs) > 1
                and expected[0]["answer"] in outputs
                and len(outputs) == len(results))

    source_checks = {}
    for sid, candidates in clue_sets.items():
        without = set.intersection(*[
            values for other, values in clue_sets.items()
            if other != sid
        ])
        results = _run_relaxed({sid})
        source_checks[sid] = {
            "remaining_candidate_count": len(without),
            "remaining_answer_count": len(results),
            "has_real_competing_answer": _ambiguous(results),
            "expected_shadow_case": target in without and len(without) > 1,
        }
    discovery_checks = {}
    for task in plan["macro_tasks"]:
        if task["role"] != "discovery":
            continue
        sids = set(task["source_scope"])
        other = [values for sid, values in clue_sets.items() if sid not in sids]
        remaining = set.intersection(*other)
        results = _run_relaxed(sids)
        discovery_checks[task["macro_id"]] = {
            "remaining_candidate_count": len(remaining),
            "remaining_answer_count": len(results),
            "has_real_competing_answer": _ambiguous(results),
            "candidate_ambiguity": target in remaining and len(remaining) > 1,
        }
    all_relaxed = _run_relaxed(set(clue_sets))
    all_bypass_blocked = _ambiguous(all_relaxed)
    passed = (
        jointly_unique
        and all(r["has_real_competing_answer"]
                and r["expected_shadow_case"]
                for r in source_checks.values())
        and all(r["has_real_competing_answer"]
                and r["candidate_ambiguity"]
                for r in discovery_checks.values())
        and all_bypass_blocked
    )
    return {
        "policy": KNOWLEDGE_REPLANNING_POLICY,
        "passed": passed,
        "identification_unique": jointly_unique,
        "identified_candidate_count": len(full),
        "identity_macro_workproduct_count": mid_count,
        "source_counterfactuals": source_checks,
        "discovery_counterfactuals": discovery_checks,
        "all_discovery_filters_removed": {
            "possible_answers": len(all_relaxed),
            "ambiguous": all_bypass_blocked,
        },
        "limitation": (
            "Bounded executable filter-relaxation and candidate-replanning "
            "checks, not a proof of globally shortest agent search."
        ),
    }


VARIABLE_POLICY = "variable_nonunique_candidate_sets_single_intersection_v2"
CORROBORATED_DISPUTE_POLICY = "reported_year_disagreement_plus_context_v1"


def _audit_corroborated_dispute(scene: dict) -> dict:
    """Two-source disagreement and an independent log each leave ambiguity.

    The final answer must be unique only after both branches are combined.
    """
    plan = scene["problem_spec"]["macro_plan"]
    feedback = {row["task_id"]: row for row in scene["task_feedback"]}
    left = {
        row["values"]["case_id"]
        for row in feedback["work_M3"]["workproduct"]
    }
    right = {
        row["values"]["case_id"]
        for row in feedback["work_M4"]["workproduct"]
    }
    overlap = left & right
    sources = scene["world"]["sources"]
    goal_field = plan["goal"]["answer_field"]
    resolved = {}
    for case_id in left | right:
        key = case_id
        for i in range(1, plan["index_hops"] + 1):
            prior = "case_id" if i == 1 else f"record_key_{i-1}"
            next_key = f"record_key_{i}"
            options = [
                row["values"][next_key]
                for row in sources[f"index_{i}"]["observations"]
                if row["values"][prior] == key
            ]
            if len(options) != 1:
                raise ValueError("corroborated_dispute_index_not_unique")
            key = options[0]
        final_field = (
            f"record_key_{plan['index_hops']}"
            if plan["index_hops"] else "case_id"
        )
        from .goal_resolution import materialized_answer
        resolved[case_id] = materialized_answer(
            plan, sources, final_field, key,
        )
    target = next(iter(overlap), None)
    profile = plan["candidate_profile"]
    profile_matches = (
        len(left) == profile["left_candidate_count"]
        and len(right) == profile["right_candidate_count"]
        and len(left | right) == profile["union_candidate_count"]
        and len(overlap) == profile["intersection_candidate_count"]
    )
    distinct = (
        target is not None and all(
            resolved[key] != resolved[target]
            for key in (left | right) - overlap
        )
    )
    passed = (
        len(left) >= 2 and len(right) >= 2 and len(overlap) == 1
        and bool(left - right) and bool(right - left)
        and feedback["work_M5"]["result_rows"] == 1
        and profile_matches and distinct
    )
    return {
        "policy": CORROBORATED_DISPUTE_POLICY,
        "left_candidate_count": len(left),
        "right_candidate_count": len(right),
        "intersection_count": len(overlap),
        "left_unique_candidates": len(left - right),
        "right_unique_candidates": len(right - left),
        "both_branches_discriminate": bool(left - right) and bool(right - left),
        "left_alone_insufficient": len(left) >= 2,
        "right_alone_insufficient": len(right) >= 2,
        "jointly_unique": len(overlap) == 1,
        "competing_goals_distinct": distinct,
        "candidate_count_matches_profile": profile_matches,
        "fused_marker_count": feedback["work_M5"]["result_rows"],
        "passed": passed,
        "interpretation": (
            "Actual report-disagreement versus independent context-log "
            "candidate discrimination, not a globally optimal search proof."
        ),
    }


def audit_candidate_information(scene: dict) -> dict:
    """Require both independent branches to remove real alternatives."""
    spec = scene["problem_spec"]
    plan = spec["macro_plan"]
    policy = plan.get("information_necessity_policy")
    if policy == "hybrid_world_replanning_v3":
        from .hybrid_world import audit_hybrid_information
        return audit_hybrid_information(scene)
    if policy == KNOWLEDGE_REPLANNING_POLICY:
        return _audit_knowledge_replanning(scene)
    if policy == CORROBORATED_DISPUTE_POLICY:
        return _audit_corroborated_dispute(scene)
    if policy not in {POLICY, VARIABLE_POLICY}:
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
        from .goal_resolution import materialized_answer
        key_to_goal[candidate_id] = materialized_answer(
            plan, sources, final_key, key,
        )
    if policy == VARIABLE_POLICY:
        # Distinct from the TARGET is sufficient for answer sensitivity.
        # Multiple alternatives can share a legitimate status label.
        target = next(iter(overlap), None)
        competing_goals_distinct = (
            target is not None
            and all(key_to_goal[other] != key_to_goal[target]
                    for other in (left | right) - overlap)
        )
        profile = plan["candidate_profile"]
        candidate_count_matches_profile = (
            len(left) == profile["left_candidate_count"]
            and len(right) == profile["right_candidate_count"]
            and len(left | right) == profile["union_candidate_count"]
            and len(overlap) == profile["intersection_candidate_count"]
        )
    else:
        competing_goals_distinct = (
            len(set(map(str, key_to_goal.values()))) == len(key_to_goal)
        )
        candidate_count_matches_profile = True
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
        "candidate_count_matches_profile": candidate_count_matches_profile,
        "policy": policy,
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
        and result["candidate_count_matches_profile"]
        and result["fused_marker_count"] == 1
    )
    return result
