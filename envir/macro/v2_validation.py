"""Semantic-contract and structural checks for macro TaskPlan v2.

This rejects DAG edges that carry the wrong named work product, unowned
micro actions or disconnected work. It does NOT claim source factual
grounding, natural question truth, or verified Sub-Agent feedback.
"""
from __future__ import annotations

import re
from collections import Counter

from .graph import compile_waves, MacroPlanError
from .v2_workbank import BY_OUTPUT, GOAL_TYPES, OUTPUT_FIELDS, ROOT_SOURCE_DESCRIPTIONS


def validate_v2(plan: dict) -> list[str]:
    issues = []
    if plan.get("schema") != "maestro.macro_taskplan.v2":
        issues.append("schema_mismatch")
    graph = plan.get("macro_graph") or {}
    tasks = graph.get("tasks") or []
    edges = graph.get("edges") or []
    by_id = {t["task_id"]: t for t in tasks}
    if len(by_id) != len(tasks):
        issues.append("duplicate_task_id")
    if not 3 <= len(tasks) <= 9:
        issues.append("invalid_macro_task_count")
    if plan.get("construction_status") != "structural_candidate_not_fact_grounded":
        issues.append("ungrounded_claim")
    chosen = graph.get("provenance_recipe_routes") or {}
    if len(chosen) != len(tasks):
        issues.append("bad_recipe_route_provenance")
    for task in tasks:
        matched = [r for recipes in BY_OUTPUT.values() for r in recipes
                   if r.name == task.get("recipe_id")]
        if len(matched) != 1:
            issues.append("unknown_work_recipe")
            continue
        recipe = matched[0]
        output_slot = OUTPUT_FIELDS[recipe.output]
        if task.get("outputs") != {output_slot: recipe.output}:
            issues.append("work_output_contract_mismatch")
        if task.get("input_schema") != recipe.input_schema():
            issues.append("work_input_contract_mismatch")
        if chosen.get(recipe.output) != recipe.name:
            issues.append("provenance_recipe_drift")
        expected_sources = [recipe.source_role] if recipe.source_role else []
        actual_sources = task.get("source_requirements") or []
        if [s.get("role") for s in actual_sources] != expected_sources:
            issues.append("source_role_mismatch")
        if any(s.get("access") != "not_instantiated" or
               not s.get("synthetic") for s in actual_sources):
            issues.append("source_falsely_claimed_real")
        if task.get("grounding_status") != "not_grounded":
            issues.append("premature_grounding_claim")
        if not task.get("deliverable") or not task.get("goal"):
            issues.append("empty_work_goal")
        if len(task.get("internal_execution_outline") or []) < 2:
            issues.append("no_internal_execution_outline")
        if any(k in task for k in ("answer", "summary", "summary_trace", "gold_answer")):
            issues.append("premature_subagent_feedback")
        inputs = task.get("inputs") or {}
        if set(inputs) != set(recipe.input_schema()):
            issues.append("unbound_work_requirement")
        for slot, required_type, why in recipe.inputs:
            binding = inputs.get(slot) or {}
            producer = by_id.get(binding.get("from_task"))
            field = binding.get("output_slot")
            if (producer is None or producer.get("outputs", {}).get(field) != required_type
                or binding.get("type") != required_type):
                issues.append("incorrect_typed_dependency")
            expected_edges = [e for e in edges
                              if e.get("to") == task["task_id"]
                              and e.get("input_slot") == slot]
            if (len(expected_edges) != 1 or
                expected_edges[0].get("from") != binding.get("from_task") or
                expected_edges[0].get("output_slot") != field or
                expected_edges[0].get("handoff_reason") != why):
                issues.append("missing_semantic_handoff_rationale")
        expected_parents = {b["from_task"] for b in inputs.values() if "from_task" in b}
        if set(task.get("depends_on") or []) != expected_parents:
            issues.append("dependency_contract_drift")
    valid_pairs = set()
    for edge in edges:
        src, dst = edge.get("from"), edge.get("to")
        slot = edge.get("input_slot")
        if src == dst or src not in by_id or dst not in by_id:
            issues.append("invalid_macro_edge")
            continue
        triplet = (src, dst, slot)
        if triplet in valid_pairs:
            issues.append("duplicate_macro_edge")
        valid_pairs.add(triplet)
        if (edge.get("kind") != "required_data"
            or edge.get("activation") != "producer_completed"):
            issues.append("unsupported_or_nontyped_dependency")
    try:
        waves = compile_waves(tasks)
        if waves != graph.get("reference_waves"):
            issues.append("incorrect_reference_waves")
        if plan["complexity"]["max_ready_parallelism"] != max(map(len, waves), default=0):
            issues.append("incorrect_parallelism")
        if plan["complexity"]["macro_dependency_depth"] < 2:
            issues.append("trivial_macro_depth")
        if plan["complexity"]["macro_task_count"] != len(tasks):
            issues.append("complexity_node_count_mismatch")
        if plan["complexity"]["concurrency_limit"] != 10:
            issues.append("unexpected_concurrency")
    except (MacroPlanError, KeyError, TypeError):
        issues.append("cycle_or_missing_dependency")
    terminal = (graph.get("terminal_handoffs") or [{}])
    if len(terminal) != 1:
        issues.append("multiple_terminal_handoffs")
        terminal = [{}]
    terminal = terminal[0]
    expected_goal_type = GOAL_TYPES.get(plan.get("goal", {}).get("intent"))
    if terminal.get("type") != expected_goal_type:
        issues.append("question_and_terminal_type_mismatch")
    if terminal.get("from_task") not in by_id:
        issues.append("missing_terminal_work")
    else:
        last = by_id[terminal["from_task"]]
        if last.get("outputs", {}).get(terminal.get("output_slot")) != terminal.get("type"):
            issues.append("terminal_output_not_produced")
    sinks = {t["task_id"] for t in tasks
             if not any(e.get("from") == t["task_id"] for e in edges)}
    if sinks != {terminal.get("from_task")}:
        issues.append("multiple_or_disconnected_sinks")
    visited = set()
    def visit(tid):
        if tid in visited:
            return
        visited.add(tid)
        for dep in by_id[tid].get("depends_on", []):
            if dep in by_id:
                visit(dep)
    if terminal.get("from_task") in by_id:
        visit(terminal["from_task"])
    if visited != set(by_id):
        issues.append("work_not_needed_for_final_deliverable")
    if len({t.get("work_type") for t in tasks}) < 3:
        issues.append("work_type_diversity_low")
    if len({t.get("goal") for t in tasks}) != len(tasks):
        issues.append("duplicate_macro_work_goal")
    if plan.get("coarsening", {}).get("collapsed_internal_steps", 0) < len(tasks):
        issues.append("no_real_micro_to_macro_collapse")
    sched = plan.get("scheduler") or {}
    if (sched.get("max_concurrent_subagents") != 10 or
        sched.get("runtime_variants_generated") is not False):
        issues.append("unsupported_runtime_variants")
    q = str(plan.get("question") or "")
    issue = str(plan.get("scenario", {}).get("issue") or "")
    if issue not in q or len(q) < 80:
        issues.append("question_missing_specific_scenario")
    if re.search(r"\b(?:S\d+|Subtask|KoPL|agent\s*\d+)\b", q, re.I):
        issues.append("question_leaks_internal_execution")
    if q != plan.get("goal", {}).get("user_request"):
        issues.append("question_goal_mismatch")
    public = plan.get("public") or {}
    if public.get("question") != q or public.get("task_id") != plan.get("task_id"):
        issues.append("public_question_mismatch")
    if set(public) != {"task_id", "question", "resource_descriptions", "notice", "construction_status"}:
        issues.append("public_contains_private_keys")
    if "answer" in public or "macro_graph" in public:
        issues.append("public_leaks_private_answer_or_graph")
    if public.get("construction_status") != "unrealized":
        issues.append("public_falsely_claimed_grounded")
    roots = [s for t in tasks for s in t.get("source_requirements", [])]
    if public.get("resource_descriptions") != [
        {"role": s["role"], "description": s["description"],
         "candidate_modalities": s["candidate_modalities"]} for s in roots
    ]:
        issues.append("public_source_projection_mismatch")
    if any(not s.get("description") for s in roots):
        issues.append("missing_evidence_role_description")
    return sorted(set(issues))
