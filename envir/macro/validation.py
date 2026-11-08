"""Structural proof obligations for coarse-grained task plans.

Fact soundness, feedback correctness, and question semantic equivalence require
World Grounding and are intentionally *not* asserted here.
"""
from __future__ import annotations

import re
from collections import Counter

from .graph import MAX_CONCURRENCY, MacroPlanError, compile_waves, complexity


def validate_plan(plan: dict) -> list[str]:
    issues = []
    graph = plan.get("macro_graph") or {}
    tasks = graph.get("tasks") or []
    edges = graph.get("edges") or []
    by_id = {t["task_id"]: t for t in tasks}
    if len(by_id) != len(tasks):
        issues.append("duplicate_macros")
    if not 3 <= len(tasks) <= 6:
        issues.append("macro_task_count_out_of_range")
    if any(len(t.get("internal_execution_outline") or []) < 2 for t in tasks):
        issues.append("micro_operations_promoted_or_missing")
    if len({t["goal"] for t in tasks}) != len(tasks):
        issues.append("duplicate_work_goals")
    for task in tasks:
        if not task.get("deliverable") or not task.get("outputs"):
            issues.append("missing_work_deliverable")
        if task.get("grounding_status") != "not_grounded":
            issues.append("false_grounding_claim")
        schema = task.get("input_schema") or {}
        inputs = task.get("inputs") or {}
        if set(schema) != set(inputs):
            issues.append("missing_input_binding")
        for slot, binding in inputs.items():
            producer = by_id.get(binding.get("from_task"))
            if (producer is None or
                producer["outputs"].get(binding.get("output_slot")) !=
                    schema.get(slot) or
                binding.get("type") != schema.get(slot)):
                issues.append("invalid_typed_handoff")
    actual = set()
    for edge in edges:
        src, dst = edge.get("from"), edge.get("to")
        if src not in by_id or dst not in by_id or src == dst:
            issues.append("invalid_edge")
            continue
        actual.add((src, dst, edge.get("input_slot")))
        binding = by_id[dst].get("inputs", {}).get(edge.get("input_slot"))
        if not binding or binding.get("from_task") != src:
            issues.append("edge_binding_disagreement")
    if len(actual) != len(edges):
        issues.append("duplicate_edges")
    for task in tasks:
        incoming = {src for src, dst, _ in actual if dst == task["task_id"]}
        if incoming != set(task.get("depends_on", [])):
            issues.append("dependency_contract_mismatch")
    try:
        waves = compile_waves(tasks)
        if graph.get("reference_waves") != waves:
            issues.append("bad_reference_waves")
        coarse = plan.get("coarsening") or {}
        expected = complexity(tasks, waves, coarse)
        if plan.get("complexity") != expected:
            issues.append("complexity_calculation_drift")
    except (MacroPlanError, KeyError, TypeError):
        issues.append("invalid_or_cyclic_dag")
    sinks = {
        t["task_id"] for t in tasks
        if not any(e.get("from") == t["task_id"] for e in edges)
    }
    terminal = graph.get("terminal_handoffs") or []
    if len(sinks) != 1 or len(terminal) != 1 or terminal[0].get("from_task") not in sinks:
        issues.append("unconnected_or_multiple_final_work_products")
    if len(tasks) > 1 and len(edges) < len(tasks) - 1:
        issues.append("weakly_disconnected_work_packages")
    all_capabilities = {cap for t in tasks for cap in t.get("capability_hints", [])}
    if len(all_capabilities) < 2:
        issues.append("no_capability_diversity")
    question = str(plan.get("question") or "")
    if not question or re.search(r"\bS[1-9]\b|\bfirst (?:subtask|step)\b", question, re.I):
        issues.append("question_exposes_execution_structure")
    public = plan.get("public") or {}
    if public.get("question") != question or "macro_graph" in public or "source_priors" in public:
        issues.append("private_plan_leaks_to_public")
    if plan.get("scheduler") != {
        "max_concurrent_subagents": MAX_CONCURRENCY,
        "environment_profile": "fixed_default",
        "runtime_variants_generated": False,
    }:
        issues.append("unexpected_runtime_variability")
    if plan.get("construction_status") != "structural_candidate_not_fact_grounded":
        issues.append("false_certification_claim")
    return sorted(set(issues))
