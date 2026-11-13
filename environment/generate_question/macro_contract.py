"""Macro-task diversity and frozen, executable question contracts.

The Program and compiled Macro Graph are the operational source of truth.
No private World values, Gold rows, or candidate answers are sent to an LLM.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter


def _primitive_steps(task: dict) -> list[dict]:
    return list(task.get("operator_chain") or [
        {"op": task["operation"], **task.get("operation_args", {})}
    ])


def macro_graph_signature(graph: dict) -> str:
    """Naming-invariant fingerprint for source-labelled executable DAGs.

    Parent order matters for directional operations (e.g. joins). The
    fingerprint encodes each task's local operator sequence, authorized
    source roles, all dependency edges, and the terminal output. It does
    not include task names, topic, media decorations, or sampled facts.
    """
    tasks = {row["task_id"]: row for row in graph["tasks"]}
    memo, visiting = {}, set()

    def encode(task_id):
        if task_id in memo:
            return memo[task_id]
        if task_id in visiting:
            raise ValueError("cycle_in_macro_graph")
        if task_id not in tasks:
            raise ValueError("unknown_macro_dependency:" + task_id)
        visiting.add(task_id)
        task = tasks[task_id]
        local = (
            tuple(task["source_ids"]),
            tuple((
                step["op"],
                step.get("comparison"),
                step.get("descending"),
                step.get("field") if step["op"] in {
                    "filter_eq", "filter_gte", "sort", "group_count"
                } else None,
                step.get("left_on"), step.get("right_on"),
                step.get("left_field"), step.get("right_field"),
                step.get("limit"),
                step.get("source_id") if step["op"] == "lookup_by_key" else None,
                step.get("input_key_field"), step.get("source_key_field"),
                step.get("value") if step["op"] in {
                    "filter_eq", "filter_gte"
                } else None,
            ) for step in _primitive_steps(task)),
        )
        value = (local, tuple(encode(parent) for parent in task["depends_on"]))
        visiting.remove(task_id)
        memo[task_id] = value
        return value

    terminal = encode(graph["terminal_task_id"])
    # Include disconnected tasks as a separate multiset to detect spurious
    # work. Valid compiled graphs normally have no unused nodes.
    other = sorted(
        (repr(encode(task_id)) for task_id in tasks),
        key=str,
    )
    content = json.dumps([repr(terminal), other], ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:20]


def macro_graph_profile(graph: dict) -> dict:
    tasks = graph["tasks"]
    return {
        "signature": macro_graph_signature(graph),
        "macro_task_count": len(tasks),
        "edge_count": sum(len(t["depends_on"]) for t in tasks),
        "critical_path_depth": graph["critical_path_depth"],
        "source_task_count": sum(bool(t["source_ids"]) for t in tasks),
        "keyed_acquisition_task_count": sum(
            any(step["op"] == "lookup_by_key" for step in _primitive_steps(t))
            for t in tasks
        ),
        "parallel_wave_count": len(graph["reference_waves"]),
        "composite_task_count": sum(
            len(_primitive_steps(t)) > 1 for t in tasks
        ),
        "primitive_operation_count": sum(
            len(_primitive_steps(t)) for t in tasks
        ),
    }


def _flatten_operations(graph: dict) -> list[dict]:
    return [step for task in graph["tasks"]
            for step in _primitive_steps(task)]


def macro_question_contract(spec: dict, graph: dict) -> dict:
    """A graph-derived Question contract, validated against frozen Intent.

    This prevents Author/Critic from silently requesting a different
    operator, source scope, or output shape than the compiled Macro Graph.
    """
    steps = _flatten_operations(graph)
    used_sources = {
        sid for task in graph["tasks"] for sid in task["source_ids"]
    }
    authorized = {row["source_id"] for row in spec["sources"]}
    if used_sources != authorized:
        raise ValueError("macro_question_source_contract_mismatch")
    ops = Counter(step["op"] for step in steps)
    pattern = spec.get("reasoning_pattern", "approval")
    family = spec["family"]
    compares = [step.get("comparison") for step in steps
                if step["op"] == "compare_fields"]
    if pattern == "macro_first_v2":
        plan = spec["macro_plan"]
        if (len(graph["tasks"]) != len(plan["macro_tasks"])
                or graph.get("plan_origin") != "macro_first"):
            raise ValueError("macro_first_dag_not_frozen")
        if [(t["task_id"], t["depends_on"], t["source_ids"]) for t in graph["tasks"]] != [
            ("work_" + t["macro_id"],
             ["work_" + parent for parent in t["depends_on"]],
             t["source_scope"]) for t in plan["macro_tasks"]
        ]:
            raise ValueError("macro_first_macro_contract_mismatch")
        if (graph["terminal_task_id"] != "work_" + plan["terminal_macro_id"]
                or not ops["lookup_by_key"]):
            raise ValueError("macro_first_terminal_or_lookup_missing")
    if family == "evidence_reconcile" and not ops["attach_evidence"]:
        raise ValueError("macro_question_missing_evidence_output")
    sorts = [step for step in steps if step["op"] == "sort"]
    takes = [step for step in steps if step["op"] == "take"]
    if family in {"priority", "chronology"}:
        if not (sorts and takes):
            raise ValueError("macro_question_missing_selection_operation")
        wanted_field = (
            "recorded_priority" if family == "priority" else "relative_step"
        )
        descending = family == "priority"
        if not any(
            step.get("field") == wanted_field
            and step.get("descending") is descending
            for step in sorts
        ):
            raise ValueError("macro_question_wrong_selection_direction")
        if not any(step.get("limit") == 1 for step in takes):
            raise ValueError("macro_question_wrong_selection_cardinality")
    if family in {"coverage_count", "category_distribution"} and not (
        ops["count"] or ops["group_count"]
    ):
        raise ValueError("macro_question_missing_aggregation")
    final = next(
        task for task in graph["tasks"]
        if task["task_id"] == graph["terminal_task_id"]
    )
    output_fields = final["outputs"]["typed_relation"]
    answer_kind = spec["requirements"]["answer_kind"]
    if answer_kind == "entry_evidence" and not (
        {"entry_label", "evidence_refs"} <= set(output_fields)
    ):
        raise ValueError("macro_question_wrong_evidence_output_schema")
    if answer_kind == "macro_value" and output_fields != [
        spec["macro_plan"]["goal"]["answer_field"]
    ]:
        raise ValueError("macro_first_wrong_goal_output")
    if answer_kind == "accession_year" and output_fields != ["accession_year"]:
        raise ValueError("macro_question_wrong_accession_year_schema")
    if answer_kind == "count" and "count" not in output_fields:
        raise ValueError("macro_question_wrong_count_output_schema")
    if answer_kind == "category_counts" and "count" not in output_fields:
        raise ValueError("macro_question_wrong_group_count_schema")
    if answer_kind in {"entry_list", "single_entry", "entry_evidence"} and (
        "entry_label" not in output_fields
    ):
        raise ValueError("macro_question_wrong_entry_output_schema")
    comparison_contract = [
        {
            "comparison": step["comparison"],
            "left_field": step["left_field"],
            "right_field": step["right_field"],
        } for step in steps if step["op"] == "compare_fields"
    ]
    selection_contract = (
        {
            "field": sorts[0]["field"],
            "descending": sorts[0]["descending"],
            "limit": takes[0]["limit"],
        } if sorts and takes else None
    )
    source_roles = {
        row["source_id"]: row["modality"] for row in spec["sources"]
    }
    return {
        "semantic_pattern": pattern,
        "answer_kind": answer_kind,
        "output_fields": list(output_fields),
        "required_source_roles": [
            {"source_id": sid, "modality": source_roles[sid]}
            for sid in sorted(used_sources)
        ],
        "required_operations": dict(ops),
        "comparison_contract": comparison_contract,
        "selection_contract": selection_contract,
        "macro_task_count": len(graph["tasks"]),
        "macro_dependency_depth": graph["critical_path_depth"],
        "graph_fingerprint": macro_graph_signature(graph),
        "no_extra_question_conditions": True,
        "no_private_world_values_disclosed": True,
    }


def validate_handoffs(scene: dict) -> dict:
    """Prove each frozen edge matters under a controlled empty handoff.

    Removing the *edge input product* differs from removing a source. Every
    edge must affect the terminal Gold Answer, not merely the execution graph.
    """
    from .compiler import execute_task
    from .program_ir import Relation
    from .dag import compile_waves

    graph = scene["macro_graph"]
    tasks = {row["task_id"]: row for row in graph["tasks"]}
    world = scene["world"]["sources"]
    expected = scene["gold_answer"]
    passed, edges = 0, []
    for child in graph["tasks"]:
        for parent_id in child["depends_on"]:
            products = {}
            for wave in graph["reference_waves"]:
                for tid in wave:
                    task = tasks[tid]
                    inputs = [
                        Relation(products[p].columns, tuple())
                        if tid == child["task_id"] and p == parent_id
                        else products[p]
                        for p in task["depends_on"]
                    ]
                    products[tid] = execute_task(
                        task, input_products=inputs,
                        visible_sources={sid: world[sid]
                                         for sid in task["source_ids"]},
                    )
            terminal = products[graph["terminal_task_id"]]
            got = sorted([row["values"] for row in terminal.rows],
                         key=lambda x: str(sorted(x.items())))
            sensitive = got != expected
            passed += int(sensitive)
            edges.append({"upstream": parent_id,
                          "downstream": child["task_id"],
                          "blank_handoff_changes_answer": sensitive})
    return {"edge_count": len(edges), "sensitive_edges": passed,
            "all_handoffs_necessary": passed == len(edges),
            "edges": edges}
