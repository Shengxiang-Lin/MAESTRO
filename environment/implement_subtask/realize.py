"""Stage 2: turn frozen Macro programs into evidence-grounded Subtask feedback.

This is an *oracle teacher materializer*: it does not run an LLM Sub-Agent.
Each answer is copied from independently checked executable feedback, and
every trace step is constructed from the exact deterministic Program run.
"""
from __future__ import annotations

import json
from copy import deepcopy

from environment.generate_question.compiler import run_macro_tasks
from environment.generate_question.program_ir import evaluate_program, answer_values
from environment.generate_question.quality import validate_replay
from environment.generate_question.evidence_tools import simulate_acquisition_trace


def _fact_index(world: dict) -> dict[str, dict]:
    index = {}
    for source_id, source in world["sources"].items():
        for observation in source["observations"]:
            for fact_ref in observation["fact_refs"]:
                if fact_ref in index:
                    raise ValueError("duplicate_fact_ref:" + fact_ref)
                index[fact_ref] = {
                    "fact_ref": fact_ref,
                    "source_id": source_id,
                    "observation_id": observation["observation_id"],
                    "values": dict(observation["values"]),
                    **({"document_context": dict(
                        observation["document_context"]
                    )} if "document_context" in observation else {}),
                    **({"record_excerpt": observation["record_excerpt"]}
                       if "record_excerpt" in observation else {}),
                }
    return index


def _describe_operation(node: dict) -> str:
    op = node["op"]
    if op == "scan":
        return f"Read the {node['source_id']} source to obtain relevant observations"
    if op == "lookup_by_key":
        return (
            f"Retrieve only the {node['source_id']} record addressed by "
            f"upstream {node['input_key_field']} via {node['source_key_field']}"
        )
    if op == "filter_eq":
        return (f"Keep rows whose {node['field']} equals "
                f"{node['value']!r}")
    if op == "filter_gte":
        return (f"Keep rows whose {node['field']} is at least "
                f"{node['value']!r}")
    if op == "join":
        return (
            f"Align the two input workproducts by "
            f"{node['left_on']} = {node['right_on']}"
        )
    if op == "claim_relation_filter":
        return (
            f"Check that {node['left_field']} differs from "
            f"{node['right_field']} for each {node['subject_field']} "
            "using the two independent source fact references"
        )
    if op == "compare_fields":
        comparison = {
            "gte": "is at least", "lte": "is at most", "eq": "equals",
            "ne": "differs from", "gt": "exceeds", "lt": "is below",
        }.get(node.get("comparison"), node.get("comparison", "compares with"))
        return (f"Check that {node['left_field']} {comparison} "
                f"{node['right_field']}")
    if op == "sort":
        return (f"Order by {node['field']} "
                f"{'descending' if node.get('descending') else 'ascending'}")
    if op == "take":
        return f"Select the first {node['limit']} row(s)"
    if op == "project":
        renames = node.get("rename_fields", {})
        return (
            "Keep only the output field(s): " + ", ".join(
                renames.get(field, field) for field in node["fields"]
            )
        )
    if op == "unique":
        return "Remove duplicate output rows"
    if op == "count":
        return "Count the remaining rows"
    if op == "group_count":
        return f"Count rows within each {node['field']} group"
    if op == "derive_comparison":
        return (
            f"Compare {node['left_field']} with {node['right_field']} "
            "and identify the independent source with the higher count"
        )
    if op == "derive_percentage":
        return (
            f"Compute 100 * {node['left_field']} / {node['right_field']} "
            "as a whole percentage rounded down"
        )
    if op == "derive_arithmetic":
        return (
            f"Calculate {node['operation']} from the independently "
            f"observed {node['left_field']} and {node['right_field']}; "
            f"round to {node['round_digits']} decimal places"
        )
    if op == "derive_date_gap":
        return (
            f"Calculate elapsed calendar days between the dated "
            f"{node['left_field']} and {node['right_field']} milestones"
        )
    if op == "derive_great_circle":
        return (
            "Calculate the great-circle distance in kilometers between "
            "the two independently sourced coordinate pairs"
        )
    if op == "compose_typed_result":
        return (
            "Compose the requested structured result from the resolved "
            "entity, comparison winner and absolute count difference"
        )
    if op == "goal_compute":
        goal_op = node["goal_operation"]
        labels = {
            "arithmetic": "Calculate the requested arithmetic expression",
            "temporal_difference": "Calculate elapsed days between source dates",
            "temporal_midpoint": "Find the midpoint date between source dates",
            "geographic_distance": "Calculate the great-circle site distance",
            "compare_select": "Select the higher of two source counts",
            "verify": "Check the count comparison using both independent facts",
        }
        return (labels[goal_op["op"]] + " and retain its input evidence "
                "before producing " + node["output_field"])
    if op == "goal_format":
        return (
            "Render the frozen " + node["output_kind"] +
            " answer contract with fields " +
            ", ".join(node["field_types"])
        )
    if op == "attach_evidence":
        return "Attach the source fact references supporting each result"
    raise ValueError("unsupported_subtask_operation:" + op)


def _reachable_source_ids(task_id: str, tasks: dict, memo: dict) -> set[str]:
    if task_id in memo:
        return memo[task_id]
    task = tasks[task_id]
    result = set(task["source_ids"])
    for parent in task["depends_on"]:
        result.update(_reachable_source_ids(parent, tasks, memo))
    memo[task_id] = result
    return result


def _operator_evidence(
    node: dict, product, trace_row: dict, fact_index: dict,
    accessible_sources: set[str], *, evidence_limit: int,
    source_roles: dict[str, dict] | None = None,
) -> dict:
    actual_refs = sorted({
        ref for row in product.rows for ref in row["support"]
    })
    if trace_row["source_fact_refs"] != actual_refs:
        raise ValueError("primitive_trace_fact_refs_mismatch:" + node["id"])
    for fact_ref in actual_refs:
        fact = fact_index.get(fact_ref)
        if fact is None:
            raise ValueError("unknown_fact_ref:" + fact_ref)
        if fact["source_id"] not in accessible_sources:
            raise ValueError("unauthorized_fact_source:" + fact_ref)
    simulated_calls = []
    if ("evidence_acquisition" in node and source_roles is not None
            and node["source_id"] in source_roles):
        key_values = (
            sorted({
                str(row["values"][node["input_key_field"]])
                for row in product.rows
            }) if node["op"] == "lookup_by_key" else None
        )
        simulated_calls = simulate_acquisition_trace(
            node, source_roles[node["source_id"]],
            actual_refs, key_values=key_values,
        )
    return {
        "node_id": node["id"],
        "operation": node["op"],
        "input_nodes": list(node["inputs"]),
        "action": _describe_operation(node),
        "observed_result": (
            f"Produced {len(product.rows)} row(s) with "
            + ", ".join(product.columns)
        ),
        "output_schema": list(product.columns),
        "output_row_count": len(product.rows),
        "supporting_fact_refs": actual_refs,
        # Prioritize support from the earliest *output rows*, rather than
        # lexicographically smallest fact IDs. For sorted/take Macros this
        # keeps the highest-ranked case in abbreviated evidence samples.
        # The complete authoritative relation and fact-ref set are unchanged.
        "evidence_atoms": [
            fact_index[ref] for ref in list(dict.fromkeys(
                ref for row in product.rows for ref in row["support"]
            ))[:evidence_limit]
        ],
        **({"evidence_acquisition": deepcopy(node["evidence_acquisition"]),
            "tool_executed": False,
            **({"simulated_tool_calls": simulated_calls}
               if simulated_calls else {})}
           if "evidence_acquisition" in node else {}),
        "lookup_keys": ({
            "source_id": node["source_id"],
            "key_field": node["input_key_field"],
            "key_values": sorted({
                str(row["values"][node["input_key_field"]])
                for row in product.rows
            }),
        } if node["op"] == "lookup_by_key" else None),
        "evidence_atoms_total": len(actual_refs),
        "evidence_atoms_truncated": len(actual_refs) > evidence_limit,
    }


def realize_scene(scene: dict, *, evidence_limit: int = 12,
                  strict: bool = True) -> dict:
    """Build a complete, grounded Sub-Agent response for every Macro.

    The Stage-1 scene is read-only. STRICT mode replays its frozen World,
    Program, MacroGraph, Gold and feedback, before materializing any rows.
    """
    if not isinstance(evidence_limit, int) or evidence_limit < 0:
        raise ValueError("evidence_limit must be a nonnegative integer")
    if strict:
        check = validate_replay(scene)
        if check["status"] != "valid":
            raise ValueError("source_scene_replay_invalid:" + repr(check["issues"]))
    graph = scene["macro_graph"]
    nodes = {n["id"]: n for n in scene["gold_program"]["nodes"]}
    execution = evaluate_program(
        scene["gold_program"], sources=scene["world"]["sources"]
    )
    trace = {row["node_id"]: row for row in execution["trace"]}
    facts = _fact_index(scene["world"])
    source_roles = {
        item["source_id"]: item
        for item in scene["problem_spec"]["sources"]
    }
    oracle = run_macro_tasks(graph, sources=scene["world"]["sources"])
    if oracle["task_feedback"] != scene["task_feedback"]:
        raise ValueError("macro_task_feedback_drift")
    if oracle["answer"] != scene["gold_answer"]:
        raise ValueError("macro_terminal_answer_drift")
    tasks = {t["task_id"]: t for t in graph["tasks"]}
    feedback = {item["task_id"]: item for item in oracle["task_feedback"]}
    memo = {}
    records = []
    verified_claims_by_macro = {}
    for wave in graph["reference_waves"]:
        for macro_id in wave:
            macro = tasks[macro_id]
            result = feedback[macro_id]
            accessible = _reachable_source_ids(macro_id, tasks, memo)
            evidence_steps = []
            for op in macro["operator_chain"]:
                row = trace[op["id"]]
                evidence_steps.append(_operator_evidence(
                    nodes[op["id"]], execution["products"][op["id"]],
                    row, facts, accessible, evidence_limit=evidence_limit,
                    source_roles=source_roles,
                ))
            # Verified claims can be inherited only from actual dependency
            # parents, never from the overall Question or future Macros.
            claims = set().union(*(
                verified_claims_by_macro[parent]
                for parent in macro["depends_on"]
            )) if macro["depends_on"] else set()
            if (
                {"report_a", "report_b"}.issubset(accessible)
                and any(
                    step["operation"] in {
                        "compare_fields", "claim_relation_filter",
                    }
                    and "reported_year_a" in step["action"]
                    and "reported_year_b" in step["action"]
                    and "differs from" in step["action"]
                    for step in evidence_steps
                )
            ):
                claims.add("independent_report_year_disagreement")
            verified_claims_by_macro[macro_id] = claims
            terminal_schema = list(macro["outputs"]["typed_relation"])
            if result["output_schema"] != terminal_schema:
                raise ValueError("subtask_output_schema_mismatch:" + macro_id)
            answer_with_provenance = result["workproduct"]
            refs = sorted({
                ref for item in answer_with_provenance
                for ref in item["source_fact_refs"]
            })
            if set(refs) - set(facts):
                raise ValueError("subtask_result_unknown_fact_refs:" + macro_id)
            descriptions = [
                step["action"] + f" ({step['output_row_count']} row(s))"
                for step in evidence_steps
            ]
            summary = (
                "To produce " + ", ".join(terminal_schema) + ", "
                + "; then ".join(
                    part[0].lower() + part[1:] for part in descriptions
                ) + ". "
                + f"The resulting workproduct has {result['result_rows']} row(s), "
                + f"supported by {len(refs)} source fact reference(s)."
            )
            records.append({
                "scene_task_id": scene["task_id"],
                **({"evidence_trust_version":
                    scene["problem_spec"]["evidence_trust_version"]}
                   if scene["problem_spec"].get("evidence_trust_version") else {}),
                "macro_task_id": macro_id,
                "depends_on": list(macro["depends_on"]),
                "authorized_source_ids": list(macro["source_ids"]),
                **({"required_tools": list(macro["required_tools"])}
                   if "required_tools" in macro else {}),
                "output_schema": terminal_schema,
                "answer": [
                    dict(item["values"]) for item in answer_with_provenance
                ],
                "answer_with_provenance": answer_with_provenance,
                "trace_summary": {
                    "objective": "Produce " + ", ".join(terminal_schema),
                    "summary": summary,
                    "steps": evidence_steps,
                    "upstream_macro_tasks": list(macro["depends_on"]),
                    "direct_source_ids": list(macro["source_ids"]),
                    "available_evidence_source_ids": sorted(accessible),
                    "answer_supporting_fact_refs": refs,
                    **({"verified_stage_claims": sorted(claims)}
                       if scene["problem_spec"].get(
                           "question_fidelity_version"
                       ) else {}),
                },
                "grounding": "exact_deterministic_oracle_not_llm_subagent",
            })
    if len(records) != graph["macro_task_count"]:
        raise ValueError("incomplete_macro_task_feedback")
    terminal = next(
        row for row in records
        if row["macro_task_id"] == graph["terminal_task_id"]
    )
    if sorted(map(_canonical, terminal["answer"])) != sorted(
        map(_canonical, scene["gold_answer"])
    ):
        raise ValueError("stage2_terminal_answer_mismatch")
    return {
        "scene_task_id": scene["task_id"],
        "question": scene["question"],
        "terminal_macro_task_id": graph["terminal_task_id"],
        "reference_waves": graph["reference_waves"],
        "macro_tasks": records,
        "macro_task_count": len(records),
        "provenance": "frozen_stage1_world_and_program_exact_replay",
    }


def _canonical(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
