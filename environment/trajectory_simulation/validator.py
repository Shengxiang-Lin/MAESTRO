"""Strict validation of gold Main/Sub-Agent orchestration records."""
from __future__ import annotations

import json
from typing import Any, Mapping

from .compiler import json_value


def _equal(left: Any, right: Any) -> bool:
    return json.dumps(json_value(left), sort_keys=True, ensure_ascii=False) == json.dumps(
        json_value(right), sort_keys=True, ensure_ascii=False
    )


def validate_parallel_plan(
    output: Mapping[str, Any], ready: list[Mapping[str, Any]],
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    assigned = output.get("parallel_subtasks")
    if not isinstance(assigned, list):
        return False, ["parallel_subtasks_missing"]
    wanted = {task["subtask_id"]: task for task in ready}
    ids = [str(row.get("subtask_id")) for row in assigned if isinstance(row, Mapping)]
    if len(assigned) != len(ready) or len(ids) != len(set(ids)) or set(ids) != set(wanted):
        issues.append("parallel_group_mismatch")
    for row in assigned:
        if not isinstance(row, Mapping):
            issues.append("malformed_subtask")
            continue
        task = wanted.get(str(row.get("subtask_id")))
        if task is None:
            continue
        if str(row.get("agent")) != task["agent"]:
            issues.append("specialist_mismatch")
        if str(row.get("resource_id")) != task["resource_id"]:
            issues.append("resource_mismatch")
        if not str(row.get("instruction", "")).strip():
            issues.append("empty_instruction")
    return not issues, sorted(set(issues))


def validate_subagent_result(
    result: Mapping[str, Any], task: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    issues = []
    if result.get("subtask_id") != task["subtask_id"]:
        issues.append("subtask_id_mismatch")
    if result.get("status") != "success":
        issues.append("subtask_not_successful")
    if result.get("evidence_source_ids") != [task["resource_id"]]:
        issues.append("evidence_source_mismatch")
    if not _equal(result.get("answer"), task["gold_answer_claims"]):
        issues.append("answer_claims_not_grounded")
    if not str(result.get("trajectory_summary", "")).strip():
        issues.append("empty_compressed_trajectory")
    if not str(result.get("answer_summary", "")).strip():
        issues.append("empty_subtask_answer_summary")
    return not issues, issues


def validate_main_summary(result: Mapping[str, Any]) -> tuple[bool, list[str]]:
    if not str(result.get("main_summary", "")).strip():
        return False, ["empty_main_summary"]
    return True, []


def validate_final(result: Mapping[str, Any], oracle: Any) -> tuple[bool, list[str]]:
    issues = []
    if result.get("action") != "complete":
        issues.append("final_action_not_complete")
    if not _equal(result.get("answer"), oracle):
        issues.append("final_answer_mismatch")
    if not str(result.get("summary", "")).strip():
        issues.append("empty_final_summary")
    return not issues, issues


def validate_gold_trajectory(
    plan: Mapping[str, Any], rounds: list[Mapping[str, Any]],
    final: Mapping[str, Any],
) -> dict[str, Any]:
    task_map = {task["subtask_id"]: task for task in plan["tasks"]}
    completed: set[str] = set()
    seen_resources: set[str] = set()
    issues: list[str] = []
    for i, round_data in enumerate(rounds):
        ready = [task for task in plan["tasks"]
                 if task["subtask_id"] not in completed
                 and set(task["depends_on"]).issubset(completed)]
        ok, errors = validate_parallel_plan(round_data.get("main_action", {}), ready)
        if not ok:
            issues.extend(f"round_{i+1}:{error}" for error in errors)
        subtasks = round_data.get("subagent_results", [])
        if not isinstance(subtasks, list) or len(subtasks) != len(ready):
            issues.append(f"round_{i+1}:result_count_mismatch")
            continue
        for result in subtasks:
            if not isinstance(result, Mapping):
                issues.append(f"round_{i+1}:malformed_result")
                continue
            task = task_map.get(result.get("subtask_id"))
            if task is None or task["subtask_id"] not in {
                item["subtask_id"] for item in ready
            }:
                issues.append(f"round_{i+1}:unknown_or_unready_result")
                continue
            valid, errors = validate_subagent_result(result, task)
            if not valid:
                issues.extend(f"round_{i+1}:{error}" for error in errors)
            completed.add(task["subtask_id"])
            seen_resources.add(task["resource_id"])
        valid_summary, errors = validate_main_summary(round_data)
        if not valid_summary:
            issues.extend(f"round_{i+1}:{error}" for error in errors)
    if completed != set(task_map):
        issues.append("incomplete_official_task_graph")
    if not set(plan["proof_source_ids"]).issubset(seen_resources):
        issues.append("incomplete_gold_proof")
    final_ok, final_issues = validate_final(final, plan["final_answer"])
    if not final_ok:
        issues.extend(final_issues)
    return {
        "accepted": not issues,
        "issues": issues,
        "completed_subtasks": len(completed),
        "expected_subtasks": len(task_map),
        "observed_resource_ids": sorted(seen_resources),
        "proof_source_ids": list(plan["proof_source_ids"]),
        "final_answer_verified": final_ok,
    }


__all__ = [
    "validate_parallel_plan", "validate_subagent_result",
    "validate_main_summary", "validate_final", "validate_gold_trajectory",
]
