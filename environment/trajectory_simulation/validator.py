"""Strict validation of gold Main/Sub-Agent orchestration records."""
from __future__ import annotations

import json
import re
from decimal import Decimal,InvalidOperation
from typing import Any, Mapping

from .compiler import json_value, ready_subtasks
from .grounding import grouped_evidence


def _equal(left: Any, right: Any) -> bool:
    return json.dumps(json_value(left), sort_keys=True, ensure_ascii=False) == json.dumps(
        json_value(right), sort_keys=True, ensure_ascii=False
    )


def _number(value):
    if isinstance(value,bool):raise ValueError("boolean is not a numeric answer")
    token=str(value).strip()
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?",token):token=token.replace(",","")
    try:result=Decimal(token)
    except InvalidOperation as exc:raise ValueError("not a finite number") from exc
    if not result.is_finite():raise ValueError("not a finite number")
    return result


def answers_equal(left,right,contract=None):
    contract=contract or {}
    if isinstance(right,bool):
        if isinstance(left,str) and left.strip().lower() in {"true","false"}:
            left=left.strip().lower()=="true"
        return isinstance(left,bool) and left==right
    if isinstance(right,(int,float)):
        try:return _number(left)==_number(right)
        except (ValueError,TypeError):return False
    if isinstance(right,(list,tuple,set)):
        # O1's complete tool transports answers as strings.
        if isinstance(left,str):
            try:left=json.loads(left)
            except json.JSONDecodeError:
                if len(right)==1 and contract.get("collection_mode")=="set":
                    left=[left]
        if not isinstance(left,(list,tuple,set)):return False
        numeric=bool(right) and all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in right)
        try:
            key=lambda v: str(_number(v).normalize()) if numeric else json.dumps(json_value(v),sort_keys=True,ensure_ascii=False)
            a,b=[key(v) for v in left],[key(v) for v in right]
        except (ValueError,TypeError):return False
        mode=contract.get("collection_mode","multiset")
        if mode=="ordered":return a==b
        if mode=="set":return set(a)==set(b)
        return sorted(a)==sorted(b)
    return _equal(left,right)


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
        if set(wanted)-set(ids):
            issues.append("missing_ready_subtasks:"+",".join(sorted(set(wanted)-set(ids))))
        if set(ids)-set(wanted):
            issues.append("unexpected_or_unready_subtasks:"+",".join(sorted(set(ids)-set(wanted))))
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
    result: Mapping[str, Any], task: Mapping[str, Any], *, question="",
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
    from .grounding import text_issues
    support=[*task["gold_answer_claims"],{"source_observation":task.get("observation",{})}]
    for field in ("trajectory_summary","answer_summary"):
        issues.extend(field+":"+e for e in text_issues(result.get(field),support,allowed_ids=[task["resource_id"]],question=question))
    return not issues, issues


def validate_main_summary(result,*,claims=None,pending_ids=None,allowed_ids=None,question=""):
    from .grounding import text_issues
    issues=text_issues(result.get("main_summary"),claims,pending_ids=pending_ids,allowed_ids=allowed_ids,question=question)
    return not issues,issues


def validate_final(result: Mapping[str, Any], oracle: Any, answer_contract=None) -> tuple[bool, list[str]]:
    issues = []
    if result.get("action") != "complete":
        issues.append("final_action_not_complete")
    if not answers_equal(result.get("answer"), oracle,answer_contract):
        issues.append("final_answer_mismatch")
    if not isinstance(result.get("summary"),str) or not result["summary"].strip():
        issues.append("empty_final_summary")
    return not issues, issues


def validate_gold_trajectory(
    plan: Mapping[str, Any], rounds: list[Mapping[str, Any]],
    final: Mapping[str, Any],
) -> dict[str, Any]:
    task_map = {task["subtask_id"]: task for task in plan["tasks"]}
    completed: set[str] = set()
    seen_resources: set[str] = set()
    observed_results=[]
    issues: list[str] = []
    for i, round_data in enumerate(rounds):
        if "completed_before" in round_data and set(round_data["completed_before"])!=completed:
            issues.append(f"round_{i+1}:completed_before_mismatch")
        ready = ready_subtasks(plan,completed)
        if not ready:issues.append(f"round_{i+1}:unexpected_empty_round")
        ok, errors = validate_parallel_plan(round_data.get("main_action", {}), ready)
        if not ok:
            issues.extend(f"round_{i+1}:{error}" for error in errors)
        subtasks = round_data.get("subagent_results", [])
        if not isinstance(subtasks, list) or len(subtasks) != len(ready):
            issues.append(f"round_{i+1}:result_count_mismatch")
            continue
        result_ids=[r.get("subtask_id") for r in subtasks if isinstance(r,Mapping)]
        if len(set(result_ids))!=len(result_ids) or set(result_ids)!={t["subtask_id"] for t in ready}:
            issues.append(f"round_{i+1}:result_group_mismatch")
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
            valid, errors = validate_subagent_result(result, task,question=plan.get("question",""))
            if not valid:
                issues.extend(f"round_{i+1}:{error}" for error in errors)
            completed.add(task["subtask_id"])
            observed_results.append(result)
            if task.get("kind","inspect")=="inspect":seen_resources.add(task["resource_id"])
        if "completed_after" in round_data and set(round_data["completed_after"])!=completed:
            issues.append(f"round_{i+1}:completed_after_mismatch")
        if "pending_after" in round_data and set(round_data["pending_after"])!=set(task_map)-completed:
            issues.append(f"round_{i+1}:pending_after_mismatch")
        known_claims,_=grouped_evidence(task_map,observed_results)
        pending_ids=set(task_map)-completed
        valid_summary, errors = validate_main_summary(round_data,claims=known_claims,
            pending_ids=pending_ids,allowed_ids={task_map[t]["resource_id"] for t in completed},question=plan.get("question",""))
        if not valid_summary:
            issues.extend(f"round_{i+1}:{error}" for error in errors)
    if completed != set(task_map):
        issues.append("incomplete_official_task_graph")
    if not set(plan["proof_source_ids"]).issubset(seen_resources):
        issues.append("incomplete_gold_proof")
    final_ok, final_issues = validate_final(final, plan["final_answer"],plan.get("answer_contract"))
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
