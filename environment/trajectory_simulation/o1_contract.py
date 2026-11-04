"""O1 wire contracts and private, source-scoped simulation bindings."""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Mapping

from .grounding import _numeric_tokens, text_issues
from .validator import _equal as _equal_json, answers_equal

TOOLS = (
    "GoogleSearchAction", "ExtractUrlContentAction", "ExecuteCodeAction",
    "ImageAnalysisAction", "AudioAnalysisAction", "VideoAnalysisAction",
)


def action_schema(payload):
    task = {
        "type": "object", "properties": {
            "task_instruction": {"type": "string", "minLength": 1},
            "context": {"type": "string"},
            "model": {"type": "string", "enum": ["model_1"]},
            "tools": {"type": "array", "minItems": 1,
                      "items": {"type": "string", "enum": list(TOOLS)}},
        }, "required": ["task_instruction", "context", "model", "tools"],
        "additionalProperties": False,
    }
    binding = {
        "type": "object", "properties": {
            "task_index": {"type": "integer", "minimum": 0},
            "subtask_ids": {"type": "array", "minItems": 1, "items": {
                "type": "string", "enum": [
                    item["simulation_subtask_id"] for item in payload["feasible_work"]]}},
        }, "required": ["task_index", "subtask_ids"], "additionalProperties": False,
    }
    complete = {
        "type": "object", "properties": {
            "action": {"type": "string", "enum": ["complete"]},
            "reasoning": {"type": "string"},
            "params": {"type": "object", "properties": {
                "answer": {"type": "string", "minLength": 1}},
                "required": ["answer"], "additionalProperties": False},
        }, "required": ["action", "reasoning", "params"], "additionalProperties": False,
    }
    if not payload["feasible_work"]:
        return complete
    delegate = {
        "type": "object", "properties": {
            "action": {"type": "string", "enum": ["delegate_task"]},
            "reasoning": {"type": "string"},
            "params": {"type": "object", "properties": {
                "tasks": {"type": "array", "minItems": 1, "items": task}},
                "required": ["tasks"], "additionalProperties": False},
            "task_bindings": {"type": "array", "minItems": 1, "items": binding},
        }, "required": ["action", "reasoning", "params", "task_bindings"],
        "additionalProperties": False,
    }
    return {"anyOf": [delegate, complete]} if payload["evidence_sufficient"] else delegate


def evidence_prefix(scopes):
    """Public frozen evidence, with no oracle answer or private scope handles."""
    lines = []
    for task in scopes:
        lines.append(f"{task['resource_id']} ({task['observation']['title']}):")
        if task["kind"] == "discover":
            lines.append("Resource located; contents not inspected.")
        else:
            lines.extend(
                f"{claim['subject']} | {claim['predicate']} | {claim['object']}"
                for claim in task["gold_answer_claims"])
    return "\n".join(lines)


def feedback_schema(payload):
    claims = {task["subtask_id"]: {"enum": [task["gold_answer_claims"]]}
              for task in payload["scopes"]}
    text = {"type": "string", "minLength": 1}
    return {
        "type": "object", "properties": {
            "finish_result": {"type": "object", "properties": {
                "result": {"type": "string", "pattern":
                           "^" + re.escape(evidence_prefix(payload["scopes"]))
                           + r"(?:\n[\s\S]*)?$"},
                "status": {"type": "string", "enum": ["done"]},
                "summary": text},
                "required": ["result", "status", "summary"], "additionalProperties": False},
            "trace_summary": text,
            "grounded_claims": {"type": "object", "properties": claims,
                                "required": list(claims), "additionalProperties": False},
        }, "required": ["finish_result", "trace_summary", "grounded_claims"],
        "additionalProperties": False,
    }


def answer_string(answer: Any) -> str:
    return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)


def suggested_tools(task: Mapping[str, Any]) -> list[str]:
    if task["kind"] == "discover":
        return ["GoogleSearchAction"]
    source_type = task["source_type"].lower()
    for prefix, tool in (
        ("image", "ImageAnalysisAction"), ("audio", "AudioAnalysisAction"),
        ("video", "VideoAnalysisAction"), ("table", "ExecuteCodeAction"),
        ("database", "ExecuteCodeAction"),
    ):
        if source_type.startswith(prefix):
            return [tool]
    return ["ExtractUrlContentAction"]


def work_hint(task: Mapping[str, Any]) -> dict[str, Any]:
    hint = {
        "simulation_subtask_id": task["subtask_id"],
        "kind": task["kind"], "resource_id": task["resource_id"],
        "source_type": task["source_type"], "suggested_tools": suggested_tools(task),
    }
    if task["kind"] == "discover":
        hint["known_discovery_terms"] = task.get("discovery_terms", [])
    else:
        hint["resource_title"] = task["observation"]["title"]
    return hint


def main_prompt(payload: Mapping[str, Any]) -> str:
    return (
        "Generate the NEXT Orchestra-o1 Main action, not a whole trajectory. "
        "Use the question and previous public Sub-Agent answers/trace_summary to decide "
        "which work to delegate NOW, how to group it, and how many independent tasks to run "
        "in parallel. You may choose ANY NONEMPTY SUBSET of currently feasible work, or "
        "combine several feasible resources into one task. You need NOT dispatch all work. "
        "Do not invent observations or schedule infeasible work. The private teacher hint "
        "contains feasible scope handles, NOT evidence or the final answer. "
        "Return delegate_task as {\"action\":\"delegate_task\",\"reasoning\":str,"
        "\"params\":{\"tasks\":[{\"task_instruction\":str,\"context\":str,"
        "\"model\":\"model_1\",\"tools\":[registered tool names]}]},"
        "\"task_bindings\":[{\"task_index\":0,\"subtask_ids\":[feasible scope IDs]}]}. "
        "IMPORTANT: task_bindings is a TOP-LEVEL sibling of action, reasoning and params. "
        "params contains ONLY tasks; NEVER put task_bindings inside params. "
        "Exactly one private binding per task; a scope may occur only once in this round. "
        "task_bindings are stripped from the O1 action and never shown as Sub-Agent feedback. "
        "Do not put private subtask:/discover: handles in reasoning, task_instruction or context; "
        "refer to public resource IDs/titles instead. "
        "Choose tools appropriate to each scope from suggested_tools. "
        "Inspection tasks should gather ALL facts in their source, including relations and "
        "contextual facts, rather than discard facts based on an early candidate filter. "
        "For discover scopes, request locating the resource from known clues, NOT reading "
        "its contents or solving the question; inspection must wait for a later round. "
        "Public reasoning describes the task strategy, never private teacher hints. "
        "Alternatively, if evidence is sufficient, return "
        "{\"action\":\"complete\",\"reasoning\":str,\"params\":{\"answer\":str}}. "
        "complete.answer is a STRING: numbers/booleans use their JSON spelling, and a "
        "collection is a JSON array encoded inside the string. JOIN facts about the same "
        "public entity across different sources; attributes need NOT coexist in one source. "
        "A resource title is an alias of its exact resource ID, not a different resource. "
        "Do not conclude from resource co-location or from unobserved facts. "
        "Keep reasoning brief. Return JSON only.\n"
        + json.dumps({
            "public_state": payload["public_state"],
            "private_teacher_hint": {
                "feasible_work": payload["feasible_work"],
                "evidence_sufficient": payload["evidence_sufficient"],
                "answer_format": payload["answer_contract"],
            },
            "models": ["model_1"], "registered_tools": TOOLS,
        }, ensure_ascii=False, separators=(",", ":"))
    )


def feedback_prompt(payload: Mapping[str, Any]) -> str:
    evidence = [{
        "simulation_subtask_id": task["subtask_id"], "kind": task["kind"],
        "resource_id": task["resource_id"], "source_observation": task["observation"],
        "authoritative_claims": task["gold_answer_claims"],
    } for task in payload["scopes"]]
    return (
        "Simulate ONE Sub-Agent reply for the Main task below, not a whole trajectory "
        "and not one reply per source. All assigned scopes belong to this one task. "
        "Use only assigned observations and already observed context. For discover, report "
        "the returned descriptor only: contents have NOT been inspected. "
        "The assigned scope kind is authoritative even if the Main asks for more: a discover "
        "scope locates a resource and does NOT inspect it. Unread content is unknown, NOT "
        "empty and NOT evidence of no matching entities. Do not give a zero/no-match answer "
        "from discovery. Say that inspection is still needed. For inspect, report "
        "observed facts faithfully. A title in an observation is an alias of THAT resource ID. "
        "Combine information about the same entity across assigned sources when useful. "
        "Report ALL supplied facts, including relevant contextual facts, without inventing "
        "values, other sources, real tool-call counts, URLs, costs, or errors. You simulate "
        "the compressed reply; do NOT generate fictitious raw tool traces. "
        "Return JSON {\"finish_result\":{\"result\":str,\"status\":\"done\",\"summary\":str},"
        "\"trace_summary\":str,\"grounded_claims\":{scope_ID:exact_authoritative_claim_list}}. "
        "grounded_claims is a private audit field: copy EVERY assigned scope's claims exactly "
        "(same order, values and types). It is removed from public feedback. "
        "Do not put private subtask:/discover: handles in the public result or summaries. "
        "finish_result.result is a source-attributed factual answer to THIS delegated task; "
        "It MUST include EVERY assigned authoritative claim in public text. For each inspect "
        "claim write its EXACT subject, predicate label, and object value, for example "
        "'source_01: Alex | birth date | 1687'. Do not replace the facts with a no-match/zero "
        "answer or omit facts outside the Main's narrow filter. Other sources may supply "
        "missing attributes or relation links. You may add a brief task-specific conclusion "
        "after reporting the evidence, but missing data in this source is not global absence. "
        "Start finish_result.result with the exact required_evidence_prefix supplied below; "
        "then optionally add your task-specific answer after a newline. The prefix contains "
        "only assigned evidence, not a final answer. Write summary and trace_summary yourself. "
        "trace_summary briefly describes its simulated work and any remaining discovery/read "
        "boundary. Completing this task need not finish the user's whole question. "
        "Do not answer the original question from outside knowledge. Return JSON only.\n"
        + json.dumps({
            "question": payload["question"], "assigned_task": payload["assigned_task"],
            "assigned_evidence": evidence,
            "required_evidence_prefix": evidence_prefix(payload["scopes"]),
            "already_observed_context": payload["past_evidence"],
        }, ensure_ascii=False, separators=(",", ":"))
    )


def public_action(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.get(key, "" if key == "reasoning" else {})
            for key in ("action", "reasoning", "params")}


def validate_action(value, ready, sufficient, answer, contract):
    issues = []
    params = value.get("params")
    if not isinstance(params, dict):
        return False, ["params_must_be_object"]
    if not isinstance(value.get("reasoning", ""), str):
        issues.append("reasoning_must_be_string")
    elif any(marker in value.get("reasoning", "") for marker in ("subtask:", "discover:")):
        issues.append("private_scope_id_in_reasoning")
    if value.get("action") == "complete":
        candidate = params.get("answer")
        if not isinstance(candidate, str) or not candidate.strip():
            issues.append("complete_answer_must_be_nonempty_string")
        elif not answers_equal(candidate, answer, contract):
            issues.append("final_answer_mismatch")
        if not sufficient:
            issues.append("decision_evidence_incomplete")
        if value.get("task_bindings"):
            issues.append("complete_cannot_bind_subtasks")
        return not issues, issues
    if value.get("action") != "delegate_task":
        return False, ["action_must_be_delegate_task_or_complete"]
    tasks = params.get("tasks")
    bindings = value.get("task_bindings")
    if not isinstance(tasks, list) or not tasks:
        return False, ["delegate_tasks_must_be_nonempty_list"]
    if not isinstance(bindings, list) or len(bindings) != len(tasks):
        if "task_bindings" in params:
            return False, ["move_task_bindings_to_top_level_outside_params"]
        return False, ["one_private_binding_required_per_task"]
    if set(params) != {"tasks"}:
        issues.append("delegate_params_must_contain_only_tasks")
    by_id = {task["subtask_id"]: task for task in ready}
    seen = []
    indices = []
    for binding in bindings:
        if not isinstance(binding, dict):
            issues.append("binding_must_be_object")
            continue
        index = binding.get("task_index")
        if type(index) is not int or not 0 <= index < len(tasks):
            issues.append("invalid_binding_task_index")
            continue
        indices.append(index)
        ids = binding.get("subtask_ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(sid, str) for sid in ids):
            issues.append("binding_scopes_must_be_nonempty_string_list")
            continue
        seen.extend(ids)
        if set(ids) - set(by_id):
            issues.append("unready_or_unknown_scope:" + ",".join(sorted(set(ids) - set(by_id))))
        task = tasks[index]
        if not isinstance(task, dict):
            issues.append("task_must_be_object")
            continue
        if not isinstance(task.get("task_instruction"), str) or not task["task_instruction"].strip():
            issues.append("empty_task_instruction")
        if not isinstance(task.get("context", ""), str):
            issues.append("context_must_be_string")
        for field in ("task_instruction", "context"):
            if isinstance(task.get(field), str) and any(
                marker in task[field] for marker in ("subtask:", "discover:")
            ):
                issues.append("private_scope_id_in_" + field)
        if task.get("model") != "model_1":
            issues.append("unknown_model_alias")
        tools = task.get("tools")
        if not isinstance(tools, list) or not tools or not all(isinstance(tool, str) and tool in TOOLS for tool in tools):
            issues.append("invalid_registered_tools")
        else:
            for sid in ids:
                if sid in by_id and not set(suggested_tools(by_id[sid])) & set(tools):
                    issues.append("tool_scope_mismatch:" + sid)
    if set(indices) != set(range(len(tasks))) or len(indices) != len(set(indices)):
        issues.append("binding_indices_must_cover_tasks_once")
    if any(count > 1 for count in Counter(seen).values()):
        issues.append("scope_assigned_more_than_once")
    return not issues, issues


def validate_feedback(value, payload):
    scopes = payload["scopes"]
    expected = {task["subtask_id"]: task["gold_answer_claims"] for task in scopes}
    if not _equal_json(value.get("grounded_claims"), expected):
        return False, ["scoped_claims_not_grounded"]
    finish = value.get("finish_result")
    if not isinstance(finish, dict):
        return False, ["finish_result_must_be_object"]
    issues = []
    if finish.get("status") != "done":
        issues.append("clean_task_status_must_be_done")
    result = finish.get("result", "")
    if isinstance(result, str):
        normalized = result.casefold()
        numbers = _numeric_tokens(result)
        for task in scopes:
            if task["resource_id"].casefold() not in normalized:
                issues.append("result_missing_source_attribution:" + task["resource_id"])
            if task["kind"] == "discover":
                if task["observation"]["title"].casefold() not in normalized:
                    issues.append("result_missing_discovered_title")
                continue
            for index, claim in enumerate(task["gold_answer_claims"]):
                for field in ("subject", "predicate", "object"):
                    surface = str(claim[field])
                    present = surface.casefold() in normalized
                    if field == "object" and claim["object_kind"] in {
                        "number", "quantity", "year",
                    }:
                        tokens = _numeric_tokens(surface)
                        present = bool(tokens) and tokens.issubset(numbers)
                    if not present:
                        issues.append(
                            f"result_omits_assigned_fact:{task['resource_id']}:{index}:{field}")
    support = [
        *payload["past_evidence"],
        *[{"resource_id": task["resource_id"],
           "title": task["observation"]["title"],
           "claims": task["gold_answer_claims"]} for task in scopes],
    ]
    allowed = set(payload["known_resource_ids"]) | {task["resource_id"] for task in scopes}
    for field, text in (
        ("result", finish.get("result")), ("summary", finish.get("summary")),
        ("trace_summary", value.get("trace_summary")),
    ):
        if not isinstance(text, str) or not text.strip():
            issues.append("empty_" + field)
        else:
            if any(marker in text for marker in ("subtask:", "discover:")):
                issues.append("private_scope_id_in_" + field)
            issues.extend(field + ":" + issue for issue in text_issues(
                text, support, allowed_ids=allowed, question=payload["question"],
                max_length=12000 if field == "result" else 3000))
    return not issues, issues
