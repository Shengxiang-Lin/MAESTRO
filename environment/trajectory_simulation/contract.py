"""Small O1 wire contracts for fragment-guided clean teacher trajectories.

LLMs never copy private graph handles or fact lists into public O1 actions.
Every delegated task is bound to a ready gold fragment by its OUTPUT INDEX.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .compiler import json_value

TOOLS = (
    "GoogleSearchAction", "ExtractUrlContentAction", "ExecuteCodeAction",
    "ImageAnalysisAction", "AudioAnalysisAction", "VideoAnalysisAction",
)


def answer_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(json_value(value), ensure_ascii=False)


def _number(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    token = str(value).strip()
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", token):
        token = token.replace(",", "")
    result = Decimal(token)
    if not result.is_finite():
        raise InvalidOperation
    return result


def answers_equal(actual: Any, expected: Any, contract: Mapping[str, Any]) -> bool:
    if isinstance(expected, bool):
        if isinstance(actual, str) and actual.strip().lower() in {"true", "false"}:
            actual = actual.strip().lower() == "true"
        return isinstance(actual, bool) and actual == expected
    if isinstance(expected, (int, float)):
        try:
            return _number(actual) == _number(expected)
        except (InvalidOperation, ValueError, TypeError):
            return False
    if isinstance(expected, (list, tuple, set)):
        if isinstance(actual, str):
            try:
                actual = json.loads(actual)
            except json.JSONDecodeError:
                if contract.get("collection_mode") == "set" and len(expected) == 1:
                    actual = [actual]
        if not isinstance(actual, (list, tuple, set)):
            return False
        numerical = bool(expected) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in expected
        )
        def key(value: Any) -> str:
            return (str(_number(value).normalize()) if numerical
                    else json.dumps(json_value(value), sort_keys=True, ensure_ascii=False))
        try:
            left = [key(x) for x in actual]
            right = [key(x) for x in expected]
        except (ValueError, InvalidOperation, TypeError):
            return False
        mode = contract.get("collection_mode", "multiset")
        if mode == "set":
            return set(left) == set(right)
        if mode == "ordered":
            return left == right
        return sorted(left) == sorted(right)
    return actual == expected


def action_schema(guidance: Mapping[str, Any]) -> dict[str, Any]:
    text = {"type": "string"}
    if guidance["may_complete"]:
        return {
            "type": "object", "properties": {
                "action": {"type": "string", "enum": ["complete"]},
                "reasoning": text,
                "params": {"type": "object", "properties": {
                    "answer": {"type": "string"},
                }, "required": ["answer"], "additionalProperties": False},
            }, "required": ["action", "reasoning", "params"],
            "additionalProperties": False,
        }
    task = {
        "type": "object", "properties": {
            "task_instruction": text, "context": text,
            "model": {"type": "string", "enum": ["model_1"]},
            "tools": {"type": "array", "minItems": 1,
                      "items": {"type": "string", "enum": list(TOOLS)}},
        }, "required": ["task_instruction", "context", "model", "tools"],
        "additionalProperties": False,
    }
    return {
        "type": "object", "properties": {
            "action": {"type": "string", "enum": ["delegate_task"]},
            "reasoning": text,
            "params": {"type": "object", "properties": {
                "tasks": {"type": "array", "items": task},
            }, "required": ["tasks"], "additionalProperties": False},
        }, "required": ["action", "reasoning", "params"], "additionalProperties": False,
    }


def feedback_schema() -> dict[str, Any]:
    return {
        "type": "object", "properties": {
            "summary": {"type": "string"},
            "trace_summary": {"type": "string"},
        }, "required": ["summary", "trace_summary"], "additionalProperties": False,
    }


def main_prompt(payload: Mapping[str, Any]) -> str:
    public = payload["public_state"]
    guidance = payload["gold_guidance"]
    output = (
        "Return an O1 complete action with params.answer as a STRING. "
        "Use only the completed public subtask history to determine the answer."
        if guidance["may_complete"] else
        "Return an O1 delegate_task action with EXACTLY ONE task for EACH "
        "ready fragment, in the ORDER of ready[]. The tasks run in parallel. "
        "Use the private fragment operations and evidence descriptors to write "
        "specific actionable task_instruction/context and select appropriate tools. "
        "Do not invent, drop, merge or reorder fragments."
    )
    return (
        "You are producing ONE official Orchestra-o1 Main action at this point in "
        "the interaction, not a whole trajectory. The O1-visible state is limited "
        "to Question, completed Sub-Agent History and available tools. "
        "PRIVATE gold guidance is teacher supervision only; its fragment IDs, "
        "source selection and graph internals MUST NOT appear in the public action. "
        "Do not mention future observations or claim tasks are already completed. "
        + output + " Return JSON only.\n"
        + json.dumps({
            "public_o1_state": public, "private_gold_guidance": guidance,
        }, ensure_ascii=False, separators=(",", ":"))
    )


def feedback_prompt(payload: Mapping[str, Any]) -> str:
    """The answer is fixed by SymbolicExecutor; LLM writes narrative only."""
    return (
        "Simulate ONE clean Sub-Agent's concise textual report for the assigned "
        "composed reasoning fragment. Its answer is already fixed by the first-stage "
        "symbolic executor; DO NOT try to replace the answer or enumerate raw fact "
        "JSON in your response. Combine the supplied, source-scoped evidence with "
        "completed predecessor answers to describe the actual inference. "
        "Write summary as a brief, grounded interpretation of this SUBTASK answer; "
        "write trace_summary as a compressed account of evidence retrieval and the "
        "specified ProgramIR operations. Do not claim real tools were executed, "
        "invent sources, report future results, or solve unrelated tasks. "
        "Return exactly JSON {\"summary\":str,\"trace_summary\":str}.\n"
        + json.dumps({
            "question": payload["question"],
            "task_instruction": payload["task_instruction"],
            "dependency_results": payload["dependency_results"],
            "gold_operations": payload["fragment"]["operations"],
            "authoritative_subtask_answer": payload["fragment"]["expected_answer"],
            "assigned_source_observations": payload["fragment"]["observations"],
        }, ensure_ascii=False, separators=(",", ":"))
    )


def _hidden_identifier(text: str) -> bool:
    return bool(re.search(
        r"(?:fragment:\d+|fragment:terminal|subtask:|discover:|entity:[A-Za-z0-9_-]+)",
        text,
    ))


def validate_action(value: Any, guidance: Mapping[str, Any],
                    oracle: Any, answer_contract: Mapping[str, Any],
                    *, public_state: Mapping[str, Any] | None = None) -> tuple[bool, list[str]]:
    if not isinstance(value, Mapping):
        return False, ["action_not_object"]
    issues: list[str] = []
    if set(value) != {"action", "reasoning", "params"}:
        issues.append("unexpected_action_fields")
    if not isinstance(value.get("reasoning"), str) or _hidden_identifier(value.get("reasoning", "")):
        issues.append("invalid_public_reasoning")
    params = value.get("params")
    if not isinstance(params, Mapping):
        return False, ["params_not_object"]
    public_sources = set(re.findall(
        r"source_\d+", json.dumps(public_state or {}, ensure_ascii=False)
    ))
    visible_text = str(value.get("reasoning", ""))
    if guidance["may_complete"]:
        if set(re.findall(r"source_\d+", visible_text)) - public_sources:
            issues.append("unobserved_resource_in_main_reasoning")
        if value.get("action") != "complete":
            issues.append("must_complete_after_gold_fragments")
        if not answers_equal(params.get("answer"), oracle, answer_contract):
            issues.append("final_answer_mismatch")
        if set(params) != {"answer"}:
            issues.append("unexpected_complete_params")
        return not issues, issues
    if value.get("action") != "delegate_task":
        issues.append("must_delegate_ready_gold_fragments")
    tasks = params.get("tasks")
    ready = guidance["ready"]
    if not isinstance(tasks, list) or len(tasks) != len(ready):
        return False, issues + ["parallel_gold_fragment_count_mismatch"]
    if set(params) != {"tasks"}:
        issues.append("unexpected_delegate_params")
    for index, (task, fragment) in enumerate(zip(tasks, ready)):
        if not isinstance(task, Mapping):
            issues.append(f"task_{index}:not_object")
            continue
        if set(task) != {"task_instruction", "context", "model", "tools"}:
            issues.append(f"task_{index}:unexpected_fields")
        public_text = str(task.get("task_instruction", "")) + " " + str(task.get("context", ""))
        visible_text += " " + public_text
        for key in ("task_instruction", "context"):
            content = task.get(key)
            if not isinstance(content, str) or not content.strip() or _hidden_identifier(content):
                issues.append(f"task_{index}:invalid_{key}")
        if task.get("model") != "model_1":
            issues.append(f"task_{index}:invalid_model")
        tools = task.get("tools")
        if not isinstance(tools, list) or not tools or any(x not in TOOLS for x in tools):
            issues.append(f"task_{index}:invalid_tools")
        elif not set(fragment["suggested_tools"]).intersection(tools):
            issues.append(f"task_{index}:tools_not_grounded")
    if set(re.findall(r"source_\d+", visible_text)) - public_sources:
        issues.append("future_resource_id_in_public_action")
    return not issues, issues


def validate_feedback(value: Any, fragment: Mapping[str, Any]) -> tuple[bool, list[str]]:
    if not isinstance(value, Mapping):
        return False, ["feedback_not_object"]
    issues = []
    if set(value) != {"summary", "trace_summary"}:
        issues.append("unexpected_feedback_fields")
    supported = json.dumps({
        "answer": fragment["expected_answer"],
        "operations": fragment["operations"],
        "evidence": fragment["observations"],
    }, ensure_ascii=False)
    for key in ("summary", "trace_summary"):
        content = value.get(key)
        if not isinstance(content, str) or not content.strip():
            issues.append("empty_" + key)
            continue
        if len(content) > (600 if key == "summary" else 1400):
            issues.append(key + "_not_compressed")
        if _hidden_identifier(content):
            issues.append(key + "_leaks_private_fragment_id")
        # Prevent references to resources outside this task's scoped packet.
        mentioned = set(re.findall(r"source_\d+", content))
        assigned = {obs["resource_id"] for obs in fragment["observations"]}
        if mentioned - assigned:
            issues.append(key + "_unobserved_source")
        # Numeric assertions may only reuse supplied observed/public values.
        def numeric(text: str) -> set[Decimal]:
            tokens = re.findall(r"(?<!\w)[+-]?\d+(?:,\d+)*(?:\.\d+)?", text)
            result = set()
            for token in tokens:
                try:
                    result.add(_number(token))
                except (InvalidOperation, ValueError, TypeError):
                    pass
            return result
        # Source IDs are provenance handles, not numeric observations.
        masked = re.sub(r"source_\d+", "", content)
        if numeric(masked) - numeric(supported):
            issues.append(key + "_unsupported_numeric_claim")
    return not issues, sorted(set(issues))


def public_action(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ("action", "reasoning", "params")}


__all__ = [
    "TOOLS", "action_schema", "feedback_schema", "main_prompt", "feedback_prompt",
    "answer_string", "answers_equal", "validate_action", "validate_feedback",
    "public_action",
]
