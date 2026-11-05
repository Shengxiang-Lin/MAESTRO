"""Natural O1 language over a separately verified, public executable contract.

This module never executes a gold program, changes a task dependency, selects
tools, or supplies an undiscovered answer to Main. The writer changes only
Main reasoning/task wording or Sub-Agent report wording. A deterministic
guard AND a separate semantic/evidence review must accept a rewrite before
the trajectory may expose it. Failure falls back to the already checked
canonical language; the caller records that fallback.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Mapping

from .contract import answer_string, validate_action, validate_feedback


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _numbers(value: str) -> set[str]:
    """Numbers in prose, not instruction/round/source handles."""
    value = re.sub(r"\b(?:Attempt|subtask|instruction|step)\s+\d+\b", "", value, flags=re.I)
    value = re.sub(r"\bsource_\d+\b", "", value)
    value = re.sub(r"(?m)^\s*\d+\.\s+", "", value)
    return set(re.findall(r"(?<!\w)-?\d+(?:,\d{3})*(?:\.\d+)?", value))


def _required_literals(fragment: Mapping[str, Any]) -> list[str]:
    required = []
    for op in fragment["operations"]:
        name, args = op["op"], op["inputs"]
        if name == "FIND" and op.get("find_kind") != "type":
            candidate = str(args[0])
            if not candidate.startswith("the entity matching "):
                required.append(candidate)
        if name.startswith("FILTER_") and name != "FILTER_TYPE" and len(args) >= 3:
            required.append(str(args[-1]))
        if name.startswith("VERIFY_") and len(args) >= 2:
            required.append(str(args[-1]))
    return list(dict.fromkeys(required))


def _language_guard(text: Any, *, limit: int, allow_empty: bool = False) -> None:
    if not isinstance(text, str) or (not text.strip() and not allow_empty):
        raise ValueError("natural_language_missing")
    if len(text) > limit:
        raise ValueError("natural_language_too_long")
    if re.search(
        r"\b(?:ProgramIR|SymbolicExecutor|private gold|private fragment|gold guidance|"
        r"oracle answer|synthesis controller|teacher-forced)\b",
        text, re.I,
    ):
        raise ValueError("natural_language_private_terminology")


def main_writer_prompt(payload: Mapping[str, Any], canonical: Mapping[str, Any]) -> str:
    public = {
        "question": payload["public_state"]["question"],
        "completed_subtasks": [{
            "reference": result.get("public_ref"), "answer": result["answer"],
            "summary": result.get("summary", ""),
        } for result in payload.get("completed_subtask_contracts", [])],
        "available_reference_handles": payload.get("visible_resources", []),
        "canonical_assignments": [{
            "required_checks": task["task_instruction"],
            "assigned_tools": task["tools"],
        } for task in canonical["params"].get("tasks", [])],
        "action": canonical["action"],
    }
    if canonical["action"] == "complete":
        public["final_public_subtask_result"] = canonical["params"]["answer"]
    return (
        "You are an O1 Main Agent phrasing a correct, already planned orchestration action. "
        "The canonical assignments are mandatory; you CANNOT alter their number, order, "
        "relations, directions, operators, thresholds, or output targets. "
        "Rewrite each assignment as a natural, self-contained delegation in 1-4 sentences: "
        "state the information goal, where to look, constraints and expected deliverable. "
        "DO NOT reproduce the numbered checklist or say 'entities satisfying the specified constraints'. "
        "Preserve named starting anchors, exact values, relation ownership, AND/OR and comparison direction. "
        "Use no results from future subtasks, and do not assume evidence not yet inspected. "
        "For each task, provide only a task_instruction; the controller supplies tools, context, "
        "references and the exact executable contract. Main reasoning should briefly explain why "
        "this batch is appropriate, or be empty if no reasoning adds value. "
        "If action is complete, return only reasoning; the controller fixes the final answer. "
        "Avoid mentioning gold plans, private IDs, or artificial simulation. Return JSON ONLY: "
        '{"reasoning":"...","task_instructions":["..."]} for delegate_task, '
        'or {"reasoning":"..."} for complete.\n' + _json(public)
    )


def realize_main(
    payload: Mapping[str, Any], canonical: Mapping[str, Any], proposal: Any,
) -> dict[str, Any]:
    """Build an O1 wire action; immutable contract fields are copied, not generated."""
    if not isinstance(proposal, dict) or not isinstance(proposal.get("reasoning"), str):
        raise ValueError("invalid_main_natural_language_plan")
    _language_guard(proposal["reasoning"], limit=600, allow_empty=True)
    output = copy.deepcopy(canonical)
    output["reasoning"] = proposal["reasoning"]
    if canonical["action"] == "complete":
        if set(proposal) != {"reasoning"}:
            raise ValueError("invalid_natural_completion_fields")
    else:
        if set(proposal) != {"reasoning", "task_instructions"}:
            raise ValueError("invalid_natural_delegation_fields")
        texts = proposal["task_instructions"]
        tasks = output["params"]["tasks"]
        fragments = payload["gold_guidance"]["ready"]
        if not isinstance(texts, list) or len(texts) != len(tasks):
            raise ValueError("natural_task_count_mismatch")
        for i, (candidate, task, fragment) in enumerate(zip(texts, tasks, fragments)):
            _language_guard(candidate, limit=2000)
            if re.search(r"(?m)^\s*\d+\.\s+", candidate):
                raise ValueError("natural_task_repeats_machine_checklist")
            if "Required checks:" in candidate:
                raise ValueError("natural_task_repeats_machine_checklist")
            for literal in _required_literals(fragment):
                if literal.casefold() not in candidate.casefold():
                    raise ValueError(f"natural_task_missing_literal:{i}:{literal[:60]}")
            allowed = (
                _numbers(task["task_instruction"])
                | _numbers(payload["public_state"]["question"])
            )
            if _numbers(candidate) - allowed:
                raise ValueError(f"natural_task_unsupported_number:{i}")
            task["task_instruction"] = candidate
    okay, issues = validate_action(
        output, payload["gold_guidance"], payload["offline_oracle_answer"],
        payload["answer_contract"], public_state=payload["public_state"],
    )
    if not okay:
        raise ValueError("natural_main_contract:" + ",".join(issues))
    return output


def main_reviewer_prompt(
    payload: Mapping[str, Any], canonical: Mapping[str, Any], natural: Mapping[str, Any],
) -> str:
    return (
        "Audit a natural-language O1 Main delegation against an immutable, fully executable "
        "public contract. Require STRICT semantic equivalence: each entity owner, relation "
        "direction, complete hop chain, filter/comparator, join (AND/OR), quantifier, "
        "and requested output must remain the same. Natural paraphrases, intermediate "
        "step omission, and concise goal-based wording are allowed ONLY when the complete "
        "semantic computation is unambiguously preserved. No invented answer clues, dates, "
        "resource facts or hidden future results. A task may ask for evidence without "
        "asserting it was already obtained. Return JSON ONLY: "
        '{"equivalent":true|false,"issues":["brief reason"]}. '
        "If uncertain, equivalent=false.\n"
        + _json({
            "question": payload["public_state"]["question"],
            "canonical_tasks": [t["task_instruction"] for t in canonical["params"].get("tasks", [])],
            "rewritten_tasks": [t["task_instruction"] for t in natural["params"].get("tasks", [])],
            "completed_subtasks": [
                {"reference": r.get("public_ref"), "answer": r["answer"]}
                for r in payload.get("completed_subtask_contracts", [])
            ],
        })
    )


def feedback_writer_prompt(payload: Mapping[str, Any], canonical: Mapping[str, Any]) -> str:
    fragment = payload["fragment"]
    public = {
        "assigned_task": payload["task_instruction"],
        "exact_result": fragment["expected_answer"],
        "verified_steps": [{
            "operation": op["op"], "arguments": op["inputs"],
            "input_values": item.get("input_values", []), "output": item["answer"],
        } for op, item in zip(fragment["operations"], fragment["operation_results"])],
        "published_observations": [{
            "source_id": o["resource_id"], "title": o["title"],
            "claims": o["claims"],
        } for o in [*fragment["observations"], *fragment.get("known_observations", [])]],
        "verified_reference_report": canonical,
    }
    return (
        "You are an O1 Sub-Agent composing a grounded answer to the ASSIGNED SUBTASK, "
        "not the entire original Main question. The controller has already verified "
        "every operation against the available public evidence and fixed the exact result. "
        "Write fluent, compact report text that could plausibly be returned to the Main Agent. "
        "summary (one or two sentences): start with the finding or the absence of matches. "
        "trace_summary (2-5 brief sentences or bullets): explain the relevant evidence "
        "and why it satisfies the constraints, including necessary relation hops and "
        "source names if supported. Do not narrate individual FILTER_TYPE or INTERSECT "
        "operators, or copy a sequence of generic operator sentences. "
        "DO NOT claim live web searches, clicks, tool executions, transcriptions, or "
        "observations that are absent from the record; you are reporting verified "
        "source-backed findings, not inventing a tool trace. No hidden graph IDs, "
        "new entities, numbers, dates, or unsupported causal conclusions. "
        "Do not change the fixed result. Return JSON ONLY: "
        '{"summary":"...","trace_summary":"..."}.\n' + _json(public)
    )


def realize_feedback_natural(payload: Mapping[str, Any], proposal: Any) -> dict[str, str]:
    if not isinstance(proposal, dict) or set(proposal) != {"summary", "trace_summary"}:
        raise ValueError("invalid_natural_report_fields")
    _language_guard(proposal["summary"], limit=600)
    _language_guard(proposal["trace_summary"], limit=1400)
    joined = proposal["summary"] + " " + proposal["trace_summary"]
    value = payload["fragment"]["expected_answer"]
    if isinstance(value, list) and len(value) <= 3:
        for item in value:
            if isinstance(item, str) and item.casefold() not in joined.casefold():
                raise ValueError("natural_report_omits_result_entity")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if str(value) not in joined:
            raise ValueError("natural_report_omits_numeric_result")
    if re.search(
        r"\b(?:I (?:searched|browsed|clicked|opened|ran|transcribed)|"
        r"(?:web search|browser|python|script|tool call) (?:was |were )?"
        r"(?:executed|performed|run))\b", joined, re.I,
    ):
        raise ValueError("unverified_tool_action_claim")
    okay, issues = validate_feedback(proposal, payload["fragment"], strict=False)
    if not okay:
        raise ValueError("natural_report_unsupported_claim:" + ",".join(issues))
    return {"summary": proposal["summary"], "trace_summary": proposal["trace_summary"]}


def feedback_reviewer_prompt(
    payload: Mapping[str, Any], natural: Mapping[str, str],
) -> str:
    f = payload["fragment"]
    return (
        "Audit this Sub-Agent report strictly against the supplied PUBLIC observations "
        "and verified operations. Check every named entity, date, number, relation "
        "direction, attributed source fact, and final result. Reject invented "
        "facts or claims of tool executions not shown in the record. Natural prose "
        "and concise omission of irrelevant steps are fine; do not require literal "
        "operator names or exact quotation. Return JSON ONLY: "
        '{"grounded":true|false,"issues":["brief reason"]}. '
        "If uncertain, grounded=false.\n"
        + _json({
            "task": payload["task_instruction"],
            "result": f["expected_answer"],
            "operations": [
                {"op": op["op"], "args": op["inputs"], "answer": result["answer"]}
                for op, result in zip(f["operations"], f["operation_results"])
            ],
            "observations": [{
                "resource_id": o["resource_id"], "title": o["title"],
                "claims": o["claims"],
            } for o in [*f["observations"], *f.get("known_observations", [])]],
            "candidate_report": natural,
        })
    )


def check_reviewer(verdict: Any, *, flag: str) -> None:
    if not isinstance(verdict, dict) or verdict.get(flag) is not True or verdict.get("issues") != []:
        raise ValueError("natural_semantic_review_failed:" + str(verdict)[:220])


__all__ = [
    "main_writer_prompt", "realize_main", "main_reviewer_prompt",
    "feedback_writer_prompt", "realize_feedback_natural",
    "feedback_reviewer_prompt", "check_reviewer",
]
