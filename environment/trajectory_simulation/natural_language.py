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
from .evidence import MARKER


def natural_public_context(context: str, instruction: str) -> str:
    """Preserve source handles but keep exact executable checks out of prose.

    The v3 packet is public metadata, not a private oracle; the independent
    replay must still derive every result from published observations.
    """
    if MARKER not in context:
        raise ValueError("canonical_task_without_public_packet")
    prefix, serialized = context.split(MARKER, 1)
    packet = json.loads(serialized)
    if packet.get("version") != 2:
        raise ValueError("naturalization_requires_v2_public_packet")
    packet = dict(packet, version=3, execution_checks=instruction)
    # Strip the irrelevant instruction-list sentence from human context.
    prefix = prefix.replace(
        "Execute the instructions in dependency order; each input refers only to the result explicitly named in that instruction.",
        "Use the supplied reference materials and previously completed results; report only supported findings.",
        1,
    )
    return prefix + MARKER + _json(packet)


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


def main_single_task_prompt(
    payload: Mapping[str, Any], canonical: Mapping[str, Any], task_index: int,
) -> str:
    """Ask for ONE task to avoid a small LLM dropping parallel siblings.

    Real O1 assignments are short, specific requests: an evidence target,
    a source or tool to inspect, constraints, and what to return. Later rounds
    may refer to discoveries from completed subtasks, but not future results.
    """
    tasks = canonical["params"].get("tasks", [])
    if canonical["action"] != "delegate_task" or not 0 <= task_index < len(tasks):
        raise ValueError("invalid natural Main task index")
    task = tasks[task_index]
    packet = {
        "main_question": payload["public_state"]["question"],
        "task_index": task_index,
        "total_parallel_tasks": len(tasks),
        "required_public_checks": task["task_instruction"],
        "assigned_tools": task["tools"],
        "available_reference_handles": payload.get("visible_resources", []),
        "completed_subtask_findings": [{
            "reference": r.get("public_ref"), "answer": r["answer"],
            "summary": r.get("summary", ""),
        } for r in payload.get("completed_subtask_contracts", [])],
        "context_for_this_task": task["context"].split("\n[Public task packet]\n", 1)[0],
    }
    return (
        "Write ONE natural-language O1 Main-to-Sub-Agent task instruction "
        "for this already-fixed assignment. Output EXACTLY one JSON object "
        'with one string field: {"task_instruction":"..."}; no arrays, '
        "reasoning, context, explanation, or other keys. "
        "Use 1-4 focused sentences (about 40-110 words if needed). "
        "Start with a concrete goal (identify, verify, compare, retrieve, investigate), "
        "name the available source or modality and useful public clues, and finish "
        "with what evidence or value the Sub-Agent should return. "
        "For a later round, use the completed findings to target unresolved or "
        "previously ambiguous details; do not invent failures or new clues. "
        "For independent initial tasks, give each task a distinct, focused goal. "
        "Preserve EVERY necessary hop, relation direction, constraint owner, "
        "quantifier, comparator, exact date, anchor and output type in the "
        "canonical checks, including UNION versus INTERSECTION. "
        "You may express intermediate steps as connected language, but never "
        "drop a necessary condition or add an unsupported fact. "
        "Do not copy numbered operator instructions, mention internal graph "
        "IDs or pretend you inspected sources. A specific public record handle "
        "is not a local file path; do not invent web URLs, media paths or tool calls. "
        "This is a delegated SUBTASK, not a demand to solve the entire Main question. "
        "Use natural O1 phrasing like 'Examine the supplied video for the named "
        "product and return the exact year with evidence' or 'The earlier report "
        "gave two candidate dates; check the announcement source and return "
        "the verified date.' These examples demonstrate *style only*, not task facts. "
        "Return valid JSON ONLY.\n" + _json(packet)
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
            # A rejected single-task rewrite may retain its exact canonical
            # task. This preserves good sibling rewrites in the same round.
            if candidate == task["task_instruction"]:
                continue
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
            task["context"] = natural_public_context(
                task["context"], task["task_instruction"]
            )
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
    *, task_index: int | None = None,
) -> str:
    originals = canonical["params"].get("tasks", [])
    rewrites = natural["params"].get("tasks", [])
    if task_index is not None:
        if not 0 <= task_index < len(originals):
            raise ValueError("invalid_reviewer_task_index")
        originals, rewrites = [originals[task_index]], [rewrites[task_index]]
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
            "canonical_tasks": [t["task_instruction"] for t in originals],
            "rewritten_tasks": [t["task_instruction"] for t in rewrites],
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
        "reconstructed_tool_observations": payload.get("tool_provenance", {
            "origin": "no_tool_observations_available",
            "live_tool_calls": False, "steps": [],
        }),
    }
    return (
        "You are an O1 Sub-Agent composing a grounded answer to the ASSIGNED SUBTASK, "
        "not the entire original Main question. The controller has already verified "
        "every operation against the available public evidence and fixed the exact result. "
        "Write fluent, compact report text that could plausibly be returned to the Main Agent. "
        "summary (one or two sentences): start with the finding or the absence of matches. "
        "Sound like a real Sub-Agent reporting to Main, not a grading rubric or execution log. "
        "Acknowledge missing or ambiguous evidence only if the verified record actually supports it. "
        "trace_summary (3-7 concise bullets when supported): resemble a real O1 "
        "Sub-Agent action/observation report. Tell the chronological story: native "
        "tool type and which resource it inspected, the specific facts observed, "
        "how a second reference or previous subtask corroborated them, the "
        "reasoning/comparison needed, and the exact finding returned. "
        "Reference the appropriate tool NAMES (VideoAnalysisAction, "
        "GoogleSearchAction, ExtractUrlContentAction, ParseAudioAction, "
        "ImageAnalysisAction, ExecuteCodeAction) ONLY if the supplied reconstructed "
        "tool observations actually include those tools. "
        "These are SIMULATED action/observation events reconstructed from frozen "
        "public witnesses, NOT logs of tools run live. Describe what the evidence "
        "supports in natural past-tense execution-report style; DO NOT invent "
        "literal queries, clicks, timestamps, file paths, execution failures, "
        "transcripts, confidence scores or URLs beyond the provided records. "
        "A previously available reference is REUSED, not newly searched. "
        "When there are no tool observations, explain what was deduced from "
        "previously completed results and the verified operations; do not pretend "
        "that a new tool was called. "
        "Avoid self-evaluation ('successfully initiated', 'no errors', "
        "'finished efficiently') and grading the whole Main problem; report "
        "only this assigned subtask. Never list FILTER_TYPE/INTERSECT as robotic "
        "operator logs if plain-language reasoning suffices. No hidden graph IDs, "
        "new entities, numbers, dates, or unsupported causal conclusions. 
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
    from .tool_provenance import permitted_tool_names
    from .tools import tool_profile
    plan = payload.get("tool_provenance") or {"steps": []}
    allowed_tools = permitted_tool_names(plan)
    known_tools = set(tool_profile(payload["fragment"].get("scene", "omnigaia")).available)
    explicitly_mentioned = {tool for tool in known_tools
                            if re.search(r"(?<!\w)" + re.escape(tool) + r"(?!\w)", joined)}
    if explicitly_mentioned - allowed_tools:
        raise ValueError("unverified_tool_action_claim:" +
                         ",".join(sorted(explicitly_mentioned - allowed_tools)))
    # Refuse action claims without a corresponding simulated observation.
    if re.search(r"\b(?:I (?:browsed|clicked|opened|ran|transcribed)|"
                 r"(?:browser|python|script|tool call) (?:was |were )?"
                 r"(?:executed|performed|run))\b", joined, re.I):
        raise ValueError("unverified_tool_action_claim")
    if re.search(r"\b(?:I searched|web search (?:was |were )?(?:executed|performed|run))\b",
                 joined, re.I) and "GoogleSearchAction" not in allowed_tools:
        raise ValueError("unverified_search_action_claim")
    if ("trace_summary" in proposal and allowed_tools
            and not any(name in proposal["trace_summary"] for name in allowed_tools)):
        raise ValueError("tool_trace_does_not_reflect_supported_tool_use")
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
        "facts or claims of tool actions absent from the reconstructed witness-backed "
        "tool observations. The tool steps are simulated from frozen PUBLIC sources, "
        "not live API/browser logs; allow accurate O1-style action/observation "
        "narrative for these enumerated tools only. A source discovery tool alone "
        "does not prove a fact; attribute factual claims to source observations. "
        "Reject invented calls, search phrases, video timestamps, errors, or "
        "unsupported attribution. Natural prose and concise omission of irrelevant "
        "steps are fine; do not require literal 
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
            "reconstructed_tool_observations": payload.get("tool_provenance", {
                "origin": "no_tool_observations_available",
                "live_tool_calls": False, "steps": [],
            }),
        })
    )


def check_reviewer(verdict: Any, *, flag: str) -> None:
    if not isinstance(verdict, dict) or verdict.get(flag) is not True or verdict.get("issues") != []:
        raise ValueError("natural_semantic_review_failed:" + str(verdict)[:220])


__all__ = [
    "natural_public_context", "main_writer_prompt", "main_single_task_prompt",
    "realize_main", "main_reviewer_prompt",
    "feedback_writer_prompt", "realize_feedback_natural",
    "feedback_reviewer_prompt", "check_reviewer",
]
