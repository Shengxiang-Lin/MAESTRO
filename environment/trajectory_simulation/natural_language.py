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

# This is the *wire-level* report contract. Bullet points belong inside the
# trace_summary JSON string, separated by "\n", not in a JSON array.
SUBAGENT_REPORT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "minLength": 1},
        "trace_summary": {"type": "string", "minLength": 1},
    },
    "required": ["summary", "trace_summary"],
    "additionalProperties": False,
}


def normalize_feedback_shape(proposal: Any) -> tuple[Any, tuple[str, ...]]:
    """Losslessly coerce a list of bullet strings into one JSON string.

    Only a nonempty list of NONEMPTY strings may be normalized. The output
    passes the exact same deterministic and semantic evidence guards as a
    directly generated string. Never fill a missing answer with templates.
    """
    if not isinstance(proposal, dict):
        return proposal, ()
    result = dict(proposal)
    normalized = []
    steps = result.get("trace_summary")
    if (isinstance(steps, list) and steps
            and all(isinstance(item, str) and item.strip() for item in steps)):
        result["trace_summary"] = "\n".join(
            item.strip() if re.match(r"^(?:[-*]|\d+[.)])\s", item.strip())
            else "- " + item.strip()
            for item in steps
        )
        normalized.append("trace_summary")
    summary = result.get("summary")
    if (isinstance(summary, list) and len(summary) == 1
            and isinstance(summary[0], str) and summary[0].strip()):
        result["summary"] = summary[0].strip()
        normalized.append("summary")
    return result, tuple(normalized)



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
    """Express fixed public source findings in real-O1-style narrative.

    The grouped source packet prevents source/tool attribution drift: each
    witness's exact claims sit beside the only simulated access tools
    permitted for that source. Class names remain optional in prose.
    """
    from .tool_provenance import source_evidence_guide

    fragment = payload["fragment"]
    plan = payload.get("tool_provenance") or {"steps": []}
    public = {
        "assigned_subtask": payload["task_instruction"],
        "fixed_result": fragment["expected_answer"],
        "source_evidence_guide": source_evidence_guide(fragment, plan),
        "verified_reasoning_steps": [{
            "operation": op["op"],
            "arguments": op["inputs"],
            "input_values": item.get("input_values", []),
            "output": item["answer"],
        } for op, item in zip(fragment["operations"], fragment["operation_results"])],
        "simulated_tool_observations": plan,
    }
    return (
        "You are an Orchestra-o1 Sub-Agent writing a report to the Main Agent "
        "on the ASSIGNED SUBTASK only. The exact answer and public source facts "
        "have already been verified; do not alter them. "
        "STRICT JSON OUTPUT CONTRACT: Return exactly one JSON object with "
        "exactly two keys. summary MUST be one nonempty JSON STRING; "
        "trace_summary MUST be one nonempty JSON STRING, NOT A JSON ARRAY. "
        "If the trace has multiple bullet steps, put ALL steps inside that "
        "single string separated by escaped newline characters (\\n). "
        "Never output a list, null, extra keys or markdown fences. "
        "For example: "
        '{"summary":"Alice matches the requested year.",'
        '"trace_summary":"- Inspected the relevant reference.\\n- '
        'Confirmed Alice matches the recorded year."}. '
        "SUMMARY: one or two direct sentences stating the result and the most "
        "relevant clue. Do not grade your performance or the whole Main problem. "
        "TRACE_SUMMARY: write 2-6 concise, connected bullets or short paragraphs "
        "in the style of a real O1 execution recap: what you examined, what "
        "the actual source showed, what you checked next (if applicable), how "
        "the verified facts satisfy the constraint, and what you returned. "
        "Describe tools in NATURAL LANGUAGE where useful: 'examined the clip', "
        "'searched for a supporting reference', 'checked the webpage', "
        "'compared the prior results'. The quoted phrases demonstrate tone "
        "ONLY and do not authorize those actions for this task. "
        "Full tool class names like VideoAnalysisAction or GoogleSearchAction "
        "are OPTIONAL and generally unnecessary; never list class names merely "
        "to prove a tool was used. Likewise do not enumerate every available "
        "source when only one or two provide decisive support. "
        "The source_evidence_guide groups every source's exact supported "
        "claims and allowed access steps. Attribute each fact to its OWN "
        "source; a search discovery event alone never proves a factual claim. "
        "The simulated_tool_observations reconstruct permitted tool activity "
        "from frozen witnesses; they are NOT a record of live tool executions. "
        "Do not claim any action absent from that record. Previously available "
        "evidence is REUSED, not newly fetched. For a pure aggregation task "
        "with no newly observed sources, discuss combining earlier results "
        "without inventing a search or code run. "
        "Do not invent literal search queries, URLs, timestamps, "
        "transcripts, errors, confidence scores, extra sources, new entities, "
        "relations or numbers. Never expose internal observation keys or "
        "claim IDs in the natural report. "
        "Avoid robotic FILTER_TYPE/INTERSECT operator logs, unsupported "
        "tool-action storytelling, and 'no errors/all checks completed' praise. "
        "When an important exact result has multiple entities, name them "
        "concisely; maintain the Gold answer exactly. "
        "Return JSON ONLY: "
        '{"summary":"...","trace_summary":"..."}.\n' + _json(public)
    )

def feedback_repair_prompt(
    payload: Mapping[str, Any], canonical: Mapping[str, Any],
) -> str:
    """A much smaller evidence-complete prompt for malformed/empty JSON retry.

    Only predicates used in the verified operations are retained from each
    source; if there is no predicate lookup (e.g. a pure FIND or aggregation),
    the verified step inputs/outputs carry the exact reasoning contract.
    Fallback claims provide a little context without dumping irrelevant
    entity-wide metadata. No new/private evidence is introduced.
    """
    from .tool_provenance import source_evidence_guide

    fragment = payload["fragment"]
    plan = payload.get("tool_provenance") or {"steps": []}
    predicates = {
        str(op["inputs"][0])
        for op in fragment["operations"]
        if op.get("inputs") and op["op"] in {
            "FILTER_STRING", "FILTER_NUMBER", "FILTER_DATE", "FILTER_YEAR",
            "FILTER_ATTR", "FILTER_VALUE", "RELATE", "QUERY_ATTR",
            "QUERY_RELATION", "VERIFY_ATTR", "VERIFY_RELATION",
        }
    }
    sources = []
    for item in source_evidence_guide(fragment, plan):
        claims = item["claims"]
        relevant = [claim for claim in claims
                    if str(claim.get("predicate")) in predicates]
        # A FIND-only task can have no relation predicates at all; do not
        # send zero context to the writer for a source-backed identification.
        if not predicates:
            relevant = claims[:4]
        sources.append({
            "id": item["source_id"],
            "type": item["source_type"],
            "access": item["source_access"],
            "tools": item["permitted_tools"],
            "facts": [{
                "s": c.get("subject"), "p": c.get("predicate"),
                "o": c.get("object"),
            } for c in relevant],
        })
    public = {
        "assigned_subtask": payload["task_instruction"],
        "fixed_answer": fragment["expected_answer"],
        "relevant_public_source_facts": sources,
        "verified_steps": [{
            "op": op["op"], "args": op["inputs"],
            "inputs": result.get("input_values", []),
            "result": result["answer"],
        } for op, result in zip(
            fragment["operations"], fragment["operation_results"]
        )],
    }
    return (
        "Write the O1 Sub-Agent's brief report for this assigned task. "
        "Your previous JSON response was empty or invalid. "
        "BOTH required fields must be NONEMPTY strings, not null or placeholders. "
        "Use exactly this structure: "
        '{"summary":"A concrete supported finding.",'
        '"trace_summary":"Examined the available evidence, verified the '
        'relevant constraints, and returned the exact supported result."}. '
        "Write the REAL task's findings, not the example's generic wording. "
        "summary: 1-2 factual sentences with the exact answer. "
        "trace_summary: 2-4 short bullets explaining relevant source/tool "
        "inspection if permitted, observed facts, verification and conclusion. "
        "Use natural tool verbs, NOT mandatory class names. "
        "Tools are SIMULATED witness-supported steps, not actual live calls. "
        "If no source inspection is permitted, describe reuse of previous "
        "results instead. Do not invent sources, tool calls, entities, dates, "
        "numbers, search queries, URLs or timestamps. "
        "Only use the following verified public facts; JSON ONLY.\n"
        + _json(public)
    )


def feedback_output_diagnostic(proposal: Any) -> str:
    """Privacy-preserving format diagnostics, safe in the private bank audit."""
    if not isinstance(proposal, dict):
        return "response=" + type(proposal).__name__
    fields = []
    for name in ("summary", "trace_summary"):
        value = proposal.get(name)
        if value is None:
            status = "null" if name in proposal else "missing"
        elif not isinstance(value, str):
            status = type(value).__name__
        elif not value.strip():
            status = "blank"
        else:
            status = "text(" + str(len(value)) + ")"
        fields.append(name + "=" + status)
    extras = set(proposal) - {"summary", "trace_summary"}
    if extras:
        fields.append("extra_keys=" + str(len(extras)))
    return ";".join(fields)


def realize_feedback_natural(payload: Mapping[str, Any], proposal: Any) -> dict[str, str]:
    if not isinstance(proposal, dict) or set(proposal) != {"summary", "trace_summary"}:
        raise ValueError("invalid_natural_report_fields")
    proposal, _ = normalize_feedback_shape(proposal)
    _language_guard(proposal["summary"], limit=600)
    _language_guard(proposal["trace_summary"], limit=1400)
    joined = proposal["summary"] + " " + proposal["trace_summary"]
    if re.search(r"\b(?:observation[ _-]?key|claim[ _-]?id|claim_key)\b",
                 joined, re.I):
        raise ValueError("natural_report_leaks_internal_evidence_handle")
    if re.search(r"\b(?:searched|searching|search)\s+(?:for|using)\s+"
                 r"[\x27\x22\x60][^\x27\x22\x60]+[\x27\x22\x60]",
                 joined, re.I):
        # The frozen source witness has no original literal search query.
        raise ValueError("unverified_literal_search_query")
    value = payload["fragment"]["expected_answer"]
    if isinstance(value, list) and len(value) <= 3:
        for item in value:
            if isinstance(item, str) and item.casefold() not in joined.casefold():
                raise ValueError("natural_report_omits_result_entity")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if str(value) not in joined:
            raise ValueError("natural_report_omits_numeric_result")
    from .tool_provenance import unsupported_tool_actions
    plan = payload.get("tool_provenance") or {"steps": []}
    unsupported = unsupported_tool_actions(joined, plan)
    if unsupported:
        raise ValueError(
            "unverified_tool_action_claim:" + ",".join(sorted(unsupported))
        )
    # Real O1 does not require literal XXXAction tokens in a natural recap.
    # Source/tool attribution and factual claims are still reviewed below.
    okay, issues = validate_feedback(proposal, payload["fragment"], strict=False)
    if not okay:
        raise ValueError("natural_report_unsupported_claim:" + ",".join(issues))
    return {"summary": proposal["summary"], "trace_summary": proposal["trace_summary"]}


def feedback_reviewer_prompt(
    payload: Mapping[str, Any], natural: Mapping[str, str],
) -> str:
    from .tool_provenance import source_evidence_guide

    f = payload["fragment"]
    plan = payload.get("tool_provenance") or {"steps": []}
    return (
        "Review this Sub-Agent natural-language report against the frozen, "
        "PUBLIC and source-replayed facts for the ASSIGNED SUBTASK. "
        "Every named entity, numeric/date value, attributed predicate, relation "
        "direction and asserted observation must be supported by its specific "
        "source or verified reasoning step. Check the exact result. "
        "The provided tool observations are SIMULATED source-acquisition and "
        "inspection affordances, NOT live API logs. Allow equivalent plain "
        "language, e.g. 'examined the video' instead of VideoAnalysisAction, "
        "IF that video source has the corresponding permitted tool activity. "
        "Do NOT demand tool class names, one bullet per tool, or descriptions "
        "of every available reference. A tool-free aggregation of previously "
        "verified results is valid. Do not treat search discovery alone as "
        "factual evidence. Reject actions with no permitted source mapping, "
        "fabricated searches/queries, website URLs, media timestamps or "
        "unsupported facts, even if the final answer is correct. "
        "Treat the verified_operations output values as authoritative "
        "public-replay conclusions. A derived result from FIND, FILTER_TYPE, "
        "COUNT, INTERSECT, UNION, SELECT_BETWEEN or a relation traversal "
        "does NOT need to be separately stated verbatim in a source claim. "
        "In a backward relation traversal, the operation reverses which "
        "endpoint is being discovered; do not demand a source assertion "
        "in the traversal direction if the underlying relation is recorded "
        "in the opposite orientation. Still require every underlying factual "
        "source assertion (such as a specific population/date/relation) to "
        "be supported by the correct reference, and reject facts that "
        "contradict verified operations. A search/discovery event by itself "
        "does not support a factual claim. "
        "Do not require exact wording or literal operator names. "
        "Return JSON ONLY: "
        '{"grounded":true|false,"issues":["brief issue"]}. '
        "Use grounded=true with issues=[] when fully supported, and "
        "grounded=false when any material claim lacks support.\n" + _json({
            "task": payload["task_instruction"],
            "fixed_result": f["expected_answer"],
            "verified_operations": [
                {"op": op["op"], "args": op["inputs"],
                 "input_values": result.get("input_values", []),
                 "answer": result["answer"]}
                for op, result in zip(f["operations"], f["operation_results"])
            ],
            "source_evidence_guide": source_evidence_guide(f, plan),
            "simulated_tool_observations": plan,
            "candidate_report": natural,
        })
    )

def check_reviewer(verdict: Any, *, flag: str) -> None:
    if not isinstance(verdict, dict) or verdict.get(flag) is not True or verdict.get("issues") != []:
        raise ValueError("natural_semantic_review_failed:" + str(verdict)[:220])


__all__ = [
    "natural_public_context", "main_writer_prompt", "main_single_task_prompt",
    "realize_main", "main_reviewer_prompt",
    "feedback_writer_prompt", "feedback_repair_prompt",
    "feedback_output_diagnostic", "SUBAGENT_REPORT_JSON_SCHEMA",
    "normalize_feedback_shape", "realize_feedback_natural",
    "feedback_reviewer_prompt", "check_reviewer",
]
