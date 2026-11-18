"""Main-Agent action surface authoring and independent guarded review.

Teacher determines which Macro slots are legal; the Author writes only
O1-compatible language. The Author never receives current/future results.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy

from environment.generate_question.execution.evidence_tools import (
    legacy_tools_from_task, registered_o1_tool_names,
    tools_from_primitive_steps,
)
from .generation_meta import GENERATION_PROCESS_RE

_AUTHOR_SYSTEM = """Write one realistic Orchestra-o1 Main-Agent decision.
The action MUST delegate exactly the supplied independent tasks in this round,
using action='delegate_task', params.tasks with task_instruction, context, model,
and tools. Only write reasoning, task_instruction and context prose.
The model/tools fields are IMMUTABLE teacher-owned fields, not your choices.
The deterministic compiler will supply them to the final O1 action.
This is an offline evidence simulator: workspace:// is a source handle,
not a physical file, and GoogleSearchAction/ExtractUrlContentAction are
valid simulated access tools for synthetic text/table evidence.
Never change the number or order of assigned tasks. Do not write Macro IDs.
Reflect previous Sub-Agent findings accurately. Include concrete identifiers
from *already returned* results when they are needed for keyed lookups.
Do not mention Oracle, Teacher, Gold DAG, synthetic, or hidden future evidence.
You are describing tasks in a controlled source workspace; do not claim real
web browsing or media decoding has already happened. Never fabricate a result.
Use JSON only with action, reasoning, params. The reasoning should explain
why these particular tasks can run together now and what remains unresolved.
"""

_CRITIC_SYSTEM = """Independently assess ONLY THE GENERATED PROSE for
an Orchestra-o1 Main decision in an OFFLINE simulated evidence environment.

All action structure, task count/order, model routing, source permissions and
O1 tool names have already been validated by deterministic code. They are
FROZEN: You MUST NOT evaluate, modify, or reject tools, model, cardinality,
tool necessity, or extra hidden source_ids fields. Your candidate payload
intentionally OMITS them. A task that only synthesizes previously returned
facts needs no tools and is fully legal.

Environment semantics: workspace:// tokens are SIMULATED source handles,
NOT physical files that the agent must open. GoogleSearchAction and
ExtractUrlContentAction can denote simulated search/extract evidence access
in the teacher trace, including reserved *.example.invalid URL fixtures.
No actual network access is performed. You MUST NOT reject a task merely
because a synthetic source is not an actual website or because it uses
GoogleSearchAction on simulated text/table observations. All tool
feasibility has ALREADY been fixed by the source-access contract.

Review only: Is the instruction specific, natural and semantically aligned
with the provided assignment? Does the context accurately reuse previously
returned Sub-Agent findings, without inventing facts from the future?
Does the reasoning respect the current independence/dependency situation?
An internal Macro ID or fabricated result is an error. Do not demand
identifiers which are private metadata rather than part of O1 output.
Return JSON only:
{"verdict":"pass"|"fail","grounded":true|false,
"assignment_fidelity":true|false,"o1_style":true|false,
"reason":"short reason","repair_hint":"specific prose-only change"}
"""


# V1.4 trusted orchestration fixes each dependency-safe Main dispatch.
# The Author writes grounded reasoning ONLY, not task count or tool routing.
STAGE3_REASONING_AUTHOR_SYSTEM = """You are the Main-Agent deciding the next
investigation from the user question and the Sub-Agent feedback ACTUALLY
returned so far. Output JSON with ONLY {"reasoning":"..."}.

The assigned investigations are supplied as context. Their number, order,
instructions, source permissions, tools and model are fixed. Explain WHY
these tasks are appropriate at this round and what remains to be checked;
do not author an action, task list or new assignment. Use ONLY evidence
visible in prior Sub-Agent results. Never assert an earlier retrieval or
Sub-Agent failed unless the provided history shows that failure. A completed
intermediate lookup is NOT a failure just because downstream work remains.
Contradictory independent source documents do NOT imply a malfunctioning
Sub-Agent. Do not claim to have executed pending verification or seen
future results. Do not mention internal reference actions or model-review
machinery. Respond with the reasoning field only.
"""

STAGE3_REASONING_CRITIC_SYSTEM = _CRITIC_SYSTEM + """
The Stage-3 model authored ONLY reasoning; task instructions, count, order,
tools and model are fixed by the executable plan. Review factual claims in
reasoning against actually Main-visible Sub-Agent results and the supplied
CURRENT-ROUND tasks. Reject invented earlier errors/retries, completed checks
or premature findings.

IMPORTANT MACRO SCOPE: The overall user question is a MULTI-STEP goal.
The fixed current-round tasks may perform just one intermediate key lookup,
even when the eventual answer requires comparing two later sources. An index
lookup that returns the record key for a dependent index is legitimate:
DO NOT demand that the Main skip this dependency and retrieve the final
metric tables now. Validate the author's explanation against the exact
current_round_task_contract and what has already been returned, NOT against
a proposed alternative plan or immediate completion of the entire question.
A valid criticism must identify a concrete ungrounded claim in the authored
reasoning. Never object merely that a later task remains necessary.

If current_round_evidence_contract.observed_identifier_set_relations is
nonempty, its literal overlap was calculated deterministically from two
ALREADY RETURNED public Result lists. The correct set intersection is
exactly shared_identifiers. Do not hallucinate membership in either list.
This proves only the textual overlap, NOT that the source claims are true,
nor that a downstream archival answer has already been obtained. If a
reasoning draft disputes this overlap, reject it; if the reasoning uses
the overlap faithfully, do not invent a mathematical objection.
Do not ask for changes to task prose or frozen dispatch. Return the normal
three axes.
"""

STAGE3_TASK_PROSE_AUTHOR_SYSTEM = """You are an Orchestra-o1 Main-Agent
writing the next investigation decision and its Sub-Agent instructions.

The upcoming investigations are listed in the provided order. Write
exactly one task_instruction and context per listed investigation.
Preserve the task count, order, allowed sources, output fields and models;
do not add, skip or substitute tasks or select new tools.

Return EXACTLY:
{"reasoning":"...",
 "tasks":[{"task_instruction":"...","context":"..."}]}

For each task_instruction, give an actionable request: which public source
to inspect, which specific clue/identifier/date/relationship from the
question or earlier Sub-Agent feedback to use, and which values to return.
Mention each permitted source ID literally and retain its public name.
Parallel investigations need distinct source-specific instructions, not
a shared generic sentence. Do not copy reference instructions verbatim.
Do not write tool names, models or internal task identifiers.

For each context, write ONE natural handoff of 25–700 characters: explain
why the investigation is needed, using available user clues or ALREADY
returned Sub-Agent results. On the first round, do not imply a previous
Sub-Agent retrieval. Previously returned result strings are attached
separately after generation; there is no need to copy them in full.
Do not invent source content, outcomes, errors, retries or future facts.

For reasoning, speak as the investigator: explain the evidence currently
available, why these particular inquiries can proceed now, and what they
will help establish. Do not describe the internal construction or
scheduling process, procedural machinery, or why software chose a step.
An earlier successful intermediate lookup is not a failed task.
Only claim facts visible in the original question and earlier feedback.
Return JSON only, with reasoning and tasks, no extra fields.
"""

STAGE3_TASK_PROSE_CRITIC_SYSTEM = _CRITIC_SYSTEM + """
This is Stage-3 trusted, fixed-dispatch natural-language authoring.
The number/order of slots, model, tool routing, source permissions
and required output fields are code-verified and IMMUTABLE.
Assess the authored reasoning AND EACH task_instruction/context:
Does each assignment explain the actual local evidence question and
requested fields, name only permitted sources, correctly use existing
Sub-Agent feedback, and avoid pretending later dependencies or checks
have already completed? Reject generic boilerplate that could apply
to any source, wrong entity-to-field attribution, invented source
content, fake earlier errors, or future task results.
A short verbatim earlier public result may be appended by the code for
evidence fidelity; this is not model-generated hallucination.
Do not change or criticize frozen dispatch, tools, model or slot order.
In particular, an intermediate key lookup is valid even if the overall
question eventually requires later downstream evidence.
Return the normal JSON verdict and boolean review axes.
"""


# A narrow check for a claim of a previous failed tool/Sub-Agent execution
# in trusted history consisting entirely of completed Sub-Agent reports.
_PAST_FAILED_ATTEMPT = re.compile(
    r"\b(?:previous|prior|earlier|last|initial|upstream)\s+"
    r"(?:\w+\s+){0,4}"
    r"(?:search|lookup|retrieval|sub[ -]?agent|task|attempt|"
    r"investigation|result|step)\s+"
    r"(?:\w+\s+){0,5}"
    r"(?:failed|errored|timed\s+out|was\s+unsuccessful|"
    r"returned\s+an?\s+error)\b"
    r"|\b(?:retry|re-?run|repeat)\s+"
    r"(?:\w+\s+){0,6}"
    r"(?:because|due\s+to|since)\s+"
    r"(?:\w+\s+){0,5}"
    r"(?:failed|error|unsuccessful|timed\s+out)\b",
    re.I,
)


# These are implementation terms, not Main-visible reasons for a decision.
# Keep the pattern narrow: "fixed deadline", "dependency" and genuine graph
# evidence in a user question are valid domain concepts.
_STAGE3_ORCHESTRATION_META = re.compile(
    r"\b(?:fixed|frozen|locked|pre[- ]?assigned|pre[- ]?determined)\s+"
    r"(?:(?:task|dependency|macro|investigation|execution)\s+){0,2}"
    r"(?:graph|batch|slots?|dispatch|schedule|workflow|plan|steps?)\b"
    r"|\b(?:current|next|assigned)\s+(?:fixed\s+)?(?:task\s+)?slot\b"
    r"|\b(?:the\s+)?(?:task|dependency)\s+graph\s+"
    r"(?:requires|mandates|dictates|specifies)\b"
    r"|\b(?:teacher[- ]owned|code[- ]owned)\s+"
    r"(?:tasks?|steps?|plans?|batch|dispatch)\b",
    re.I,
)


def stage3_reasoning_issues(reasoning: str, delivered: dict) -> list[str]:
    """Check Main-visible decisions, not the private generation machinery."""
    if not isinstance(reasoning, str):
        return ["invalid_main_reasoning"]
    issues = []
    if re.search(
        r"\b(?:reference_action_for_structure_only|"
        r"current_round_assignments_in_required_order)\b",
        reasoning, re.I,
    ):
        issues.append("internal_reference_action_leak")
    if _STAGE3_ORCHESTRATION_META.search(reasoning):
        issues.append("internal_orchestration_language")
    if _PAST_FAILED_ATTEMPT.search(reasoning):
        statuses = [
            record.get("finish_result", {}).get("status")
            for record in delivered.values()
        ]
        if not statuses or all(status == "done" for status in statuses):
            issues.append("invented_prior_subagent_failure")
    return issues


_FORBIDDEN = re.compile(
    r"\b(?:work_[A-Za-z0-9_]+|oracle|gold\s+(?:program|dag|answer)|"
    r"teacher\s+(?:task|answer|state)|macro\s*(?:task|id)|"
    r"synthetic|simulator)\b", re.I,
)


# Shared with the independent Stage-4 publication guard.
_GENERATION_META = GENERATION_PROCESS_RE


def _critic_review_passes(review: dict) -> bool:
    return (
        isinstance(review, dict)
        and review.get("verdict") == "pass"
        and all(review.get(key) is True for key in (
            "grounded", "assignment_fidelity", "o1_style",
        ))
    )


def _review_only_frozen_tool_objection(review: dict) -> bool:
    """An invalid tool critique is not a reason to reject reviewed prose.

    Re-review from a tool-neutral view; do not auto-accept the action.
    """
    if not isinstance(review, dict) or review.get("verdict") != "fail":
        return False
    reason = str(review.get("reason", "")) + " " + str(
        review.get("repair_hint", "")
    )
    return (
        bool(re.search(
            r"GoogleSearchAction|ExtractUrlContentAction|ExecuteCodeAction|"
            r"tool\s+(?:choice|selection|feasibility|availability)|"
            r"workspace://|simulated\s+source\s+tools", reason, re.I,
        ))
        and bool(re.search(
            r"(?:inappropriate|invalid|should\s+(?:not|instead)|"
            r"replace|cannot|violat|not\s+applicable|not\s+allowed|"
            r"wrong\s+tool)", reason, re.I,
        ))
    )


def _tool_neutral_review_payload(prompt: str, assignments: list[dict],
                                 candidate: dict) -> dict:
    # Remove tool-routing instructions only for the *second* reviewer
    # perspective. The published Action itself is not modified.
    question = prompt.split("==== QUESTION ====", 1)[-1].split(
        "====", 1
    )[0].strip()
    return {
        "current_user_question": question[:1800],
        "assignment_prose_goals_in_required_order": [{
            "expected_output_fields": a["output_fields"],
            "needs_upstream": a["needs_upstream"],
            "expected_context": a["context"],
        } for a in assignments],
        "candidate_prose_only": {
            "reasoning": candidate["reasoning"],
            "tasks": [{
                "task_instruction": re.sub(
                    r"(?:GoogleSearchAction|ExtractUrlContentAction|"
                    r"ExecuteCodeAction)", "[authorized tool]",
                    task["task_instruction"],
                ),
                "context": task["context"],
            } for task in candidate["params"]["tasks"]],
        },
        "fixed_tool_feasibility_already_checked": True,
        "review_rule": (
            "Review semantic task goal, accuracy, and visible history only. "
            "Tool choice and simulated source accessibility are outside scope."
        ),
    }


def _short(value, limit=500) -> str:
    text = str(value).strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def _prior_context(
    target: dict, delivered: dict[str, dict], *, max_chars: int = 3200,
) -> str:
    parents = target["depends_on"]
    if not parents:
        return ("Original user question and the named workspace source "
                "descriptions; this assignment requires no earlier result.")
    fragments = []
    for parent in parents:
        if parent not in delivered:
            raise ValueError("stage3_undelivered_parent_for_context")
        item = delivered[parent]
        result = item["finish_result"]
        fragments.append(
            "Previously returned result: " + _short(result["result"], 1400)
            + "; Sub-Agent summary: " + _short(result["summary"], 280)
        )
    return _short(" ".join(fragments), max_chars)


def _assignment(target: dict, catalog: dict[str, dict],
                delivered: dict[str, dict], *,
                model_alias: str = "model_1") -> dict:
    source_ids = list(target["source_ids"])
    output = ", ".join(target["outputs"]["typed_relation"])
    sources = [
        f"{sid} ({catalog[sid]['title']}; "
        f"{catalog[sid]['modality_hint']}"
        + (f"; known URL: {catalog[sid]['resource_url']}"
           if catalog[sid].get("resource_url") else "")
        + ")"
        for sid in source_ids
    ]
    source_intro = (
        "Examine the assigned workspace source(s): " + "; ".join(sources)
        if sources else
        "Cross-reference the previously returned Sub-Agent findings"
    )
    has_keyed = any(
        step["op"] == "lookup_by_key"
        for step in target["operator_chain"]
    )
    if has_keyed:
        action = (
            "Use the record key provided by an earlier result to retrieve "
            "only the matching observation. Do not read an entire keyed table."
        )
    elif source_ids:
        action = (
            "Identify the candidates meeting the evidence described in "
            "the user question. Preserve their original identifiers."
        )
    else:
        action = (
            "Combine the returned records by their matching identifiers; "
            "do not invent missing evidence or apply future-stage filters."
        )
    workflows = []
    for step in target["operator_chain"]:
        contract = step.get("evidence_acquisition") or {}
        strategy = contract.get("strategy")
        if strategy == "search_then_extract":
            workflows.append(
                "First use GoogleSearchAction to locate a relevant page and "
                "obtain its URL; then use ExtractUrlContentAction with that "
                "returned URL and an evidence-specific browse_query. Do not "
                "assume the URL before searching."
            )
        elif strategy == "direct_extract":
            workflows.append(
                "Use ExtractUrlContentAction to inspect the supplied known "
                "source URL with a targeted browse_query."
            )
        elif strategy == "search_only":
            workflows.append(
                "Use GoogleSearchAction to retrieve supporting search "
                "snippets; a separate page extraction is not required."
            )
    instruction = (
        f"{source_intro}. {action} "
        + " ".join(dict.fromkeys(workflows))
        + f" Return {output} and briefly explain "
          "which available observations support the finding."
    )
    if "required_tools" in target:
        tools = list(target["required_tools"])
        if tools != tools_from_primitive_steps(target["operator_chain"]):
            raise ValueError("stage3_macro_acquisition_tool_drift")
    else:
        if any("evidence_acquisition" in step
               for step in target["operator_chain"]):
            raise ValueError("stage3_frozen_acquisition_tools_missing")
        # Pre-contract Scene snapshots remain usable, but should be
        # regenerated at Stage 1 for fully audited tool requirements.
        tools = legacy_tools_from_task(target, catalog)
    if any(name not in registered_o1_tool_names() for name in tools):
        raise ValueError("stage3_unknown_o1_tool_in_assignment")
    return {
        "target_instruction": instruction,
        "context": _prior_context(target, delivered),
        "model": model_alias,
        "tools": tools,
        "source_ids": source_ids,
        "public_source_descriptions": [{
            "source_id": sid,
            "title": catalog[sid]["title"],
            "modality": catalog[sid]["modality_hint"],
        } for sid in source_ids],
        "output_fields": list(target["outputs"]["typed_relation"]),
        "needs_upstream": bool(target["depends_on"]),
        "keyed_lookup": has_keyed,
    }


_STAGE3_IDENTIFIER = re.compile(
    r"\b(?:Case-\d+|Dossier-[A-Za-z0-9]+|PRJ-[A-Za-z0-9]+|"
    r"Entity-[A-Za-z0-9]+|MK-\d+|RK-\d+|Entry-\d+)\b", re.I,
)


def _stage3_task_prose_issues(
    draft: dict, assignments: list[dict], *, prompt: str,
) -> list[str]:
    """Fail closed on authored task prose before the fixed O1 slots are built.

    Source identities and task count are code-owned. Author can phrase a
    request, never fabricate a new lookup, upstream key, or output schema.
    The independent Critic separately reviews semantic correctness.
    """
    if not isinstance(draft, dict) or set(draft) != {"reasoning", "tasks"}:
        return ["stage3_task_prose_shape"]
    tasks = draft.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(assignments):
        return ["stage3_task_prose_slot_count"]
    allowed_identifiers = {
        token.casefold()
        for token in _STAGE3_IDENTIFIER.findall(
            prompt + "\n" + "\n".join(
                item["target_instruction"] + "\n" + item["context"]
                for item in assignments
            )
        )
    }
    issues = []
    for i, (item, expected) in enumerate(zip(tasks, assignments)):
        if not isinstance(item, dict) or set(item) != {
            "task_instruction", "context",
        }:
            issues.append(f"task_{i}_prose_shape")
            continue
        instruction = item["task_instruction"]
        context = item["context"]
        if not isinstance(instruction, str) or not 50 <= len(
            instruction.strip()
        ) <= 1200:
            issues.append(f"task_{i}_authored_instruction_invalid")
            continue
        if not isinstance(context, str):
            issues.append(f"task_{i}_authored_context_invalid")
            continue
        if not 25 <= len(context.strip()) <= 700:
            # Still check the instruction for other independent errors
            # (e.g. generic templates): a single revision should repair
            # both issues instead of burning two model retries.
            issues.append(f"task_{i}_authored_context_invalid")
        if instruction.strip() == expected["target_instruction"].strip():
            issues.append(f"task_{i}_instruction_not_naturalized")
        if re.search(
            r"\b(?:identify the candidates meeting the evidence described "
            r"in the user question|combine the returned records by their "
            r"matching identifiers)\b", instruction, re.I,
        ):
            issues.append(f"task_{i}_generic_template_repeated")
        for sid in expected["source_ids"]:
            if not re.search(
                r"(?<![\w])" + re.escape(sid) + r"(?![\w])",
                instruction + " " + context,
                re.I,
            ):
                issues.append(f"task_{i}_source_scope_missing:{sid}")
        for identifier in _STAGE3_IDENTIFIER.findall(
            instruction + " " + context
        ):
            if identifier.casefold() not in allowed_identifiers:
                issues.append(f"task_{i}_invented_visible_identifier")
                break
        if _FORBIDDEN.search(instruction + " " + context):
            issues.append(f"task_{i}_private_contract_leak")
        if _GENERATION_META.search(instruction + " " + context):
            issues.append(f"task_{i}_generation_process_leak")
    return sorted(set(issues))


def _stage3_task_prose_repair_guidance(
    failures: list[dict], assignments: list[dict],
) -> list[str]:
    """Turn a rejected public-language draft into actionable, scoped feedback.

    Only task indices and CURRENT-ROUND public contracts appear here. No
    private DAG IDs, future results or verbatim Critic dialogue are exposed.
    """
    guidance = []
    for failure in failures[-2:]:
        if failure.get("stage") != "stage3_task_prose_guard":
            continue
        for issue in failure.get("errors", []):
            match = re.fullmatch(r"task_(\d+)_(.+)", issue)
            if not match:
                continue
            index = int(match.group(1))
            kind = match.group(2)
            if not 0 <= index < len(assignments):
                continue
            task = assignments[index]
            label = f"Task {index + 1}: "
            if kind == "authored_context_invalid":
                guidance.append(
                    label + "Return context as a string of 25–700 "
                    "characters. Write one sentence explaining the "
                    "specific clue/source and WHY this task follows; "
                    "do not paste the full earlier result because the "
                    "application appends the evidence separately."
                )
            elif kind in {"generic_template_repeated",
                          "instruction_not_naturalized"}:
                names = ", ".join(
                    s.get("title", s["source_id"])
                    for s in task.get("public_source_descriptions", [])
                )
                sources = ", ".join(task["source_ids"]) or (
                    "previously returned evidence"
                )
                fields = ", ".join(task["output_fields"])
                guidance.append(
                    label + "Rewrite the instruction as a concrete, "
                    "distinct investigation, not a generic template. "
                    f"Name current permitted source IDs ({sources}), "
                    + (f"public source titles ({names}), " if names else "")
                    + f"and specifically request fields ({fields}). "
                    "Use only already provided clues, never future facts."
                )
    return list(dict.fromkeys(guidance))


def _stage3_compile_task_prose(
    draft: dict, assignments: list[dict], template: dict,
) -> dict:
    """Keep exact fixed slots; edit only publicly visible task prose.

    The source/field contract is appended by code only when necessary.
    The original already-visible parent Result is attached verbatim to
    dependent contexts, avoiding accidental information loss in paraphrase.
    """
    tasks = deepcopy(template["params"]["tasks"])
    for i, (item, expected) in enumerate(zip(draft["tasks"], assignments)):
        instruction = item["task_instruction"].strip()
        missing = [
            field.replace("_", " ")
            for field in expected["output_fields"]
            if field.casefold() not in instruction.casefold()
            and field.replace("_", " ").casefold() not in instruction.casefold()
        ]
        if missing:
            instruction = instruction.rstrip().rstrip(".") + (
                ". Report " + ", ".join(missing) + "."
            )
        context = item["context"].strip()
        if expected["needs_upstream"]:
            evidence = expected["context"]
            # _prior_context has a 3200-character bound; the O1 action
            # contract permits 3500 characters in total.
            available = 3500 - len(evidence) - 2
            context = context[:available].rstrip() + "\n\n" + evidence
        tasks[i]["task_instruction"] = instruction
        tasks[i]["context"] = context
    return {
        "action": "delegate_task",
        "reasoning": draft["reasoning"],
        "params": {"tasks": tasks},
    }


def proposed_template_action(assignments: list[dict]) -> dict:
    if not assignments:
        raise ValueError("stage3_no_assignments")
    if len(assignments) == 1:
        reason = (
            "The next investigation depends on the currently available "
            "evidence, so I will delegate this prerequisite before "
            "planning the following step."
        )
    else:
        reason = (
            "These investigations are independent at this point. I will "
            "delegate them together, then use the returned findings to "
            "plan the dependent follow-up work."
        )
    return {
        "action": "delegate_task",
        "reasoning": reason,
        "params": {"tasks": [{
            "task_instruction": item["target_instruction"],
            "context": item["context"],
            "model": item["model"],
            "tools": item["tools"],
        } for item in assignments]},
    }


def action_issues(action: dict, assignments: list[dict],
                  *, prompt: str) -> list[str]:
    errors = []
    if not isinstance(action, dict) or action.get("action") != "delegate_task":
        return ["expected_delegate_task_action"]
    reasoning = action.get("reasoning")
    if not isinstance(reasoning, str) or not 25 <= len(reasoning) <= 1600:
        errors.append("invalid_main_reasoning")
    params = action.get("params")
    tasks = params.get("tasks") if isinstance(params, dict) else None
    if not isinstance(tasks, list) or len(tasks) != len(assignments):
        return errors + ["wrong_parallel_batch_cardinality"]
    all_action_text = json.dumps(action, ensure_ascii=False)
    if _FORBIDDEN.search(all_action_text):
        errors.append("internal_teacher_language")
    # The machine-readable action label is legitimately "delegate_task".
    # Scan natural-language fields ONLY: internal planning vocabulary in
    # Main prose is forbidden, but the O1 action schema is not prose.
    prose = [reasoning] if isinstance(reasoning, str) else []
    prose.extend(
        value for task in tasks if isinstance(task, dict)
        for value in (task.get("task_instruction"), task.get("context"))
        if isinstance(value, str)
    )
    if any(_GENERATION_META.search(text) for text in prose):
        errors.append("generation_process_leak")
    for i, (task, expected) in enumerate(zip(tasks, assignments)):
        if not isinstance(task, dict):
            errors.append(f"task_{i}_not_an_object")
            continue
        instruction = task.get("task_instruction")
        context = task.get("context")
        model = task.get("model")
        tools = task.get("tools")
        if not isinstance(instruction, str) or not 35 <= len(instruction) <= 1500:
            errors.append(f"task_{i}_invalid_instruction")
        if not isinstance(context, str) or not 10 <= len(context) <= 3500:
            errors.append(f"task_{i}_invalid_context")
        if model != expected["model"]:
            errors.append(f"task_{i}_model_mismatch")
        if tools != expected["tools"]:
            errors.append(f"task_{i}_tool_scope_mismatch")
        if isinstance(instruction, str) and expected["source_ids"]:
            if not all(sid in instruction + " " + str(context)
                       for sid in expected["source_ids"]):
                errors.append(f"task_{i}_source_identifiers_missing")
        if expected["needs_upstream"] and isinstance(context, str):
            # Include at least one real upstream identity when available;
            # generic references to "earlier findings" are insufficient.
            identifiers = set(re.findall(
                r"(?:Case-\d+|MK-\d+|RK-\d+|Entry-\d+)",
                expected["context"],
            ))
            if identifiers and not any(
                token in context for token in identifiers
            ):
                errors.append(f"task_{i}_upstream_identifier_missing")
            elif not identifiers and "Previously returned result:" not in context:
                errors.append(f"task_{i}_upstream_result_not_transmitted")
    return errors


def completion_issues(action: dict, *, expected_answer: str,
                      delivered: dict) -> list[str]:
    """Guard teacher-verified completion against unsupported prose.

    The final answer stays frozen by the trusted Scene. Only preceding
    public Finish Results can justify the completion rationale.
    """
    if not isinstance(action, dict) or action.get("action") != "complete":
        return ["invalid_completion_action"]
    if action.get("params") != {"answer": expected_answer}:
        return ["completion_answer_not_teacher_verified"]
    reasoning = action.get("reasoning")
    if not isinstance(reasoning, str) or not 45 <= len(reasoning) <= 1600:
        return ["invalid_completion_reasoning"]
    issues = []
    if _FORBIDDEN.search(reasoning) or _GENERATION_META.search(reasoning):
        issues.append("completion_generation_or_private_leak")
    visible_result_quotes = [
        row["finish_result"]["result"].strip().replace("\n", " ")[:360].rstrip()
        for row in delivered.values()
    ]
    if not any(quote and quote in reasoning
               for quote in visible_result_quotes):
        issues.append("completion_not_grounded_in_visible_result")
    return issues


_COMPLETE_AUTHOR_SYSTEM = """Write one concise, naturally phrased
Orchestra-o1 Main-Agent COMPLETE rationale. The final answer is already
teacher-verified and must NOT be changed. Only use earlier public Sub-Agent
Finish Results; never mention private Oracle, revision feedback, a Critic,
or future observations. Explain why those results are sufficient to answer.
Return JSON: {"reasoning": "one or two evidence-grounded sentences"}.
"""


_COMPLETE_CRITIC_SYSTEM = """Review ONLY the final Main-Agent completion
reasoning for factual support and O1 style. The final answer and all earlier
public Sub-Agent results have been independently verified. Do NOT review
tools, source-access routes, model choices, or hidden Macro metadata.
Reject invented evidence, generator revisions or unsupported conclusions.
Return JSON only: {"verdict":"pass"|"fail","grounded":true|false,
"assignment_fidelity":true|false,"o1_style":true|false,
"reason":"brief reason","repair_hint":"concrete prose-only change"}.
"""


def author_complete_action(*, template: dict, prompt: str,
                           delivered: dict, author=None, critic=None,
                           max_revisions: int = 2) -> tuple[dict, dict]:
    """Naturalize a grounded completion; never weaken the frozen answer.

    An explicitly labeled deterministic evidence-quoting completion is a
    safe fallback if the language author/reviewer cannot agree. The fallback
    is NOT represented as independent LLM-reviewed language.
    """
    correct_answer = template["params"]["answer"]
    template_errors = completion_issues(
        template, expected_answer=correct_answer, delivered=delivered,
    )
    if template_errors:
        raise ValueError("invalid_grounded_complete_template:" +
                         ",".join(template_errors))
    if author is None:
        return deepcopy(template), {
            "status": "accepted", "mode": "teacher_verified_completion",
            "attempts": 0,
        }
    if critic is None:
        raise ValueError("stage3_completion_author_requires_critic")
    public_evidence = [{
        "result": report["finish_result"]["result"],
        "summary": report["finish_result"]["summary"],
    } for report in delivered.values()]
    failures = []
    for attempt in range(max_revisions + 1):
        request = {
            "current_main_visible_prompt": prompt,
            "public_evidence_from_previously_completed_subtasks":
                public_evidence,
            "reference_action_for_structure_only": template,
            "previous_prose_feedback": failures[-2:],
        }
        try:
            proposed = author.complete_json(
                _COMPLETE_AUTHOR_SYSTEM,
                json.dumps(request, ensure_ascii=False),
            )
        except Exception as exc:
            failures.append({"stage": "completion_author",
                             "reason": str(exc)[:200]})
            continue
        if (not isinstance(proposed, dict)
                or proposed.get("action", "complete") != "complete"
                or not isinstance(proposed.get("reasoning"), str)):
            failures.append({"stage": "completion_shape"})
            continue
        authored = proposed["reasoning"].strip()
        if _GENERATION_META.search(authored) or _FORBIDDEN.search(authored):
            failures.append({"stage": "completion_guard",
                             "errors": ["generation_process_leak"]})
            continue
        # Always attach the exact prior public result quotation so that a
        # paraphrase alone cannot invent completion evidence.
        if authored == template["reasoning"]:
            full_reason = authored
        else:
            full_reason = authored[:420].rstrip() + " " + template["reasoning"]
        candidate = {
            "action": "complete", "reasoning": full_reason,
            "params": {"answer": correct_answer},
        }
        errors = completion_issues(
            candidate, expected_answer=correct_answer, delivered=delivered,
        )
        if errors:
            failures.append({"stage": "completion_guard", "errors": errors})
            continue
        try:
            review = critic.complete_json(
                _COMPLETE_CRITIC_SYSTEM,
                json.dumps({
                    "current_main_visible_prompt": prompt,
                    "public_evidence_from_previously_completed_subtasks":
                        public_evidence,
                    "candidate_prose_only": {
                        "reasoning": candidate["reasoning"],
                        "tasks": [],
                    },
                    "assignment_prose_goals_in_required_order": [],
                    "fixed_answer_verified_by_code": True,
                }, ensure_ascii=False),
            )
        except Exception as exc:
            failures.append({"stage": "completion_critic_backend",
                             "reason": str(exc)[:200]})
            continue
        if _critic_review_passes(review):
            return candidate, {
                "status": "accepted", "mode": "llm_reviewed_completion",
                "attempts": attempt + 1, "review": review,
            }
        failures.append({
            "stage": "completion_critic",
            "reason": str(review.get("reason", ""))[:250]
                      if isinstance(review, dict) else "invalid_review",
            "repair_hint": str(review.get("repair_hint", ""))[:250]
                           if isinstance(review, dict) else "",
        })
    return deepcopy(template), {
        "status": "accepted", "mode": "teacher_grounded_completion_fallback",
        "attempts": max_revisions + 1, "failures": failures,
        "reviewed": False,
    }


def author_delegate_action(
    *, assignments: list[dict], prompt: str, author=None, critic=None,
    max_revisions: int = 2,
    author_system: str | None = None,
    critic_system: str | None = None,
    freeze_assignment_prose: bool = False,
    reasoning_only_stage3: bool = False,
    naturalize_assignment_prose: bool = False,
    round_evidence_contract: dict | None = None,
    reasoning_goals: list[dict] | None = None,
    reasoning_guard=None,
    decision_contract: dict | None = None,
    critic_conflict_objection=None,
    validation_prompt: str | None = None,
) -> tuple[dict | None, dict]:
    # Separate runtime-specific prose instructions from the shared O1
    # deterministic structure/Guard. Existing Base callers are unchanged.
    author_system = author_system or _AUTHOR_SYSTEM
    critic_system = critic_system or _CRITIC_SYSTEM
    # Stage 3 can author/review against a compact evidence-only prompt while
    # validating identifier reuse against the original exact O1 prompt.
    validation_prompt = prompt if validation_prompt is None else validation_prompt
    template = proposed_template_action(assignments)
    if author is None:
        errors = action_issues(template, assignments, prompt=validation_prompt)
        if errors:
            raise ValueError("stage3_invalid_template:" + ",".join(errors))
        return template, {"status": "accepted", "mode": "template",
                          "attempts": 0}
    if critic is None:
        raise ValueError("stage3_llm_author_requires_independent_critic")
    failures = []
    conflict_rereviewed = False
    visible_set_relation_rereviewed = False
    for revision in range(max_revisions + 1):
        request = {
            "current_main_visible_prompt": prompt,
            "current_round_assignments_in_required_order": assignments,
            "reference_action_for_structure_only": template,
            "previous_feedback": failures[-2:],
        }
        if freeze_assignment_prose and reasoning_only_stage3:
            # Stage-3 Author sees public Main history and pending work,
            # never the private reference action or Critic dialogue.
            request = {
                "question_and_subagent_history": prompt,
                "pending_investigations": [
                    ({
                        "reference_instruction": item["target_instruction"],
                        "prior_visible_context": item["context"],
                        "public_source_descriptions": item.get(
                            "public_source_descriptions", []
                        ),
                        "allowed_source_ids": item["source_ids"],
                        "required_output_fields": item["output_fields"],
                        "requires_upstream_result": item["needs_upstream"],
                        "investigation_type": (
                            "lookup_by_returned_key"
                            if item.get("keyed_lookup") else
                            ("source_investigation" if item["source_ids"]
                             else "combine_prior_findings")
                        ),
                    } if naturalize_assignment_prose else {
                        "instruction": item["target_instruction"],
                        "context": item["context"],
                    })
                    for item in assignments
                ],
                "output_format": (
                    {"reasoning": "explain the evidence for the next investigations",
                     "tasks": [{
                         "task_instruction": "specific investigative request",
                         "context": "natural handoff and known clues",
                     } for _ in assignments]}
                    if naturalize_assignment_prose else
                    {"reasoning": "grounded explanation of next investigations"}
                ),
                **({
                    "round_evidence_contract":
                        round_evidence_contract["public"],
                } if round_evidence_contract is not None else {}),
                "writing_guidance": (
                    "Explain only CURRENT-round investigations, using "
                    "the exact pending task instructions and previously "
                    "returned Sub-Agent facts. If the evidence contract "
                    "contains observed_identifier_set_relations, use "
                    "exactly the public shared identifiers, but do NOT "
                    "confuse candidate selection with confirmed archival "
                    "year or claim the current batch performs later steps. The overall question may "
                    "require additional dependent source lookups later; "
                    "do not claim all sources must be queried now, or "
                    "that a previous lookup failed just because another "
                    "key must still be resolved. No invented past failures, "
                    "completed checks, or hidden future results. "
                    + ("Write reasoning and one task_instruction/context "
                       "pair per listed investigation; keep permitted "
                       "source identities and output fields unchanged."
                       if naturalize_assignment_prose else
                       "Return only the reasoning field.")
                ),
            }
            if failures:
                request["revision_issue_types"] = [
                    {"errors": row.get("errors", []),
                     "stage": row.get("stage", "")}
                    for row in failures[-2:]
                ]
                # Deliver only general issue-specific repair guidance.
                # Do not expose private reviewer dialogue to the Author.
                repairs = {
                    "reasoning_visible_intersection_mismatch":
                        "Recalculate the overlap using only the returned "
                        "identifier sets in the visible evidence contract.",
                    "reasoning_dispatch_task_count_mismatch":
                        "Describe the number of delegated tasks accurately; "
                        "the number of evidence sources may differ.",
                    "reasoning_current_output_field_mismatch":
                        "Only describe this round's output fields as "
                        "returned, not later dependent fields.",
                    "reasoning_premature_final_step":
                        "Do not claim the investigation finishes in this "
                        "round while further work remains.",
                    "invented_prior_subagent_failure":
                        "Do not describe a previously completed Sub-Agent "
                        "handoff as a failed retrieval.",
                }
                request["repair_guidance"] = list(dict.fromkeys(
                    [repairs[code]
                     for failed in failures[-2:]
                     for code in failed.get("errors", [])
                     if code in repairs]
                    + (_stage3_task_prose_repair_guidance(
                        failures, assignments,
                    ) if naturalize_assignment_prose else [])
                ))
        if freeze_assignment_prose and not reasoning_only_stage3:
            request["authoring_scope"] = (
                "Generate natural grounded reasoning ONLY. The assigned "
                "task instructions and contexts are already determined "
                "by the evidence planner and cannot be revised here."
            )
            if reasoning_goals is not None:
                request["main_visible_evidence_requests"] = reasoning_goals
                request["reasoning_obligation"] = (
                    "Explain unresolved evidence requests as unresolved. "
                    "Never claim a pending cross-check already succeeded."
                )
            if decision_contract is not None:
                # Stage-4-only author surface: do not put private
                # compiler/teacher vocabulary, action names, reference
                # actions, or raw Critic dialogue in the model input.
                # The Critic receives the full contract separately.
                visible = decision_contract["visible_evidence"]
                request = {
                    "question_and_subagent_history": prompt,
                    "observed_disagreements": visible[
                        "confirmed_result_summary_conflicts"
                    ],
                    "recent_delivery_gaps": visible["delivery_gaps"],
                    "earlier_field_requests": decision_contract.get(
                        "prior_field_activity", []
                    ),
                    "information_still_needed": [
                        {
                            "fields": item.get("target_fields", []),
                            "reason": item.get(
                                "main_visible_justification", ""
                            ),
                        } for item in reasoning_goals or []
                    ],
                    "work_to_explain": [
                        {
                            "instruction": item["target_instruction"],
                            "context": item["context"],
                        } for item in assignments
                    ],
                    "output_format": {
                        "reasoning": "natural explanation of the next investigation"
                    },
                    "writing_guidance": (
                        "Write as the investigator making a decision from "
                        "the observations above. Distinguish observed "
                        "disagreements from verified truth, explain why "
                        "the requested investigations are needed, and "
                        "avoid inventing completed checks or results. "
                        "Do not say an earlier retrieval for a requested "
                        "field failed unless earlier Sub-Agent feedback "
                        "actually shows that request; a field not yet "
                        "requested is simply outstanding. Avoid synthetic "
                        "workspace handles and placeholder URLs in prose. "
                        "Return only the reasoning field."
                    ),
                }
        try:
            candidate = author.complete_json(
                author_system, json.dumps(request, ensure_ascii=False)
            )
        except Exception as exc:
            failures.append({"stage": "author", "reason": str(exc)[:200]})
            continue
        # V10.4 only: reasoning-only JSON has no action choice to fix;
        # an explicitly authored complete action is still rejected below.
        authored_reasoning = (
            candidate.get("reasoning") if isinstance(candidate, dict) else None
        )
        if (decision_contract is not None and isinstance(candidate, dict)
                and "action" not in candidate):
            candidate = {**candidate, "action": "delegate_task"}
        if (freeze_assignment_prose and reasoning_only_stage3
                and naturalize_assignment_prose):
            task_prose_errors = _stage3_task_prose_issues(
                candidate, assignments, prompt=validation_prompt,
            )
            if task_prose_errors:
                authored_tasks = (
                    candidate.get("tasks", []) if isinstance(candidate, dict)
                    else []
                )
                failures.append({
                    "stage": "stage3_task_prose_guard",
                    "errors": task_prose_errors,
                    "task_prose_diagnostics": [{
                        "task_index": i,
                        "context_type": type(item.get("context")).__name__,
                        "context_chars": (
                            len(item["context"].strip())
                            if isinstance(item.get("context"), str) else None
                        ),
                        "instruction_chars": (
                            len(item["task_instruction"].strip())
                            if isinstance(item.get("task_instruction"), str)
                            else None
                        ),
                    } for i, item in enumerate(authored_tasks)
                       if isinstance(item, dict)],
                })
                continue
            candidate = _stage3_compile_task_prose(
                candidate, assignments, template,
            )
        # Stage-4C can explicitly freeze the entire evidence dispatch
        # contract. The Author writes *reasoning*, while all task language,
        # handoff context, tool routing and parallelism come from the
        # verified evidence plan. This avoids silently publishing an
        # Author's unrelated deliverable (e.g. a winner instead of the
        # planned marker key) as a valid executor-bound assignment.
        if (freeze_assignment_prose and not naturalize_assignment_prose
                and isinstance(candidate, dict) and (
                candidate.get("action", "delegate_task") == "delegate_task"
        )):
            candidate = {
                "action": "delegate_task",
                "reasoning": candidate.get("reasoning"),
                "params": deepcopy(template["params"]),
            }
        # Default Stage-3 and legacy callers remain free to author the
        # task prose, subject to the original O1 Guard and critic.
        elif (not naturalize_assignment_prose
              and isinstance(candidate, dict)
              and candidate.get("action") == "delegate_task"):
            parameters = candidate.get("params")
            generated = (parameters.get("tasks")
                         if isinstance(parameters, dict) else None)
            if (isinstance(generated, list)
                    and len(generated) == len(assignments)
                    and all(isinstance(task, dict) for task in generated)):
                compiled_tasks = []
                for task, expected in zip(generated, assignments):
                    instruction = task.get("task_instruction")
                    context = task.get("context")
                    # The source scope and previously delivered results are
                    # facts known to Main. Preserve them programmatically;
                    # a prose author must not accidentally erase handoffs.
                    if isinstance(instruction, str) and expected["source_ids"]:
                        missing = [
                            sid for sid in expected["source_ids"]
                            if sid not in instruction + " " + str(context)
                        ]
                        if missing:
                            instruction = (instruction.rstrip()
                                + " Examine source(s): " + ", ".join(missing) + ".")
                    if isinstance(context, str) and expected["needs_upstream"]:
                        upstream = expected["context"]
                        if "Previously returned result:" not in context:
                            context = upstream + " " + context
                        context = context[:3500]
                    compiled_tasks.append({
                        "task_instruction": instruction,
                        "context": context,
                        "model": expected["model"],
                        "tools": list(expected["tools"]),
                    })
                candidate = {
                    "action": "delegate_task",
                    "reasoning": candidate.get("reasoning"),
                    "params": {"tasks": compiled_tasks},
                }
        errors = action_issues(candidate, assignments, prompt=validation_prompt)
        if errors:
            failures.append({"stage": "guard", "errors": errors})
            continue
        if freeze_assignment_prose and reasoning_guard is not None:
            semantic_errors = reasoning_guard(candidate["reasoning"])
            if semantic_errors:
                failures.append({
                    "stage": "evidence_reasoning_guard",
                    "errors": semantic_errors,
                })
                continue
        try:
            if freeze_assignment_prose:
                # The LLM critic cannot alter code-frozen dispatch,
                # source scope, model or tool routing. In Stage 3 it also
                # reviews authored instructions/contexts for semantic
                # fidelity; Stage 4's existing reasoning-only path is
                # unchanged.
                review_payload = {
                    "current_main_visible_prompt": prompt,
                    "expected_current_evidence_goals": [{
                        "target_fields": item.get(
                            "focus_fields", item["output_fields"]
                        ),
                        "needs_upstream": item["needs_upstream"],
                    } for item in assignments],
                    **({
                        # These are the exact PUBLIC task instructions which
                        # Main is about to dispatch, never future DAG tasks.
                        # Without them the critic may try to jump straight
                        # to the final user goal and reject a legal indexed
                        # intermediate lookup.
                        "current_round_task_contract": [{
                            "task_instruction": item["target_instruction"],
                            "context": item["context"],
                            "source_ids": item["source_ids"],
                            "output_fields": item["output_fields"],
                            "needs_upstream": item["needs_upstream"],
                        } for item in assignments],
                        "review_scope_priority": (
                            "First compare reasoning with CURRENT-ROUND "
                            "task contract and visible prior history. "
                            "Do not replace this round with the final "
                            "question's later dependent tasks. Reject "
                            + ("unsupported reasoning and task prose."
 if naturalize_assignment_prose else
 "only unsupported claims in authored reasoning.")
                        ),
                    } if freeze_assignment_prose
                    and reasoning_only_stage3 else {}),
                    **({
                        "current_round_evidence_contract":
                            round_evidence_contract["public"],
                    } if reasoning_only_stage3
                    and round_evidence_contract is not None else {}),
                    "candidate_reasoning_only": (
                        authored_reasoning if decision_contract is not None
                        else candidate["reasoning"]
                    ),
                    **({
                        "candidate_task_prose_only": [{
                            "task_instruction": task["task_instruction"],
                            "context": task["context"],
                        } for task in candidate["params"]["tasks"]],
                        "task_prose_constraints": [{
                            "allowed_source_ids": item["source_ids"],
                            "required_output_fields": item["output_fields"],
                            "public_source_descriptions": item.get(
                                "public_source_descriptions", []
                            ),
                            "prior_visible_context": item["context"],
                            "requires_upstream": item["needs_upstream"],
                        } for item in assignments],
                    } if naturalize_assignment_prose else {}),
                    "published_reasoning": candidate["reasoning"],
                    "main_visible_evidence_requests": reasoning_goals or [],
                    **({
                        "locked_main_visible_decision_facts": decision_contract,
                    } if decision_contract is not None else {}),
                    "assignment_contract_validated_by_code": True,
                    "review_scope": (
                        "Evaluate factual claims ADDED by the Author and "
                        "their coherence with the code-owned visible facts. "
                        "The confirmed observed conflicts, if present, "
                        "come from explicit field comparisons of returned "
                        "Sub-Agent reports; do not deny their existence "
                        "or treat Result as independently verified. "
                        "Never approve an invented source, premature "
                        "verification, hidden reviewer dialogue or an "
                        "incorrect dependency claim. "
                        + ("Review authored task_instruction/context for "
                           "evidence fidelity and specificity, never "
                           "slot order, model or tools. "
                           if naturalize_assignment_prose else
                           "Do not assess or rewrite task_instruction, "
                           "context, model, tools, or the overall frozen "
                           "delegation plan. ") +
                        "In trusted Stage-3 review, distinguish this round's "
                        "returned fields from later dependent evidence, "
                        "and distinguish the number of sources from the "
                        "number of independently delegated tasks. "
                        "Reject unsupported final-step claims. "
                        "When a public literal set-intersection witness "
                        "exists, verify membership against that witness "
                        "rather than inventing extra shared Case IDs. "
                        "The witness is NOT proof that the final answer "
                        "or a source's truth has been verified."
                    ),
                }
            else:
                review_payload = {
                    "current_main_visible_prompt": prompt,
                    "assignment_prose_goals_in_required_order": [{
                        "target_instruction": item["target_instruction"],
                        "expected_context": item["context"],
                        "needs_upstream": item["needs_upstream"],
                        "output_fields": item["output_fields"],
                    } for item in assignments],
                    "candidate_prose_only": {
                        "reasoning": candidate["reasoning"],
                        "tasks": [{
                            "task_instruction": task["task_instruction"],
                            "context": task["context"],
                        } for task in candidate["params"]["tasks"]],
                    },
                    "fixed_tool_and_model_fields_checked_by_code": True,
                }
            review = critic.complete_json(
                critic_system, json.dumps(review_payload, ensure_ascii=False)
            )
        except Exception as exc:
            failures.append({
                "stage": "critic_backend", "reason": str(exc)[:200],
            })
            continue
        # Reviewers can miscompute even a trivial set overlap. If a
        # rejection disputes a publicly witnessed intersection, request
        # another INDEPENDENT judgment against the SAME public evidence.
        # Never auto-pass or erase a genuine unsupported Author claim.
        if (freeze_assignment_prose and reasoning_only_stage3
                and round_evidence_contract is not None
                and not _critic_review_passes(review)
                and round_evidence_contract["public"].get(
                    "observed_identifier_set_relations"
                )):
            critique = (
                str(review.get("reason", "")) + " "
                + str(review.get("repair_hint", ""))
                if isinstance(review, dict) else ""
            ).casefold()
            if any(word in critique for word in (
                "intersection", "overlap", "both lists",
                "both sets", "shared identifier",
            )):
                visible_set_relation_rereviewed = True
                try:
                    review = critic.complete_json(
                        critic_system + (
                            "\nSECOND INDEPENDENT REVIEW: The provided "
                            "observed_identifier_set_relations were "
                            "computed deterministically from TWO literal "
                            "Main-visible Sub-Agent Result lists. Check "
                            "whether the disputed identifier is actually "
                            "in both lists before alleging faulty "
                            "arithmetic. The witness is NOT proof of "
                            "source truth or the downstream final answer. "
                            "Reject any other unsupported factual claim "
                            "or incorrect task boundary as usual. "
                            "Do not auto-accept just because the "
                            "intersection itself is correct.\n"
                        ),
                        json.dumps(review_payload, ensure_ascii=False),
                    )
                except Exception as exc:
                    failures.append({
                        "stage": "visible_set_relation_rereview_backend",
                        "reason": str(exc)[:200],
                    })
                    continue
        # The Critic cannot overturn a code-confirmed *visible* conflict
        # solely by asserting that the two delivered statements agree.
        # Invalid critiques trigger a second independent review, never
        # automatic acceptance. Existing Stage-3/legacy behavior is unchanged.
        if freeze_assignment_prose and reasoning_goals and (
            not _critic_review_passes(review)
        ):
            critique = (str(review.get("reason", "")) + " " +
                        str(review.get("repair_hint", ""))).casefold()
            if decision_contract is not None:
                mistaken_conflict_objection = (
                    critic_conflict_objection is not None
                    and critic_conflict_objection(review)
                )
            else:
                mistaken_conflict_objection = (
                    any(g.get("operation") == "verify_conflicting_evidence"
                        for g in reasoning_goals)
                    and any(word in critique for word in (
                        "no conflict", "no discrepancy", "no mismatch",
                        "both the result", "invented discrepancy",
                        "fabricated discrepancy", "does not exist",
                    ))
                )
            if mistaken_conflict_objection:
                conflict_rereviewed = True
                try:
                    review = critic.complete_json(
                        critic_system + (
                            "\nSECOND INDEPENDENT REVIEW: The attached "
                            "locked visible-fact contract proves the "
                            "specified Result/Summary values came from "
                            "the same public Sub-Agent report. Do not deny "
                            "their discrepancy. Decide whether the Author "
                            "ADDED unsupported facts, claimed a false "
                            "verification or stated the next step badly. "
                            "Never approve merely because the core is valid.\n"
                        ),
                        json.dumps(review_payload, ensure_ascii=False),
                    )
                except Exception as exc:
                    failures.append({
                        "stage": "critic_conflict_rereview_backend",
                        "reason": str(exc)[:200],
                    })
                    continue
        if not freeze_assignment_prose and _review_only_frozen_tool_objection(review):
            try:
                review = critic.complete_json(
                    critic_system + "\nSECOND REVIEW: ignore all frozen tool "
                    "routing, judge only grounded Task prose.\n",
                    json.dumps(
                        _tool_neutral_review_payload(
                            prompt, assignments, candidate,
                        ), ensure_ascii=False,
                    ),
                )
            except Exception as exc:
                failures.append({
                    "stage": "critic_scope_rereview_backend",
                    "reason": str(exc)[:200],
                })
                continue
        if _critic_review_passes(review):
            return deepcopy(candidate), {
                "status": "accepted",
                "mode": ("llm_reviewed_contract_frozen"
                         if freeze_assignment_prose else "llm_reviewed"),
                "assignment_contract_frozen": freeze_assignment_prose,
                "authored_fields": (
                    ["reasoning"] if freeze_assignment_prose
                    and not naturalize_assignment_prose else
                    ["reasoning", "task_instruction", "context"]
                ),
                "naturalized_task_prose": naturalize_assignment_prose,
                "attempts": revision + 1, "review": review,
                **({
                    "reasoning_fidelity_rejections": [
                        {"errors": list(failed.get("errors", []))}
                        for failed in failures
                        if failed.get("stage") == "evidence_reasoning_guard"
                    ],
                    "visible_set_relation_rereviewed":
                        visible_set_relation_rereviewed,
                } if reasoning_only_stage3 else {}),
                **({
                    "decision_contract_version": decision_contract.get(
                        "contract_version", "unknown"
                    ),
                    "confirmed_visible_conflicts": len(
                        decision_contract["visible_evidence"][
                            "confirmed_result_summary_conflicts"
                        ]
                    ),
                    "critic_conflict_rereviewed": conflict_rereviewed,
                } if decision_contract is not None else {}),
            }
        failures.append({"stage": "critic",
                         "reason": str(review.get("reason", ""))[:250],
                         "repair_hint": str(review.get("repair_hint", ""))[:250]})
    return None, {
        "status": "failed",
        "mode": ("llm_reviewed_contract_frozen"
                 if freeze_assignment_prose else "llm_reviewed"),
        "attempts": max_revisions + 1, "failures": failures,
        **({
            "decision_contract_version": decision_contract.get(
                "contract_version", "unknown"
            ),
            "confirmed_visible_conflicts": len(
                decision_contract["visible_evidence"][
                    "confirmed_result_summary_conflicts"
                ]
            ),
            "critic_conflict_rereviewed": conflict_rereviewed,
        } if decision_contract is not None else {}),
    }
