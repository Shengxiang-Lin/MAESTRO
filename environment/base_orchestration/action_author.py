"""Main-Agent action surface authoring and independent guarded review.

Teacher determines which Macro slots are legal; the Author writes only
O1-compatible language. The Author never receives current/future results.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy

from environment.generate_question.evidence_tools import (
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
reasoning against actually Main-visible Sub-Agent results. Reject invented
earlier errors/retries, completed checks or premature findings. Do not ask
for changes to task prose or frozen dispatch. Return the normal three axes.
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


def stage3_reasoning_issues(reasoning: str, delivered: dict) -> list[str]:
    """Validate only unsupported claims refutable from public report status."""
    if not isinstance(reasoning, str):
        return ["invalid_main_reasoning"]
    if re.search(
        r"\b(?:reference_action_for_structure_only|"
        r"current_round_assignments_in_required_order)\b",
        reasoning, re.I,
    ):
        return ["internal_reference_action_leak"]
    if not _PAST_FAILED_ATTEMPT.search(reasoning):
        return []
    statuses = [
        record.get("finish_result", {}).get("status")
        for record in delivered.values()
    ]
    if not statuses or all(status == "done" for status in statuses):
        return ["invented_prior_subagent_failure"]
    return []


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
        "output_fields": list(target["outputs"]["typed_relation"]),
        "needs_upstream": bool(target["depends_on"]),
        "keyed_lookup": has_keyed,
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
    reasoning_goals: list[dict] | None = None,
    reasoning_guard=None,
    decision_contract: dict | None = None,
    critic_conflict_objection=None,
) -> tuple[dict | None, dict]:
    # Separate runtime-specific prose instructions from the shared O1
    # deterministic structure/Guard. Existing Base callers are unchanged.
    author_system = author_system or _AUTHOR_SYSTEM
    critic_system = critic_system or _CRITIC_SYSTEM
    template = proposed_template_action(assignments)
    if author is None:
        errors = action_issues(template, assignments, prompt=prompt)
        if errors:
            raise ValueError("stage3_invalid_template:" + ",".join(errors))
        return template, {"status": "accepted", "mode": "template",
                          "attempts": 0}
    if critic is None:
        raise ValueError("stage3_llm_author_requires_independent_critic")
    failures = []
    conflict_rereviewed = False
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
                    {"instruction": item["target_instruction"],
                     "context": item["context"]}
                    for item in assignments
                ],
                "output_format": {
                    "reasoning": "grounded explanation of next investigations"
                },
                "writing_guidance": (
                    "Explain only current evidence needs using already "
                    "returned Sub-Agent facts. No invented past failures, "
                    "completed checks, or hidden future results. "
                    "Return only the reasoning field."
                ),
            }
            if failures:
                request["revision_issue_types"] = [
                    {"errors": row.get("errors", []),
                     "stage": row.get("stage", "")}
                    for row in failures[-2:]
                ]
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
        # Stage-4C can explicitly freeze the entire evidence dispatch
        # contract. The Author writes *reasoning*, while all task language,
        # handoff context, tool routing and parallelism come from the
        # verified evidence plan. This avoids silently publishing an
        # Author's unrelated deliverable (e.g. a winner instead of the
        # planned marker key) as a valid executor-bound assignment.
        if freeze_assignment_prose and isinstance(candidate, dict) and (
            candidate.get("action", "delegate_task") == "delegate_task"
        ):
            candidate = {
                "action": "delegate_task",
                "reasoning": candidate.get("reasoning"),
                "params": deepcopy(template["params"]),
            }
        # Default Stage-3 and legacy callers remain free to author the
        # task prose, subject to the original O1 Guard and critic.
        elif isinstance(candidate, dict) and candidate.get("action") == "delegate_task":
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
        errors = action_issues(candidate, assignments, prompt=prompt)
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
                # The LLM critic must not attempt to rewrite frozen
                # evidence acquisition instructions or tool routing.
                # It reviews only Main's *reasoning* against the visible
                # history and the STRUCTURAL goal summaries. The code
                # validates and emits the executable task assignments.
                review_payload = {
                    "current_main_visible_prompt": prompt,
                    "expected_current_evidence_goals": [{
                        "target_fields": item.get(
                            "focus_fields", item["output_fields"]
                        ),
                        "needs_upstream": item["needs_upstream"],
                    } for item in assignments],
                    "candidate_reasoning_only": (
                        authored_reasoning if decision_contract is not None
                        else candidate["reasoning"]
                    ),
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
                        "incorrect dependency claim. Do not assess or "
                        "rewrite task_instruction, context, model, tools, "
                        "or the overall frozen delegation plan."
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
                    ["reasoning"] if freeze_assignment_prose else
                    ["reasoning", "task_instruction", "context"]
                ),
                "attempts": revision + 1, "review": review,
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
