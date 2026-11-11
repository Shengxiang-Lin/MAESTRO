"""Stage-2-style Author/Guard/Critic for executed untrusted Sub-Agent attempts.

The frozen World and its Gold answer are NEVER available to this author or
reviewer. An actual (possibly wrong or partial) WorkProduct is authoritative
for the report. The LLM writes full Main-facing result, summary and trace
prose, while the deterministic Guard checks field values, identifiers,
execution scope, status and report completeness. No result is fabricated
when an execution produced no deliverable.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re

from environment.implement_subtask.reporting import (
    _normalize_author_result, _readable_verified_answer,
)
from .state_machine import deterministic_report, Attempt, relation_rows


_AUTHOR_SYSTEM = """Write the FINAL message that a specialized Orchestra-o1
Sub-Agent would return after THIS specific delegated attempt.

Only the observations and operation steps supplied in the user payload
were available. They might be incomplete or confidently mistaken: report
exactly what this attempt established, NOT a corrected benchmark answer.
Do not inspect or infer the ground truth. Never compare this output with a
hidden clean reference or suggest an unobserved correction.

Return JSON ONLY:
{"result": "a concise, actionable, naturally phrased finding (or null if
 no result was delivered)",
 "summary": "1-2 grounded sentences about what this attempt did and learned",
 "trace_summary": "1-2 grounded sentences about observed operations",
 "status": "done|partial|missing"}

Result should identify concrete returned entities, keys and values where
feasible. For a small one-row lookup, mention its actual lookup key and
numeric measure; a large intermediate handoff can summarize its record set.
Status is immutable: done is NOT proof of correctness, and partial is NOT
the same as no usable result. Missing means no finish_result was delivered:
result must be null and you must not invent evidence. If only some fields
are available, describe those without claiming the task is fully solved.

Describe this stage only; do not claim a later filter, join, verification,
approval, web search, real audio/video perception, or tool call that the
provided executed operations did not perform. Never invent source IDs,
entity names, units, measurements, dates, provenance or numerical facts.
Do not mention fault injection, Oracle, Teacher, Gold, the simulator,
previous drafts or the Critic. You are a Sub-Agent reporting normally to
its Main-Agent, not an omniscient judge of the attempt.
"""


_CRITIC_SYSTEM = """Independently review the Main-facing Sub-Agent report
against the CURRENT ATTEMPT'S ACTUAL RETURNED RECORDS and executed
operator scope. The run may have produced a WRONG key or a partial
result; that is not itself a failure of the language report. Do not
correct it against the global Gold or imaginary source contents.

Check:
1. grounded: all stated values, identities and source claims are supported
   by this attempt's returned records and accessible execution context.
2. scope_preserved: status and findings are stage-local; no invented
   downstream filter, full completion, or independent verification.
3. natural: the result, summary and trace read like authentic concise
   Sub-Agent feedback, not a raw backend dump.
4. useful: a single-row answer names its returned key and key measurement;
   partial results distinguish available and unresolved fields.

Output JSON ONLY:
{"verdict":"pass|fail","grounded":true|false,
 "scope_preserved":true|false,"natural":true|false,"useful":true|false,
 "reason":"short factual explanation","repair_hint":"specific prose-only change"}
The result's status, structured rows and available fields are immutable.
Do not reject simply because a returned observation is inconsistent with
a separate clean reference which is NOT included in this payload.
"""


_TOKEN = re.compile(
    r"\b(?:KEY[12]-\d+|MK-\d+|RK-\d+|Case-\d+|Entry-\d+"
    r"|\d+(?:\.\d+)?)\b", re.I,
)
_ENTITY = re.compile(
    r"\b(?:KEY[12]-\d+|MK-\d+|RK-\d+|Case-\d+|Entry-\d+)\b", re.I,
)
_PRIVATE = re.compile(
    r"\b(?:oracle|gold\s+(?:answer|program|dag)|teacher|simulat\w*|"
    r"fault\s+inject\w*|previous\s+draft|critic\s+(?:said|noted)|"
    r"benchmark\s+answer|clean\s+(?:reference|result))\b", re.I,
)
_UNPERFORMED = re.compile(
    r"\b(?:i\s+watched|i\s+listened|i\s+searched\s+(?:the\s+)?web|"
    r"i\s+browsed\s+(?:the\s+)?internet)\b", re.I,
)


def _observed_contract(task: dict, attempt: Attempt, instruction: str) -> dict:
    """No hidden Gold, fault flag, or clean Oracle rows in LLM context."""
    rows = relation_rows(attempt.product)
    steps = []
    for node in task["operator_chain"]:
        steps.append({
            "operation": node["op"],
            "direct_source": node.get("source_id"),
            "input_key_field": node.get("input_key_field"),
            "source_key_field": node.get("source_key_field"),
            "comparison_keys": [
                node.get("left_on"), node.get("right_on")
            ] if node["op"] == "join" else None,
            "filter_field": node.get("field"),
            "projected_fields": node.get("fields"),
        })
    return {
        "assigned_task_instruction": instruction,
        "executed_operations_in_order": steps,
        "actual_delivered_rows": rows[:32],
        "total_delivered_rows": len(rows),
        "rows_omitted_from_prompt": max(0, len(rows) - 32),
        "output_fields": (list(attempt.product.columns)
                          if attempt.product else []),
        "available_fields": list(attempt.available_fields),
        "unresolved_fields": list(attempt.runtime.get("missing_fields", [])),
        "rows_complete": attempt.runtime.get("rows_complete", True),
        "execution_ended": True,
        "delivery_status": attempt.status,
        "finish_result_exists": attempt.product is not None,
    }


def _guard(draft: dict, contract: dict, attempt: Attempt) -> list[str]:
    if not isinstance(draft, dict):
        return ["invalid_report_json"]
    issues = []
    if set(draft) != {"result", "summary", "trace_summary", "status"}:
        issues.append("invalid_report_fields")
    if draft.get("status") != attempt.status:
        issues.append("reported_status_not_equal_actual_attempt")
    missing = attempt.product is None
    result = draft.get("result")
    if missing:
        if result is not None:
            issues.append("missing_result_must_be_null")
    elif not isinstance(result, (str, dict, list)):
        issues.append("invalid_delivered_result")
    normalized = None
    exact_rows = [row["values"] for row in relation_rows(attempt.product)]
    if not missing and isinstance(result, (str, dict, list)):
        normalized, errors = _normalize_author_result(
            result, exact_rows, list(attempt.product.columns),
        )
        issues.extend("result_" + e for e in errors)
    summary = draft.get("summary")
    trace = draft.get("trace_summary")
    if not isinstance(summary, str) or not 20 <= len(summary.strip()) <= 750:
        issues.append("invalid_summary")
    if not isinstance(trace, str) or not 20 <= len(trace.strip()) <= 1000:
        issues.append("invalid_trace_summary")
    prose = " ".join(
        v for v in (normalized, summary, trace) if isinstance(v, str)
    )
    if _PRIVATE.search(prose):
        issues.append("internal_generation_or_teacher_leak")
    if _UNPERFORMED.search(prose):
        issues.append("claims_unperformed_media_or_web_action")

    # A new value or identifier is never licensed merely because it
    # appears in the overall question or clean Stage 2 report.
    permitted = set(_TOKEN.findall(json.dumps(
        contract["actual_delivered_rows"], ensure_ascii=False,
    )))
    permitted |= {
        str(contract["total_delivered_rows"]),
        str(len(contract["executed_operations_in_order"])),
    }
    # The task instruction can mention *upstream* identifiers and numbers.
    # Such terms may be used in summary/trace as context, but result must
    # describe this attempt's actual deliverable, not the clean target.
    context_tokens = set(_TOKEN.findall(
        contract["assigned_task_instruction"]
    ))
    for field, value in (("result", normalized), ("summary", summary),
                         ("trace_summary", trace)):
        if not isinstance(value, str):
            continue
        allowed = permitted if field == "result" else permitted | context_tokens
        # Traces may name an operator's fixed row count, not invent values.
        unknown = set(_TOKEN.findall(value)) - allowed
        if unknown:
            issues.append(field + "_unobserved_numeric_or_entity")
    if not missing and isinstance(normalized, str):
        identities = set(_ENTITY.findall(json.dumps(
            contract["actual_delivered_rows"], ensure_ascii=False,
        )))
        # A short handoff must not silently switch a reported identity.
        if len(exact_rows) == 1 and len(identities) == 1:
            if not identities <= set(_ENTITY.findall(normalized)):
                issues.append("result_missing_returned_identity")
        if attempt.status == "partial":
            unresolved = contract["unresolved_fields"]
            if unresolved and not any(
                name.replace("_", " ") in (summary or "").lower()
                or name in (summary or "")
                or name in (normalized or "")
                for name in unresolved
            ):
                issues.append("partial_missing_field_not_acknowledged")
    return sorted(set(issues))


def naturalize_report(task: dict, attempt: Attempt, *,
                      instruction: str, model_alias: str,
                      reviewed: dict | None,
                      author=None, critic=None,
                      max_revisions: int = 2) -> tuple[dict, dict]:
    """Generate all three public prose fields from this executed attempt.

    Template fallback is explicitly labeled; in LLM dataset mode the
    caller can fail closed rather than treating fallback as reviewed.
    """
    base = deterministic_report(
        task, attempt, instruction=instruction,
        model_alias=model_alias, reviewed=reviewed,
    )
    provenance = base.pop("_report_mode")
    if author is None or provenance == "reviewed_clean":
        return base, {"status": "accepted", "mode": provenance,
                      "attempts": 0}
    if critic is None:
        raise ValueError("subagent_naturalization_requires_critic")
    contract = _observed_contract(task, attempt, instruction)
    failures = []
    for revision in range(max_revisions + 1):
        request = {
            "observed_current_attempt": contract,
            "previous_prose_feedback": failures[-2:],
        }
        try:
            candidate = author.complete_json(
                _AUTHOR_SYSTEM,
                json.dumps(request, ensure_ascii=False),
            )
        except Exception as exc:
            failures.append({"stage": "author_backend", "reason": str(exc)[:200]})
            continue
        issues = _guard(candidate, contract, attempt)
        if issues:
            failures.append({"stage": "guard", "errors": issues})
            continue
        # The structured WorkProduct remains in the execution ledger.
        # LLM's structured answer, when supplied, must match exactly and is
        # normalized to Stage-2-style concise readable result prose.
        result = candidate["result"]
        if result is not None and not isinstance(result, str):
            result, _ = _normalize_author_result(
                result,
                [row["values"] for row in relation_rows(attempt.product)],
                list(attempt.product.columns),
            )
        try:
            decision = critic.complete_json(
                _CRITIC_SYSTEM,
                json.dumps({
                    "observed_current_attempt": contract,
                    "candidate_main_facing_report": {
                        "result": result,
                        "summary": candidate["summary"],
                        "trace_summary": candidate["trace_summary"],
                        "status": candidate["status"],
                    },
                }, ensure_ascii=False),
            )
        except Exception as exc:
            failures.append({"stage": "critic_backend", "reason": str(exc)[:200]})
            continue
        if (isinstance(decision, dict)
                and decision.get("verdict") == "pass"
                and all(decision.get(k) is True for k in (
                    "grounded", "scope_preserved", "natural", "useful",
                ))):
            report = deepcopy(base)
            if report.get("finish_result") is not None:
                report["finish_result"] = {
                    "status": attempt.status,
                    "result": result,
                    "summary": candidate["summary"].strip(),
                }
                report["trace_summary"] = candidate["trace_summary"].strip()
            else:
                # Preserve real missing finish_result. Place the status
                # explanation in the observable trace, never invent one.
                report["trace_summary"] = (
                    candidate["summary"].strip() + " " +
                    candidate["trace_summary"].strip()
                )
            return report, {
                "status": "accepted", "mode": "llm_reviewed",
                "attempts": revision + 1,
                "review": decision,
                "actual_workproduct_immutable": True,
            }
        failures.append({
            "stage": "critic",
            "reason": str(decision.get("reason", ""))[:250]
                      if isinstance(decision, dict) else "invalid_review",
            "repair_hint": str(decision.get("repair_hint", ""))[:250]
                           if isinstance(decision, dict) else "",
        })
    return base, {
        "status": "fallback",
        "mode": "grounded_template_fallback",
        "attempts": max_revisions + 1,
        "failures": failures,
        "actual_workproduct_immutable": True,
    }
