"""Turn frozen Oracle Macro evidence into realistic Sub-Agent finish reports.

The LLM may only verbalize results. Exact typed answers, source fact IDs,
operation traces, and completion status are immutable teacher-owned data.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re


AUTHOR_SYSTEM = """You are writing the FINAL message that a real specialized
Sub-Agent would return to its Main-Agent after completing a delegated task.

Follow the actual multi-agent finish convention:
  {"result": "...", "summary": "...", "status": "done"}

"result": concrete findings the Main-Agent can use immediately, with entity
names/values and short evidence explanation. "summary": 1-3 natural sentences
saying what work was performed, what it established, and relevant limitations.
Write professional, specific, compact English. Vary wording by work type.
Do NOT produce a database log, schema report, generic 'To produce...' boilerplate,
or a step-by-step internal chain of thought. The Main-Agent needs findings and
an evidence-backed operational summary, not hidden reasoning.

Your input is a deterministic, FICTIONAL observation and execution record,
not actual image/video/audio perception. You may say 'checked the supplied
video annotation' or 'compared the source records' but NEVER claim to have
watched video, transcribed audio, searched the internet, read a real website,
or used a tool not actually present in the supplied operator trace.
No invented names, dates, numeric observations, facts, URLs, quotes, motives,
sources, or eligibility rules. Do not add 'approved' when it only says matched.
Use only source_fact_refs from the given available evidence and upstream context.
If no rows meet conditions, this is a valid completed result: say so, with
status done, not partial. If the reported relation is a handoff, say what the
handoff contains instead of falsely claiming it answers the overall question.

CRITICAL STAGE BOUNDARY:
Each Macro is only ONE stage of a multi-step DAG. The overall_question is
background, NOT a list of filters you already executed. Read the actual
observed_operator_steps. For a scan-only Macro, report the unfiltered source
rows; for a join-only Macro, report joined candidates. Do NOT claim that
every joined entry is active/approved/above threshold or meets final
eligibility requirements unless the supplied steps actually established it.
A join does NOT itself check active, measured_value, time windows, approval
or clearance. Some candidate rows can have active=false or value=0.
Later Macro Tasks perform those filters. NEVER alter, omit or pre-filter the
EXACT Oracle output rows, even when they fail the overall question's later
criteria. If a reviewer objects to a premature claim, correct ONLY the prose
summary, never the result rows.

The result field may be (A) a concise grounded STRING describing the
workproduct, or (B) EXACTLY the frozen structured answer rows as a JSON
array. If the answer consists ONLY of entry_label, a JSON array of those
exact entry labels is also accepted. For a single-row result, one exact
object is acceptable. Do NOT invent a filtered subset or an extra row:
structured output is programmatically compared to the full exact answer.
Do not copy/modify JSON if you can simply describe the result naturally.
The program always attaches exact verified answer values afterward.
Output ONLY JSON with keys result, summary, status; status must be done.

Examples of VALID result shapes:
{"result":"Identified two records matching the assigned condition.",
 "summary":"Checked the supplied annotations against the register and retained
 the two supported records.","status":"done"}
{"result":[{"entry_label":"Entry-02"}],
 "summary":"Compared the supplied event steps against the session windows and
 found one matching record.","status":"done"}
"""


CRITIC_SYSTEM = """You are an INDEPENDENT factual and stylistic reviewer
for a Sub-Agent's final message to Main-Agent. The candidate is NOT a
new solver; its facts MUST be entailed by the frozen oracle outputs and
the specific source-scoped operation/evidence observations supplied.

Check separately:
- grounded: all names, numbers, filtering claims, source facts, provenance
  and reported observations have support; no fabricated source or media action;
- answer_scope_preserved: does not alter exact result semantics or confuse
  an intermediate handoff with the final answer; no missing/extra filters;
- evidence_scope_respected: only authorized local/upstream facts, no source
  that was not accessible to this Macro; no unsupported timestamps/URLs;
- natural: reads like a concise specialized Sub-Agent completion message,
  not raw backend schema, templated "To produce X", or an overlong narration.

STAGE-LOCAL REVIEW: Judge against THIS Macro's permitted operator steps,
not the final overall question. A scan of a glossary is not an approval or
clearance check. A join that includes active=false is CORRECT when it
precedes an active filter. The row set may include candidates that fail
LATER checks; never demand filtering them here.
If a candidate summary overclaims final qualification, require that the
SUMMARY be corrected to accurately describe the current stage, NOT that
the already-exact structured result be filtered or rewritten. Your
repair_hint must NEVER suggest deleting, filtering, adding or changing
the frozen answer rows.

For empty results, a genuine complete "no matches" is correct, not partial.
For a structured candidate result, the application has already verified
an EXACT match to the full frozen answer rows; do not reject it merely
because it is rendered as a list/array. Review the proposed natural
summary and any prose for actual factual errors.
Do not demand all raw fact IDs in the user-facing summary.
Do not invent any issue merely to be critical; pass accurate natural reports.
Return ONLY JSON:
{"verdict":"pass"|"fail","grounded":true|false,
 "answer_scope_preserved":true|false,
 "evidence_scope_respected":true|false,"natural":true|false,
 "reason":"specific reason (short)","repair_hint":"one focused fix"}
"""


_TOOL_HALLUCINATION = re.compile(
    r"\b(?:GoogleSearchAction|VideoAnalysisAction|ParseAudioAction|"
    r"ImageAnalysisAction|ExtractUrlContentAction)\b"
    r"|\b(?:searched the (?:web|internet)|watched the video|"
    r"transcribed the audio|visited the website)\b",
    re.I,
)
_FACT_ID = re.compile(r"\b[A-Za-z][A-Za-z0-9_-]*:F[0-9A-Za-z_-]+\b")


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _teacher_contract(record: dict, *, question: str | None,
                      is_terminal: bool) -> dict:
    trace = record["trace_summary"]
    rows = record["answer"]
    if not isinstance(rows, list) or not isinstance(trace, dict):
        raise ValueError("invalid_teacher_answer_or_trace")
    if rows != [entry["values"] for entry in record["answer_with_provenance"]]:
        raise ValueError("oracle_answer_provenance_mismatch")
    steps = trace["steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("oracle_has_no_evidence_steps")
    allowed = set(trace["available_evidence_source_ids"])
    all_refs = {
        ref for step in steps for ref in step["supporting_fact_refs"]
    } | set(trace["answer_supporting_fact_refs"])
    if any(ref.split(":", 1)[0] not in allowed for ref in all_refs):
        raise ValueError("oracle_unauthorized_source_ref")
    for step in steps:
        if any(atom["fact_ref"] not in step["supporting_fact_refs"]
               for atom in step["evidence_atoms"]):
            raise ValueError("oracle_atom_reference_inconsistent")
    return {
        "overall_question_for_context_only": question,
        "macro_task_id": record["macro_task_id"],
        "macro_is_terminal": is_terminal,
        "upstream_handoffs": record["depends_on"],
        "authorized_direct_sources": record["authorized_source_ids"],
        "available_provenance_sources": sorted(allowed),
        "terminal_output_schema_of_this_macro": record["output_schema"],
        "exact_oracle_answer": rows,
        "executed_operations_in_this_macro": [
            step["operation"] for step in steps
        ],
        "stage_scope_note": (
            "Only these operators were executed in this Macro. "
            "An intermediate join/scan does not imply later eligibility "
            "filters; preserve the complete input rows and report only "
            "what THIS Macro established."
        ),
        "answer_supporting_fact_refs": trace["answer_supporting_fact_refs"],
        "observed_operator_steps": [
            {
                "action": step["action"],
                "operation": step["operation"],
                "observed_result": step["observed_result"],
                "output_row_count": step["output_row_count"],
                "source_fact_refs": step["supporting_fact_refs"][:16],
                "sampled_evidence_atoms": step["evidence_atoms"][:4],
                "remaining_atoms_omitted": max(
                    0, step["evidence_atoms_total"] - 4
                ),
            }
            for step in steps
        ],
        "note": (
            "Ground truth is the exact oracle relation. Evidence is from "
            "fictional structured sources, not actual media/tool inspection. "
            "A Macro may produce an intermediate relation rather than the "
            "final overall question answer."
        ),
    }


def _readable_value(value) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if value is None:
        return "not specified"
    return str(value)


def _readable_verified_answer(rows: list[dict]) -> str:
    """Render the EXACT Oracle relation without needing an LLM rewrite.

    All machine-readable rows remain separately available as answer.
    This sentence can be shown directly to Main-Agent.
    """
    if not rows:
        return "No matching records were found."
    rendered = []
    for row in rows:
        if set(row) == {"entry_label"}:
            rendered.append(_readable_value(row["entry_label"]))
        elif set(row) == {"count"}:
            rendered.append(_readable_value(row["count"]))
        elif set(row) == {"category_name", "count"}:
            rendered.append(
                f"{_readable_value(row['category_name'])}: "
                f"{_readable_value(row['count'])}"
            )
        else:
            rendered.append("; ".join(
                f"{key}: {_readable_value(value)}"
                for key, value in row.items()
            ))
    return (
        f"{len(rows)} row(s) returned: " + " | ".join(rendered)
    )


def _normalize_author_result(value, exact_rows: list[dict],
                             output_schema: list[str]) -> tuple[str | None, list[str]]:
    """Accept the natural real-world variation of Sub-Agent result payloads.

    The model can write prose OR provide exact structured records. Any
    structured payload MUST be an exact multiset match to the frozen Oracle.
    A filtered subset, reordered field meaning, or fabricated rows FAIL.
    """
    if isinstance(value, str):
        if not 12 <= len(value.strip()) <= 1600:
            return None, ["result_invalid_length_or_type"]
        return value.strip(), []
    if isinstance(value, dict):
        proposed = [value]
    elif isinstance(value, list):
        proposed = value
    elif (type(value) in {int, float, bool}
          and output_schema == ["count"]):
        proposed = [{"count": value}]
    else:
        return None, ["result_invalid_type"]
    if (len(output_schema) == 1 and proposed
            and all(not isinstance(row, dict) for row in proposed)):
        proposed = [{output_schema[0]: row} for row in proposed]
    if not all(isinstance(row, dict) for row in proposed):
        return None, ["structured_result_invalid_rows"]
    if sorted(map(_canonical, proposed)) != sorted(map(_canonical, exact_rows)):
        return None, ["structured_result_not_equal_to_oracle"]
    return _readable_verified_answer(exact_rows), []


_ASSERTED_FILTER_ACTION = re.compile(
    r"\b(?:filtered|excluded|removed|dropped|"
    r"screened\s+out|retained\s+only|"
    r"filtering\s+out|applied\s+(?:an?\s+)?filter)\b",
    re.I,
)
_FILTER_NEGATION_OR_DEFERMENT = re.compile(
    r"\b(?:without|not|never|no|none|neither|"
    r"did\s+not|was\s+not|were\s+not|has\s+not|have\s+not|"
    r"before|prior\s+to|pending|awaiting|rather\s+than|"
    r"left\s+for|reserved\s+for|deferred\s+to|"
    r"for\s+(?:later|future|subsequent|downstream))"
    r"(?:\s+[\w-]+){0,7}\s*$",
    re.I,
)


def _scan_claims_completed_filter(text: str) -> bool:
    """Only flag *asserted* filtering during a read-only source scan.

    References to a FUTURE filtering step or negated operations (e.g.
    "without filtering" and "no rows were excluded") are NOT claims
    that the current Macro filtered anything. Prior lexical matching
    incorrectly rejected such honest stage-boundary summaries.
    """
    for action in _ASSERTED_FILTER_ACTION.finditer(text):
        prefix = text[max(0, action.start() - 115):action.start()]
        prefix = re.split(r"[.;:!?\n]", prefix)[-1]
        if _FILTER_NEGATION_OR_DEFERMENT.search(prefix):
            continue
        # "Rows to be filtered in the next stage" describes intended
        # future work rather than work the current scan completed.
        if re.search(r"\b(?:to\s+be|will\s+be|would\s+be|"
                     r"can\s+be|may\s+be)\s*$", prefix, re.I):
            continue
        return True
    return False


def _stage_scope_issues(text: str, contract: dict) -> list[str]:
    """High-precision claims contradicted by the frozen Macro output/ops.

    A narrow evidence guard, not a complete semantic entailment checker.
    Only reject explicit strong claims with no present execution support.
    """
    rows = contract["exact_oracle_answer"]
    operations = set(contract["executed_operations_in_this_macro"])
    fields = set(contract["terminal_output_schema_of_this_macro"])
    terminal = contract["macro_is_terminal"]
    errors = []
    if "active" in fields and any(row.get("active") is False for row in rows):
        if re.search(
            r"\b(?:all|every|each)\b.{0,85}\bactive\b"
            r"|\bonly\s+active\s+(?:entries|records|rows)\b"
            r"|\b(?:records|entries|rows)\s+(?:are|were)\s+all\s+active\b",
            text, re.I,
        ):
            errors.append("claims_all_active_with_inactive_rows")
    if "approved" in fields and any(row.get("approved") is False for row in rows):
        if re.search(
            r"\b(?:all|every|each)\b.{0,85}\bapproved\b"
            r"|\bonly\s+approved\s+(?:entries|records|rows|categories)\b",
            text, re.I,
        ):
            errors.append("claims_all_approved_with_unapproved_rows")

    # A standalone source read cannot have filtered data. Upstream
    # workproducts are not involved in scan-only Macro Tasks.
    if operations == {"scan"} and not contract["upstream_handoffs"]:
        if _scan_claims_completed_filter(text):
            errors.append("scan_claims_unexecuted_filter")
        if "active" not in fields and re.search(
            r"\b(?:scanned|retrieved|extracted|identified|listed)\b"
            r".{0,80}\bactive\s+(?:entries|records|items)\b",
            text, re.I,
        ):
            errors.append("scan_claims_unobserved_active_status")
        if "approved" not in fields and re.search(
            r"\b(?:scanned|retrieved|extracted|identified)\b"
            r".{0,100}\bapproved\s+(?:translated\s+)?"
            r"(?:category\s+)?(?:labels|entries|records)\b",
            text, re.I,
        ):
            errors.append("scan_claims_unobserved_approval")
        if "cleared" not in fields and re.search(
            r"\b(?:satisfied|met|verified|passed)\b"
            r".{0,65}\bclearance\b",
            text, re.I,
        ):
            errors.append("scan_claims_unchecked_clearance")

    # An intermediate join that has no interval boundaries and no
    # compare step cannot establish that temporal conditions passed.
    if (not terminal and "compare_fields" not in operations
            and not any("step_start" in f or "step_end" in f
                        for f in fields)):
        if re.search(
            r"\b(?:verified|confirmed|checked|satisfied|passed)\b"
            r".{0,95}\b(?:event[\s_-]*step\s+(?:constraints|criteria)|"
            r"interval\s+(?:bounds|constraints)|"
            r"session\s+(?:window|interval)|time\s+(?:window|bounds))\b",
            text, re.I,
        ):
            errors.append("claims_unexecuted_temporal_verification")
    return sorted(set(errors))


def _guard(candidate: dict, contract: dict) -> list[str]:
    errors = []
    if not isinstance(candidate, dict):
        return ["report_not_object"]
    if candidate.get("status") != "done":
        errors.append("status_not_done_for_complete_oracle")
    result_text, result_errors = _normalize_author_result(
        candidate.get("result"), contract["exact_oracle_answer"],
        contract["terminal_output_schema_of_this_macro"],
    )
    errors.extend(result_errors)
    summary = candidate.get("summary")
    if (not isinstance(summary, str)
            or not 25 <= len(summary.strip()) <= 750):
        errors.append("summary_invalid_length_or_type")
    for field, value in (("result", result_text), ("summary", summary)):
        if not isinstance(value, str):
            continue
        if _TOOL_HALLUCINATION.search(value):
            errors.append(field + "_claims_unperformed_tool_or_media_action")
        if any(ref not in contract["all_evidence_refs"]
               for ref in _FACT_ID.findall(value)):
            errors.append(field + "_invented_fact_ref")
        if re.search(r"\b(?:gold program|teacher world|oracle|macro taskgraph|"
                     r"synthetic ground truth)\b", value, re.I):
            errors.append(field + "_internal_teacher_jargon")
        errors.extend(
            field + "_" + issue
            for issue in _stage_scope_issues(value, contract)
        )
    return sorted(set(errors))


def author_critic_report(
    teacher_record: dict, *, question: str | None,
    author, critic, max_revisions: int = 2,
    is_terminal: bool = False,
) -> tuple[dict | None, dict]:
    """Generate, check and repair a Main-facing finish result.

    Failure is FAILURE, never a deterministic report silently counted as
    accepted. The oracle output and source trace are never LLM-edited.
    """
    if type(max_revisions) is not int or not 0 <= max_revisions <= 8:
        raise ValueError("max_revisions must be 0..8")
    contract = _teacher_contract(
        teacher_record, question=question, is_terminal=is_terminal
    )
    contract["all_evidence_refs"] = sorted({
        ref for step in teacher_record["trace_summary"]["steps"]
        for ref in step["supporting_fact_refs"]
    } | set(teacher_record["trace_summary"]["answer_supporting_fact_refs"]))
    failures = []
    for revision in range(max_revisions + 1):
        request = {
            "frozen_task_and_evidence": contract,
            "previous_review_feedback": [{
                "stage": f.get("stage"),
                "errors": f.get("errors", []),
                "reason": f.get("reason"),
                "repair_hint": (
                    "Revise the prose to describe only operations actually "
                    "executed by this Macro; leave ALL exact Oracle answer "
                    "rows unchanged. Do not apply future-stage filters."
                    if f.get("stage") == "critic" else None
                ),
            } for f in failures[-2:]],
            "style": (
                "Act like a real Sub-Agent returning finish to Main-Agent. "
                "result = actionable finding; summary = actual work and evidence. "
                "Describe ONLY this Macro's executed operations and stage-local "
                "findings. Preserve ALL exact Oracle answer rows including "
                "candidates that may fail later filters. When repairing, "
                "correct only prose, not the answer."
            ),
        }
        try:
            candidate = author.complete_json(
                AUTHOR_SYSTEM, json.dumps(request, ensure_ascii=False)
            )
            problems = _guard(candidate, contract)
            if problems:
                failures.append({
                    "stage": "guard", "errors": problems,
                    "preview": str(candidate)[:220],
                    "candidate_result_type": type(
                        candidate.get("result")
                    ).__name__ if isinstance(candidate, dict) else None,
                    # Keep full short natural-language text only in the
                    # PRIVATE per-Macro review audit. The old preview lost
                    # summaries behind long structured result arrays.
                    "candidate_summary": (
                        candidate["summary"][:1000]
                        if isinstance(candidate, dict)
                        and isinstance(candidate.get("summary"), str)
                        else None
                    ),
                })
                continue
            normalized_result, _ = _normalize_author_result(
                candidate["result"], contract["exact_oracle_answer"],
                contract["terminal_output_schema_of_this_macro"],
            )
            decision = critic.complete_json(
                CRITIC_SYSTEM, json.dumps({
                    "frozen_task_and_evidence": contract,
                    "candidate_finish": {
                        "result": normalized_result,
                        "summary": candidate["summary"],
                        "status": "done",
                    },
                    "author_result_was_structured": not isinstance(
                        candidate["result"], str
                    ),
                }, ensure_ascii=False)
            )
            axes = (
                "grounded", "answer_scope_preserved",
                "evidence_scope_respected", "natural",
            )
            if (decision.get("verdict") == "pass"
                    and all(decision.get(axis) is True for axis in axes)):
                result_text = normalized_result
                summary_text = candidate["summary"].strip()
                # Structured reports have been checked against every exact
                # answer row. For prose reports, programmatically append the
                # complete canonical answer (never the LLM's guessed rows).
                verified_result = (
                    result_text
                    if not isinstance(candidate["result"], str)
                    else result_text + "\n\nVerified result: "
                         + _canonical(teacher_record["answer"])
                )
                row = deepcopy(teacher_record)
                row["answer_text"] = verified_result
                row["trace_summary"]["narrative"] = summary_text
                row["finish_result"] = {
                    "result": verified_result,
                    "status": "done",
                    "summary": summary_text,
                }
                row["report_authoring"] = {
                    "status": "accepted",
                    "attempts": revision + 1,
                    "review": decision,
                    "exact_answer_immutable": True,
                    "grounding_note": (
                        "LLM narrative of deterministic oracle evidence, "
                        "NOT independently observed LLM Sub-Agent actions"
                    ),
                }
                return row, row["report_authoring"]
            failures.append({
                "stage": "critic", "reason": decision.get("reason"),
                "candidate_summary": candidate.get("summary"),
                "repair_hint": (
                    "Correct stage-local summary/prose only. Preserve the "
                    "entire frozen answer exactly; do not pre-filter or "
                    "alter any result rows."
                ),
                "axes": {key: decision.get(key) for key in axes},
                "preview": str(candidate)[:220],
            })
        except Exception as exc:
            failures.append({
                "stage": "backend",
                "error_type": type(exc).__name__,
                "error": str(exc)[:220],
            })
    return None, {
        "status": "failed",
        "attempts": max_revisions + 1,
        "failures": failures,
        "exact_answer_immutable": True,
    }
