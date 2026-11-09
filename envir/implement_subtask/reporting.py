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
    return errors


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
            "previous_review_feedback": failures[-2:],
            "style": (
                "Act like a real Sub-Agent returning finish to Main-Agent. "
                "result = actionable finding; summary = actual work and evidence. "
                "Do not add facts beyond the frozen observations."
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
                "repair_hint": decision.get("repair_hint"),
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
