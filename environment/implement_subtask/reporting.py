"""Turn frozen Oracle Macro evidence into realistic Sub-Agent finish reports.

The LLM may only verbalize results. Exact typed answers, source fact IDs,
operation traces, and completion status are immutable teacher-owned data.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re

from environment.generate_question.authoring.surface_quality import SCHEMA_LABELS
from .stage_local_claims import (
    compile_claim_contract, stage_local_claim_issues,
    normalize_report_language,
)

PRESENTATION = "clean_finish_v2"


AUTHOR_SYSTEM = """You are writing the FINAL message that a real specialized
Sub-Agent would return to its Main-Agent after completing a delegated task.

Follow the actual multi-agent finish convention:
  {"result": "...", "summary": "...", "status": "done"}

"result": put the actionable finding first in a short sentence. For a large
handoff, say what the records contain and give the most useful grounded detail.
Full verified records are carried separately. Avoid row dumps, raw field names
and generic "rows returned" language.
"summary": describe the local work and the supporting observation in 1–2
natural sentences. Distinguish findings established here from findings received
upstream. If this stage only projects fields or removes duplicates, say what
you did with the upstream result and avoid implying a fresh eligibility check.
Use source roles and display labels. Refer to earlier results by their role
(e.g. the mapped entries), without printing work_* identifiers, operator names,
or names of raw fields. Keep the report useful for a colleague
taking the next step.

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
criteria. If this stage runs a keyed lookup, use the EXACT input key field and
values in observed_operator_steps.lookup_keys. Do not claim that the archive
was queried using an earlier-stage marker when the actual archive query
used an accession identifier.
If a reviewer objects to a premature claim, correct ONLY the prose
summary, never the result rows.

Prefer a concise grounded string for result. Exact structured answer rows
remain accepted if provided, but the runtime carries them independently of the
prose. Preserve every exact row in the structured form and use the summary for
a short explanation.
Output ONLY JSON with keys result, summary, status; status must be done.

Examples of valid finish messages:
{"result":"Entry-02 is the only entry within its assigned session window.",
 "summary":"I compared the recorded event steps with the session boundaries. Entry-02 occurred at step 68, inside its session's 63–88 interval.","status":"done"}
{"result":"Entry-02 remains after removing duplicates.",
 "summary":"I deduplicated the entry labels received from the earlier stage. The returned list has one distinct entry.","status":"done"}
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
For keyed retrievals, check observed_operator_steps.lookup_keys: the
query key used against a source must be the one actually supplied to that
source, not an earlier upstream marker.
For judging a structured result, treat exact_oracle_answer as the
COMPLETE authoritative relation. sampled_evidence_atoms is an abbreviated
display for explanation, NOT an exhaustive list of records. NEVER call
an exact verified result fabricated just because its rows are absent
from the sampled evidence atoms. Critique unsupported claims in the
summary instead, using exact_oracle_answer as ground truth.
For a structured candidate result, the application has already verified
an EXACT match to the full frozen answer rows; do not reject it merely
because it is rendered as a list/array. Review the proposed natural
summary and any prose for actual factual errors.
The exact Oracle rows travel separately from the finish prose. Do not demand
that result repeat all columns or that the summary enumerate raw fact IDs.
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
from .oracle_review import (
    full_entry_list_matches, entry_fact_index, positive_entry_provenance,
    mismatched_entry_refs, critic_oracle_contradiction,
)

VERIFIED_ENTRY_CRITIC_SYSTEM = CRITIC_SYSTEM + """
THE COMPLETE ENTRY LIST IS DETERMINISTICALLY VERIFIED AGAINST ORACLE.
Do not dispute enumeration membership or reassign FactRefs. The full
Entry/FactRef/active index is authoritative, not truncated sample atoms.
Review only naturalness, stage-local source claims and unsupported prose.
Reject genuinely ungrounded prose. Return the normal four boolean axes.
"""


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
        "stage_local_claim_contract": compile_claim_contract(record),
        "overall_question_for_context_only": question,
        "macro_task_id": record["macro_task_id"],
        "macro_is_terminal": is_terminal,
        "upstream_handoffs": record["depends_on"],
        "authorized_direct_sources": record["authorized_source_ids"],
        "available_provenance_sources": sorted(allowed),
        "terminal_output_schema_of_this_macro": record["output_schema"],
        "exact_oracle_answer": rows,
        "entry_fact_index": entry_fact_index(record),
        "positive_entry_provenance": positive_entry_provenance(record),
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
                "lookup_keys": step.get("lookup_keys"),
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


def _public_field_text(text: str, fields) -> str:
    names = sorted({field for field in fields if "_" in field})
    if not names:
        return text
    pattern = r"\b(?:" + "|".join(map(re.escape, names)) + r")\b"
    return re.sub(
        pattern, lambda match: SCHEMA_LABELS.get(
            match[0].lower(), match[0].lower().replace("_", " ")
        ), text, flags=re.I,
    )


def _readable_verified_answer(rows: list[dict]) -> str:
    """Summarize an exact relation. The full verified rows travel separately."""
    if not rows:
        return "No records matched the assigned task."
    if len(rows) == 1 and set(rows[0]) == {"count"}:
        return f"The count is {_readable_value(rows[0]['count'])}."
    if all(set(row) == {"entry_label"} for row in rows):
        return "Entries: " + ", ".join(_readable_value(row["entry_label"]) for row in rows) + "."
    if all(set(row) == {"category_name", "count"} for row in rows):
        return "Counts by category: " + ", ".join(
            f"{row['category_name']} ({row['count']})" for row in rows
        ) + "."
    count = len(rows)
    unit = "record" if count == 1 else "records"
    labels = list(dict.fromkeys(str(row["entry_label"]) for row in rows if "entry_label" in row))
    if labels and len(labels) <= 3:
        return f"The handoff contains {count} {unit} for " + ", ".join(labels) + "."
    if labels:
        return f"The handoff contains {count} {unit}, including " + ", ".join(labels[:3]) + "."
    return f"The handoff contains {count} {unit} from the assigned task."


_PAST_ACTION = {
    "Read": "Read", "Keep": "Kept", "Align": "Aligned", "Check": "Checked",
    "Order": "Ordered", "Select": "Selected", "Remove": "Removed",
    "Count": "Counted", "Attach": "Attached",
}


def trace_narrative(trace: dict) -> str:
    """Describe the actual operator events independently of finish prose."""
    events = []
    for step in trace["steps"]:
        action = step["action"]
        first, _, rest = action.partition(" ")
        action = (
            _PAST_ACTION[first] + " " + rest if first in _PAST_ACTION
            else "Executed " + step["operation"].replace("_", " ")
        )
        fields = set(step["output_schema"]) | set(SCHEMA_LABELS)
        action = _public_field_text(action, fields)
        action = action.replace("row(s)", "records")
        action = action.replace("Checked that", "Checked whether")
        action = action.replace("Kept only the output field(s): ", "Kept only ")
        join = re.fullmatch(r"Aligned the two input workproducts by (.+) = (.+)", action)
        if join:
            left, right = join.groups()
            action = (f"Aligned the inputs by their shared {left}" if left == right
                      else f"Matched {left} in the first input to {right} in the second")
        count = step["output_row_count"]
        unit = "record" if count == 1 else "records"
        events.append(f"{action}. This produced {count} {unit}.")
    return " ".join(events)


def _normalize_author_result(value, exact_rows: list[dict],
                             output_schema: list[str]) -> tuple[str | None, list[str]]:
    """Accept the natural real-world variation of Sub-Agent result payloads.

    The model can write prose OR provide exact structured records. Any
    structured payload MUST be an exact multiset match to the frozen Oracle.
    A filtered subset, reordered field meaning, or fabricated rows FAIL.
    """
    if isinstance(value, str):
        if not 1 <= len(value.strip()) <= 1600:
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


# Detect only a POSITIVE assertion that the current relation contains
# exclusively active rows. "all rows contain the active field", "all
# records, including inactive ones" and "all active/inactive statuses"
# are NOT such assertions. Older broad \b all ... active \b matching
# falsely rejected 42 otherwise accurate Author attempts in one pilot.
_ALL_ACTIVE_ASSERTION = re.compile(
    r"\b(?:all|every|each)\s+(?:(?:of\s+)?the\s+)?"
    r"(?:(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|returned|retrieved|matched|selected|output|candidate|qualifying)\s+){0,3}"
    r"(?:records|rows|entries|items|candidates)\s+"
    r"(?:are|were|is|have\s+been|has\s+been)\s+"
    r"(?:marked\s+as\s+|flagged\s+as\s+|confirmed\s+as\s+)?active\b"
    r"|\b(?:only|exclusively)\s+active\s+"
    r"(?:rows|records|entries|items|candidates)\b"
    r"|\b(?:all|every|each)\s+(?:record|row|entry|item|candidate)\s+"
    r"(?:is|was|has\s+been)\s+(?:marked\s+as\s+)?active\b"
    r"|\b(?:all|every|each)\s+"
    r"(?:(?:returned|retrieved|matched|selected)\s+)?"
    r"(?:records|rows|entries|items|candidates)\s+"
    r"(?:have|had)\s+active\s+(?:=|status\s+(?:=|of|as))\s*"
    r"(?:true|yes)\b"
    r"|\bactive\s*(?:=|equals|is)\s*(?:true|yes)\s+"
    r"(?:for|across|in)\s+all\b"
    r"|\b(?:returned|retained|selected|identified|produced|"
    r"extracted|includes?|consists\s+of|contains?)\s+"
    r"(?:only|exclusively)\s+(?:records|rows|entries|items|candidates)\s+"
    r"(?:that\s+(?:are|were)\s+)?(?:marked\s+as\s+)?active\b"
    r"|\b(?:records|rows|entries|items|candidates)\s+"
    r"(?:are|were)\s+all\s+active\b",
    re.I,
)


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
        if _ALL_ACTIVE_ASSERTION.search(text):
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
    # Some joins only align the codebook and annotations. Their provenance
    # has no register observation to support an asserted status disagreement.
    sources = contract.get("available_provenance_sources")
    if sources is not None and "register" not in sources and re.search(
        r"\b(?:contains?|returned|found|identified|retained|selected)\b[^.!?]{0,100}"
        r"\b(?:observed|reported|binary)\s+status\s+"
        r"(?:differs?|disagrees?|conflicts?)\s+(?:from|with)\s+"
        r"(?:the\s+)?(?:master\s+)?(?:register|ledger)\b",
        text, re.I,
    ):
        errors.append("claims_comparison_without_register_evidence")
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
        if re.search(r"\bwork_[a-z0-9_]+\b", value, re.I):
            errors.append(field + "_internal_macro_identifier")
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
        errors.extend(
            field + "_" + issue
            for issue in stage_local_claim_issues(
                value, contract.get("stage_local_claim_contract", {})
            )
        )
        if mismatched_entry_refs(
            value, contract.get("entry_fact_index", {}),
            contract.get("positive_entry_provenance"),
        ):
            errors.append(field + "_entry_fact_ref_mismatch")
        # Narrow, high-confidence check: claiming an archive was queried
        # *with the marker* is false when the verified lookup used a
        # downstream accession key. A marker mentioned as historical
        # context without this direct-query claim remains allowed.
        if any(
            step.get("lookup_keys")
            and step["lookup_keys"]["source_id"] == "archive"
            and step["lookup_keys"]["key_field"] != "marker_code"
            for step in contract["observed_operator_steps"]
        ):
            direct_marker = (
                r"\b(?:using|with|by)\s+(?:the\s+)?(?:upstream\s+)?"
                r"marker(?:\s+code)?\b[^.!?]{0,70}"
                r"\b(?:against|in|on)\s+(?:the\s+)?archive\b"
                r"|\barchive\s+(?:lookup|query)\b[^.!?]{0,65}"
                r"\b(?:using|with|by)\s+(?:the\s+)?(?:upstream\s+)?marker\b"
            )
            if re.search(direct_marker, value, re.I):
                errors.append(field + "_wrong_archive_lookup_key")
    return sorted(set(errors))




CODE_OWNED_AUTHOR_SYSTEM = """You are a specialized Sub-Agent describing
ONLY the work already performed for ONE delegated task. Return JSON with
ONLY {"summary":"..."} containing 1-2 concise, grounded sentences.

The program will supply Result and status from the current Macro's frozen
executed WorkProduct. DO NOT write a result or status.
You must respect the local operation and provenance contract supplied in
observed_operator_steps: a context-log filter yields context candidates,
NOT cases independently verified to have disagreeing report years.
Do not claim a downstream year disagreement or final eligibility condition
has been established until a step here explicitly checked it. A different
independent Macro may do that later.

Only describe local evidence and actual upstream handoffs. Do not invent
facts, sources, numeric values, checks or tool/media execution. The trace
represents simulated evidence; never claim to have actually searched
the web, watched a video or visited a real source. Empty results mean a
successfully executed step with no matching records, not a failed lookup.
"""

CODE_OWNED_CRITIC_SYSTEM = CRITIC_SYSTEM + """
candidate_finish.result is PROGRAM-OWNED and has already been rendered from
the complete frozen current Macro output. It was not model-authored. Review
ONLY the model-authored summary for grounding, stage-local evidence scope and
naturalness. Do not demand changes to the program-owned Result, add filters,
or infer the overall question was completed by an intermediate Macro.
Return the same four boolean axes.
"""


def code_owned_result(record: dict) -> str:
    """Render all delivered Macro rows as neutral, stage-local observations.

    Never replace a context candidate with a claimed year disagreement.
    Fail rather than truncate an oversized relation: hiding rows would
    corrupt downstream Main handoffs.
    """
    rows, fields = record["answer"], record["output_schema"]
    if record.get("result_encoding") == "typed_evidence_json_v1":
        # Main must receive the same complete typed rows and provenance that
        # Stage 4 later parses. The Summary remains natural-language prose.
        cited = record.get("answer_with_provenance")
        if (not isinstance(cited, list) or len(cited) != len(rows)
                or any(item.get("values") != row
                       for item, row in zip(cited, rows))):
            raise ValueError("code_owned_provenance_does_not_match_rows")
        result = json.dumps([{
            "values": item["values"],
            "source_fact_refs": list(item["source_fact_refs"]),
        } for item in cited], ensure_ascii=False, separators=(",", ":"))
        # Keep full typed evidence even for one-to-many observation logs.
        # Size is a separate prompt-budget concern, not a license to drop rows.
        return result
    if not isinstance(rows, list) or not fields:
        raise ValueError("code_owned_invalid_oracle_relation")
    if any(not isinstance(row, dict) or set(row) != set(fields)
           for row in rows):
        raise ValueError("code_owned_oracle_schema_mismatch")
    if not rows:
        return "No records matched the assigned task."
    if fields == ["case_id"]:
        text = (
            f"The current task returned {len(rows)} case identifiers: "
            + ", ".join(str(row["case_id"]) for row in rows) + "."
        )
    elif fields == ["entry_label"]:
        text = "Entries returned by this task: " + ", ".join(
            str(row["entry_label"]) for row in rows
        ) + "."
    else:
        def label(field: str) -> str:
            return SCHEMA_LABELS.get(field, field.replace("_", " "))
        def render(row: dict) -> str:
            return ", ".join(
                f"{label(field)}: {_readable_value(row[field])}"
                for field in fields
            )
        text = (
            f"The current task returned {len(rows)} record(s): "
            + "; ".join(render(row) for row in rows) + "."
        )
    if len(text) > 1600:
        raise ValueError("code_owned_result_exceeds_finish_budget")
    return text

def author_critic_report(
    teacher_record: dict, *, question: str | None,
    author, critic, max_revisions: int = 2,
    is_terminal: bool = False,
    result_policy: str = "llm_authored",
) -> tuple[dict | None, dict]:
    """Generate, check and repair a Main-facing finish result.

    Failure is FAILURE, never a deterministic report silently counted as
    accepted. The oracle output and source trace are never LLM-edited.
    """
    if type(max_revisions) is not int or not 0 <= max_revisions <= 8:
        raise ValueError("max_revisions must be 0..8")
    if result_policy not in {"llm_authored", "code_owned"}:
        raise ValueError("invalid_report_result_policy")
    contract = _teacher_contract(
        teacher_record, question=question, is_terminal=is_terminal
    )
    contract["all_evidence_refs"] = sorted({
        ref for step in teacher_record["trace_summary"]["steps"]
        for ref in step["supporting_fact_refs"]
    } | set(teacher_record["trace_summary"]["answer_supporting_fact_refs"]))
    frozen_result = (
        code_owned_result(teacher_record)
        if result_policy == "code_owned" else None
    )
    author_contract = deepcopy(contract)
    if result_policy == "code_owned":
        # The overall question often mentions downstream conditions that
        # are not established in the current source-only Macro.
        author_contract.pop("overall_question_for_context_only", None)
    failures = []
    for revision in range(max_revisions + 1):
        request = {
            "frozen_task_and_evidence": author_contract,
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
                CODE_OWNED_AUTHOR_SYSTEM if result_policy == "code_owned"
                else AUTHOR_SYSTEM,
                json.dumps(request, ensure_ascii=False),
            )
            ignored_model_result = bool(
                isinstance(candidate, dict)
                and result_policy == "code_owned"
                and "result" in candidate
            )
            if isinstance(candidate, dict) and result_policy == "code_owned":
                # Neither a model's wrong result nor an invented status
                # may overwrite the actual delivered WorkProduct.
                candidate = {
                    "result": frozen_result,
                    "summary": candidate.get("summary"),
                    "status": candidate.get("status", "done"),
                }
            surface_rewrites = []
            if isinstance(candidate, dict):
                fields = {
                    field for step in teacher_record["trace_summary"]["steps"]
                    for field in step["output_schema"]
                } | set(SCHEMA_LABELS) | {
                    step["operation"] for step in teacher_record["trace_summary"]["steps"]
                }
                candidate = {
                    key: _public_field_text(value, fields)
                    if (key in {"result", "summary"}
                        and isinstance(value, str)
                        and not (key == "result"
                                 and result_policy == "code_owned"
                                 and teacher_record.get("result_encoding") ==
                                 "typed_evidence_json_v1")) else value
                    for key, value in candidate.items()
                }
                # Surface-only repair of known internal jargon. Preserve
                # claims, rows and provenance; all subsequent guards and
                # Critic review still run on the revised prose.
                for field in ("result", "summary"):
                    if (field == "result" and result_policy == "code_owned"
                            and teacher_record.get("result_encoding") ==
                            "typed_evidence_json_v1"):
                        # Never rewrite machine-readable JSON keys or refs.
                        continue
                    if isinstance(candidate.get(field), str):
                        revised, replacements = normalize_report_language(
                            candidate[field]
                        )
                        candidate[field] = revised
                        surface_rewrites.extend(
                            {"field": field, "pattern": x}
                            for x in replacements
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
            verified_entry_list = full_entry_list_matches(
                normalized_result, contract["exact_oracle_answer"],
            )
            decision = critic.complete_json(
                CODE_OWNED_CRITIC_SYSTEM if result_policy == "code_owned"
                else (VERIFIED_ENTRY_CRITIC_SYSTEM if verified_entry_list
                      else CRITIC_SYSTEM), json.dumps({
                    "frozen_task_and_evidence": contract,
                    "candidate_finish": {
                        "result": normalized_result,
                        "summary": candidate["summary"],
                        "status": "done",
                    },
                    "author_result_was_structured": not isinstance(
                        candidate["result"], str
                    ),
                    "result_policy": result_policy,
                }, ensure_ascii=False)
            )
            axes = (
                "grounded", "answer_scope_preserved",
                "evidence_scope_respected", "natural",
            )
            corrected_false_negative = False
            false_claim = (
                critic_oracle_contradiction(
                    decision, normalized_result,
                    contract["exact_oracle_answer"],
                    contract["entry_fact_index"],
                    contract["positive_entry_provenance"],
                ) if verified_entry_list and decision.get("verdict") != "pass"
                else None
            )
            if false_claim:
                # A known factual error in critic reasoning merits one narrow
                # independent re-review. It never auto-accepts the report.
                decision = critic.complete_json(
                    VERIFIED_ENTRY_CRITIC_SYSTEM, json.dumps({
                        "frozen_task_and_evidence": contract,
                        "result_was_deterministically_verified": True,
                        "verified_full_result": normalized_result,
                        "candidate_summary_for_review": candidate["summary"],
                        "verified_critic_error": false_claim,
                    }, ensure_ascii=False)
                )
                corrected_false_negative = (
                    decision.get("verdict") == "pass"
                    and all(decision.get(axis) is True for axis in axes)
                )
            if (decision.get("verdict") == "pass"
                    and all(decision.get(axis) is True for axis in axes)):
                summary_text = candidate["summary"].strip()
                row = deepcopy(teacher_record)
                row["answer_text"] = normalized_result
                row["trace_summary"]["narrative"] = trace_narrative(
                    teacher_record["trace_summary"]
                )
                # Code-owned, source-scoped observations are added AFTER
                # the LLM's natural-language prose has been independently
                # reviewed. They are taken from THIS Macro's delivered answer
                # rows, never from an unseen Source or downstream Gold.
                from .visible_evidence import witnessed_source_summary
                witnessed = (
                    witnessed_source_summary(teacher_record)
                    if teacher_record.get("evidence_trust_version") else ""
                )
                row["finish_result"] = {
                    "result": normalized_result,
                    "status": "done",
                    "summary": (
                        summary_text + " " + witnessed
                        if witnessed else summary_text
                    ),
                }
                row["report_authoring"] = {
                    "status": "accepted",
                    "result_policy": result_policy,
                    "ignored_model_result": ignored_model_result,
                    "attempts": revision + 1,
                    "surface_rewrites": surface_rewrites,
                    "review": decision,
                    "critic_oracle_contradiction_rechecked":
                        corrected_false_negative,
                    "exact_answer_immutable": True,
                    "presentation": PRESENTATION,
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

# Stage-local claims are checked against each Macro's own executed operations.
# A source scan cannot assert cross-report disagreement merely because the
# overall question mentions conflicting reports. A later compare_fields(ne)
# Macro may establish it after the two sources are joined.
