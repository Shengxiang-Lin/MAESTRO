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

PRESENTATION = "o1_answer_and_trace_v4"


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
    entry_fact_index, positive_entry_provenance, mismatched_entry_refs,
)

def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _source_scoped_filter_steps(steps: list[dict]) -> list[dict]:
    """Attribute a local filter to the most recent direct evidence source.

    Source identifiers live on source-reading operations. Subsequent
    filter/project operations usually have source_id=None, so a Critic
    looking only at the LAST filter can misattribute earlier filters.
    Join/aggregation breaks the single-source scope; never guess a source
    for those downstream operations.
    """
    active_source = None
    scoped = []
    for i, step in enumerate(steps, 1):
        operation = step["operation"]
        if step.get("source_id") is not None:
            active_source = step["source_id"]
        elif operation in {
            "join", "unique", "count_aggregate", "compare_select",
            "derive_comparison", "derive_percentage", "goal_compute",
            "goal_format",
        }:
            active_source = None
        if active_source is not None and operation.startswith("filter_"):
            scoped.append({
                "step_number": i,
                "source_id": active_source,
                "condition": step["action"],
            })
    return scoped


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
        # Grounded names and source roles come from Stage 1, not SQL op IDs.
        "source_context": trace.get("source_context", []),
        "result_overview": {
            "row_count": len(rows),
            "sample_rows": rows[:3],
            "additional_rows_not_shown": max(0, len(rows) - 3),
        },
        "terminal_output_schema_of_this_macro": record["output_schema"],
        "exact_oracle_answer": rows,
        "entry_fact_index": entry_fact_index(record),
        "positive_entry_provenance": positive_entry_provenance(record),
        "executed_operations_in_this_macro": [
            step["operation"] for step in steps
        ],
        # Review-only disambiguation: each checked condition is tied to
        # the source that was actually read, not the last filter in a list.
        "source_scoped_filter_steps": _source_scoped_filter_steps(steps),
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
                "source_id": step.get("source_id"),
                "source_title": next((
                    src["title"] for src in trace.get("source_context", [])
                    if src["source_id"] == step.get("source_id")
                ), None),
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


def _o1_answer_requirements(record: dict) -> list[str]:
    """Distinct delivered scalar values a natural answer must preserve."""
    values = []
    def collect(value):
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, dict):
            for member in value.values():
                collect(member)
            return
        if isinstance(value, (tuple, list)):
            for member in value:
                collect(member)
            return
        if isinstance(value, (str, int, float)):
            literal = str(value).strip()
            if literal and literal not in values:
                values.append(literal)
    for row in record["answer"]:
        collect(row)
    return values


def _exact_single_field_scalar_text(value, contract: dict) -> str | None:
    """Canonical text only for a one-row, one-field, exact Oracle scalar.

    Intermediate Macro handoffs often return an opaque PRJ-/ENT-/EVT- key,
    and terminal answers may be short dates or numbers. A word count is not
    a meaningful evidence-quality test for either. This does NOT authorize
    shortened multi-field/multi-row results or altered identifiers.
    """
    rows = contract.get("exact_oracle_answer")
    if (not isinstance(rows, list) or len(rows) != 1
            or not isinstance(rows[0], dict) or len(rows[0]) != 1
            or type(value) not in {str, int, float}):
        return None
    original = next(iter(rows[0].values()))
    if type(original) not in {str, int, float}:
        return None
    exact = str(original).strip()
    return exact if exact and str(value).strip() == exact else None


def _exact_structured_o1_answer(value, contract: dict) -> str | None:
    """Render an EXACT typed Oracle match as the O1 finish(result) string.

    Current task-plan terminal goals often have a single outer 'answer'
    field whose value is a multi-field dict. Qwen may emit that inner dict
    instead of the requested prose. Never silently repair a merely
    similar/subset dict, wrong value/type or invented extra field.
    """
    rows = contract.get("exact_oracle_answer")
    if not isinstance(rows, list) or not all(
        isinstance(row, dict) for row in rows
    ):
        return None
    proposed = None
    if type(value) is dict:
        if (len(rows) == 1 and set(rows[0]) == {"answer"}
                and type(rows[0]["answer"]) is dict
                and _canonical(value) == _canonical(rows[0]["answer"])):
            proposed = [rows[0]]
        else:
            proposed = [value]
    elif type(value) is list and all(type(row) is dict for row in value):
        proposed = value
    else:
        return None
    if sorted(map(_canonical, proposed)) != sorted(map(_canonical, rows)):
        return None

    def format_row(row):
        if set(row) == {"answer"} and type(row["answer"]) is dict:
            row = row["answer"]
        return ", ".join(
            f"{SCHEMA_LABELS.get(field, field.replace('_', ' '))}: "
            f"{_readable_value(data)}"
            for field, data in row.items()
        )

    text = "; ".join(format_row(row) for row in proposed)
    if (not text or len(text) > 6500
            or _o1_answer_issues(text, contract)):
        return None
    return text


def _o1_answer_issues(answer: str, contract: dict) -> list[str]:
    if not isinstance(answer, str):
        return ["answer_invalid_length_or_type"]
    answer_text = answer.strip()
    # O1 finish(result) may be an exact key, short number, date, or concise
    # answer. Length is not evidence quality. The literal/association,
    # provenance and independent Critic checks below still apply.
    if not answer_text or len(answer_text) > 6500:
        return ["answer_invalid_length_or_type"]
    missing = []
    for literal in contract.get("answer_required_literals", []):
        # A trailing sentence period is punctuation, not part of an
        # identifier. For numbers only, reject decimal continuations.
        number = re.fullmatch(r"-?\d+(?:\.\d+)?", literal)
        left = r"(?<![A-Za-z0-9_.-])" if number else r"(?<![A-Za-z0-9_-])"
        right = (r"(?![A-Za-z0-9_-]|\.\d)" if number
                 else r"(?![A-Za-z0-9_-])")
        pattern = left + re.escape(literal) + right
        if not re.search(pattern, answer, re.I):
            missing.append(literal)
    return (
        ["answer_missing_required_values:" + ", ".join(missing[:8])]
        if missing else []
    )


def _guard(candidate: dict, contract: dict) -> list[str]:
    errors = []
    if not isinstance(candidate, dict):
        return ["report_not_object"]
    if candidate.get("status") != "done":
        errors.append("status_not_done_for_complete_oracle")
    if contract.get("requires_o1_answer"):
        result_text = candidate.get("result")
        result_errors = _o1_answer_issues(result_text, contract)
    elif contract.get("result_encoding") == "typed_evidence_json_v1":
        result_text = candidate.get("result")
        result_errors = (
            [] if result_text == contract.get("code_owned_expected_result")
            else ["code_owned_typed_result_changed"]
        )
    else:
        result_text, result_errors = _normalize_author_result(
            candidate.get("result"), contract["exact_oracle_answer"],
            contract["terminal_output_schema_of_this_macro"],
        )
    errors.extend(result_errors)
    summary = candidate.get("summary")
    if (not isinstance(summary, str)
            or not 25 <= len(summary.strip()) <= 750):
        errors.append("summary_invalid_length_or_type")
    narrative = (candidate.get("trace_summary")
                 if contract.get("requires_o1_answer")
                 else candidate.get("trace_narrative"))
    if contract.get("requires_o1_answer"):
        if (not isinstance(narrative, str)
                or not 75 <= len(narrative.strip()) <= 2800):
            errors.append("trace_summary_invalid_length_or_type")
    elif contract.get("requires_natural_trace"):
        if (not isinstance(narrative, str)
                or not 60 <= len(narrative.strip()) <= 1800):
            errors.append("trace_narrative_invalid_length_or_type")
    for field, value in (
        ("result", result_text), ("summary", summary),
        *((("trace_summary", narrative),) if
          contract.get("requires_o1_answer") else
          ((("trace_narrative", narrative),) if
           contract.get("requires_natural_trace") else ())),
    ):
        if not isinstance(value, str):
            continue
        if re.search(r"\bwork_[a-z0-9_]+\b", value, re.I):
            errors.append(field + "_internal_macro_identifier")
        if (contract.get("requires_natural_trace")
                or contract.get("requires_o1_answer")):
            if re.search(
                r"\b(?:macro(?:s|\s+task)?|workproducts?|"
                # 'project' alone names a real source/domain entity, e.g.
                # 'Project event ledger' and 'project identifier'. Ban only
                # verb/adjective operator forms, not the domain noun.
                r"scan(?:ned)?|project(?:ed|ion)|join(?:ed)?|uniqued|"
                r"lookup_by_key|filter_eq|filter_gte|row\(s\)|"
                r"m\d+_op\d+)\b",
                value, re.I,
            ):
                errors.append(field + "_technical_execution_jargon")
        # Distinguish public source handles (e.g. M3_a, relation_2)
        # from invented or private identifiers (e.g. M3_op01). Only IDs
        # explicitly carried by this Macro's visible evidence sources pass.
        visible_source_ids = (
            set(contract.get("available_provenance_sources", []))
            & {
                source.get("source_id")
                for source in contract.get("source_context", [])
                if isinstance(source, dict)
            }
        )
        for source_token in re.findall(
            r"\b(?:M\d+_[a-z]\w*|relation_\d+)\b", value, re.I,
        ):
            if (re.fullmatch(r"M\d+_op\d+", source_token, re.I)
                    or source_token not in visible_source_ids):
                errors.append(field + "_technical_execution_jargon")
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




SUBAGENT_REPORT_AUTHOR_SYSTEM = """You are a specialist Sub-Agent sending
the completed delegated-task feedback to the Main-Agent. Use the provided
verified task-local evidence; the source observations are simulated.

Return ONLY a JSON object with THREE authored text fields:
{"answer":"...", "summary":"...", "trace_summary":"..."}

ANSWER: directly answer the DELEGATED task. State the useful finding FIRST.
For ANY one-field task, including intermediate identifier lookups,
you may return the exact single value directly (e.g. PRJ-/ENT-/EVT- ID,
number, date, choice or place name); use TRACE_SUMMARY for explanation.
Do not pad a correct short identifier with invented facts merely to make
the answer longer.
All distinct entity identifiers, dates, measurements, scalar fields, and
entity-to-value associations in exact_oracle_answer must be preserved
verbatim in the answer, including when there are many rows. Use compact
readable lists rather than machine JSON, and include the number of matched
records when applicable. If you need to compare filters from multiple
sources, associate each condition with the specific source it came from.
If this is an intermediate task, never claim future eligibility,
comparisons or checks were completed. The complete evidence/provenance
rows are held separately by the application.

SUMMARY: 1-3 concise sentences describing THIS delegated task: what
evidence or observations were examined, what was established or returned,
and any important unresolved limitation if one actually remains. Include
the key result (e.g. an identifier, date, count or measured value) where
it is useful to Main. The summary is a short finish handoff, NOT a
copy of the first trace bullet, a generic "task complete" sentence,
or a description of work that belongs to a later task. Base every claim
on the same current-Macro evidence as ANSWER and TRACE_SUMMARY. Do not
invent tool actions, external media access or uncertainty.

TRACE_SUMMARY: write a grounded, chronological execution recap in
concise dash-prefixed bullet points (typically 2-8, as needed by task
complexity). Simple lookups may need only a few points; multi-source
investigations should identify the evidence used, comparisons or
calculations actually carried out, returned findings and any unresolved
limitations. A clean, successfully completed investigation can naturally
describe progress in affirmative language (for example, that a clue was
successfully recovered, a value was correctly identified, or the requested
handoff was completed smoothly). A brief, proportionate self-assessment
that no problem or apparent inconsistency was encountered is acceptable
when the supplied evidence and trace support it. Such language is OPTIONAL:
vary wording freely; do not copy a fixed success checklist, repeat the
same boilerplate in every bullet, or add success claims solely for length.
Never claim to have run a hallucination detector, audited all possible
sources, used unprovided tools, measured efficiency or verified downstream
tasks. Do not claim a trouble-free outcome when the current evidence
shows genuine gaps or unresolved contradictions.
Use descriptive source titles, modalities and observed_content from
the provided source_context and sampled_evidence_atoms. The source
observations are SIMULATED: you may describe 'the supplied photograph
annotation' or 'recording description' but NEVER claim to have actually
watched video, listened to audio, searched the web, visited pages or
used a tool. Only state tool execution when tool_executed=true and
the trace itself confirms it. Do not invent errors, retries or omissions.

None of these fields may expose raw SQL/graph operator names, field-schema
labels, Macro/work_M identifiers, FactRefs or 'workproduct', 'row(s)'.
Describe the observation in ordinary investigation language: "reviewed
the supplied annotations", "cross-referenced the candidate lists", and
"retained the requested values". Do NOT narrate SQL-style actions as
"scanned", "joined", "projected", "rows" or "workproducts". Prefer public source titles over cryptic identifiers; when multiple
visible sources need disambiguation, you may reference a source ID that
appears in authorized source_context (such as M3_a versus M3_b), but
never invent source IDs or expose private operator IDs. Never claim
validation against an Oracle, private ground truth or an output schema.
Natural source titles such as "Project event ledger" and phrases such as
"project identifier" ARE legitimate domain names, not SQL projection.
Preserve source titles instead of avoiding the noun "project".
Keep evidence at its actual grain: an observation count grouped by an
entity/location is not necessarily a count belonging to an upstream
dossier. Do not reattribute measured results to a dossier, project or
other entity unless the supplied observation explicitly links them.
Avoid mechanical repetition. Natural expressions of successful progress
and completion are welcome when supported by THIS Macro's observations;
prefer specific findings useful for Main's next decision over stock praise.

STYLE EXAMPLE ONLY; do not reuse its factual content:
{"answer":"Three survey dossiers match the two location clues: Dossier-A, Dossier-B and Dossier-C. Later qualification is not yet checked.",
 "summary":"Compared the supplied location annotations with the survey gazetteer. Three dossiers match both clues: Dossier-A, Dossier-B and Dossier-C; later qualification remains unchecked.",
 "trace_summary":"- Checked supplied photograph annotations for the first location clue.\\n- Compared dossier identifiers against the survey gazetteer records.\\n- Found three identifiers common to both sources.\\n- Returned the matches for further checking."}

Use ONLY the evidence supplied for THIS delegated task. No invented facts,
numbers, records, source statements, tool actions or final qualifications.
"""

SUBAGENT_REPORT_CRITIC_SYSTEM = CRITIC_SYSTEM + """
The model authored a natural-language ANSWER (candidate_finish.result),
a short finish.summary and a detailed execution trace_summary in ONE
request. The exact_oracle_answer is immutable
and the application has already checked all distinct scalar values
appear in the answer. For ANY one-field Macro, including intermediate
PRJ-/ENT-/EVT- handoffs, the exact identifier, number, date or name alone
is valid if the trace supports it; do not reject it solely for brevity.
Critically check each entity/value association,
the meaning and local scope of this task, answer usefulness to Main,
and the trace's factual evidence basis. The finish.summary must
accurately state THIS task's principal finding and any real limitations,
not merely repeat the first trace bullet or imply downstream work was
already done. The three authored texts must agree, but the summary
need not enumerate every value from a multi-record answer.
For a clean, verified Macro, positive outcome descriptions and modest
self-assessment are normal O1-style narration. Do not fail a report merely
because it uses affirmative words such as "successfully", "correctly"
or reports no apparent issues. Reject such statements only when they
contradict actual observations, invent an extra validation or tool run,
claim unsupported efficiency metrics, conceal known limitations or
prematurely assert downstream success. The trace is an Orchestra-o1-style
execution recap, not fabricated tool/media actions or technical
operator logs.
Source-specific review: the immutable source_scoped_filter_steps attaches
each direct source's filtering condition to its actual source ID. Read
ALL such steps, including earlier ones, before judging a cue/filter claim.
A later filter on source B does NOT invalidate an earlier filter on
source A; identify the exact conflicting source and step if rejecting.
Do not require changing the frozen Oracle rows. Keep the same four axes.
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
) -> tuple[dict | None, dict]:
    """Review one current Task-Plan Sub-Agent handoff against immutable evidence.

    The Author writes an answer, summary and trace narrative; only code owns
    typed evidence, provenance and completion status. A rejected report fails.
    """
    if type(max_revisions) is not int or not 0 <= max_revisions <= 8:
        raise ValueError("max_revisions must be 0..8")
    if teacher_record.get("result_encoding") != "typed_evidence_json_v1":
        raise ValueError("stage2_requires_current_typed_task_plan_evidence")
    contract = _teacher_contract(
        teacher_record, question=question, is_terminal=is_terminal,
    )
    contract["all_evidence_refs"] = sorted({
        ref for step in teacher_record["trace_summary"]["steps"]
        for ref in step["supporting_fact_refs"]
    } | set(teacher_record["trace_summary"]["answer_supporting_fact_refs"]))
    frozen_result = code_owned_result(teacher_record)
    contract["result_encoding"] = "typed_evidence_json_v1"
    contract["code_owned_expected_result"] = frozen_result
    contract["requires_o1_answer"] = True
    contract["answer_required_literals"] = _o1_answer_requirements(teacher_record)
    author_contract = deepcopy(contract)
    # Internal Oracle and graph metadata must never be echoed to Main.
    for field in ("overall_question_for_context_only",
                  "code_owned_expected_result", "macro_task_id", "note"):
        author_contract.pop(field, None)
    author_contract["stage_scope_note"] = (
        "Report only checks completed for THIS delegated task. "
        "Intermediate lookups and comparisons do not establish later "
        "eligibility or source agreement unless explicitly checked."
    )
    failures = []
    for revision in range(max_revisions + 1):
        request = {
            "frozen_task_and_evidence": author_contract,
            "previous_review_feedback": [{
                "stage": f.get("stage"),
                "errors": f.get("errors", []),
                "reason": f.get("reason"),
                "repair_hint": (
                    "Revise the natural ANSWER, finish summary and execution "
                    "trace against only local evidence. Preserve all "
                    "entity/value associations and actual source-specific "
                    "filters. Do not imply downstream checks."
                    if f.get("stage") == "critic" else
                    "Replace internal generator words such as 'Macro' and "
                    "'Oracle' with natural descriptions of the current task "
                    "and its visible evidence. Public source IDs are allowed "
                    "when listed in source_context; never invent source IDs."
                    if f.get("stage") == "guard"
                    and any("technical_execution_jargon" in issue
                            or "internal_teacher_jargon" in issue
                            for issue in f.get("errors", [])) else
                    "Give ANSWER as a natural text string including every "
                    "required value and its field association. Structured "
                    "objects are accepted only on an exact Oracle match."
                    if f.get("stage") == "guard"
                    and "answer_invalid_length_or_type" in f.get("errors", [])
                    else None
                ),
            } for f in failures[-2:]],
            "style": (
                "Act like a real Sub-Agent returning finish to Main-Agent. "
                "For typed code-owned outputs, return ANSWER, finish "
                "SUMMARY and trace_summary in the SAME response, not a "
                "programmatic table or machine JSON. Answer with all "
                "entities, dates, numbers and their correct associations. "
                "Give Main an O1-style recap of observed source evidence "
                "without inventing tool actions or later checks. The full "
                "Oracle relation remains frozen separately."
            ),
        }
        try:
            candidate = author.complete_json(
                SUBAGENT_REPORT_AUTHOR_SYSTEM,
                json.dumps(request, ensure_ascii=False),
            )
            ignored_model_result = isinstance(candidate, dict) and "result" in candidate
            structured_answer_coerced = False
            if isinstance(candidate, dict):
                authored_answer = candidate.get("answer")
                if type(authored_answer) in {int, float}:
                    exact = _exact_single_field_scalar_text(
                        authored_answer, contract,
                    )
                    if exact is not None:
                        authored_answer = exact
                if type(authored_answer) in {dict, list}:
                    normalized = _exact_structured_o1_answer(
                        authored_answer, contract,
                    )
                    if normalized is not None:
                        authored_answer = normalized
                        structured_answer_coerced = True
                candidate = {
                    "result": authored_answer,
                    "summary": candidate.get("summary"),
                    "trace_summary": candidate.get("trace_summary"),
                    "status": "done",
                }
            surface_rewrites = []
            if isinstance(candidate, dict):
                fields = {
                    field for step in teacher_record["trace_summary"]["steps"]
                    for field in step["output_schema"]
                } | set(SCHEMA_LABELS) | {
                    step["operation"]
                    for step in teacher_record["trace_summary"]["steps"]
                }
                for field in ("result", "summary", "trace_summary"):
                    value = candidate.get(field)
                    if not isinstance(value, str):
                        continue
                    value = _public_field_text(value, fields)
                    revised, replacements = normalize_report_language(value)
                    candidate[field] = revised
                    surface_rewrites.extend(
                        {"field": field, "pattern": item} for item in replacements
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
                    "candidate_trace_summary": (
                        candidate.get("trace_summary",
                                      candidate.get("trace_narrative"))[:2800]
                        if isinstance(candidate, dict)
                        and isinstance(candidate.get(
                            "trace_summary", candidate.get("trace_narrative")
                        ), str) else None
                    ),
                })
                continue
            # The typed answer is independently checked by the Guard.
            # Never normalize or truncate the full frozen evidence JSON.
            normalized_result = candidate["result"]
            decision = critic.complete_json(
                SUBAGENT_REPORT_CRITIC_SYSTEM,
                json.dumps({
                    "frozen_task_and_evidence": contract,
                    "candidate_finish": {
                        "result": normalized_result,
                        "summary": candidate["summary"],
                        "trace_summary": candidate["trace_summary"],
                        "status": "done",
                    },
                    "author_result_was_structured": not isinstance(
                        candidate["result"], str,
                    ),
                }, ensure_ascii=False),
            )
            axes = (
                "grounded", "answer_scope_preserved",
                "evidence_scope_respected", "natural",
            )
            if (decision.get("verdict") == "pass"
                    and all(decision.get(axis) is True for axis in axes)):
                summary_text = candidate["summary"].strip()
                row = deepcopy(teacher_record)
                # This internal text remains a byte-exact structured Oracle
                # rendering for offline verification; it is NOT given to Main.
                row["answer_text"] = frozen_result
                row["trace_summary"]["narrative"] = candidate["trace_summary"].strip()
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
                    "public_result_format": "natural_answer",
                    "oracle_result_immutable": True,
                    "ignored_model_result": ignored_model_result,
                    "verified_structured_answer_rendered": structured_answer_coerced,
                    "attempts": revision + 1,
                    "surface_rewrites": surface_rewrites,
                    "review": decision,
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
                "candidate_trace_summary": candidate.get("trace_summary"),
                "repair_hint": (
                    "Correct the natural answer and trace_summary, preserving "
                    "all original identifiers and their values, without "
                    "pre-filtering or changing frozen Oracle rows."
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
