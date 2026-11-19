"""Pure deterministic Stage-4 report fidelity guard.

Checks *delivered WorkProduct* separately from observed intermediate steps.
Never compares a potentially untrusted return with hidden clean Oracle data.
The user-facing report-only faults bypass trusted naturalization upstream.
"""
from __future__ import annotations

import json
import re
from environment.implement_subtask.reporting import (
    _normalize_author_result, _stage_scope_issues,
)
from .state_machine import Attempt, relation_rows
from .report_contract import render_delivered_result


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

def claim_tokens(value: str) -> set[str]:
    """Normalize thousands separators, not facts or entity identifiers."""
    value = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value)
    return {token.casefold() for token in _TOKEN.findall(value)}

_MONTHS = {
    name.casefold(): number for number, name in enumerate((
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ), start=1)
}
_CALENDAR_DATE = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})"
    r"(?:,?\s+(\d{4}))?\b", re.I,
)
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")


def grounded_calendar_text(text: str, observed_text: str) -> str:
    """Normalize calendar wording *only* if that exact date was observed.

    E.g. 'July 2, 2011' and '2011-07-02' are equivalent, but a proposed
    July 3 is NOT authorized. A month/day without a year is normalized only
    when exactly one matching full date exists in the delivered/observed trace.
    Unrelated numbers are untouched, so this does not relax numeric guards.
    """
    known = {}
    for year, month, day in _ISO_DATE.findall(observed_text):
        iso = f"{year}-{month}-{day}"
        known.setdefault((int(month), int(day)), set()).add(iso)

    def subst(match: re.Match) -> str:
        month = _MONTHS[match.group(1).casefold()]
        day = int(match.group(2))
        year = match.group(3)
        dates = known.get((month, day), set())
        if year:
            dates = {date for date in dates if date.startswith(year + "-")}
        return next(iter(dates)) if len(dates) == 1 else match.group(0)

    return _CALENDAR_DATE.sub(subst, text)


def explicit_final_value_issues(text: str, contract: dict) -> list[str]:
    """Distinguish a claimed delivered field value from observed provenance.

    Numeric values inside an actually executed filter/lookup condition may
    legitimately be discussed in `result`, even if the final projection
    drops that field. But a value explicitly assigned to a FINAL output
    field must belong to the actual delivered rows, not just an intermediate
    preview. This especially protects null fields in partial handoffs.
    """
    issues = []
    rows = [entry["values"] for entry in contract["actual_delivered_rows"]]
    for field in contract["output_fields"]:
        forms = {field, field.replace("_", " ")}
        valid = {
            str(row[field]).casefold() for row in rows
            if row.get(field) is not None
        }
        for name in forms:
            # Match declarative field-value syntax, not incidental context
            # such as "marker code field remains unavailable" or a
            # source/filter operation that did not deliver the field.
            pattern = re.compile(
                r"\b" + re.escape(name) +
                r"\b\s*(?::|=|is|was)\s*"
                r"(?P<value>KEY[12]-\d+|MK-\d+|RK-\d+|"
                r"Case-\d+|Entry-\d+|-?\d+(?:\.\d+)?)\b",
                flags=re.I,
            )
            for match in pattern.finditer(text):
                value = match.group("value").casefold()
                if value not in valid:
                    issues.append("claimed_final_field_value_not_delivered")
    return sorted(set(issues))


def validate_report_draft(draft: dict, contract: dict, attempt: Attempt) -> list[str]:
    if not isinstance(draft, dict):
        return ["invalid_report_json"]
    issues = []
    required = {"result", "summary", "trace_summary"}
    # Old Stage-4 Authors may still supply status. It is ignored rather than
    # compared with the executor, and never appears in the final report.
    if not required <= set(draft) or set(draft) - (required | {"status"}):
        issues.append("invalid_report_fields")
    missing = attempt.product is None
    result = draft.get("result")
    exact_rows = [row["values"] for row in relation_rows(attempt.product)]
    normalized = None
    typed_delivery = (
        contract.get("result_encoding") in {
            "typed_evidence_json_v1", "cited_record_prose_v1",
        } and not missing
    )
    if missing:
        if result is not None:
            issues.append("missing_result_must_be_null")
    elif typed_delivery:
        # Executor-owned, exact provenance-bearing JSON may exceed the
        # old prose length cap. Do not silently truncate or reconstruct it.
        expected = render_delivered_result(contract)
        if isinstance(result, str) and result == expected:
            normalized = result
        else:
            issues.append("result_not_exact_delivered_evidence")
    else:
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
        v for v in ((None if typed_delivery else normalized), summary, trace)
        if isinstance(v, str)
    )
    if _PRIVATE.search(prose):
        issues.append("internal_generation_or_teacher_leak")
    if _UNPERFORMED.search(prose):
        issues.append("claims_unperformed_media_or_web_action")

    # Reuse Stage 2's narrow stage-local factual scope guard. The reference
    # is this ATTEMPT's delivered relation, never the frozen Gold.
    stage_local = {
        "exact_oracle_answer": exact_rows,
        "executed_operations_in_this_macro":
            contract["executed_operations_in_order"],
        "terminal_output_schema_of_this_macro": contract["output_fields"],
        "macro_is_terminal": False,
        "upstream_handoffs": contract["upstream_handoffs"],
        "available_provenance_sources":
            contract["available_provenance_sources"],
    }
    for field, text in (
        ("result", None if typed_delivery else normalized),
        ("summary", summary), ("trace_summary", trace),
    ):
        if isinstance(text, str):
            issues.extend(
                field + "_" + issue
                for issue in _stage_scope_issues(text, stage_local)
            )

    # Keep exact structured result comparison above and protect against
    # hallucinated IDs/measurements in prose. Unlike the old Stage 4 check,
    # actual observed intermediate steps, lookup keys and output counts
    # also authorize a statement; none are drawn from the clean Oracle.
    observed = contract["observed_operator_steps"]
    available = (
        json.dumps(contract["actual_delivered_rows"], ensure_ascii=False)
        + " " + json.dumps(observed, ensure_ascii=False)
    )
    result_tokens = claim_tokens(json.dumps(
        contract["actual_delivered_rows"], ensure_ascii=False,
    ))
    result_tokens.add(str(contract["total_delivered_rows"]))
    # Stage 2 normalizes schema names like record_key_1 to "record key 1".
    # The digit 1 is a FIELD LABEL, not a fabricated numeric observation.
    # Authorize only actual output/observed schema labels, not arbitrary
    # numbers from the question or hidden clean WorkProduct.
    schema_fields = set(contract["output_fields"]) | {
        field for step in observed
        for field in step.get("output_schema", [])
    }
    result_tokens.update(claim_tokens(
        " ".join(name.replace("_", " ") for name in schema_fields)
    ))
    allowed_tokens = claim_tokens(available)
    allowed_tokens |= result_tokens | {
        str(len(contract["executed_operations_in_order"])),
    }
    instruction_tokens = claim_tokens(
        contract["assigned_task_instruction"]
    )
    # An upstream identity can be used as an explanatory parent/context
    # label in a result, but never invented out of thin air. Only explicit
    # entity-shaped IDs already visible in both the instruction AND the
    # current actual observation are admitted; independent numeric values
    # remain scoped to the final delivered WorkProduct.
    contextual_entities = (
        {x.casefold() for x in _ENTITY.findall(
            contract["assigned_task_instruction"]
        )}
        & {x.casefold() for x in _ENTITY.findall(available)}
    )
    for field, text in (
        ("result", None if typed_delivery else normalized),
        ("summary", summary), ("trace_summary", trace),
    ):
        if not isinstance(text, str):
            continue
        # A clean upstream identifier may be visible in a preceding
        # operation, but it must not replace the actually DELIVERED result.
        # A result may explain a grounded threshold, observed lookup key,
        # or intermediate row count as context. Those tokens originate in
        # the ACTUAL executed trace, not from the future clean World.
        # The separate final-field check prevents intermediate values
        # from masquerading as the delivered output.
        allowed = (allowed_tokens if field == "result"
                   else allowed_tokens | instruction_tokens)
        # Authored calendar wording can omit a leading zero or use a
        # month name. Normalize only dates literally observed in this
        # Attempt; never grant a generic numeric exemption.
        grounded_text = grounded_calendar_text(text, available)
        if claim_tokens(grounded_text) - allowed:
            issues.append(field + "_unobserved_numeric_or_entity")
        if field == "result":
            issues.extend("result_" + issue for issue in
                          explicit_final_value_issues(text, contract))

    if attempt.status == "partial":
        missing_fields = contract["unresolved_fields"]
        descriptive = " ".join(
            x for x in (normalized, summary, trace)
            if isinstance(x, str)
        ).lower()
        if missing_fields and not any(
            field in descriptive or field.replace("_", " ") in descriptive
            for field in missing_fields
        ) and not any(word in descriptive for word in (
            "incomplete", "unresolved", "missing", "not returned",
            "not delivered", "partial",
        )):
            issues.append("partial_missing_field_not_acknowledged")
    if attempt.status == "missing" and contract["observed_operator_steps"]:
        # Stage-local check on truthful failed-acquisition description.
        if re.search(r"\b(?:completed|successfully|finished)\b.{0,75}"
                     r"\b(?:lookup|extraction|search|retrieval)\b",
                     prose, re.I):
            issues.append("missing_claims_completed_acquisition")
    return sorted(set(issues))

