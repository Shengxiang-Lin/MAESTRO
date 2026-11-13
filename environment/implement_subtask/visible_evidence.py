"""Code-owned, source-scoped assertion surfaces for Sub-Agent feedback.

Only facts in the current *delivered* Macro WorkProduct are rendered. When
recovering claims from Main-visible text, parse explicit literals and fail
closed on absent or ambiguous prose. No hidden Gold lookup is performed.
"""
from __future__ import annotations

import json
import re

YEAR_FIELDS = {"report_a": "reported_year_a", "report_b": "reported_year_b"}
_LABELS = {"report_a": "A", "report_b": "B"}
_CASE = re.compile(r"^Case-\d{3}$")
_DIGEST = re.compile(
    r"\bSource\s+([AB])\s+reported\s+years:\s*([^.]+)\.",
    re.I,
)
_PAIR = re.compile(r"\b(Case-\d{3})=(\d{4})\b")


def witnessed_source_summary(record: dict) -> str:
    """Produce truthful report-year summaries from THIS Macro's answer rows."""
    scope = record.get("authorized_source_ids")
    if not isinstance(scope, list) or len(scope) != 1:
        return ""
    source_id = scope[0]
    field = YEAR_FIELDS.get(source_id)
    if field is None:
        return ""
    if set(record.get("output_schema", [])) != {"case_id", field}:
        return ""
    rows = record.get("answer")
    if not isinstance(rows, list) or not rows:
        return ""
    seen, entries = set(), []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"case_id", field}:
            raise ValueError("public_claim_source_output_schema_drift")
        case, year = row["case_id"], row[field]
        if (not isinstance(case, str) or not _CASE.fullmatch(case)
                or type(year) is not int or not 1000 <= year <= 2999
                or case in seen):
            raise ValueError("public_claim_source_values_not_grounded")
        seen.add(case)
        entries.append((case, year))
    values = "; ".join(f"{case}={year}" for case, year in sorted(entries))
    return f"Source {_LABELS[source_id]} reported years: {values}."


def visible_year_claims(report: dict, source_id: str) -> list[dict]:
    """Extract only explicit public claims; never project private World rows.

    A genuine summary digest or a structured delivered Result can ground an
    assertion. Result/Summary contradictions remain as separate claims and
    must not be silently replaced with the executor's true answer.
    """
    field = YEAR_FIELDS.get(source_id)
    if field is None:
        return []
    finish = report.get("finish_result")
    if not isinstance(finish, dict) or finish.get("status") != "done":
        return []
    result = finish.get("result")
    summary = finish.get("summary")
    values = []
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, list):
            for item in parsed:
                row = item.get("values", item) if isinstance(item, dict) else None
                if (isinstance(row, dict) and
                        isinstance(row.get("case_id"), str) and
                        _CASE.fullmatch(row["case_id"]) and
                        type(row.get(field)) is int):
                    values.append((row["case_id"], row[field], "result"))
    if isinstance(summary, str):
        for match in _DIGEST.finditer(summary):
            label, raw = match.groups()
            if label.upper() != _LABELS[source_id]:
                continue
            for case, year in _PAIR.findall(raw):
                values.append((case, int(year), "summary"))
    seen, claims = set(), []
    for case, year, surface in values:
        key = (case, year)
        if key in seen:
            continue
        seen.add(key)
        claims.append({"subject": case, "value": year,
                       "surface": surface, "source_id": source_id})
    return sorted(claims, key=lambda x: (x["subject"], x["value"], x["surface"]))
