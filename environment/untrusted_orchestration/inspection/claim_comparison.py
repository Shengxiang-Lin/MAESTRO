"""Conservative comparisons of claims that Main can actually read.

Only explicit Result/Summary values from one delivered report are considered.
A multi-row summary without an unambiguous subject does not prove a conflict.
Never inspect Gold, fault provenance, or an unpublished WorkProduct here.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
import re

from environment.untrusted_orchestration.reporting.surface import delivered_records

_SUBJECT_FIELDS = (
    "case_id", "candidate_id", "entry_label", "accession_id", "project_id",
    "event_id", "entity_final", "entity_0", "entity_1", "entity_2",
    "entity_3", "record_key_1", "record_key_2", "marker_code",
)
_ENTITY = re.compile(
    r"(?:Case-\d+|Entry-\d+|Dossier-[A-Za-z0-9]+|PRJ-[A-Za-z0-9]+|"
    r"ENT-[A-Za-z0-9]+|Record\s+\d+)", re.I,
)
_DECODER = json.JSONDecoder()
_ID_VALUE = re.compile(
    r"(?:Case-\d+|Entry-\d+|Dossier-[A-Za-z0-9]+|PRJ-[A-Za-z0-9]+|"
    r"ENT-[A-Za-z0-9]+|MK-[A-Za-z0-9]+|RK-[A-Za-z0-9]+|"
    r"KEY[12]-[A-Za-z0-9]+)", re.I,
)


def _display(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return value.casefold() if isinstance(value, str) else str(value)



def _same(a, b) -> bool:
    if type(a) is bool or type(b) is bool:
        return type(a) is type(b) and a == b
    if type(a) in (int, float) and type(b) in (int, float):
        try:
            return Decimal(str(a)) == Decimal(str(b))
        except InvalidOperation:
            return False
    return type(a) is type(b) and a == b


def _report_finish(report: dict) -> dict:
    nested = report.get("finish_result")
    return nested if isinstance(nested, dict) else report


def _rows(report: dict) -> list[dict] | None:
    finish = _report_finish(report)
    rows = delivered_records(finish.get("result"))
    if not isinstance(rows, list) or not rows:
        return None
    if any(not isinstance(row, dict)
           or not isinstance(row.get("values"), dict) for row in rows):
        return None
    return rows


def _subject_rows(prefix: str, rows: list[dict]) -> list[dict] | None:
    """Return only unambiguously selected rows, else None (unknown scope)."""
    candidates = list(_ENTITY.finditer(prefix[-100:]))
    if candidates:
        match = candidates[-1]
        # The subject must directly qualify the field in the same clause.
        tail = prefix[-100:][match.end():]
        if not re.fullmatch(r"\s*(?:(?:[:,]|contains|has)\s*)?", tail, re.I):
            return None
        token = match.group(0)
        if re.fullmatch(r"Record\s+\d+", token, re.I):
            index = int(re.search(r"\d+", token).group()) - 1
            return [rows[index]] if 0 <= index < len(rows) else None
        selected = [
            row for row in rows
            if any(str(row["values"].get(key, "")).casefold() == token.casefold()
                   for key in _SUBJECT_FIELDS)
        ]
        return selected if len(selected) == 1 else None
    return rows if len(rows) == 1 else None


def _explicit_values(summary: str, field: str):
    # Anchor on an actual field of a delivered Result, not arbitrary prose.
    label = "[_ ]".join(re.escape(part) for part in field.split("_"))
    pattern = re.compile(r"(?<!\w)" + label + r"\s*(?:=|:)\s*", re.I)
    for match in pattern.finditer(summary):
        fragment = summary[match.end():].lstrip()
        try:
            value, end = _DECODER.raw_decode(fragment)
        except (ValueError, TypeError):
            identity = _ID_VALUE.match(fragment)
            boolean = re.match(r"(?:true|false)\b", fragment, re.I)
            if identity:
                value, end = identity.group(), identity.end()
            elif boolean:
                value, end = boolean.group().casefold() == "true", boolean.end()
            else:
                continue
        if type(value) not in (str, int, float, bool):
            continue
        suffix = fragment[end:]
        # A parser that reads "46" from "46.653836" is never accepted.
        if suffix and not (suffix[0].isspace()
                           or re.match(r"^[;,.!?](?:\s|$)", suffix)):
            continue
        yield match, value


def report_claim_conflicts(
    report: dict, *, allowed_fields: set[str] | None = None,
) -> list[dict]:
    """Find justified same-subject, same-field Result/Summary disagreements.

    A bare field claim against several result rows remains ambiguous even if
    its value is absent from the table. A compatible clean report must never
    become a conflict merely because its summary uses a different sentence.
    """
    if not isinstance(report, dict):
        return []
    finish = _report_finish(report)
    summary = finish.get("summary")
    rows = _rows(report)
    if not rows or not isinstance(summary, str):
        return []
    fields = {name for row in rows for name in row["values"]}
    if allowed_fields is not None:
        normalized = {name.strip().replace(" ", "_").casefold()
                      for name in allowed_fields}
        fields = {name for name in fields if name.casefold() in normalized}
    conflicts, seen = [], set()
    for field in sorted(fields, key=lambda name: (-len(name), name)):
        for match, claimed in _explicit_values(summary, field):
            prefix = summary[max(0, match.start() - 100):match.start()]
            selected = _subject_rows(prefix, rows)
            if (selected is None and field in _SUBJECT_FIELDS
                    and re.search(r"\bsummary\s+records?\s*$", prefix, re.I)):
                # An explicit returned-identifier list is a set-membership
                # assertion. Its missing identifier is observable without
                # guessing which measurement belongs to which row.
                selected = rows
            if selected is None:
                continue
            values = [
                row["values"][field] for row in selected
                if field in row["values"] and row["values"][field] is not None
            ]
            if not values or any(_same(claimed, value) for value in values):
                continue
            key = (field, match.start(), repr(claimed))
            if key in seen:
                continue
            seen.add(key)
            conflicts.append({
                "field": field,
                "subject": (str(next(
                    (selected[0]["values"][name] for name in _SUBJECT_FIELDS
                     if name in selected[0]["values"]), "single_record"))
                    if len(selected) == 1 else "returned_record_set"),
                "result_value": ", ".join(_display(x) for x in values[:8]),
                "summary_value": _display(claimed),
                "source_fact_refs": sorted({
                    ref for row in selected
                    for ref in row.get("source_fact_refs", [])
                }),
                "basis": "explicit same-object Result/Summary values",
                "state": "unverified",
            })
    return conflicts


# A repeated subject alone is NOT evidence of contradiction: independent
# observations at different times may legitimately return different values.
# A shared explicit FactRef (or an identical declared record key) establishes
# a concrete same-record witness without consulting hidden World rows.
_RECORD_KEYS = ("record_key_1", "record_key_2", "accession_id", "marker_code")
_SCOPE_KEYS = {
    "observed_at", "observation_date", "observation_year", "timestamp",
    "valid_from", "valid_to", "valid_scope", "effective_date",
    "period", "period_start", "period_end", "record_version", "version",
}


def result_internal_claim_conflicts(
    report: dict, *, allowed_fields: set[str] | None = None,
) -> list[dict]:
    """Compare two *delivered Result rows* from the same record.

    Rows must explicitly name the same subject AND have common cited FactRefs
    or an identical strong record key. Different dates/versions/scopes do not
    imply an error in a measurement. Unknown provenance abstains.
    """
    rows = _rows(report)
    if not rows or len(rows) < 2:
        return []
    allowed = (None if allowed_fields is None else {
        field.strip().replace(" ", "_").casefold()
        for field in allowed_fields
    })
    grouped = {}
    for number, row in enumerate(rows, 1):
        values = row["values"]
        subject_field = next((
            field for field in _SUBJECT_FIELDS
            if isinstance(values.get(field), str) and values[field].strip()
        ), None)
        if subject_field is None:
            continue
        subject = values[subject_field].strip()
        grouped.setdefault((subject_field, subject.casefold()), []).append(
            (number, subject, row)
        )
    conflicts, seen = [], set()
    for group in grouped.values():
        for i, (left_number, subject, left) in enumerate(group):
            for right_number, _, right in group[i + 1:]:
                a, b = left["values"], right["values"]
                refs_a = set(left.get("source_fact_refs") or [])
                refs_b = set(right.get("source_fact_refs") or [])
                shared_refs = refs_a & refs_b
                same_key = any(
                    key in a and key in b and a[key] is not None
                    and _same(a[key], b[key]) for key in _RECORD_KEYS
                )
                if not shared_refs and not same_key:
                    continue
                # If an explicit time/version qualifier differs, the rows
                # may be different real observations, not conflicting claims.
                if any(
                    k in a and k in b and not _same(a[k], b[k])
                    for k in _SCOPE_KEYS
                ):
                    continue
                for field in sorted(a.keys() & b.keys()):
                    if field in _SUBJECT_FIELDS or field in _RECORD_KEYS:
                        continue
                    if allowed is not None and field.casefold() not in allowed:
                        continue
                    value_a, value_b = a[field], b[field]
                    if (value_a is None or value_b is None or
                            type(value_a) not in (str, int, float, bool) or
                            type(value_b) not in (str, int, float, bool) or
                            _same(value_a, value_b)):
                        continue
                    key = (subject.casefold(), field, repr(value_a), repr(value_b))
                    if key in seen:
                        continue
                    seen.add(key)
                    conflicts.append({
                        "kind": "intra_result_conflict",
                        "field": field, "subject": subject,
                        "result_value": _display(value_a),
                        "summary_value": _display(value_b),
                        "observed_values": [value_a, value_b],
                        "result_row_numbers": [left_number, right_number],
                        "source_fact_refs": sorted(shared_refs),
                        "basis": "two explicitly returned same-record Result rows",
                        "state": "unverified",
                    })
    return conflicts


def public_report_conflicts(
    report: dict, *, allowed_fields: set[str] | None = None,
) -> list[dict]:
    """One Main-visible comparison entrypoint for planning and 4C review."""
    old = report_claim_conflicts(report, allowed_fields=allowed_fields)
    inside = result_internal_claim_conflicts(
        report, allowed_fields=allowed_fields,
    )
    return [
        {**finding, "kind": "result_summary_disagreement"}
        for finding in old
    ] + inside
