"""Extract explicit typed claims from the Sub-Agent text Main actually received.

This parser does not inspect the World, Oracle, private execution product, or
future Sub-Agent output. It intentionally abstains on free-form narratives.
The caller may attach source provenance from the known delegated task, but
must not mistake provenance for correctness or independent verification.
"""
from __future__ import annotations

import json
import re

from environment.implement_subtask.visible_evidence import visible_year_claims
from environment.generate_question.authoring.surface_quality import SCHEMA_LABELS

SUBJECT_FIELDS = (
    "case_id", "candidate_id", "entry_label", "accession_id",
    "record_key_1", "record_key_2", "marker_code",
    "project_id", "event_id", "entity_final", "entity_0",
    "entity_1", "entity_2", "entity_3",
)
_SCALAR = (str, int, float, bool)
_SUMMARY_PAIR = re.compile(
    r"\b(?P<subject>(?:(?:Case|Entry)-\d{2,5}|Dossier-[A-Za-z0-9]+))\s*"
    r"(?::|,)\s*(?P<field>[A-Za-z][A-Za-z0-9_]*)\s*"
    r"=\s*(?P<value>-?\d+(?:\.\d+)?|true|false)\b", re.I,
)
_CASE_LIST = re.compile(
    r"^The current task returned (?P<count>\d+) case identifiers?:\s*"
    r"(?P<ids>(?:(?:Case-\d+|Dossier-[A-Za-z0-9]+)"
    r"(?:,\s*(?:Case-\d+|Dossier-[A-Za-z0-9]+))*))\.$", re.I,
)
_REPORTED_YEAR = {"reported_year_a", "reported_year_b"}


_NATURAL_RECORDS = re.compile(
    r"^(?:The current task returned (?P<count>\d+) record\(s\)|"
    r"The handoff contains (?P<delivered>\d+) delivered records): "
    r"(?P<body>.+)\.$", re.I,
)
_NATURAL_SINGLE = re.compile(
    r"^The delivered record contains (?P<body>.+)\.$", re.I,
)


def _parse_current_record_prose(result: str, fields: set[str]) -> list[dict] | None:
    """Conservatively recover full rows from code-owned natural Result text.

    This compatibility path is for already-generated Stage-2/4 reports.
    It never reconstructs fields from the private World or Gold.
    """
    found = _NATURAL_RECORDS.fullmatch(result.strip())
    single = _NATURAL_SINGLE.fullmatch(result.strip()) if not found else None
    if not found and not single:
        return None
    body = (found or single).group("body")
    expected = (int(found.group("count") or found.group("delivered"))
                if found else 1)
    labels = {}
    for field in fields:
        labels[field.lower().replace("_", " ")] = field
        labels[SCHEMA_LABELS.get(field, field.replace("_", " ")).lower()] = field
    rows = []
    for record in body.split("; "):
        values = {}
        for part in record.split(", "):
            if ": " not in part:
                return None
            label, value = part.split(": ", 1)
            field = labels.get(label.lower())
            if not field or field in values or not value.strip():
                return None
            raw = value.strip()
            if re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
                parsed = _parse_scalar(raw)
            elif raw.lower() in ("yes", "no"):
                parsed = raw.lower() == "yes"
            else:
                parsed = raw
            values[field] = parsed
        if not values:
            return None
        rows.append(values)
    return rows if len(rows) == expected else None


def _typed_scalar(value):
    return type(value) in _SCALAR and (
        not isinstance(value, str) or bool(value.strip())
    )


def _normalize_attribute(field: str) -> tuple[str, str]:
    if field in _REPORTED_YEAR:
        return "reported_event_year", "same_reported_event"
    return field, "same_record"


def _parse_scalar(text: str):
    lower = text.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if "." in text:
        return float(text)
    return int(text)


def extract_visible_claims(report: dict, *, output_fields: list[str],
                           source_id: str | None) -> tuple[list[dict], dict]:
    """Return literal claims and parser diagnostics for one public delivery.

    JSON result rows are included only if the main-visible Result contains
    the typed value and identifier. A source-specific, explicit year digest
    may contribute additional Summary claims, including contradictions
    to Result claims. No private relation rows are substituted.
    """
    finish = report.get("finish_result")
    if not isinstance(finish, dict) or finish.get("status") not in {
        "done", "partial",
    }:
        return [], {"status": (finish or {}).get("status", "missing"),
                    "parsed": False, "parse_mode": "not_completed"}
    result, summary = finish.get("result"), finish.get("summary")
    claims = []
    declared = set(output_fields)
    modes = set()

    def add(subject, field, value, surface, refs=None):
        if (not isinstance(subject, str) or not subject.strip()
                or not isinstance(field, str) or not field
                or not _typed_scalar(value)):
            return
        attr, scope = _normalize_attribute(field)
        claims.append({
            "subject": subject, "attribute": attr,
            "original_field": field, "valid_scope": scope,
            "value": value, "surface": surface,
            "source_id": source_id,
            "fact_refs": list(refs or []),
            "declared_output_field": field in declared,
        })

    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            modes.add("structured_result")
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                row = item.get("values", item)
                if not isinstance(row, dict):
                    continue
                subject_field = next((
                    k for k in SUBJECT_FIELDS
                    if isinstance(row.get(k), str) and row[k].strip()
                ), None)
                if subject_field is None:
                    # Single-source scalar statistics often return only
                    # e.g. {metric_a: 7}, without an entity key. Retain
                    # them as SOURCE-SCOPED observations, not as a shared
                    # entity that could produce false corroboration.
                    scalar_fields = [
                        field for field, value in row.items()
                        if field in declared and _typed_scalar(value)
                    ]
                    if (source_id is None or len(declared) != 1
                            or len(scalar_fields) != 1):
                        continue
                    subject = "source:" + source_id
                else:
                    subject = row[subject_field]
                refs = item.get("source_fact_refs", [])
                if not isinstance(refs, list) or not all(
                    isinstance(ref, str) for ref in refs
                ):
                    refs = []
                for field, value in row.items():
                    if field != subject_field and _typed_scalar(value):
                        add(subject, field, value, "result", refs)
                if subject_field is not None and len(row) == 1:
                    if subject_field == "case_id":
                        add(subject, "identifier_present", True, "result", refs)
                    elif source_id is not None:
                        add("source:" + source_id, subject_field,
                            row[subject_field], "result", refs)
        else:
            match = _CASE_LIST.fullmatch(result.strip())
            if match:
                ids = [x.strip() for x in match.group("ids").split(",")]
                if len(ids) == int(match.group("count")) and len(ids) == len(set(ids)):
                    modes.add("literal_identifier_list")
                    for case in ids:
                        add(case, "identifier_present", True, "result")
            else:
                natural_rows = _parse_current_record_prose(result, declared)
                if natural_rows is not None:
                    modes.add("code_owned_record_prose")
                    for row in natural_rows:
                        subject_field = next((
                            key for key in SUBJECT_FIELDS
                            if isinstance(row.get(key), str) and row[key].strip()
                        ), None)
                        if subject_field is not None:
                            subject = row[subject_field]
                        elif len(declared) == 1 and source_id is not None:
                            subject = "source:" + source_id
                        else:
                            continue
                        for field, value in row.items():
                            if field != subject_field and _typed_scalar(value):
                                add(subject, field, value, "result")
                        if subject_field is not None and len(row) == 1:
                            if subject_field == "case_id":
                                add(subject, "identifier_present", True, "result")
                            elif source_id is not None:
                                add("source:" + source_id, subject_field,
                                    row[subject_field], "result")

    for surface, prose in (("result", result), ("summary", summary)):
        if not isinstance(prose, str):
            continue
        for match in _SUMMARY_PAIR.finditer(prose):
            # A free-form assertion must explicitly name its subject
            # and a field in this delegated task's output contract.
            # This cannot infer missing fields from task/World metadata.
            if match.group("field") in declared:
                add(match.group("subject"), match.group("field"),
                    _parse_scalar(match.group("value")), surface)
                modes.add("explicit_" + surface + "_pair")

    if source_id in {"report_a", "report_b"}:
        for item in visible_year_claims(report, source_id):
            # The helper also parses structured Result rows. Keep exact
            # surface provenance so Result/Summary contradictions survive.
            field = "reported_year_a" if source_id == "report_a" else "reported_year_b"
            add(item["subject"], field, item["value"], item["surface"])
            modes.add("source_year_digest")

    dedup, seen = [], set()
    for claim in claims:
        key = (claim["subject"], claim["attribute"], claim["valid_scope"],
               type(claim["value"]).__name__, repr(claim["value"]),
               claim["surface"], claim["source_id"])
        if key not in seen:
            seen.add(key)
            dedup.append(claim)
    dedup.sort(key=lambda x: (
        x["subject"], x["attribute"], x["surface"], repr(x["value"]),
    ))
    return dedup, {
        "status": finish.get("status"),
        "parsed": bool(dedup),
        "parse_mode": "+".join(sorted(modes)) or "unparsed",
        "explicit_claim_count": len(dedup),
    }
