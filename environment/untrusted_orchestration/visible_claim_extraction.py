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

SUBJECT_FIELDS = (
    "case_id", "candidate_id", "entry_label", "accession_id",
    "record_key_1", "record_key_2", "marker_code",
)
_SCALAR = (str, int, float, bool)
_SUMMARY_PAIR = re.compile(
    r"\b(?P<subject>(?:Case|Entry)-\d{2,5})\s*"
    r"(?::|,)\s*(?P<field>[A-Za-z][A-Za-z0-9_]*)\s*"
    r"=\s*(?P<value>-?\d+(?:\.\d+)?|true|false)\b", re.I,
)
_CASE_LIST = re.compile(
    r"^The current task returned (?P<count>\d+) case identifiers?:\s*"
    r"(?P<ids>Case-\d+(?:,\s*Case-\d+)*)\.$", re.I,
)
_REPORTED_YEAR = {"reported_year_a", "reported_year_b"}


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
        else:
            match = _CASE_LIST.fullmatch(result.strip())
            if match:
                ids = [x.strip() for x in match.group("ids").split(",")]
                if len(ids) == int(match.group("count")) and len(ids) == len(set(ids)):
                    modes.add("literal_identifier_list")
                    for case in ids:
                        add(case, "identifier_present", True, "result")

    if isinstance(summary, str):
        for match in _SUMMARY_PAIR.finditer(summary):
            # A free-form summary pair must explicitly name its subject
            # and one of this delegated task's output fields.
            if match.group("field") in declared:
                add(match.group("subject"), match.group("field"),
                    _parse_scalar(match.group("value")), "summary")
                modes.add("explicit_summary_pair")

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
