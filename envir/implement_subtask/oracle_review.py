"""Deterministic Entry/FactRef/active-status verification for Stage 2.

A complete Oracle list is authoritative. LLM Critics may judge narrative
quality, but cannot reassign a FactRef to a different Entry.
"""
from __future__ import annotations

import re
from collections import Counter

_ENTRY = re.compile(r"\bEntry-\d+\b", re.I)
_REF = re.compile(r"\b[A-Za-z][\w-]*:F[\w-]+\b")
_NEAR = re.compile(
    r"\b(Entry-\d+)\b\s*(?:\(|:|,|\s){0,8}"
    r"([A-Za-z][\w-]*:F[\w-]+)", re.I,
)
_REVERSE = re.compile(
    r"\b([A-Za-z][\w-]*:F[\w-]+)\b\s*(?:\(|:|,|\s){0,8}"
    r"(Entry-\d+)\b", re.I,
)
_FALSE_CLAIM = re.compile(
    r"\b(?:inactive|not active|active\s*=\s*false|"
    r"active\s+is\s+false|marked\s+false)\b", re.I
)


def full_entry_list_matches(text: str, rows: list[dict]) -> bool:
    if not isinstance(text, str) or not rows or not all(
        set(row) == {"entry_label"}
        and _ENTRY.fullmatch(str(row["entry_label"]))
        for row in rows
    ):
        return False
    expected = Counter(str(row["entry_label"]).casefold() for row in rows)
    observed = Counter(x.casefold() for x in _ENTRY.findall(text))
    return bool(observed) and observed == expected


def entry_fact_index(teacher_record: dict) -> dict:
    """Map only observed FactRefs; no inference from unrelated source rows."""
    result = {}
    for step in teacher_record["trace_summary"]["steps"]:
        for atom in step.get("evidence_atoms", []):
            ref, values = atom.get("fact_ref"), atom.get("values", {})
            if (ref and isinstance(values, dict)
                    and isinstance(values.get("entry_label"), str)):
                record = {
                    "entry_label": values["entry_label"],
                    "active": values.get("active"),
                }
                if ref in result and result[ref] != record:
                    raise ValueError("conflicting_oracle_fact_ref:" + ref)
                result[ref] = record
    return result


def mismatched_entry_refs(text: str, fact_index: dict) -> list[dict]:
    if not isinstance(text, str):
        return []
    pairs = [(a, b) for a, b in _NEAR.findall(text)]
    pairs.extend((entry, ref) for ref, entry in _REVERSE.findall(text))
    return [
        {"stated_entry": entry, "cited_ref": ref,
         "actual_entry": fact_index[ref]["entry_label"]}
        for entry, ref in pairs
        if ref in fact_index
        and fact_index[ref]["entry_label"].casefold() != entry.casefold()
    ]


def critic_oracle_contradiction(
    review: dict, result_text: str, rows: list[dict], fact_index: dict
) -> dict | None:
    """Return a confirmed Critic factual error; never auto-accept the report."""
    if not full_entry_list_matches(result_text, rows):
        return None
    reason = str(review.get("reason", ""))
    mismatches = mismatched_entry_refs(reason, fact_index)
    if mismatches:
        return {"type": "critic_entry_fact_ref_mismatch",
                "mismatches": mismatches}
    listed = {str(row["entry_label"]).casefold() for row in rows}
    mentioned = {v.casefold() for v in _ENTRY.findall(reason)}
    if not (mentioned & listed):
        return None
    if _FALSE_CLAIM.search(reason):
        # Only criticize an Oracle-listed entry's active status when the
        # Critic's claim is explicit, not when it merely quotes a counterexample.
        return {"type": "critic_oracle_entry_status_contradiction",
                "listed_entries": sorted(mentioned & listed)}
    if ("oracle" in reason.lower()
            and re.search(
                r"\b(?:not\s+in|not\s+included|absent|missing|"
                r"does\s+not\s+contain|doesn.?t\s+contain)\b",
                reason, re.I
            )):
        return {"type": "critic_oracle_membership_contradiction",
                "listed_entries": sorted(mentioned & listed)}
    return None
