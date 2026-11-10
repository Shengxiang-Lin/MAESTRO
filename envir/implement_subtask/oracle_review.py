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
    # Answer provenance is complete even when the evidence samples from a
    # long register scan contain only the first few atoms.
    for output in teacher_record.get("answer_with_provenance", []):
        label = output.get("values", {}).get("entry_label")
        if not isinstance(label, str):
            continue
        for ref in output.get("source_fact_refs", []):
            if not ref.startswith("register:"):
                continue
            previous = result.get(ref)
            if previous is not None and previous["entry_label"] != label:
                raise ValueError("oracle_answer_provenance_conflicts_with_register")
            if previous is None:
                result[ref] = {"entry_label": label, "active": None}
    return result


def positive_entry_provenance(teacher_record: dict) -> dict[str, list[str]]:
    """Complete register refs for listed Oracle rows, even if atoms truncate."""
    output = {}
    for row in teacher_record.get("answer_with_provenance", []):
        label = row.get("values", {}).get("entry_label")
        if isinstance(label, str):
            output[label.casefold()] = [
                ref for ref in row.get("source_fact_refs", [])
                if ref.startswith("register:")
            ]
    return output


def mismatched_entry_refs(
    text: str, fact_index: dict, positive_provenance: dict | None = None,
) -> list[dict]:
    if not isinstance(text, str):
        return []
    pairs = [(a, b) for a, b in _NEAR.findall(text)]
    pairs.extend((entry, ref) for ref, entry in _REVERSE.findall(text))
    result = []
    positive_provenance = positive_provenance or {}
    for entry, ref in pairs:
        actual = fact_index.get(ref, {}).get("entry_label")
        confirmed_refs = positive_provenance.get(entry.casefold(), [])
        if (actual is not None and actual.casefold() != entry.casefold()
                or (ref.startswith("register:")
                    and confirmed_refs and ref not in confirmed_refs)):
            result.append({
                "stated_entry": entry, "cited_ref": ref,
                "actual_entry": actual,
                "expected_register_refs": confirmed_refs,
            })
    return result


def critic_oracle_contradiction(
    review: dict, result_text: str, rows: list[dict], fact_index: dict,
    positive_provenance: dict | None = None,
) -> dict | None:
    """Return a confirmed Critic factual error; never auto-accept the report."""
    if not full_entry_list_matches(result_text, rows):
        return None
    reason = str(review.get("reason", ""))
    mismatches = mismatched_entry_refs(
        reason, fact_index, positive_provenance
    )
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
