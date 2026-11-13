"""Non-destructive Main-visible presentation of already reviewed Stage-2 reports.

Only local narrative wording is normalized. The original Stage-2 JSONL,
semantic result, status, source FactRefs, output relation and execution
trace remain unchanged. Never call this on deliberately untrusted 4B
feedback: its discrepancies are part of the evaluation stimulus.
"""
from __future__ import annotations

from copy import deepcopy
import re


# These replacements are *only* referential style edits. No assertion,
# predicate, source name, negation, numeric value or identity is rewritten.
_SURFACE_REWRITES = (
    (re.compile(r"\bthis macro\b", re.I), "this task"),
    (re.compile(r"\bthe macro\b", re.I), "the task"),
    (re.compile(r"\bthe macro's\b", re.I), "the task's"),
    (re.compile(r"\bmacro output\b", re.I), "returned evidence"),
    (re.compile(r"\bthis stage\b", re.I), "this step"),
    (re.compile(r"\bthe stage\b", re.I), "the step"),
    (re.compile(r"\bupstream stage\b", re.I), "earlier step"),
    (re.compile(r"\bdownstream stage\b", re.I), "subsequent step"),
    (re.compile(r"\bworkproducts\b", re.I), "results"),
    (re.compile(r"\bworkproduct\b", re.I), "result"),
    (re.compile(r"\boutputted\b", re.I), "returned"),
)

# The change must not corrupt any observable literal key, numeric
# measurement, quoted source FactRef or answer identifier.
_ANCHORS = re.compile(
    r"(?<![A-Za-z0-9_])(?:Case-\d+|Entry-\d+|MK-\d+|"
    r"KEY\d+-\d+|[A-Za-z][A-Za-z0-9_-]*:F[A-Za-z0-9_-]+|"
    r"\d+(?:\.\d+)?%?)(?![A-Za-z0-9_])"
)


def clean_public_prose(value: str) -> str:
    """Remove known self-references without adding or deleting any fact."""
    if not isinstance(value, str):
        raise TypeError("public_surface_requires_string")
    cleaned = value
    for pattern, replacement in _SURFACE_REWRITES:
        cleaned = pattern.sub(replacement, cleaned)
    if _ANCHORS.findall(value) != _ANCHORS.findall(cleaned):
        # Keep the original rather than silently corrupt an identifier.
        raise ValueError("public_surface_fact_anchor_changed")
    return cleaned


def reviewed_public_view(record: dict) -> dict:
    """Return a sanitized COPY of a vetted Stage-2 record.

    Never alter finish_result.result or .status; they are part of the
    precise teacher-verified handoff. Only summary and human-readable
    trace narrative may change. The original report remains immutable.
    """
    if record.get("report_authoring", {}).get("status") != "accepted":
        raise ValueError("public_surface_requires_reviewed_stage2")
    public = deepcopy(record)
    finish = public.get("finish_result")
    if not isinstance(finish, dict) or finish.get("status") != "done":
        raise ValueError("public_surface_requires_done_finish")
    if isinstance(finish.get("summary"), str):
        finish["summary"] = clean_public_prose(finish["summary"])
    trace = public.get("trace_summary")
    if isinstance(trace, dict) and isinstance(trace.get("narrative"), str):
        trace["narrative"] = clean_public_prose(trace["narrative"])
    return public
