"""Typed question-scope checks backed by executable temporal predicates.

These are deliberately conservative, deterministic checks: they reject
clear category mistakes such as confusing frame timestamp with action
duration. They are not a general natural-language theorem prover.
Older frozen V2 snapshots do not opt into this versioned validator.
"""
from __future__ import annotations

import re

QUESTION_FIDELITY_VERSION = "maestro.question_fidelity.v2"

_COORDINATED_IDENTITY_OUTPUT = re.compile(
    r"\bwhich\s+(?:(?:specific|matching|identified|particular)\s+)?"
    r"(?:(?:video|audio|film|recording|footage)\s+)?"
    r"(?:case|record|entry|event|episode|song|match|game|item|artifact|"
    r"report|person|team|exhibit|venue|clip|file|recording)\b"
    r"[^?!.]{0,260}\b(?:and|plus|as well as)\s+"
    r"(?:what|which|where|who|when)\b",
    re.I,
)
_OUTPUT_COMMAND = re.compile(
    r"\b(?:identify|name|list|return)\s+(?:the\s+)?"
    r"(?:(?:video|audio|film)\s+)?"
    r"(?:case|record|entry|event|episode|song|match|game|item|"
    r"clip|artifact|venue)\b[^?!.]{0,170}"
    r"\b(?:and|then)\s+(?:give|report|provide|return|name|state)\b",
    re.I,
)
_DURATION_CLAIM = re.compile(
    r"\b(?:footage|video|clip|action|signal|movement|event|scene)\b"
    r"[^.?!]{0,75}\b(?:lasting|lasted|lasts|"
    r"duration\s+(?:of|is|was)|continued?\s+for|"
    r"for\s+(?:a\s+)?duration\s+of)\b"
    r"[^.?!]{0,45}\b(?:\d+|one|two|half)[-\s]?"
    r"(?:seconds?|secs?|minutes?|mins?)\b",
    re.I,
)
_START_AT = re.compile(
    r"\b(?:start(?:s|ing|ed)?|began|begin(?:s|ning)?|"
    r"commenc(?:ed|es|ing)|first\s+appeared)\s+"
    r"(?:at|from|around)\s+(?:the\s+)?(?:\d+|one|two|half)"
    r"(?:[-\s]?(?:seconds?|secs?|minutes?|mins?))?\b",
    re.I,
)
_LOWER_BOUND = re.compile(
    r"\b(?:at\s+or\s+after|on\s+or\s+after|"
    r"no\s+earlier\s+than|at\s+least|"
    r"not\s+before)\b[^.?!]{0,45}\b(?:\d+|one|two|half)"
    r"\b|\bfrom\s+\d+\s+(?:seconds?|minutes?)\s+onward\b",
    re.I,
)
_NUMERIC_TIME = re.compile(
    r"\b(?:\d+|one|two|half)[-\s]?(?:seconds?|secs?|minutes?|mins?)\b",
    re.I,
)
_EXACT_TIMESTAMP = re.compile(
    r"\b(?:exactly|precisely)\s+(?:at\s+)?(?:the\s+)?"
    r"\d+[-\s]?(?:second|minute)\s*(?:mark|timestamp|point)?",
    re.I,
)
_STRICT_ORDER = re.compile(
    r"\b(?:strictly|exclusively)\s+(?:before|earlier\s+than)\b",
    re.I,
)


def question_fidelity_issues(question: str, contract: dict) -> list[str]:
    """Reject *unsupported* exact temporal/output claims from new questions."""
    if not isinstance(question, str):
        return ["semantic_question_not_text"]
    problems = []
    if (_COORDINATED_IDENTITY_OUTPUT.search(question)
            or _OUTPUT_COMMAND.search(question)):
        problems.append("semantic_additional_output_requested")
    temporal = contract.get("temporal_semantics", {})
    frame_bounds = temporal.get("frame_minimum_seconds_inclusive", [])
    if frame_bounds:
        if _DURATION_CLAIM.search(question):
            problems.append("semantic_frame_timestamp_as_duration")
        if _START_AT.search(question):
            problems.append("semantic_frame_lower_bound_as_start")
        if _EXACT_TIMESTAMP.search(question):
            problems.append("semantic_frame_lower_bound_as_exact_time")
        # A bare timestamp number without a comparison is ambiguous, so
        # ask the Author to state the actual inclusive lower bound.
        if _NUMERIC_TIME.search(question) and not _LOWER_BOUND.search(question):
            problems.append("semantic_frame_lower_bound_not_expressed")
    if (temporal.get("footage_not_after_commentary")
            and _STRICT_ORDER.search(question)):
        problems.append("semantic_non_strict_order_claimed_strict")
    return sorted(set(problems))
