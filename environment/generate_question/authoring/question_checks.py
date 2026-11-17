"""Question semantic correctness checks for archived and current task plans.

Keeps field fidelity and goal-lock checks together without changing their
versioned contracts. Historical root imports remain compatibility facades.
"""
from __future__ import annotations

# --- exact answer-field fidelity ---
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
        for sentence in re.split(r"(?<=[.!?])\s+", question):
            if (_NUMERIC_TIME.search(sentence)
                    and not _LOWER_BOUND.search(sentence)):
                problems.append("semantic_frame_lower_bound_not_expressed")
    if (temporal.get("footage_not_after_commentary")
            and _STRICT_ORDER.search(question)):
        problems.append("semantic_non_strict_order_claimed_strict")
    return sorted(set(problems))

# --- frozen goal semantic lock ---
import re


QUESTION_SEMANTIC_LOCK_VERSION = "maestro.question_semantic_lock.v1"

_ROUND_DOWN = re.compile(
    r"\b(?:round(?:ed|ing)?\s+down|"
    r"round\s+toward\s+negative\s+infinity|"
    r"floor(?:ed|ing)?|truncat(?:e|ed|ing|ion))\b",
    re.I,
)
_WHOLE_PERCENT = re.compile(
    r"\b(?:whole|integer|whole-number)\s+(?:number\s+)?"
    r"(?:percent(?:age)?|percentage\s+points?)\b",
    re.I,
)


def question_author_requirements(spec: dict) -> dict:
    if spec.get("question_semantic_lock_version") != QUESTION_SEMANTIC_LOCK_VERSION:
        raise ValueError("question_semantic_lock_not_enabled")
    plan = spec["macro_plan"]
    kind = plan["goal_resolution"]["kind"]
    requirements = {
        "topic_verbatim": spec["scenario"]["topic"],
        "single_requested_output": plan["goal"]["public_label"],
        "answer_cardinality": "exactly one field; identity is a filter, not output",
        "required_goal_meaning": plan["goal"]["meaning"],
        "no_additional_output_fields": True,
        "no_extra_eligibility_or_time_conditions": True,
        "allow_surface_rephrasing": True,
        "avoid_copying_reference_sentence": True,
    }
    if kind == "calculation":
        requirements["mandatory_computation"] = (
            "completed / total * 100; ROUND DOWN to a WHOLE PERCENT. "
            "Both the rounding direction and whole-percent unit must be "
            "explicit in the final question."
        )
    elif kind == "comparison":
        requirements["mandatory_comparison"] = (
            "Determine which independently recorded count is HIGHER: "
            "first source or second source. Ask for only this result; "
            "do not add an identity answer."
        )
    pattern = plan["pattern"]
    if pattern == "corroborated_dispute":
        requirements["mandatory_filters"] = [
            "the two independent report years differ",
            "the independent event log tag is literally "
            + repr(plan["clues"]["case_tag"]),
        ]
    elif pattern == "disputed_reports":
        requirements["mandatory_filters"] = [
            "the two independent report years differ",
        ]
    elif pattern == "multimodal_event":
        requirements["mandatory_filters"] = [
            "video action occurs no later than audio cue (ties allowed)",
            "the independent event log tag is literally "
            + repr(plan["clues"]["case_tag"]),
        ]
    return requirements


def question_semantic_lock_issues(question: str, spec: dict) -> list[str]:
    """Only add high-precision checks missed by the existing guards."""
    if spec.get("question_semantic_lock_version") != QUESTION_SEMANTIC_LOCK_VERSION:
        return []
    if not isinstance(question, str):
        return ["question_semantic_lock_not_text"]
    if spec["macro_plan"]["goal_resolution"]["kind"] != "calculation":
        return []
    issues = []
    if not _ROUND_DOWN.search(question):
        issues.append("question_calculation_round_down_missing")
    if not (_WHOLE_PERCENT.search(question)
            or re.search(r"\bwhole\s+percent(?:age)?\b", question, re.I)):
        issues.append("question_calculation_whole_percent_missing")
    return issues
