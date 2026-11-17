"""Explicit one-answer and numerical semantics for new question authoring.

The marker is opt-in so previously published Questions keep their
existing validation/replay semantics. The contract contains ONLY frozen
semantic requirements, never private World rows, hidden target values or
Gold Answers.
"""
from __future__ import annotations

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
