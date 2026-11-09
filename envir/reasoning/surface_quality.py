"""Pure, conservative language diagnostics for fictional benchmark questions.

These checks catch specifically observed high-confidence failures.
They are not semantic-equivalence proofs or a substitute for a Critic.
"""
from __future__ import annotations

import re

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
STRICT_INTERVAL = re.compile(
    r"\bstrictly\s+(?:\w+\s+){0,3}(?:within|inside|between)\b",
    re.I,
)
INCLUSIVE_INTERVAL = re.compile(
    r"\b(?:inclusive|closed)(?:\s+(?:start|end|time|interval|window|boundary|boundaries))?\b"
    r"|\bincluding\s+(?:both\s+)?(?:endpoints|boundaries)\b",
    re.I,
)


def unlicensed_calendar_years(text: str, topic: str) -> list[str]:
    """Years in the candidate but absent from the original topic anchor."""
    if not isinstance(text, str):
        return []
    return sorted(set(YEAR_RE.findall(text)) - set(YEAR_RE.findall(topic or "")))


def normalize_question_punctuation(question: str) -> str:
    """Repair harmless repeated terminal question marks, never alter predicates."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    return re.sub(r"\?{2,}", "?", question.strip())


def question_language_warnings(question: str, spec: dict) -> list[str]:
    if not isinstance(question, str):
        return ["question_not_string"]
    warnings = []
    if "??" in question:
        warnings.append("repeated_question_marks")
    if (spec.get("reasoning_pattern") == "time_window"
            and STRICT_INTERVAL.search(question)
            and INCLUSIVE_INTERVAL.search(question)):
        warnings.append("strict_vs_inclusive_time_window")
    topic = spec.get("scenario", {}).get("topic", "")
    if unlicensed_calendar_years(question, topic):
        warnings.append("unlicensed_calendar_year_not_in_topic")
    return warnings
