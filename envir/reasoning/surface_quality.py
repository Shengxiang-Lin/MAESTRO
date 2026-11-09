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


def topic_mentioned(text: str, topic: str) -> bool:
    """Require the grounded topic, allowing harmless typographic variation.

    A parenthetical *type* suffix may be omitted ("Green Lantern (film)"
    -> "Green Lantern"); not arbitrary words, invented organizations,
    unrelated aliases, or semantic/embedding-only matches.
    """
    if not isinstance(text, str) or not isinstance(topic, str):
        return False
    def canonical(value: str) -> str:
        value = value.casefold()
        value = re.sub(r"[\u2010-\u2015\u2212]", "-", value)
        value = re.sub(r"[\u2018\u2019]", "'", value)
        return re.sub(r"\s+", " ", value).strip()
    candidate = canonical(text)
    anchor = canonical(topic)
    core = re.sub(r"\s*\([^()]+\)\s*$", "", anchor).strip()
    return bool(anchor and (anchor in candidate or
                            len(core) >= 5 and core in candidate))


REQUEST_VERB = (
    r"(?:identify|list|find|determine|report|return|select|count|provide|"
    r"show|give|tell|extract|name|calculate|check|filter|compare)"
)


def is_explicit_user_request(text: str) -> bool:
    """An imperative is a valid benchmark task even without '?'.

    Do not treat an arbitrary fact description or fragment as an actionable
    question. All semantic constraints still go through the deterministic
    guards and independent Critic.
    """
    if not isinstance(text, str):
        return False
    normalized = text.strip()
    if "?" in normalized:
        return True
    if len(normalized) < 55:
        return False
    if re.search(
        rf"(?:^|[.!?]\s+|[,;:]\s+)(?:please\s+|kindly\s+)?"
        rf"{REQUEST_VERB}\b",
        normalized, re.I,
    ):
        return True
    if re.search(
        rf"\b(?:I need to|I want to|I would like to|we need to)"
        rf"\s+(?:please\s+)?{REQUEST_VERB}\b",
        normalized, re.I,
    ):
        return True
    return False
