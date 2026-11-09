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


SCHEMA_LABELS = {
    "diagram_step_start": "diagram interval start step",
    "diagram_step_end": "diagram interval end step",
    "utterance_step_start": "audio utterance start step",
    "utterance_step_end": "audio utterance end step",
    "clip_step_start": "video clip start step",
    "clip_step_end": "video clip end step",
    "measured_value": "measured value",
    "minimum_required": "minimum required threshold",
}


def normalize_question_punctuation(question: str) -> str:
    """Make only semantics-preserving text repairs BEFORE Critic review.

    The repaired text is what the Critic actually evaluates and, if
    approved, what becomes public. It never adds criteria, drops source
    requirements or changes a comparison.
    """
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    q = question.strip()
    for field, label in SCHEMA_LABELS.items():
        q = re.sub(r"\b" + re.escape(field) + r"\b", label, q)
    q = re.sub(r"\?{2,}", "?", q)
    q = re.sub(r"\.\s*\?$", "?", q)
    # Some LLMs leave a single unpaired final quote after the answer
    # instruction. Do not touch balanced quotation marks around topics.
    if q.endswith('"') and q.count('"') % 2:
        q = q[:-1].rstrip()
    return q


def unsupported_macro_predicates(question: str, spec: dict) -> list[str]:
    """High-precision extra eligibility criteria missing from the Program.

    This is intentionally narrow: any unrecognized phrasing still needs
    Critic judgement. Never infer that "active" implies "not archived",
    that a matching time-window is "approved", or that a policy group
    passing a dynamic-threshold comparison is also independently approved.
    """
    if not isinstance(question, str):
        return ["question_not_string"]
    pattern = spec.get("reasoning_pattern", "approval")
    difficulty = spec.get("difficulty_target")
    problems = []
    if re.search(
        r"\b(?:non[- ]archived|unarchived)\b"
        r"|\b(?:exclude|excluding|excluded)\b(?:\s+\w+){0,7}"
        r"\s+archived\b",
        question, re.I,
    ):
        problems.append("unlicensed_archive_status_filter")
    if pattern in {"time_window", "status_disagreement", "dynamic_threshold"}:
        if re.search(
            r"\b(?:approved|approval[- ]verified|authorized)\b"
            r"(?:\s+\w+){0,3}\s+"
            r"(?:group|session|window|interval|policy)\b",
            question, re.I,
        ):
            problems.append("unlicensed_independent_approval_filter")
    if pattern == "approval" and difficulty == "easy":
        if re.search(
            r"\bqualif(?:y|ies|ied)\b(?:\s+\w+){0,6}"
            r"\s+\bapproved\b",
            question, re.I,
        ):
            problems.append("unlicensed_easy_approval_filter")
    return sorted(set(problems))


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
    warnings.extend(unsupported_macro_predicates(question, spec))
    return sorted(set(warnings))


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
