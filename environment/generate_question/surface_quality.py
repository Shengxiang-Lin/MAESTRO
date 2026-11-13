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
    "entry_label": "entry label",
    "evidence_refs": "supporting source references",
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


# High precision patterns: a mere background mention is not a filter.
# Only these explicit criteria can be safely screened without a full
# semantic theorem prover. The independent Critic checks other paraphrases.
_EXTRA_FILTERS = {
    "archived": r"archiv(?:ed|ing)|unarchived|non[- ]archived",
    "published": r"published|unpublished",
    "certified": r"certified|uncertified",
    "funded": r"funded|unfunded",
    "deleted": r"deleted|undeleted",
}
_FILTER_PREFIX = (
    r"(?:only|no|not|without|exclude|excluding|excluded|"
    r"remove|removing|omit|omitting|filter(?:ing)?\s+out|"
    r"must\s+(?:not\s+)?be|should\s+(?:not\s+)?be|"
    r"cannot\s+be|keep\s+only|retain\s+only)"
)


def _macro_filter_fields(graph: dict | None) -> set[str]:
    if graph is None:
        return {"active", "approved", "cleared"}
    return {
        step["field"]
        for task in graph["tasks"]
        for step in task.get("operator_chain", [])
        if step["op"] in {"filter_eq", "filter_gte"}
        and "field" in step
    }


def unsupported_macro_predicates(
    question: str, spec: dict, macro_graph: dict | None = None,
) -> list[str]:
    """Detect *explicit* unlicensed eligibility conditions.

    Use the actual operator chain where possible, never infer extra
    predicates from the existence of a source. High-precision, not a
    complete NL entailment checker. Ambiguous cases go to the Critic.
    """
    if not isinstance(question, str):
        return ["question_not_string"]
    fields = _macro_filter_fields(macro_graph)
    pattern = spec["reasoning_pattern"]
    problems = []
    for name, expression in _EXTRA_FILTERS.items():
        if name in fields:
            continue
        expr = rf"\b(?:{expression})\b"
        # "no archived", "not archived", "exclude archived", including
        # "ensuring no archived records" (previously missed).
        if (re.search(rf"\b{_FILTER_PREFIX}(?:\s+\w+){{0,5}}\s+{expr}",
                      question, re.I)
                or re.search(rf"{expr}\s+(?:only|excluded)\b", question, re.I)):
            problems.append(f"unlicensed_{name}_filter")
    if "approved" not in fields:
        if re.search(
            r"\b(?:approved|approval[- ]verified|authorized)\b"
            r"(?:\s+\w+){0,3}\s+"
            r"(?:groups?|sessions?|windows?|intervals?|polic(?:y|ies))\b",
            question, re.I,
        ):
            problems.append("unlicensed_independent_approval_filter")
        if re.search(
            r"\b(?:approved|approval[- ]verified)\s+(?:entry|record)s?\b",
            question, re.I,
        ):
            problems.append("unlicensed_independent_approval_filter")
    if "cleared" not in fields and re.search(
        r"\b(?:only|must\s+be|require|requiring)\s+"
        r"(?:\w+\s+){0,3}(?:cleared|clearance[- ]verified)\b",
        question, re.I,
    ):
        problems.append("unlicensed_clearance_filter")
    return sorted(set(problems))


_OUTPUT_REQUEST = re.compile(
    r"\b(?:list|return|output|report|provide|show|name|identify|"
    r"give(?:\s+me)?|which|what\s+are)\b"
    r"(?:\s+\w+){0,7}\s+"
    r"(?:abbreviated|shortened|coded)\s+(?:entry\s+)?labels?\b"
    r"|\b(?:list|return|output|report|provide|show|name|identify|"
    r"give(?:\s+me)?|which|what\s+are)\b"
    r"(?:\s+\w+){0,7}\s+"
    r"(?:label[_ ]?codes?|route[_ ]?codes?|policy[_ ]?codes?|"
    r"category[_ ]?codes?)\b",
    re.I,
)


def question_output_issues(
    question: str, spec: dict, macro_graph: dict | None = None,
) -> list[str]:
    """Bind requested display labels to the Graph's *terminal schema*."""
    if not isinstance(question, str):
        return ["question_not_string"]
    expected = spec["requirements"]["answer_kind"]
    if macro_graph is not None:
        terminal = next(
            task for task in macro_graph["tasks"]
            if task["task_id"] == macro_graph["terminal_task_id"]
        )
        fields = set(terminal["outputs"]["typed_relation"])
    else:
        fields = {
            "count" if expected in {"count", "category_counts"}
            else "entry_label"
        }
    issues = []
    if "entry_label" in fields and _OUTPUT_REQUEST.search(question):
        issues.append("requested_code_instead_of_entry_label")
    if expected == "count" and re.search(
        r"\b(?:list|return|provide|identify)\b(?:\s+\w+){0,6}"
        r"\s+(?:entry\s+labels?|individual\s+entries)\b", question, re.I,
    ):
        issues.append("requested_list_instead_of_count")
    if expected == "category_counts" and re.search(
        r"\b(?:list|return|identify)\b(?:\s+\w+){0,6}"
        r"\s+(?:only\s+)?entry\s+labels?\b", question, re.I,
    ):
        issues.append("requested_entry_labels_instead_of_group_counts")
    return issues


def question_language_warnings(
    question: str, spec: dict, macro_graph: dict | None = None,
) -> list[str]:
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
    warnings.extend(unsupported_macro_predicates(question, spec, macro_graph))
    warnings.extend(question_output_issues(question, spec, macro_graph))
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
    r"(?:identify|list|find|locate|determine|report|return|select|count|provide|"
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
