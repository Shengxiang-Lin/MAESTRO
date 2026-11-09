"""Python-authoritative Goal semantics and question-surface constraints.

LLMs may name fields and write a motivating fictional scenario. They cannot
change ordering, boolean conjunction, threshold, category-vs-identity
semantics, source universe, answer shape or temporal direction.
"""
from __future__ import annotations

import re

GOAL_RULES = {
    "lookup": {
        "result_kind": "candidate_list",
        "selection": {"positive_flag": True},
        "order": "unspecified",
        "relation": "none",
    },
    "aggregation": {
        "result_kind": "integer_count_plus_sources",
        "selection": {"positive_flag": True},
        "order": "unspecified",
        "relation": "none",
    },
    "comparison": {
        "result_kind": "complete_candidate_ranking_and_top",
        "selection": {},
        "order": "score_descending_then_candidate_id",
        "relation": "none",
    },
    "prioritization": {
        "result_kind": "complete_candidate_priority_order",
        "selection": {},
        "order": "score_descending_then_candidate_id",
        "relation": "none",
    },
    "feasibility": {
        "result_kind": "eligible_candidate_list",
        "selection": {"positive_flag": True, "score_gte": 50,
                      "combine": "and"},
        "order": "unspecified",
        "relation": "none",
    },
    "temporal": {
        "result_kind": "complete_relative_order_and_earliest",
        "selection": {},
        "order": "relative_time_ascending_then_candidate_id",
        "relation": "relative_order_not_calendar_dates",
    },
    "intersection": {
        "result_kind": "recurring_category_groups_with_source_witnesses",
        "selection": {},
        "order": "unspecified",
        "relation": "shared_group_across_distinct_source_ids_NOT_identity",
    },
    "composition": {
        "result_kind": "explicit_cross_source_reference_links",
        "selection": {},
        "order": "unspecified",
        "relation": "directed_record_references_NOT_identity_or_shared_group",
    },
}


def goal_rules(intent: str) -> dict:
    if intent not in GOAL_RULES:
        raise ValueError("unsupported_goal:" + str(intent))
    # Entries contain immutable scalar and nested selection dictionaries;
    # return a defensive copy for serialization and caller fingerprinting.
    import copy
    return copy.deepcopy(GOAL_RULES[intent])


def goal_text_issues(intent: str, text: str) -> list[str]:
    """High-precision, one-direction-only contradiction checks.

    Absence of certain keywords is not proof of Goal drift. More nuanced
    equivalence remains an independent Critic responsibility.
    """
    q = str(text).casefold()
    issues = []
    if intent in {"comparison", "prioritization"} and re.search(
        r"\b(?:ascending|lowest\s+(?:to|first|score)|"
        r"smallest\s+to\s+largest|increasing\s+score|"
        r"from\s+low(?:est)?\s+to\s+high(?:est)?)\b", q):
        issues.append("goal_order_direction_changed")
    if intent == "temporal" and re.search(
        r"\b(?:latest\s+first|descending\s+(?:time|date|chronology)|"
        r"reverse\s+chronolog|newest\s+first)\b", q):
        issues.append("goal_time_direction_changed")
    if intent == "intersection" and re.search(
        r"\b(?:same\s+(?:individuals?|people|players?|persons?)|"
        r"identical\s+(?:entities|persons|players)|"
        r"same\s+named\s+entity)\b", q):
        issues.append("group_match_falsely_equated_with_entity_identity")
    if intent == "composition" and re.search(
        r"\b(?:same\s+group\s+proves|"
        r"identical\s+person\s+in\s+both)\b", q):
        issues.append("explicit_reference_falsely_equated_with_identity")
    if intent == "feasibility" and re.search(
        r"\b(?:score\s+(?:above|at\s+least|>=?)\s*(?:[0-4]\d|[6-9]\d)|"
        r"(?:flag|marked)\s+positive\s+or\s+"
        r"(?:score|rating))\b", q):
        issues.append("goal_feasibility_predicate_changed")
    return issues


_INTERNAL = re.compile(
    r"\b(?:PriorityReport|AttributeAuditReport|SourceLinkageReport|"
    r"ComparativeReport|ReconciliationReport|AggregateReport|"
    r"TimelineReport|FeasibilityReport|"
    r"WorkProduct|TaskGraph|EvidenceGraph|"
    r"synthetic(?:_attribute)?|simulator|"
    r"relative_time_index|candidate_key|"
    r"positive_flag|score_gte|source_id|fact_id|"
    r"source_fact_ref|source_grant|"
    r"candidate-[0-9a-f]{6,}|src-[0-9a-f]{6,}|"
    r"group-[0-9]+)\b", re.I
)

_FAKE_MEDIA = re.compile(
    r"\b(?:the\s+)?(?:image|video|photo|poster|screenshot)\b"
    r".{0,65}\b(?:proves?|visually\s+confirms?|depicts?|shows?|"
    r"demonstrates?)\b"
    r"|\b(?:visually\s+confirmed|detected\s+in\s+(?:the\s+)?image)\b",
    re.I,
)


def question_surface_issues(text: str) -> list[str]:
    issues = []
    if _INTERNAL.search(str(text)):
        issues.append("question_exposes_internal_schema_or_artifact")
    if _FAKE_MEDIA.search(str(text)):
        issues.append("question_asserts_unobserved_media")
    return issues
