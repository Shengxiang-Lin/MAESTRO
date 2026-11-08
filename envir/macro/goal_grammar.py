"""Independent Macro work goals for genuinely new TaskGraph-first user questions.

No MultiModalQA original question or original answer is an input to goal
selection. MMQA contributes scene entities, source handles/modalities and
a composition-structure prior only. These goals are designed for the typed
synthetic observations that the clean environment can actually evaluate.
"""
from __future__ import annotations

GOALS = {
    "lookup": {
        "artifact": "AttributeAuditReport",
        "assessment": "attribute_verification",
        "focus": "attribute_flags",
        "goal": "Audit source-linked candidate attribute flags and identify records with positive observations.",
        "deliverable": "A source-backed attribute audit listing matching candidates and evidence references.",
        "question": (
            "Which source-linked candidate records in the available {sources} observation "
            "summaries for {topic} have the recorded attribute flag set, and which "
            "source supports each selection?"
        ),
    },
    "composition": {
        "artifact": "SourceLinkageReport",
        "assessment": "reference_chain_verification",
        "focus": "candidate_links",
        "goal": "Resolve cross-source candidate references and show how corresponding observations link.",
        "deliverable": "A source-linkage report showing traceable candidate/reference relationships.",
        "question": (
            "How do the candidate references connect across the available {sources} "
            "observation summaries for {topic}, and which source records support "
            "each established link?"
        ),
    },
    "comparison": {
        "artifact": "ComparativeReport",
        "assessment": "comparative_assessment",
        "focus": "attribute_scores",
        "goal": "Compare the recorded synthetic attribute scores of source-linked candidates.",
        "deliverable": "A ranked comparison with source handles for each candidate observation.",
        "question": (
            "Among the candidates documented in the available {sources} observation "
            "summaries for {topic}, which has the highest recorded attribute score, "
            "how do the candidates rank, and what sources support that ranking?"
        ),
    },
    "intersection": {
        "artifact": "ReconciliationReport",
        "assessment": "record_matching",
        "focus": "group_associations",
        "goal": "Reconcile candidate reference groups appearing across independent source records.",
        "deliverable": "A reconciliation report listing matched groups and the evidence for each.",
        "question": (
            "Which candidate groups can be reconciled across the available {sources} "
            "observation summaries for {topic}, and which records provide "
            "the supporting evidence for each match?"
        ),
    },
    "aggregation": {
        "artifact": "AggregateReport",
        "assessment": "candidate_aggregation",
        "focus": "attribute_flags",
        "goal": "Count the source-linked candidate observations that meet the recorded attribute flag.",
        "deliverable": "An evidence-linked aggregate count and a source-by-source candidate register.",
        "question": (
            "How many candidate observations in the available {sources} evidence "
            "summaries for {topic} have their recorded attribute flag set, "
            "and which sources contribute to that count?"
        ),
    },
    "temporal": {
        "artifact": "TimelineReport",
        "assessment": "timeline_reconstruction",
        "focus": "relative_time_indices",
        "goal": "Reconstruct the relative order of source-linked candidate observations.",
        "deliverable": "A source-backed relative chronology identifying the earliest recorded observation.",
        "question": (
            "What is the relative order of the candidate observations across the "
            "available {sources} evidence summaries for {topic}, which observation "
            "is earliest, and which records establish the sequence?"
        ),
    },
}

ORDER = tuple(GOALS)


def select_goal(*, ordinal: int, run_seed: int, source_count: int) -> tuple[str, dict]:
    """Balanced design selection, independent of MMQA original question/intent.

    The ordinal/seed determine the design target; source count only serves as
    an eligibility constraint. A single source is never a multi-work case.
    """
    if source_count < 2:
        raise ValueError("typed multi-source goals need at least two sources")
    key = ORDER[(ordinal + run_seed) % len(ORDER)]
    return key, dict(GOALS[key])


def source_phrase(modalities: list[str]) -> str:
    values = sorted(set(modalities))
    if not values:
        raise ValueError("question must have at least one source modality")
    named = [{"image": "image", "table": "table", "text": "text"}.get(x, x)
             for x in values]
    if len(named) == 1:
        return named[0] + "-source"
    if len(named) == 2:
        return " and ".join(named) + "-source"
    return ", ".join(named[:-1]) + ", and " + named[-1] + "-source"
