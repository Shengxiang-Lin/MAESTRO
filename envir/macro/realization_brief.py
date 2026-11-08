"""Natural-language realization brief, separated from the frozen semantic contract.

The Author should not see verbose Macro Task descriptions, Recipe IDs,
synthetic field names or deterministic template questions as writing examples.
Those are strong lexical anchors and encourage template copying. The Critic
continues to receive the full frozen contract and TaskGraph independently.
"""
from __future__ import annotations

from .semantic_contract import semantic_contract


# A description of the *same* question in ordinary language. These are
# information needs, not candidate user questions or question templates.
# Do not replace synthetic scores/flags with unsupported domain semantics.
INTENT_BRIEFS = {
    "lookup": (
        "Find entries with a positive recorded mark.",
        "Identify the matching entries and where the marks are documented.",
    ),
    "composition": (
        "Determine how entries mentioned in different materials refer to or "
        "connect with one another.",
        "Describe the traceable connections and the supporting records.",
    ),
    "comparison": (
        "Compare the scores actually recorded for the entries.",
        "Identify the highest-scoring entry, give the complete score ordering, "
        "and indicate which records support it.",
    ),
    "intersection": (
        "Find which recorded groups recur across independent materials.",
        "Name the recurring groups and the records establishing each overlap.",
    ),
    "aggregation": (
        "Count entries with a positive recorded mark.",
        "Give the count and identify the records contributing to it.",
    ),
    "temporal": (
        "Arrange entries by their recorded relative sequence.",
        "Give the ordering, indicate which appears earliest, and support the "
        "sequence using the records. Do not invent calendar dates.",
    ),
    "prioritization": (
        "Place the entries in descending order of their recorded scores.",
        "Give the full priority order and the sources supporting it.",
    ),
    "feasibility": (
        "Find entries marked positive and with a recorded score of at least 50.",
        "Identify all qualifying entries and the records supporting the choice.",
    ),
}


def question_realization_brief(scene: dict) -> dict:
    """Return an Author-only non-template brief without private observations."""
    spec = semantic_contract(scene)
    intent = spec["goal_id"]
    if intent not in INTENT_BRIEFS:
        raise ValueError("unknown author intent")
    need, output = INTENT_BRIEFS[intent]
    return {
        "topic": spec["authorized_topic"],
        "information_need": need,
        "complete_answer": output,
        "available_materials": (
            "There are existing source records about this topic. "
            "Refer to these collectively as 'the records', 'the available "
            "materials', or 'these sources' if necessary. "
            "The user will see source descriptions separately. "
            "You are not required to name each source modality in the Question."
        ),
        "allowed_source_modalities": spec["allowed_modalities"],
        "explicit_conditions": spec["hard_constraints"],
        "language_guide": (
            "Use ordinary English and one focused question. Replace internal "
            "terms such as candidate_key, synthetic_attribute_flag, "
            "synthetic_attribute_score, candidate observations, source-linked, "
            "WorkProduct, Recipe or GoalAssessment with natural equivalents. "
            "Keep the *meaning* of positive marks, recorded scores and "
            "relative ordering, without turning them into new real-world facts."
        ),
        "evidence_boundary": (
            "Do not invent any event, account, person, source, observed image "
            "content, source subtype, measurement meaning, or calendar date. "
            "Neutral conversational framing ('I'm looking at the records') is "
            "allowed; claiming a new account or document is not."
        ),
    }
