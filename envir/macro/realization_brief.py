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
        "Identify all entries that are marked positive.",
    ),
    "composition": (
        "Determine how entries mentioned in different materials refer to or "
        "connect with one another.",
        "Identify the traceable connections among the entries.",
    ),
    "comparison": (
        "Compare the scores actually recorded for the entries.",
        "Identify the highest-scoring entry and the complete score ordering. "
        "",
    ),
    "intersection": (
        "Find which recorded groups recur across independent materials.",
        "Name the recurring groups and their overlaps.",
    ),
    "aggregation": (
        "Count entries with a positive recorded mark.",
        "Determine the complete count.",
    ),
    "temporal": (
        "Arrange entries by their recorded relative sequence.",
        "Give the ordering and identify which appears earliest. "
        "Do not invent calendar dates.",
    ),
    "prioritization": (
        "Place the entries in descending order of their recorded scores.",
        "Give the complete score-based priority order.",
    ),
    "feasibility": (
        "Find entries marked positive and with a recorded score of at least 50.",
        "Identify all qualifying entries under both stated conditions.",
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
        "numerical_condition": (
            "A recorded score of at least 50."
            if intent == "feasibility" else None
        ),
        "language_guide": (
            "Use ordinary English and one focused question. Replace internal "
            "technical identifiers, source-linked candidate jargon, "
            "or backend task labels with natural equivalents. "
            "Keep the *meaning* of positive marks, recorded scores and "
            "relative ordering, without turning them into new real-world facts."
        ),
        "solver_provenance_obligation": (
            "The frozen solution still verifies evidence and preserves "
            "source references. These are internal answer standards; the "
            "user's Question need not explicitly demand citations, "
            "source enumeration or verification steps."
        ),
        "evidence_boundary": (
            "Do not invent any event, account, person, source, observed image "
            "content, source subtype, measurement meaning, or calendar date. "
            "Neutral conversational framing ('I'm looking at the records') is "
            "allowed; claiming a new account or document is not."
        ),
    }
