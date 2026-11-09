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
    brief = {
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
    grounding = scene.get("scene_grounding")
    if grounding and grounding.get("status") == "accepted" and (
            grounding.get("version") == "semantic_scenario_contract_v1"):
        scenario = grounding["scenario"]
        bindings = grounding["semantic_bindings"]
        brief["hypothetical_scene"] = {
            "context": scenario["setting"],
            "user_role": scenario["user_role"],
            "motivation": scenario["motivation"],
            "candidate_kind": bindings["candidate_kind"],
            "field_meanings": {
                key: value for key, value in bindings.items()
                if key != "candidate_kind"
            },
            "goal_need": grounding["user_need"],
            "frozen_question_intent": grounding["question_intent"],
            "sample_entries": [{
                "display_label": card["candidate_label"],
                "record_role": card["source_role"],
            } for card in grounding["source_cards"]],
            "provenance": (
                "A source-bound hypothetical SYNTHETIC scenario. "
                "No original image/video/audio was observed. "
                "Use the approved object, attribute and source meanings, "
                "but never add a new operation, observation, source or "
                "historical claim."
            ),
        }
        brief["language_guide"] += (
            " Express this exact information need naturally within the "
            "approved domain Scenario Contract; preserve every formal "
            "Goal condition, answer component and candidate scope. "
            "Do not describe internal implementation steps."
        )
        brief["evidence_boundary"] = (
            "Only approved hypothetical scenario and source-bound labels "
            "are licensed, not actual historical assertions or observed media."
        )
    elif grounding and grounding.get("status") == "accepted":
        brief["hypothetical_scene"] = {
            "context": grounding["scene_context"],
            "candidate_kind": grounding["candidate_kind"],
            "field_meanings": grounding["field_meanings"],
            "goal_need": grounding["user_need"],
            "sample_entries": [
                {
                    "display_label": card["candidate_label"],
                    "record_role": card["source_role"],
                }
                for card in grounding["source_cards"]
            ],
            "provenance": (
                "These are validated fictional interpretation labels for "
                "existing simulated source values, NOT original MMQA facts. "
                "You may use these labels, but do not add new events, "
                "corrections, measurements, dates or source claims."
            ),
        }
        brief["language_guide"] += (
            " Use the domain-appropriate labels and hypothetical context "
            "when it improves clarity; do not revert to generic entries "
            "and marks unless the scenario lacks a clearer equivalent."
        )
        brief["evidence_boundary"] = (
            "The approved hypothetical_scene is allowed as fictional "
            "scenario wording, but no claims about actual historical "
            "events, images or external original documents are permitted."
        )
    world_spec = scene.get("semantic_world_spec")
    if world_spec and world_spec.get("status") == "accepted":
        scene_description = world_spec["scenario"]
        meanings = world_spec["semantic_bindings"]
        brief["hypothetical_scene"] = {
            "context": scene_description["setting"],
            "user_role": scene_description["user_role"],
            "motivation": scene_description["motivation"],
            "candidate_kind": meanings["candidate_kind"],
            "field_meanings": {
                k: v for k, v in meanings.items() if k != "candidate_kind"
            },
            "goal_need": world_spec["user_need"],
            "frozen_question_intent": world_spec["goal_contract"],
            "sample_entries": [{
                "display_label": x["candidate_label"],
                "record_role": x["source_role"],
            } for x in world_spec["source_bindings"]],
            "provenance": (
                "The hypothetical domain meanings were set BEFORE numeric "
                "source facts were generated. Existing World scores, flags, "
                "groups and time indices are simulated, not real media "
                "observations or verified historical information."
            ),
        }
        brief["language_guide"] += (
            " Express the existing Goal using the approved candidate kind, "
            "field meanings and scenario purpose. Do not introduce extra "
            "operations or historical facts. Do not copy the reference "
            "Question's abstract terminology."
        )
        brief["evidence_boundary"] = (
            "The semantic World is fictional, not an observed real source. "
            "The author may use its field meanings and fictional labels, "
            "but cannot assert new facts or image/video contents."
        )
    return brief
