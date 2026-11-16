"""Canonical, typed information intent distinct from its rendered question.

The Author sees a semantic request contract, not a fixed opening sentence.
Every substantive predicate corresponds to an executable source-scoped
operation in the approved Program. A full reference question is retained
for the independent Critic and an explicit last-chance repair prompt.
"""
from __future__ import annotations

from environment.schema import digest

PATTERN_CONSTRAINTS = {
    "approval": ("Only active entries in the register are eligible.",),
}
FAMILY_OUTPUTS = {
    "reconcile": "List the qualifying entry labels.",
    "eligibility": "List the qualifying entry labels.",
    "evidence_reconcile": (
        "For every qualifying entry label, include the supporting "
        "source fact references in the final answer."
    ),
    "coverage_count": "Return the number of distinct qualifying entries.",
    "category_distribution": (
        "Return the count of qualifying entries for each resolved category."
    ),
    "priority": (
        "Return the single qualifying entry with the highest recorded priority."
    ),
    "chronology": (
        "Return the single qualifying entry with the earliest relative step."
    ),
    "window_alignment": "List the entries whose recorded events fit the allowed time windows.",
    "status_disagreement": "List entries with a genuine disagreement between the two reported statuses.",
    "threshold_eligibility": "List entries meeting the independently defined minimum value.",
    "accession_year": "Report the year in which the identified exhibit was accessioned.",
    "evidence_answer": "Report the accession year justified by discovered evidence.",
    "macro_first": "Return the authoritative year for the uniquely resolved case.",
    "macro_first_v2": "Return the confirmed domain-specific fact for the identified case.",
    "typed_knowledge_v1": "Name the entity reached by the documented YAGO relation path.",
    "typed_knowledge_v2": "Name the entity reached by the documented YAGO relation path.",
    "typed_knowledge_v3": "Identify the fictional project, reconcile its candidate real references, then name the related YAGO entity.",
}
STYLES = (
    ("direct_lookup", "Start with a concise direct question, not a job title."),
    ("decision_framing", "Open with a practical selection decision; put source conditions afterward."),
    ("context_then_question", "Give a short situational context, then ask one question."),
    ("clarification_request", "Phrase as a colleague asking for clarification about matching records."),
    ("outcome_first", "State the required result first, then the relevant conditions."),
    ("single_sentence", "Prefer one clear, self-contained sentence and no role-play introduction."),
    ("two_sentence", "Use one short sentence for context and a second interrogative sentence."),
)


def intent_contract(spec: dict) -> dict:
    pattern = spec.get("reasoning_pattern", "approval")
    if pattern == "macro_first_v2":
        mode = spec["macro_plan"]["pattern"]
        clues = spec["macro_plan"]["clues"]
        constraints = {
            "serial_identity": (
                ["Identify the same case using BOTH visual image features."]
                if clues["visual_modality"] == "image" else
                ["Identify the video event by the named action at or after the specified time threshold."]
            ),
            "parallel_cues": [
                "Resolve the identity using an independent visual clue and"
                " a spoken or written second cue."
            ],
            "disputed_reports": [
                "Find the case for which the two independent reports disagree"
                " on a reported year; do not confuse this with the final answer field."
            ],
            "corroborated_dispute": [
                "First identify cases with conflicting reported years across"
                " two independent reports.",
                "Independently identify cases with the specified event log tag.",
                "Only the intersection of these two candidate sets identifies"
                " the case whose final goal is requested.",
            ],
            "multimodal_event": [
                "Match a video action to a spoken cue from later commentary.",
                "Also match the independent event log tag.",
                "Preserve footage-before-commentary temporal ordering."
            ],
        }[mode]
        constraints.append(
            "Return the confirmed " +
            spec["macro_plan"]["goal"]["public_label"] +
            " for exactly the identified case."
        )
    elif pattern == "typed_knowledge_v3":
        plan = spec["macro_plan"]
        constraints = [
            "Identify exactly one fictional project using all simulated clues.",
            "Use BOTH simulated candidate reference directories: neither "
            "alone establishes a unique real-world entity.",
            "Follow the real YAGO chain only after the fictional "
            "project-to-entity reference has been reconciled.",
            "Do not describe fictional project entities or their synthetic "
            "associations as historically verified facts.",
            "Return one final linked YAGO entity name.",
        ]
        constraints.extend(
            f"Identifying clue: {info['attribute']} {info['value']}."
            for info in plan["clues"].values()
        )
    elif pattern in {"typed_knowledge_v1", "typed_knowledge_v2"}:
        plan = spec["macro_plan"]
        constraints = [
            "Reconcile all independent fictional dossier observations to identify one case.",
            "Follow the documented YAGO relations from the cross-referenced entity.",
            "Do not treat simulated dossier observations as historical YAGO facts.",
            "Return only the final linked entity name, not intermediate identifiers.",
        ]
        constraints.extend(
            f"Evidence cue: {clue['attribute']} {clue['value']}."
            for clue in plan["clues"].values()
        )
    else:
        constraints = list(PATTERN_CONSTRAINTS["approval"])
    if pattern == "approval":
        constraints = constraints[:1]
    applied_clues = None
    if spec.get("semantic_evidence_contract_version"):
        from .semantic_evidence import compile_semantic_contract
        applied_clues = compile_semantic_contract(spec)[
            "necessary_question_clues"
        ]
    return {
        "visual_clue": (
            applied_clues if applied_clues is not None else
            spec["macro_plan"]["clues"] if pattern in {"macro_first_v2", "typed_knowledge_v1", "typed_knowledge_v2", "typed_knowledge_v3"} else None
        ),
        "acquisition_chain": None,
        "topic": spec["scenario"]["topic"],
        "hypothetical": pattern != "macro_first_v2",
        "setting": spec["scenario"]["setting"],
        "entity_kind": " ".join(
            spec["scenario"]["entity_kind"].replace("_", " ").split()
        ),
        "user_need": spec["scenario"]["purpose"],
        "reasoning_pattern": pattern,
        "difficulty": spec["difficulty_target"],
        "nonnegotiable_conditions": constraints,
        "output_contract": (
            "Return the confirmed " + spec["macro_plan"]["goal"]["public_label"] +
            " for the identified case."
            if pattern == "macro_first_v2" else FAMILY_OUTPUTS[spec["family"]]
        ),
        "sources": [{
            "source_id": role["source_id"],
            "modality": role["modality"],
            "purpose": role["purpose"],
        } for role in spec["sources"]],
        "semantic_equivalence": (
            "All predicates are conjunctive; no new fact, source, threshold, "
            "historical claim, modality observation, or output is allowed."
        ),
    }


def style_contract(spec: dict) -> dict:
    token = int(digest(
        spec["seed"], spec["task_id"], "question-style"
    )[:12], 16)
    name, guidance = STYLES[token % len(STYLES)]
    return {"style_id": name, "guidance": guidance,
            "avoid_opening": "As a / As an ...",
            "do_not_copy_reference_question": True}
