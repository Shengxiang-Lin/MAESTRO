"""Canonical, typed information intent distinct from its rendered question.

The Author sees a semantic request contract, not a fixed opening sentence.
Every substantive predicate corresponds to an executable source-scoped
operation in the approved Program. A full reference question is retained
for the independent Critic and an explicit last-chance repair prompt.
"""
from __future__ import annotations

from environment.schema import digest

PATTERN_CONSTRAINTS = {
    "approval": (
        "Only active entries in the register are eligible.",
        "Translate each entry's abbreviated category code using the glossary.",
        "The independent annotation must explicitly approve that resolved category.",
    ),
    "time_window": (
        "Only active entries in the register are eligible.",
        "Map each entry's code to a simulated session using the glossary.",
        "The entry's recorded event step must fall inside the matching annotation's closed start/end interval.",
        "The interval endpoints are fictional local indices, not actual historical dates.",
    ),
    "status_disagreement": (
        "Only active entries in the register are eligible.",
        "Use the glossary to align each entry's code with the annotation group.",
        "Keep entries where the entry's reported binary status differs from the independently observed group status.",
    ),
    "clue_chain": (
        "Identify the pictured exhibit using BOTH given emblem features.",
        "Read the marker code shown on the matching illustrated exhibit.",
        "Use that discovered marker to obtain its accession identifier from the keyed index.",
        "Use that accession identifier to retrieve ONLY its matching archive record.",
        "Report the exhibit's accession year. The image is a simulated observation.",
    ),
    "dynamic_threshold": (
        "Only active entries in the register are eligible.",
        "Use the glossary to map each entry's policy code to a policy group.",
        "Keep entries whose recorded measurement is at least the matching group's independently supplied minimum.",
    ),
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
    elif pattern == "macro_first":
        mode = spec["macro_plan"]["pattern"]
        constraints = {
            "serial_discovery": [
                "Identify the relevant fictional item using both visual emblem features.",
                "Return its authoritative year as specified by the final source.",
            ],
            "parallel_resolution": [
                "Resolve one identity using BOTH an independent visual color"
                " and a separate motif description.",
                "Return the authoritative year for that uniquely resolved identity.",
            ],
            "conflict_reconciliation": [
                "Locate the case where two independent inventories disagree.",
                "Return the authoritative year for the disputed case.",
            ],
        }[mode]
    elif pattern == "evidence_graph":
        archetype = spec["evidence_graph"]["archetype"]
        constraints = {
            "chain": [
                "Identify the pictured exhibit using BOTH emblem features.",
                "Obtain its authoritative accession year from available records.",
            ],
            "parallel": [
                "Use BOTH independent visual and caption clues to resolve one exhibit.",
                "Find that exhibit's authoritative accession year.",
            ],
            "disambiguation": [
                "The illustrated emblem color alone does not identify one exhibit.",
                "Use the independent motif notes to disambiguate the candidates.",
                "Find the resolved exhibit's authoritative accession year.",
            ],
            "conflict": [
                "The two inventories disagree about the recorded year of one case.",
                "Identify the disputed case and report its authoritative accession year.",
            ],
        }[archetype]
    else:
        constraints = list(PATTERN_CONSTRAINTS[pattern])
    if spec["difficulty_target"] == "easy":
        constraints = constraints[:1]
    if spec["difficulty_target"] == "hard" and pattern not in {
        "evidence_graph", "macro_first", "macro_first_v2"
    }:
        constraints.append(
            "Also require entry-specific clearance to be marked cleared "
            "in a separate source, not just group approval."
        )
    applied_clues = None
    if spec.get("semantic_evidence_contract_version"):
        from .semantic_evidence import compile_semantic_contract
        applied_clues = compile_semantic_contract(spec)[
            "necessary_question_clues"
        ]
    return {
        "visual_clue": (
            applied_clues if applied_clues is not None else
            spec["macro_plan"]["clues"] if pattern in {
                "macro_first", "macro_first_v2"
            }
            else spec.get("investigation", {}).get("visual_features")
        ),
        "acquisition_chain": (
            spec.get("investigation", {}).get("acquisition_chain")
        ),
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
            "Report the authoritative " +
            spec["macro_plan"]["goal"]["answer_field"].replace("_", " ") +
            " for the uniquely identified case."
            if pattern == "macro_first" else
            ("Return the confirmed " +
             spec["macro_plan"]["goal"]["public_label"] +
             " for the identified case." if pattern == "macro_first_v2"
             else FAMILY_OUTPUTS[spec["family"]])
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
