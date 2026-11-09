"""Canonical, typed information intent distinct from its rendered question.

The Author sees a semantic request contract, not a fixed opening sentence.
Every substantive predicate corresponds to an executable source-scoped
operation in the approved Program. A full reference question is retained
for the independent Critic and an explicit last-chance repair prompt.
"""
from __future__ import annotations

from envir.schema import digest

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
    constraints = list(PATTERN_CONSTRAINTS[pattern])
    if spec["difficulty_target"] == "easy":
        constraints = constraints[:1]
    if spec["difficulty_target"] == "hard":
        constraints.append(
            "Also require entry-specific clearance to be marked cleared "
            "in a separate source, not just group approval."
        )
    return {
        "topic": spec["scenario"]["topic"],
        "hypothetical": True,
        "setting": spec["scenario"]["setting"],
        "entity_kind": " ".join(
            spec["scenario"]["entity_kind"].replace("_", " ").split()
        ),
        "user_need": spec["scenario"]["purpose"],
        "reasoning_pattern": pattern,
        "difficulty": spec["difficulty_target"],
        "nonnegotiable_conditions": constraints,
        "output_contract": FAMILY_OUTPUTS[spec["family"]],
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
