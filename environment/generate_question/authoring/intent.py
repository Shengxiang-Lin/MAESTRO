"""Frozen user intent and deterministic language styles for current task_plan."""
from __future__ import annotations
from environment.schema import digest

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
    if spec.get("reasoning_pattern") != "task_plan":
        raise ValueError("unsupported_stage1_reasoning_pattern")
    pattern = "task_plan"
    plan = spec["macro_plan"]
    goal_kind = plan["goal_spec"]["source_profile"]
    constraints = [
        "Combine the independently recorded identifying clues.",
        "Resolve the matching subject from the supplied observations before consulting its related records.",
        (
            "Report ONLY the entity at the end of the relation chain."
            if goal_kind == "entity_resolution" else
            "Report ONLY the frozen typed Goal output: "
            + plan["goal"]["public_label"] + ". "
            + "Preserve every typed field and requested computation."
        ),
        "Write the user request as a domain-specific information investigation; "
        "do not mention generators, simulated worlds, YAGO or provenance.",
        "Do not assert unsupported external historical facts.",
    ]
    condition = plan["condition"]
    if condition:
        constraints.append(
            "Only completed events qualify."
            if condition["field"] == "event_status"
            else "Only events in or after 2025 qualify."
        )
    compact = plan.get("question_surface_policy") == "macro_goal_reference_card_v2"
    if compact:
        constraints.extend([
            "The initial public view provides a separate reference card "
            "identifying the target record; refer to those materials, "
            "not to individual feature values or source titles.",
            "Do not list source observations, retrieval steps or "
            "intermediate keys in the Question.",
            "The final question must be answerable using the supplied "
            "reference card plus discoverable source records.",
        ])
        if plan["mode"] == "verify":
            constraints.append(
                "The result must rely on a corroborated reference; do "
                "not enumerate the directories or verification operations."
            )
    else:
        if plan["mode"] == "verify":
            constraints.append(
                "Combine two candidate directories, the event eligibility "
                "criterion and a separately maintained registry relation."
            )
        constraints.extend(
            f"Identifying observation: {item['attribute']} {item['value']}."
            for item in plan["clues"].values()
        )
    if plan.get("semantic_grounding_policy") == (
        "goal_constrained_yago_grounding_v1"
    ):
        goal = plan["goal"]["semantic"]
        from environment.generate_question.planning.goal_plan import goal_question
        final_question = goal_question(plan)
        constraints.extend([
            "The precise final user goal must be semantically equivalent "
            "to: " + final_question,
            ("The Goal follows the exact recorded relationship "
             "sequence and then its frozen numeric resolution. "
             "Do not request any intermediate entity as a second output "
             "or invent extra filters or calculations."
             if goal_kind != "entity_resolution" else
             "The goal follows the exact recorded relationship sequence. "
             "Do not invent additional comparisons, eligibility conditions "
             "or relationships."),
            "Use only the supplied observations and documented relationship sources.",
            ("Keep the reference-material pointer and the exact "
             "conditional event eligibility requirement; individual "
             "feature values live in the separate public task input."
             if compact else
             "Keep all identifying observations and the exact conditional "
             "event eligibility requirement when applicable."),
        ])

    return {
        "visual_clue": (None if compact else plan["clues"]),
        "acquisition_chain": None,
        "topic": spec["scenario"]["topic"],
        "hypothetical": False,
        "setting": spec["scenario"]["setting"],
        "entity_kind": " ".join(
            spec["scenario"]["entity_kind"].replace("_", " ").split()
        ),
        "user_need": spec["scenario"]["purpose"],
        "reasoning_pattern": pattern,
        "difficulty": spec["difficulty_target"],
        "nonnegotiable_conditions": constraints,
        "output_contract": "Return one final related entity name.",
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
