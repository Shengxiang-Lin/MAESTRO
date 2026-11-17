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
    "typed_knowledge_v4": "Resolve the fictional event and its genuine secondary YAGO relation before tracing the related entity.",
    "task_graph_first_v1": "Identify the project and its completed event, reconcile competing reference records and use genuine YAGO evidence to answer.",
    "task_graph_first_v2": "Select a status- or year-qualified fictional event, reconcile event-specific references and verify genuine YAGO evidence.",
    "task_plan": "Resolve a project from simulated clues and name the entity reached by genuinely witnessed YAGO relations.",
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
    elif pattern == "task_plan":
        plan = spec["macro_plan"]
        v5 = plan.get("composer_policy") == "goal_conditioned_operation_dag_v5"
        goal_kind = plan.get("goal_spec", {}).get(
            "source_profile", plan.get("goal_spec", {}).get(
                "kind", "entity_resolution"
            )
        )
        constraints = [
            "Combine the independently recorded identifying clues.",
            "Identify the matching dossier before following its relationship records.",
            (
                "Report ONLY the entity at the end of the relation chain."
                if goal_kind == "entity_resolution" else
                "Report ONLY the frozen typed Goal output: "
                + plan["goal"]["public_label"] + ". "
                + ("Preserve every typed field and requested computation."
                   if v5 else
                   "Preserve the exact three-field structured answer."
                   if goal_kind == "structured_answer" else
                   "Do not request an additional intermediate entity.")
            ),
            "Write the user request as an ordinary archival investigation; "
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
            if v5:
                from .goal_ops_v5 import goal_question
                final_question = goal_question(plan)
            elif goal_kind != "entity_resolution":
                from .typed_goal_v4 import numeric_goal_question
                final_question = numeric_goal_question(plan)
            else:
                final_question = goal["question"]
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
                "Use only the dossier records and relationship sources provided.",
                ("Keep the reference-material pointer and the exact "
                 "conditional event eligibility requirement; individual "
                 "feature values live in the separate public task input."
                 if compact else
                 "Keep all dossier identifying cues and the exact conditional "
                 "event eligibility requirement when applicable."),
            ])
    elif pattern == "task_graph_first_v2":
        plan = spec["macro_plan"]
        relation = spec["task_first_v2_relation_sketch"]["event_relation"]
        event_constraint = (
            "Select a completed fictional project event; other event "
            "statuses do not qualify."
            if relation["field"] == "event_status" else
            "Select a fictional project event recorded in or after "
            "the year 2025; events before 2025 do not qualify."
        )
        constraints = [
            "Reconcile the simulated identifying clues for exactly one "
            "fictional project.",
            event_constraint,
            "Each of two simulated reference directories has distinct "
            "candidate hypotheses. In the event-keyed graph, event_id "
            "must be used to select the correct per-event records.",
            "Check the selected event's permitted relation objects against "
            "genuine secondary YAGO fact witnesses.",
            "Follow the documented YAGO relation chain to identify the "
            "single final related real-world entity.",
            "Project/event records and candidate claims are simulated, "
            "not externally verified historical facts.",
            "Do not expose Macro IDs or internal primitive operator names.",
        ]
        constraints.extend(
            f"Identifying observation: {item['attribute']} {item['value']}."
            for item in plan["clues"].values()
        )
    elif pattern == "task_graph_first_v1":
        plan = spec["macro_plan"]
        relation_sketch = spec["task_first_relation_sketch"]
        constraints = [
            "All necessary simulated identifying clues must refer to one fictional project.",
            "The required event-status condition is 'completed'; scheduled "
            "follow-up events do NOT count as the chosen event.",
            "Reconcile two source-specific candidate references and independently "
            "check real YAGO secondary-fork evidence.",
            "Use the documented real YAGO relationship trail to determine "
            "one final linked entity's name.",
            "Real RDF assertions are genuine YAGO witnesses. Project, "
            "event, organizer, venue and evidence statements are fictional.",
            "The internal Macro DAG, source keys, task numbers and exact "
            "primitive operator sequence must not be shown to the user.",
        ]
        constraints.extend(
            f"Required identifying observation: {item['attribute']} {item['value']}."
            for item in plan["clues"].values()
        )
        constraints.append(
            "Planning shape: " + spec["task_first_macro_graph"]["shape"]
            + " (internal orchestration contract only)."
        )
    elif pattern == "typed_knowledge_v4":
        plan = spec["macro_plan"]
        constraints = [
            "Identify exactly one fictional project from ALL simulated media clues.",
            "Investigate both simulated reference directories. Their overlap "
            "still contains TWO possible YAGO entities; neither the two "
            "directories alone nor the event record alone gives the answer.",
            "Inspect the fictional project's actual event, organizer and "
            "venue records to obtain permitted secondary real-world links.",
            "Use genuine YAGO secondary fork evidence as a required "
            "verification input before following the final real YAGO path.",
            "Do not treat synthetic projects, events, organizers, venues, "
            "or their associations as genuine YAGO historical facts.",
            "Return only the final real-world entity name; internal "
            "project IDs and candidate keys are not user-visible goals.",
        ]
        constraints.extend(
            f"Identifying clue: {item['attribute']} {item['value']}."
            for item in plan["clues"].values()
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
            (None if pattern == "task_plan" and
             spec["macro_plan"].get("question_surface_policy") ==
             "macro_goal_reference_card_v2" else
             spec["macro_plan"]["clues"])
            if pattern in {"macro_first_v2", "typed_knowledge_v1", "typed_knowledge_v2", "typed_knowledge_v3", "typed_knowledge_v4", "task_graph_first_v1", "task_graph_first_v2", "task_plan"} else None
        ),
        "acquisition_chain": None,
        "topic": spec["scenario"]["topic"],
        "hypothetical": pattern not in {"macro_first_v2", "task_plan"},
        "setting": spec["scenario"]["setting"],
        "entity_kind": " ".join(
            spec["scenario"]["entity_kind"].replace("_", " ").split()
        ),
        "user_need": spec["scenario"]["purpose"],
        "reasoning_pattern": pattern,
        "difficulty": spec["difficulty_target"],
        "nonnegotiable_conditions": constraints,
        "output_contract": (
            ("Return one final related entity name."
             if pattern == "task_plan" else
             "Return one final linked real-world entity name.")
            if pattern in {"task_graph_first_v1", "task_graph_first_v2", "task_plan"} else
            "Return the single final real YAGO-related entity name."
            if pattern == "typed_knowledge_v4" else
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
