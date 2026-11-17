"""MacroGraph-grounded Question authoring from frozen source-scoped specifications.

LLM rewriting is optional. An independent Critic and deterministic guards
can reject additions; the original executable problem remains unchanged.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from environment.generate_question.planning.spec import spec_issues, VERSION
from environment.generate_question.authoring.intent import intent_contract, style_contract
from environment.generate_question.auditing.macro_contract import macro_question_contract
from environment.generate_question.authoring.surface_quality import (
    question_language_warnings, normalize_question_punctuation,
    topic_mentioned, is_explicit_user_request,
)


def reference_question(spec: dict) -> str:
    errors = spec_issues(spec)
    if errors:
        raise ValueError("invalid_spec:" + ",".join(errors))
    return _intent_reference_question(spec)


def _intent_reference_question(spec: dict) -> str:
    """Deterministic fidelity fallback, not the Author's primary template."""
    intent = intent_contract(spec)
    context, kind, topic = intent["setting"], intent["entity_kind"], intent["topic"]
    if (spec.get("reasoning_pattern") == "task_plan"
            and spec["macro_plan"].get("semantic_grounding_policy") ==
            "goal_constrained_yago_grounding_v1"):
        # The Goal is already selected and frozen by the witnessed relation
        # grammar. Do not turn individual RDF predicates into a chain of
        # awkward nested noun phrases, and do not let the Question invent
        # new constraints or historical claims.
        plan = spec["macro_plan"]
        goal = plan["goal"]["semantic"]
        topic = spec["scenario"]["topic"]
        if plan.get("question_surface_policy") == "macro_goal_reference_card_v2":
            # User intent comes from the Task Plan's semantic Goal and
            # eligibility rule. Target-specific features belong to the
            # provided reference material, NOT the user question.
            lead = (
                f"The supplied photograph and recording descriptions concern a record in the {topic}.",
                f"A record in the {topic} is described by supplied image and audio observations.",
                f"The provided photographic and spoken descriptions concern a record in the {topic}.",
                f"The supplied visual and spoken evidence refers to a record in the {topic}.",
            )[spec["ordinal"] % 4]
            if plan.get("composer_policy") == "output_first":
                # A frozen clue can accidentally equal an ordinary phrase
                # in our deterministic lead (e.g. "catalog" in
                # "spoken-catalog"). Do not weaken the leakage guard; select
                # a neutral alternative that never quotes a private cue.
                safe_leads = (
                    lead,
                    f"The supplied reference card concerns a record in the {topic}.",
                    f"The provided observation descriptions refer to a record in the {topic}.",
                    f"The enclosed reference notes concern a record in the {topic}.",
                )
                allowed = topic + " " + goal["question"]
                def _safe_lead(candidate):
                    return not any(
                        re.search(r"\b" + re.escape(str(info["value"])) + r"\b",
                                  candidate, re.I)
                        and not re.search(
                            r"\b" + re.escape(str(info["value"])) + r"\b",
                            allowed, re.I
                        )
                        for info in plan["clues"].values()
                    )
                lead = next(
                    (candidate for candidate in safe_leads
                     if _safe_lead(candidate)), lead
                )
            if plan["goal_spec"]["output_spec"]["bindings"] != {
                "answer": "entity"
            }:
                from environment.generate_question.planning.goal_plan import goal_question
                return f"{lead} {goal_question(plan)}"
            condition = plan["condition"]
            if condition is None:
                qualifier = ""
            elif condition["field"] == "event_status":
                qualifier = "Considering only its completed review session, "
            else:
                qualifier = "Considering only its review sessions from 2025 onward, "
            if plan["mode"] == "verify":
                qualifier += "using the corroborated reference, "
            target_question = goal["question"]
            if qualifier:
                target_question = (
                    qualifier + target_question[0].lower()
                    + target_question[1:]
                )
            return f"{lead} {target_question}"
        clues = "; ".join(
            f"{info['attribute']} '{info['value']}' in {info['title']}"
            for _, info in sorted(plan["clues"].items())
        )
        condition = plan["condition"]
        if condition:
            qualifying = (
                "the completed archive session" if condition["field"] ==
                "event_status" else "the archive session recorded in or after 2025"
            )
            if plan["mode"] == "verify":
                instruction = (
                    f"Using {qualifying}, reconcile the two candidate-reference "
                    "reports with the separate registry relationship to identify "
                    "the entity documented by the dossier."
                )
            else:
                instruction = (
                    f"Use {qualifying} to locate the dossier's entity reference."
                )
        else:
            instruction = (
                "Use the matching dossier's documented entity reference."
            )
        if plan.get("question_surface_policy") == (
            "varied_archive_reference_v1"
        ):
            lead = (
                f"An archival review of the {topic} begins with these "
                f"records: {clues}.",
                f"Several entries in the {topic} refer to a dossier: "
                f"{clues}.",
                f"A researcher examining the {topic} has to reconcile "
                f"the following identifying details: {clues}.",
                f"The {topic} contains a record whose identity depends "
                f"on comparing these sources: {clues}.",
                f"To resolve a cross-reference in the {topic}, start "
                f"with these archival clues: {clues}.",
            )[spec["ordinal"] % 5]
            return f"{lead} {instruction} {goal['question']}"
        return (
            f"The {topic} contains an unresolved dossier with the "
            f"following identifying clues: {clues}. {instruction} "
            f"{goal['question']}"
        )
    raise ValueError("unsupported_stage1_reasoning_pattern")


AUTHOR_SYSTEM = """You write ONE realistic natural-language user question from a
FROZEN fictional benchmark ProblemSpec. Do not add, delete or change any
condition, information source or output. The source roles are simulations, not
actual observations of the original MultiModalQA media.

Start with the information the user needs or a concrete clue already available
in the fictional scenario. Name a supplied material when it helps the user
identify the clue or understand a comparison. Express the link in ordinary
language, e.g. a code in the event log identifies the session whose timing
should be checked. Let the deliverable follow from the user need.
Use a short context sentence when it helps. Ask for the result once.
Treat the topic label as the name of the fictional collection if it is an
ordinary phrase. Source titles can be shortened without changing their roles.
A role introduction or a recital of every operator adds little context.
No raw IDs, backend field names, TaskGraph vocabulary, step-by-step tool
instructions, or hallucinated facts about a real person/event/visual scene.
All relevant conditions must be clear to someone who will solve it:
active records, code-to-category translation and category approval when
those sources exist, separate clearance when provided, and the exact final
answer type (count, list, highest recorded priority, or earliest relative
ordering). No fabricated additional filters. A hypothetical problem about
a real topic does NOT assert historical facts. NEVER add calendar years,
seasons, dated cycles or concrete thresholds that the original topic
and frozen intent do not provide. Do not invent an additional eligibility
criterion such as a funding tier or certification that is absent from
the required predicates. Even "no archived records" is an extra filter
unless the frozen MacroGraph explicitly filters an archived field.
The FINAL ANSWER uses the terminal output schema: entry_label is an
entry display label, NOT label_code, route_code, abbreviated category
labels or codes. Never ask to list abbreviated labels if only
entry_label is projected. Specifically, do not add "non-archived" or
"exclude archived" unless the contract contains an archived field. A
time window is NOT independently "approved" merely because it exists;
a policy group meeting a threshold is NOT "approved" unless the graph
contains an explicit approval filter. Describe any synthetic source
field in readable language ("measured value", "audio interval start step"),
not raw names such as measured_value, minimum_required or diagram_step_start.

The frozen user intent specifies every answer-defining relationship and
deliverable. Distinguish active status from the reported status being compared.
An independent approval applies to the category, not the glossary translation. Use the material titles as readable source handles. Keep
simulated observations clearly fictional and avoid inventing a new clue.

For evidence_graph tasks, describe the user need and the starting evidence
clues without leaking the full retrieval algorithm. For a conflict task,
identify the discrepancy but never invent factual years. For visual clues,
retain BOTH stated features. The typed source graph is an internal contract,
not a list of step-by-step instructions that must appear in the Question.

For Task Plan problems, the archive and its records are treated as
ordinary facts *inside the scenario*. Never label the archive, its entities,
observations or source titles fictional, simulated, hypothetical or synthetic
in the public question. Provenance and generator policy remain private; do
not imply any fictional assertion is a verified real-world historical fact.

For latest-version requests, you receive a STRUCTURED FROZEN USER INTENT:
constraints, typed output, source roles and a style preference. Write an
original user question from that contract. You may vary clause order,
interrogative structure and user motivation, but every mandatory condition
must remain unambiguous. Do NOT start with "As a" or "As an". Do not merely
copy or minimally edit a reference question.

For historical requests and explicit repair attempts, a FROZEN REFERENCE
QUESTION may also be supplied. It was authored from the fictional
contract, NOT the original MultiModalQA question. Preserve its meaning.
Preserve the word "active" and exactly the cross-source relation required
by the structured intent: approval, temporal window, binary disagreement,
or per-group threshold. Do NOT substitute one operation for another.
Preserve distinct entry-specific clearance for hard cases.
Temporal boundaries are INCLUSIVE: a step equal to the start or end
qualifies. NEVER say "strictly inside" or "strictly within" when also
describing a closed or inclusive interval. Use "within the inclusive
interval" or "between the endpoints, inclusive".
Preserve counting, category grouping, priority MAXIMUM, or chronological
EARLIEST directions when applicable.

Return ONE complete, grammatical English user REQUEST (55-1200 characters).
It may be a direct question ending with '?' or an unambiguous, complete
imperative such as "Please list ...", "Identify ..." or "For ... identify ...".
Never return only a declarative description or an incomplete fragment.
When the original topic has a proper name, keep that exact proper name,
rather than inventing an unrelated company/organization or dropping it. Do not return a summary, classification, fragment, table,
answer, explanation, or commentary. If unsure, closely paraphrase the frozen
reference question without inventing facts. End with '?'.

Output JSON only: {"question":"..."}."""

CRITIC_SYSTEM = """Review the candidate user question against an immutable
fictional ProblemSpec. Never infer facts from the original MultiModalQA
question, answer, image, video or audio. Decide whether the user can infer
the same required output and all constraints, with no added evidence claims.
A statement that a simulated source has hypothetical record fields is
permitted; asserting the content of ORIGINAL media is not.

Output JSON only:
{"verdict":"pass|revise|reject",
 "all_constraints_preserved":true,
 "no_unlicensed_facts":true,
 "same_answer_scope":true,
 "natural":true,
 "reason":"one sentence"}
When a compiled Macro Task contract is supplied, check it rather than
trusting the prose Goal alone. Confirm that the Question asks for the
terminal result of the stated operator chain, not a broader inventory,
all evidence, or an unrelated computation.
Check substantive semantics independently: (a) required source mapping,
(b) the exact comparator, including inclusive temporal endpoints,
(c) hard entry-specific clearance if required, (d) exact output type,
(e) no invented calendar year or new classification/eligibility rule.
The candidate must be grounded in the frozen topic/conditions, not in
unsupported elaborations supplied by an earlier scene paraphrase.
A candidate saying "strictly inside" while also claiming inclusive
boundaries MUST be rejected. Reject added predicates such as "approved
policy group" in a threshold-only task, "approved session window" in a
temporal-only task, or excluding "archived" items when the Program only
filters active=True. A plausible fictional setting does NOT license
adding unsupported eligibility criteria. In particular distinguish
the final requested output field: entry_label denotes an entry name,
label_code/route_code denote source-specific abbreviations. Reject a
candidate that asks to LIST abbreviated codes when the terminal
MacroGraph projects entry_label. A well-formed imperative user request (e.g.
"Please identify ...") is just as valid as an interrogative question.
For chronology, the compiled sort uses ascending relative_step: "earliest"
or "first" is correct; "latest" or "most recent" is WRONG. Never hallucinate
a reversed direction. Surface fluency alone is insufficient.
For Task Plan problems, reject public questions that expose the synthetic
construction process or use words like fictional/simulated/hypothetical to
qualify the source records. Treat the archive as an ordinary scenario, but
reject claims of external historical verification that were not authorized.
For macro_first tasks, the frozen Macro DAG is an internal reference;
check the initial distinguishing clues, the exact authoritative year
deliverable and, for conflict tasks, the dispute condition. Do not
require the question to list Macro operations, retrieved keys or the
intermediate source traversal.
For evidence_graph tasks, check all frozen starting clues, multi-source
identification or conflict conditions and the accession-year deliverable.
Do NOT require a user question to name every index or archive lookup.
Only pass if all four booleans are true."""


def _supplied_reference_pointer(question: str) -> bool:
    """Recognize real pointers to initially supplied materials, not bare
    words such as 'supplied' or unrelated mentions of a record.
    """
    object_pattern = (
        r"(?:reference\s+(?:cards?|materials?|records?|descriptions?|"
        r"notes?|sheets?|documents?)|materials?|records?|observations?|"
        r"descriptions?|notes?|documents?|cards?|photos?|photographs?|"
        r"images?|audio|recordings?|videos?|evidence)"
    )
    return bool(re.search(
        r"\b(?:supplied|provided|attached|enclosed|given)\b"
        r"(?:\s+[\w-]+){0,8}\s+" + object_pattern + r"\b",
        question, flags=re.I,
    ))


def _unlicensed_task_plan_condition(question: str, plan: dict) -> list[str]:
    """Reject explicit extra filters, preserving frozen v2 Replay behavior.

    This high-precision guard checks only new v3 tasks; the independent
    LLM Critic remains responsible for subtler semantic entailment errors.
    """
    if plan.get("composer_policy") != "output_first":
        return []
    q = question.casefold()
    condition = plan["condition"]
    status_terms = re.search(
        r"\b(?:completed|scheduled|cancelled|canceled)\b", q
    )
    year_terms = re.search(r"\b(?:19|20)\d{2}\b", q)
    if condition is None:
        if (re.search(
            r"\b(?:event|session|review)s?\b.{0,65}\b"
            r"(?:completed|scheduled|cancelled|canceled|(?:19|20)\d{2})\b",
            q,
        ) or re.search(
            r"\b(?:completed|scheduled|cancelled|canceled|(?:19|20)\d{2})\b"
            r".{0,40}\b(?:event|session|review)s?\b",
            q,
        )):
            return ["task_plan_unlicensed_event_condition"]
        return []
    if condition["field"] == "event_status":
        errors = []
        # Goal-level filtering of *observation rows* is independent of the
        # event-eligibility rule. Do not reject its licensed year threshold.
        goal_years = {
            str(node["year_gte"])
            for node in plan["goal_spec"]["operations"]
            if node["op"] == "count_aggregate"
        }
        mentioned_years = set(re.findall(r"\b(?:19|20)\d{2}\b", q))
        if mentioned_years - goal_years:
            errors.append("task_plan_unlicensed_year_condition")
        if re.search(r"\b(?:not|never)\s+completed\b", q):
            errors.append("task_plan_reversed_completed_status")
        if re.search(r"\b(?:scheduled|cancelled|canceled)\b", q):
            errors.append("task_plan_conflicting_event_status")
        return errors
    errors = []
    if status_terms:
        errors.append("task_plan_unlicensed_status_condition")
    # "in or after 2025" and "on or after 2025" are INCLUSIVE;
    # a bare "after 2025" or "later than 2025" is exclusive.
    # Check the preceding context instead of a substring match.
    exclusive = any(
        not re.search(
            r"\b(?:in|on)\s+(?:or|and)\s+$",
            q[max(0, match.start() - 20):match.start()],
        )
        for match in re.finditer(
            r"\b(?:after|later than)\s+2025\b", q
        )
    )
    if exclusive:
        errors.append("task_plan_exclusive_year_threshold")
    if re.search(r"\b(?:before|prior to|earlier than)\s+2025\b", q):
        errors.append("task_plan_reversed_year_threshold")
    return errors


def guard_question(question: str, spec: dict,
                   macro_graph: dict | None = None) -> list[str]:
    """Conservative high-precision textual checks before/after the LLM Critic."""
    if not isinstance(question, str):
        return ["question_not_string"]
    q = question.strip()
    errors = []
    if len(q) < 55 or len(q) > 1200:
        errors.append("question_length_out_of_bounds")
    if not is_explicit_user_request(q):
        errors.append("question_not_actionable_request")
    errors.extend(question_language_warnings(q, spec, macro_graph))
    if not topic_mentioned(q, spec["scenario"]["topic"]):
        errors.append("topic_missing")
    pattern = spec.get("reasoning_pattern", "approval")
    if pattern == "task_plan" and (
        spec["macro_plan"].get("semantic_grounding_policy") ==
        "goal_constrained_yago_grounding_v1"
    ):
        # A deterministic draft includes the exact frozen Goal Question,
        # but a connected LLM Author is allowed to paraphrase it. Validate
        # every underlying relation concept, not exact surface wording.
        # This is a conservative semantic *surface* check in addition to
        # the existing Question Critic, never a proof of equivalence.
        from environment.generate_question.planning.task_plan import _predicate_key
        goal = spec["macro_plan"]["goal"]["semantic"]
        patterns = {
            "birthplace": r"\b(?:birth\s?place|born)\b",
            "deathplace": r"\b(?:death|died|place of death)\b",
            "doctoraladvisor": r"\b(?:advisor|adviser|supervisor)\b",
            "fieldofwork": r"\b(?:research field|field of (?:work|study)|research area|discipline)\b",
            "spouse": r"\b(?:spouse|husband|wife|partner)\b",
            "memberof": r"\b(?:member|affiliat\w*|organization)\b",
            "location": r"\b(?:locat(?:ion|ed)|where|area)\b",
            "containedinplace": r"\b(?:contain\w*|region|area|district|within)\b",
            "neighbor": r"\b(?:neighbor\w*|border\w*|adjacent)\b",
            "twinnedwith": r"\b(?:twin\w*|sister.city)\b",
            "nationality": r"\b(?:nationality|citizenship)\b",
            "capital": r"\bcapital\b",
        }
        keys = [_predicate_key(iri) for iri in
                goal["relation_sequence"]]
        if any(
            (key not in patterns or
             not re.search(patterns[key], q, flags=re.I))
            for key in keys
        ) or (
            keys.count("containedinplace") >= 2 and
            not re.search(r"\b(?:larger|broader|higher.level|second.level)\b",
                          q, flags=re.I)
        ):
            errors.append("task_plan_semantic_goal_question_missing")
    if pattern == "task_plan":
        plan = spec["macro_plan"]
        if re.search(
            r"\b(?:fictional|simulated|synthetic|hypothetical|imaginary)\b",
            q, flags=re.I,
        ):
            errors.append("task_plan_meta_fiction_framing")
        compact = plan.get("question_surface_policy") == "macro_goal_reference_card_v2"
        if compact:
            if not _supplied_reference_pointer(q):
                errors.append("task_plan_reference_material_not_identified")
            # Guard against *specific* evidence values being copied into
            # the question, not ordinary vocabulary shared by a semantic
            # goal or topic (e.g., the word "survey" in a regional survey).
            # Exact word/phrase boundaries also avoid "ridge" in "bridge".
            allowed = (
                spec["scenario"]["topic"] + " " +
                plan["goal"]["semantic"]["question"]
            )
            for info in plan["clues"].values():
                value = str(info["value"])
                phrase = r"\b" + re.escape(value) + r"\b"
                if (not re.search(phrase, allowed, flags=re.I)
                        and re.search(phrase, q, flags=re.I)):
                    errors.append("task_plan_answer_clues_leaked_into_question")
                    break
        else:
            for info in plan["clues"].values():
                if info["value"].casefold() not in q.casefold():
                    errors.append("task_plan_required_clue_missing")
        if plan["condition"]:
            if plan["condition"]["field"] == "event_status":
                if not re.search(r"\bcompleted\b", q, re.I):
                    errors.append("task_plan_required_status_missing")
            elif not re.search(r"\b2025\b", q):
                errors.append("task_plan_required_year_missing")
        errors.extend(_unlicensed_task_plan_condition(q, plan))
        if plan.get("composer_policy") == "output_first":
            goal_spec = plan["goal_spec"]
            fields = goal_spec["output_spec"]["bindings"]
            # Each output semantic has a required public description.
            clues = {
                "difference": r"\b(?:difference|subtract|absolute)\b",
                "ratio": r"\b(?:ratio|divide|divided)\b",
                "duration_days": r"\b(?:elapsed|days|duration)\b",
                "midpoint_date": r"\b(?:midpoint|middle)\b",
                "distance_km": r"\b(?:distance|great.circle)\b",
                "observation_count": r"\b(?:count|how many|number of)\b",
                "higher_source": r"\b(?:higher|greater|larger)\b",
                "conclusion": r"\b(?:supported|evidence|judgment|assessment)\b",
            }
            for semantic in fields.values():
                if semantic in clues and not re.search(
                    clues[semantic], q, re.I,
                ):
                    errors.append("goal_missing_output_semantics:" + semantic)
            if goal_spec["output_spec"]["kind"] in {
                "structured_multi_field", "evidence_based_explanation"
            }:
                for name in fields:
                    field_pattern = (
                        r"\b" + re.escape(name).replace("_", r"[_\s-]?")
                        + r"\b"
                    )
                    if not re.search(field_pattern, q, re.I):
                        errors.append("goal_output_field_missing:" + name)
            if "aggregate" in goal_spec["capabilities"]:
                if not re.search(r"\b2025\b", q):
                    errors.append("goal_required_threshold_missing")
            elif re.search(r"\b(?:19|20)\d{2}\b", q):
                # Year selection may only be licensed by an exact Goal
                # or the pre-existing v3 event condition.
                if not plan["condition"]:
                    errors.append("goal_unlicensed_year")
        if not compact and not re.search(
            r"\b(?:related|relation|entity)\b", q, re.I
        ):
            errors.append("task_plan_final_goal_missing")
    family = spec["family"]
    if family in {"coverage_count", "category_distribution"} and not re.search(r"\b(how many|count|number of)\b", q, re.I):
        errors.append("count_answer_scope_missing")
    if family == "priority" and not re.search(r"\b(highest|largest|top|maximum|greatest)\b", q, re.I):
        errors.append("priority_direction_missing")
    if family == "chronology":
        if not re.search(
            r"\b(earliest|first|lowest|minimum|oldest)\b", q, re.I
        ):
            errors.append("relative_ordinal_direction_missing")
        if re.search(
            r"\b(latest|most recent|newest|last|most current)\b", q, re.I
        ):
            errors.append("reversed_chronological_direction")
    if family == "evidence_reconcile":
        if not re.search(r"\b(evidence|source|citation|reference|support)\w*", q, re.I):
            errors.append("evidence_output_scope_missing")
    fields = {"entry_label", "evidence_refs"}
    if macro_graph is not None:
        fields.update(
            field for task in macro_graph["tasks"]
            for field in task["outputs"]["typed_relation"]
            if "_" in field
        )
    implementation_terms = r"TaskGraph|WorkProduct|candidate-\d+|source_id|fact_refs|synthetic_attribute"
    terms = implementation_terms + "|" + "|".join(map(re.escape, sorted(fields)))
    if re.search(r"\b(?:" + terms + r")\b", q, re.I):
        errors.append("internal_implementation_leak")
    if re.search(r"\b(actual photograph shows|original image depicts|original video shows)\b", q, re.I):
        errors.append("unsupported_real_media_assertion")
    return sorted(set(errors))


def naturalize_one(spec: dict, *, author, critic, max_revisions: int = 2,
                   macro_graph: dict | None = None) -> tuple[str, dict]:
    """Model cannot edit World, Program, TaskGraph, or Gold Answer."""
    initial = reference_question(spec)
    compact_task_plan = (
        spec.get("reasoning_pattern") == "task_plan"
        and spec["macro_plan"].get("question_surface_policy")
        == "macro_goal_reference_card_v2"
    )
    instruction = {
        "intent": intent_contract(spec),
        "style": style_contract(spec),
        "frozen_scenario": spec["scenario"],
    }
    graph_contract = (
        macro_question_contract(spec, macro_graph)
        if macro_graph is not None else None
    )
    if graph_contract is not None:
        # Never reveal private World values or Gold answers; the Author sees
        # only source roles, operations, dependencies and final answer type.
        instruction["compiled_macro_task_contract"] = {
            key: graph_contract[key] for key in (
                "semantic_pattern", "answer_kind", "output_fields",
                "required_source_roles", "required_operations",
                "comparison_contract", "selection_contract",
                "macro_task_count", "macro_dependency_depth",
                "no_extra_question_conditions",
            )
        }
        instruction["macro_dependency_plan"] = [
            {
                "step": i,
                "depends_on": [
                    next(j for j, upstream in enumerate(macro_graph["tasks"])
                         if upstream["task_id"] == parent)
                    for parent in task["depends_on"]
                ],
                "source_roles": task["source_ids"],
                "operations": [
                    part["op"] for part in task.get(
                        "operator_chain", [{"op": task["operation"]}]
                    )
                ],
            }
            for i, task in enumerate(macro_graph["tasks"])
        ]
    if max_revisions < 0 or max_revisions > 8:
        raise ValueError("revisions must be 0..8")
    failures = []
    for attempt in range(max_revisions + 1):
        author_contract = {
            "user_intent": {
                key: value for key, value in instruction["intent"].items()
                if key not in {"sources", "difficulty", "reasoning_pattern"}
            },
            "materials": (
                [{
                    "kind": "provided_reference_materials",
                    "modalities": sorted({
                        item["modality"] for item in spec["task_input"][
                            "observations"]
                    }),
                    "description": (
                        "The user has supplied a separate target reference "
                        "card; do not enumerate its observation values in "
                        "the question or expose internal source lookups."
                    ),
                }] if (
                    spec.get("reasoning_pattern") == "task_plan" and
                    spec["macro_plan"].get("question_surface_policy") ==
                        "macro_goal_reference_card_v2"
                ) else [
                    {"title": source["title"],
                     "modality": source["modality"],
                     "purpose": source["purpose"]}
                    for source in spec["sources"]
                ]
            ),
            "style": instruction["style"],
        }
        if compact_task_plan:
            plan = spec["macro_plan"]
            goal = plan["goal"]["semantic"]
            from environment.generate_question.planning.goal_plan import goal_question
            goal_spec = plan["goal_spec"]
            author_contract["semantic_goal_question"] = goal_question(plan)
            author_contract["goal_operations"] = [
                {"op": item["op"], "inputs": item["inputs"],
                 "output": item["output"]}
                for item in goal_spec["operations"]
            ]
            author_contract["final_output_spec"] = goal_spec["output_spec"]
            author_contract["required_literal_fields"] = list(
                goal_spec["output_spec"]["fields"]
            )
            author_contract["output_constraint"] = (
                "Keep every answer field spelled EXACTLY as in "
                "required_literal_fields, including underscores. "
                "Preserve the semantic type and calculation of each field; "
                "do not add an intermediate entity as a separate answer."
            )
            author_contract["required_event_condition"] = plan["condition"]
            author_contract["verification_mode"] = (
                plan["mode"] == "verify"
            )
            if plan.get("composer_policy") == "output_first":
                condition = plan["condition"]
                if condition and condition["field"] == "event_year":
                    author_contract["event_condition_scope"] = (
                        "Select review sessions held IN OR AFTER 2025 "
                        "(2025 inclusive). No completion, cancellation, or "
                        "other status criterion is authorized."
                    )
                elif condition and condition["field"] == "event_status":
                    author_contract["event_condition_scope"] = (
                        "Only completed review sessions qualify. "
                        "No year restriction is authorized."
                    )
                else:
                    author_contract["event_condition_scope"] = (
                        "No event eligibility condition is requested. "
                        "Do not introduce one."
                    )
        if spec.get("question_semantic_lock_version"):
            from environment.generate_question.authoring.question_checks import question_author_requirements
            author_contract["mandatory_question_semantics"] = (
                question_author_requirements(spec)
            )
        if attempt > 0:
            author_contract["repair_reference_question"] = initial
            prior = failures[-1]
            prior_errors = prior.get("errors", [])
            author_contract["repair_priority"] = [
                "Keep every mandatory semantic requirement EXACT while "
                "changing natural sentence structure.",
                "Request ONLY the final typed output; identity matching "
                "must be a subordinate clause, not a second answer.",
                "Do not copy the deterministic reference question.",
            ]
            if compact_task_plan and spec["macro_plan"].get(
                "composer_policy"
            ) == "output_first":
                condition = spec["macro_plan"]["condition"]
                if ("task_plan_unlicensed_status_condition" in prior_errors
                        and condition and
                        condition["field"] == "event_year"):
                    author_contract["repair_priority"].insert(
                        0, "The previous candidate ADDED an unauthorized "
                        "event STATUS condition (for example 'completed'). "
                        "Remove every event-status restriction. The ONLY "
                        "eligibility predicate is event_year >= 2025; "
                        "write 'in or after 2025' or 'from 2025 onward'."
                    )
                if "task_plan_exclusive_year_threshold" in prior_errors:
                    author_contract["repair_priority"].insert(
                        0, "Use an INCLUSIVE year condition: 'in or after "
                        "2025' or 'from 2025 onward'. Do not say only "
                        "'after 2025' or 'later than 2025'."
                    )
                if "task_plan_unlicensed_year_condition" in prior_errors:
                    author_contract["repair_priority"].insert(
                        0, "The frozen event condition has NO year "
                        "restriction. Remove the invented year filter."
                    )
                if "task_plan_unlicensed_event_condition" in prior_errors:
                    author_contract["repair_priority"].insert(
                        0, "There is NO event eligibility condition. "
                        "Remove invented year or status filters."
                    )
                if "task_plan_reference_material_not_identified" in prior_errors:
                    author_contract["repair_priority"].insert(
                        0, "Identify the initial supplied reference card "
                        "or provided observation descriptions without "
                        "listing any clue values."
                    )
            if spec.get("question_semantic_lock_version"):
                if spec["macro_plan"]["goal_resolution"]["kind"] == "calculation":
                    author_contract["repair_priority"].append(
                        "Explicitly state rounded DOWN to a WHOLE percent."
                    )
                if spec["macro_plan"]["pattern"] in {
                    "corroborated_dispute", "multimodal_event",
                }:
                    author_contract["repair_priority"].append(
                        "Preserve the exact independent event-log tag: "
                        + spec["macro_plan"]["clues"]["case_tag"]
                    )
                if ("too_similar_to_reference_question" in prior_errors
                        or "semantic_additional_output_requested" in prior_errors):
                    author_contract["repair_priority"].append(
                        "Use a different single-output sentence pattern: "
                        "'What [single Goal] applies to the record matching "
                        "[all evidence predicates]?'"
                    )
            if compact_task_plan and any(
                "output_field_missing:" in issue or
                "reference_material_not_identified" in issue
                for issue in prior_errors
            ):
                author_contract["repair_priority"].insert(
                    0, "Reproduce every required output field name "
                    "VERBATIM (underscores included). Explicitly mention "
                    "the supplied reference card or observation material."
                )
            if "topic_missing" in failures[-1].get("errors", ()):
                author_contract["topic_repair"] = (
                    ("Mention the exact topic naturally in the supplied "
                     "working materials: "
                     if spec.get("scene_realism_version") else
                     "Mention the exact topic as the fictional collection title: ")
                    + spec["scenario"]["topic"]
                )
        user_text = json.dumps({
            "contract": author_contract,
            "previous_failures": [{
                "errors": entry.get("errors", []),
                "critic_reason": entry.get("decision", {}).get("reason"),
            } for entry in failures[-2:]],
            "constraint": "Preserve all conditions and the exact answer contract. "
                          "Prefer a direct natural question, avoid As-a role openings. "
                          "Return a self-contained natural user request: either "
                          "a question ending in '?' or an explicit imperative "
                          "with identify/list/determine/return. Preserve the "
                          "exact grounded topic label and temporal direction.",
        }, ensure_ascii=False)
        try:
            from environment.generate_question.authoring.prompt_templates import (
                QUESTION_AUTHOR, QUESTION_CRITIC, QUESTION_AUTHOR_SINGLE_SLOT,
                TASK_PLAN_QUESTION_AUTHOR, TASK_PLAN_QUESTION_CRITIC,
            )
            v2 = spec.get("reasoning_pattern") == "macro_first_v2"
            system_prompt = (
                TASK_PLAN_QUESTION_AUTHOR if compact_task_plan
                else QUESTION_AUTHOR_SINGLE_SLOT if spec.get("scene_realism_version")
                else QUESTION_AUTHOR if v2 else AUTHOR_SYSTEM
            )
            candidate = author.complete_json(system_prompt, user_text)
            raw = candidate.get("question")
            q = (normalize_question_punctuation(raw)
                 if isinstance(raw, str) else raw)
            surface = guard_question(q, spec, macro_graph)
            if (isinstance(q, str)
                    and SequenceMatcher(None, q.strip().casefold(),
                                        initial.casefold()).ratio() > 0.96):
                surface.append("too_similar_to_reference_question")
            if surface:
                failures.append({
                    "stage": "guard", "errors": surface,
                    "candidate_length": len(q) if isinstance(q, str) else None,
                    "candidate_has_question_mark": (
                        "?" in q if isinstance(q, str) else False
                    ),
                    "candidate_preview": (
                        q[:320] if isinstance(q, str) else repr(q)[:120]
                    ),
                    "punctuation_normalized": (
                        isinstance(raw, str) and q != raw.strip()
                    ),
                })
                continue
            critic_prompt = (
                TASK_PLAN_QUESTION_CRITIC if compact_task_plan
                else QUESTION_CRITIC if v2 else CRITIC_SYSTEM
            )
            if compact_task_plan:
                # The Critic needs answer/constraint semantics, not a
                # verbose private operator/DAG walkthrough. The public
                # Question should never be forced to narrate Gold steps.
                plan = spec["macro_plan"]
                from environment.generate_question.planning.goal_plan import goal_question
                goal_spec = plan["goal_spec"]
                goal_kind = goal_spec["source_profile"]
                target_question = goal_question(plan)
                frozen_contract = {
                    "topic": spec["scenario"]["topic"],
                    "semantic_goal_question": target_question,
                    "goal_resolution_kind": goal_kind,
                    "final_output_spec": goal_spec.get("output_spec"),
                    "goal_operations": [
                        {"op": item["op"], "inputs": item["inputs"],
                         "output": item["output"]}
                        for item in goal_spec.get("operations", [])
                    ],
                    "relation_sequence": plan["goal"]["semantic"]["relation_sequence"],
                    "answer_type_hint": (
                        plan["goal"]["answer_type"]
                        if "output_spec" in goal_spec
                        or goal_kind != "entity_resolution"
                        else plan["goal"]["semantic"]["answer_type_hint"]
                    ),
                    "required_event_condition": plan["condition"],
                    "verification_mode": plan["mode"] == "verify",
                    "user_intent": instruction["intent"]["nonnegotiable_conditions"],
                    "reference_materials": "initially supplied textual observations",
                    "frozen_reference_question": initial,
                }
            else:
                frozen_contract = {
                    **instruction,
                    "frozen_reference_question": initial,
                    "frozen_program_output": spec["requirements"]["answer_kind"],
                }
            decision = critic.complete_json(critic_prompt, json.dumps({
                "frozen_contract": frozen_contract,
                "candidate_question": q,
            }, ensure_ascii=False))
            axes = ("all_constraints_preserved", "no_unlicensed_facts",
                    "same_answer_scope", "natural")
            if (decision.get("verdict") == "pass" and
                    all(decision.get(axis) is True for axis in axes)):
                return q.strip(), {
                    "status": "accepted", "attempts": attempt + 1,
                    "punctuation_normalized": (
                        isinstance(raw, str) and q != raw.strip()
                    ),
                    "review": decision,
                    "semantic_equivalence_proven": False,
                }
            failures.append({
                "stage": "critic", "decision": decision,
                "candidate_preview": q[:320],
            })
        except Exception as exc:
            # Do not invent acceptance; preserve the executable reference.
            failures.append({"stage": "backend",
                             "error_type": type(exc).__name__,
                             "error": str(exc)[:240]})
    return initial, {
        "status": "fallback", "attempts": max_revisions + 1,
        "failures": failures,
        "semantic_equivalence_proven": False,
    }
