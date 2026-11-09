"""MacroGraph-grounded Question authoring from frozen source-scoped specifications.

LLM rewriting is optional. An independent Critic and deterministic guards
can reject additions; the original executable problem remains unchanged.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from .spec import spec_issues, VERSION
from .intent import intent_contract, style_contract
from .macro_contract import macro_question_contract
from .surface_quality import (
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
    pattern = spec["reasoning_pattern"]
    constraints = {
        "approval": (
            "Consider only active register entries, resolve their category "
            "codes through the glossary, and use the annotation source to "
            "retain independently approved categories."
        ),
        "time_window": (
            "For active register entries, map their session codes with the "
            "glossary and compare each event step with the corresponding "
            "annotation's inclusive time window."
        ),
        "status_disagreement": (
            "For active register entries, resolve group codes through the "
            "glossary and compare reported statuses with independent "
            "annotation statuses; retain only disagreements."
        ),
        "dynamic_threshold": (
            "For active register entries, resolve policy codes with the "
            "glossary and compare each measured value with its group's "
            "independently recorded minimum threshold."
        ),
    }[pattern]
    if spec["difficulty_target"] == "easy":
        constraints = "Consider only active entries from the register."
    if spec["difficulty_target"] == "hard":
        constraints += (
            " Also require each selected entry to have a separate "
            "entry-specific clearance marked cleared."
        )
    outputs = {
        "reconcile": "Which entries meet these conditions?",
        "eligibility": "Which entries meet these conditions?",
        "evidence_reconcile": (
            "Which entries qualify, and what source fact references "
            "support each entry?"
        ),
        "coverage_count": "How many distinct entries qualify?",
        "category_distribution": (
            "How many qualifying entries occur in each resolved category?"
        ),
        "priority": (
            "Which qualifying entry has the highest recorded priority?"
        ),
        "chronology": (
            "Which qualifying entry has the earliest relative step?"
        ),
        "window_alignment": (
            "Which entries fall inside their matching annotation time windows?"
        ),
        "status_disagreement": (
            "Which entries have different statuses across the two records?"
        ),
        "threshold_eligibility": (
            "Which entries meet their independently defined minimum values?"
        ),
    }
    beginning = [
        f"For a fictional {topic} reference collection,",
        f"I'm examining fictional {topic} records.",
        f"Please help check the fictional {topic} collection.",
        f"The fictional {topic} register needs a cross-source review.",
        f"Regarding the fictional {topic} record set,",
    ][int(spec["ordinal"]) % 5]
    return f"{beginning} {constraints} {outputs[spec['family']]}"


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
Only pass if all four booleans are true."""


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
    if not re.search(r"\bactive\b", q, flags=re.I):
        errors.append("active_condition_missing")
    pattern = spec.get("reasoning_pattern", "approval")
    if pattern != "approval":
        if not re.search(r"\b(codes?|glossary|map|mapped|mapping|crosswalk|translat\w*|resolv\w*)\b", q, re.I):
            errors.append("cross_source_code_mapping_missing")
        if pattern == "time_window":
            if not re.search(r"\b(window|interval|range|between|within|inside|boundary|boundaries)\b", q, re.I):
                errors.append("temporal_interval_missing")
            if not re.search(r"\b(time|step|event|moment|frame|timestamp)\b", q, re.I):
                errors.append("temporal_event_missing")
        elif pattern == "status_disagreement":
            if not re.search(r"\b(disagree(?:s|ment|ments|d)?|discrepanc(?:y|ies)|"
            r"differ(?:s|ed|ing|ence|ences|ent)?|mismatch(?:es|ed)?|"
            r"conflict(?:s|ing|ed)?|contradict(?:s|ion|ions)?|"
            r"inconsisten\w*)\b", q, re.I):
                errors.append("status_disagreement_missing")
            if not re.search(r"\b(status|statuses|mark|reported|observed)\b", q, re.I):
                errors.append("binary_status_missing")
        elif pattern == "dynamic_threshold":
            if not re.search(r"\b(threshold|minimum|at least|exceed|above|meet|qualify)\w*", q, re.I):
                errors.append("dynamic_threshold_missing")
            if not re.search(r"\b(measur\w*|value\w*|rating\w*|score\w*|"
            r"metric\w*|amount\w*|quantity|reading\w*|level\w*)\b", q, re.I):
                errors.append("measured_value_missing")
    elif spec["difficulty_target"] != "easy":
        if not re.search(r"\b(code|codes|abbreviat|glossary|mapped|mapping|translat|resolv)", q, re.I):
            errors.append("alias_resolution_not_expressed")
        if not re.search(r"\b(approv|review|annotat)", q, re.I):
            errors.append("review_condition_not_expressed")
    if spec["difficulty_target"] == "hard":
        if not re.search(r"\b(clear|clearance|authoriz)", q, re.I):
            errors.append("clearance_condition_missing")
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
            "materials": [
                {"title": source["title"], "modality": source["modality"],
                 "purpose": source["purpose"]}
                for source in spec["sources"]
            ],
            "style": instruction["style"],
        }
        if attempt > 0:
            author_contract["repair_reference_question"] = initial
            if "topic_missing" in failures[-1].get("errors", ()):
                author_contract["topic_repair"] = (
                    "Mention the exact topic as the fictional collection title: "
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
            candidate = author.complete_json(AUTHOR_SYSTEM, user_text)
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
            decision = critic.complete_json(CRITIC_SYSTEM, json.dumps({
                "frozen_contract": {
                    **instruction,
                    "frozen_reference_question": initial,
                    "frozen_program_output": spec["requirements"]["answer_kind"],
                },
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
