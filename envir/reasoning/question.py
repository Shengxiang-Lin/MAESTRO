"""Scenario-aware questions from immutable V2 ProblemSpecs, never from old MMQA QAs.

LLM rewriting is optional. An independent Critic and deterministic guards
can reject additions; the original executable problem remains unchanged.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from .spec import spec_issues, VERSION
from .intent import intent_contract, style_contract


def reference_question(spec: dict) -> str:
    problems = spec_issues(spec)
    if problems:
        raise ValueError("invalid_spec:" + ",".join(problems))
    if spec.get("version") == VERSION:
        return _intent_reference_question(spec)
    scene = spec["scenario"]
    topic = scene["topic"]
    noun = " ".join(scene["entity_kind"].replace("_", " ").split())
    difficulty = spec["difficulty_target"]
    family = spec["family"]
    purpose = re.sub(r"^(to\s+)+", "", scene["purpose"].strip(), flags=re.I)
    purpose = purpose.strip().rstrip(".!;: ")
    purpose = purpose[:1].lower() + purpose[1:]
    beginning = (
        f"As a {scene['user_role']} reviewing a fictional {topic} "
        f"reference collection, I need to {purpose}."
    )
    if difficulty == "easy":
        condition = (
            f"Using the working register, consider only active {noun}."
        )
    else:
        condition = (
            f"The working register uses abbreviated category codes, a separate "
            f"glossary resolves those codes, and independent simulated "
            f"annotations mark categories as approved. "
            f"Consider only active {noun} whose resolved category is approved."
        )
    if difficulty == "hard":
        condition += (
            " Also require the entry-specific clearance note to mark each "
            "selected entry as cleared."
        )
    questions = {
        "reconcile": "Which entries meet all these conditions?",
        "coverage_count": "How many distinct entries meet all these conditions?",
        "eligibility": (
            "Which entries qualify after combining these independent conditions?"
        ),
        "priority": (
            "Which qualifying entry has the highest recorded priority value?"
        ),
        "chronology": (
            "Which qualifying entry has the earliest relative step in "
            "the working register?"
        ),
        "evidence_reconcile": (
            "Which entries meet all these conditions, and what are their "
            "supporting source observations?"
        ),
        "category_distribution": (
            "For each qualifying category, how many entries meet all "
            "these conditions?"
        ),
    }
    return " ".join([beginning, condition, questions[family]])


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

Vary the question structure and natural motivation. Specific source roles and
relationships are valuable; do not merely insert a decorative user persona.
No raw IDs, backend field names, TaskGraph vocabulary, step-by-step tool
instructions, or hallucinated facts about a real person/event/visual scene.
All relevant conditions must be clear to someone who will solve it:
active records, code-to-category translation and category approval when
those sources exist, separate clearance when provided, and the exact final
answer type (count, list, highest recorded priority, or earliest relative
ordering). No fabricated additional filters. A hypothetical problem about
a real topic does NOT assert historical facts.

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
Preserve counting, category grouping, priority MAXIMUM, or chronological
EARLIEST directions when applicable.

Return ONE complete, grammatical English question (55-1200 characters), with
a question mark. Do not return a summary, classification, fragment, table,
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
Only pass if all four booleans are true."""


def guard_question(question: str, spec: dict) -> list[str]:
    """Conservative high-precision textual checks before/after the LLM Critic."""
    if not isinstance(question, str):
        return ["question_not_string"]
    q = question.strip()
    errors = []
    if len(q) < 55 or len(q) > 1200:
        errors.append("question_length_out_of_bounds")
    if "?" not in q:
        errors.append("question_missing_interrogative")
    if spec["scenario"]["topic"].casefold() not in q.casefold():
        errors.append("topic_missing")
    if not re.search(r"\bactive\b", q, flags=re.I):
        errors.append("active_condition_missing")
    pattern = spec.get("reasoning_pattern", "approval")
    latest = spec.get("version") == VERSION
    if latest and pattern != "approval":
        if not re.search(r"\b(code|codes|glossary|map|mapped|mapping|translat|resolv)\b", q, re.I):
            errors.append("cross_source_code_mapping_missing")
        if pattern == "time_window":
            if not re.search(r"\b(window|interval|range|between|within|inside|boundary|boundaries)\b", q, re.I):
                errors.append("temporal_interval_missing")
            if not re.search(r"\b(time|step|event|moment|frame|timestamp)\b", q, re.I):
                errors.append("temporal_event_missing")
        elif pattern == "status_disagreement":
            if not re.search(r"\b(disagree|different|mismatch|conflict|contradict|inconsisten)\w*", q, re.I):
                errors.append("status_disagreement_missing")
            if not re.search(r"\b(status|statuses|mark|reported|observed)\b", q, re.I):
                errors.append("binary_status_missing")
        elif pattern == "dynamic_threshold":
            if not re.search(r"\b(threshold|minimum|at least|exceed|above|meet|qualify)\w*", q, re.I):
                errors.append("dynamic_threshold_missing")
            if not re.search(r"\b(measur|value|rating|score)\w*", q, re.I):
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
    if family == "chronology" and not re.search(r"\b(earliest|first|lowest|minimum)\b", q, re.I):
        errors.append("relative_ordinal_direction_missing")
    if latest and family == "evidence_reconcile":
        if not re.search(r"\b(evidence|source|citation|reference|support)\w*", q, re.I):
            errors.append("evidence_output_scope_missing")
    if re.search(r"\b(TaskGraph|WorkProduct|candidate-\d+|source_id|fact_refs|synthetic_attribute)\b", q, re.I):
        errors.append("internal_implementation_leak")
    if re.search(r"\b(actual photograph shows|original image depicts|original video shows)\b", q, re.I):
        errors.append("unsupported_real_media_assertion")
    return sorted(set(errors))


def naturalize_one(spec: dict, *, author, critic, max_revisions: int = 2) -> tuple[str, dict]:
    """Model cannot edit World, Program, TaskGraph, or Gold Answer."""
    initial = reference_question(spec)
    latest = spec.get("version") == VERSION
    instruction = (
        {
            "intent": intent_contract(spec),
            "style": style_contract(spec),
            "frozen_scenario": spec["scenario"],
        } if latest else {
            "frozen_reference_question": initial,
            "frozen_scenario": spec["scenario"],
            "required_source_roles": spec["sources"],
            "requirements": spec["requirements"],
            "difficulty": spec["difficulty_target"],
            "goal_family": spec["family"],
        }
    )
    if max_revisions < 0 or max_revisions > 8:
        raise ValueError("revisions must be 0..8")
    failures = []
    for attempt in range(max_revisions + 1):
        author_contract = dict(instruction)
        if latest and attempt > 0:
            author_contract["repair_reference_question"] = initial
        user_text = json.dumps({
            "contract": author_contract,
            "previous_failures": [{
                "errors": entry.get("errors", []),
                "critic_reason": entry.get("decision", {}).get("reason"),
            } for entry in failures[-2:]],
            "constraint": "Preserve all conditions and the exact answer contract. "
                          "Prefer a direct natural question, avoid As-a role openings. "
                          "Return one fully formed question ending with '?'.",
        }, ensure_ascii=False)
        try:
            candidate = author.complete_json(AUTHOR_SYSTEM, user_text)
            q = candidate.get("question")
            surface = guard_question(q, spec)
            if (latest and isinstance(q, str)
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
