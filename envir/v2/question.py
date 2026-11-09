"""Scenario-aware questions from immutable V2 ProblemSpecs, never from old MMQA QAs.

LLM rewriting is optional. An independent Critic and deterministic guards
can reject additions; the original executable problem remains unchanged.
"""
from __future__ import annotations

import json
import re

from .spec import spec_issues


def reference_question(spec: dict) -> str:
    problems = spec_issues(spec)
    if problems:
        raise ValueError("invalid_spec:" + ",".join(problems))
    scene = spec["scenario"]
    topic = scene["topic"]
    noun = scene["entity_kind"]
    difficulty = spec["difficulty_target"]
    family = spec["family"]
    beginning = (
        f"I'm working as a {scene['user_role']} on a fictional {topic} "
        f"reference collection. I need to {scene['purpose']}."
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
    if len(q) < 55 or len(q) > 1200 or "?" not in q:
        errors.append("question_out_of_bounds")
    if spec["scenario"]["topic"].casefold() not in q.casefold():
        errors.append("topic_missing")
    if not re.search(r"\bactive\b", q, flags=re.I):
        errors.append("active_condition_missing")
    if spec["difficulty_target"] != "easy":
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
    if re.search(r"\b(TaskGraph|WorkProduct|candidate-\d+|source_id|fact_refs|synthetic_attribute)\b", q, re.I):
        errors.append("internal_implementation_leak")
    if re.search(r"\b(actual photograph shows|original image depicts|original video shows)\b", q, re.I):
        errors.append("unsupported_real_media_assertion")
    return sorted(set(errors))


def naturalize_one(spec: dict, *, author, critic, max_revisions: int = 2) -> tuple[str, dict]:
    """Model cannot edit World, Program, TaskGraph, or Gold Answer."""
    initial = reference_question(spec)
    instruction = {
        "frozen_scenario": spec["scenario"],
        "required_source_roles": spec["sources"],
        "requirements": spec["requirements"],
        "difficulty": spec["difficulty_target"],
        "goal_family": spec["family"],
    }
    if max_revisions < 0 or max_revisions > 8:
        raise ValueError("revisions must be 0..8")
    failures = []
    for attempt in range(max_revisions + 1):
        user_text = json.dumps({
            "contract": instruction,
            "previous_failures": failures[-2:],
            "constraint": "Preserve all conditions; return a SINGLE question.",
        }, ensure_ascii=False)
        try:
            candidate = author.complete_json(AUTHOR_SYSTEM, user_text)
            q = candidate.get("question")
            surface = guard_question(q, spec)
            if surface:
                failures.append({"stage": "guard", "errors": surface})
                continue
            decision = critic.complete_json(CRITIC_SYSTEM, json.dumps({
                "frozen_contract": instruction,
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
            failures.append({"stage": "critic", "decision": decision})
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
