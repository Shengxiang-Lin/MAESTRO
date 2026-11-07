"""Stage 2A: lightweight Subtask-intent reviews and replayable certificates.

A pass is an audited (fallible) semantic judgement, NOT a formal proof.
The deterministic checks protect against explicit cross-branch literals, numeric operators,
branch leakage and an independently replayed reviewer verdict. No private Gold facts are sent to models.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping

from environment.delegation_blueprint import (
    _fragment_goal_packet, _validate_goal, validate_delegation_blueprint,
)
from environment.trajectory_simulation.eligibility import digest


GOAL_CERT_SCHEMA = "maestro.stage2a.certified_blueprint.v3"
GOAL_REVIEW_METHOD = "subtask_intent_alignment_critic_v3"

_REVIEW_SYSTEM = """Review ONE brief Sub-Agent SUBTASK INTENT against its
immutable local fragment contract and its declared direct dependencies.

IMPORTANT: This Goal is ONLY a high-level planning hint, not a fully
executable query. The exact step contract remains separately available
to Question and Sub-Agent writers. Do NOT demand that the Goal alone
return the same precise answer as executing every program step.

Check just THREE things:
1) RELEVANCE: Does it describe the actual main job for THIS fragment,
   rather than a different sibling, the whole Main task or a generic
   request without a meaningful target/action?
2) OUTPUT: Does it aim at the correct broad kind of result (entities vs
   count vs value, and the principal entity/information type) and respect
   its actual upstream input role?
3) NO CONTRADICTION: Does anything it explicitly CLAIM reverse a
   relation, misattribute a property, alter AND/OR, alter a comparison,
   add an unrelated branch condition, invent an answer, or narrow/expand
   the candidate pool in a way incompatible with the contract?

ALLOWED abstraction: omitted thresholds, dates, names, intermediate
steps, attribute predicates and explicit operator names. A phrase like
'people meeting the specified criteria' may stand for multiple accurate
filters whose details remain in the separate contract. Do not require
EVERY public step to be mentioned or 'INTERSECT' to be spelled out.
Do not reject faithful high-level paraphrases for brevity or style.

Not allowed: 'people born after 1840' when the contract requires before
1840 (if explicitly stated); 'A OR B' when the fragment requires A AND B;
returning people when the fragment must return their events; pretending
a sibling's output is already known; or a generic Goal such as 'find
relevant information' that does not identify an actual subtask target.

Public null labels mean the exact literal was withheld; never invent
missing identity labels. Upstream dependencies are planning inputs, not
completed findings. No private source results or Gold answers are present.

Return ONLY compact JSON:
{"verdict":"pass|revise|uncertain","issues":[],"summary":"..."}
PASS when the core intent and expected output are recognizable and NO
explicit conflicting claim is present. For REVISE, specify 1-3 genuine
core-intent/output contradictions plus the minimal fix. For genuinely
ambiguous core intent return UNCERTAIN. Do NOT turn the omission of
fine-grained implementation details into a failure.
"""

_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"enum": ["pass", "revise", "uncertain"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["verdict", "issues", "summary"],
    "additionalProperties": False,
}


def _numeric_tokens(text):
    """Numbers already public in local steps, never raw ProgramIR operands."""
    return {
        m.group(0).replace(",", "")
        for m in re.finditer(r"(?<![\w])\d[\d,]*(?:\.\d+)?(?![\w])", str(text))
    }


def deterministic_goal_issues(outline, fragment, goal):
    """Catch only explicit local intent contradictions, never missing details.

    Goals may legitimately summarize filters as 'specified criteria'.
    Exact literals and every intermediate step still live in the public
    semantic contract, so absence of a numeric/date value is NOT a failure.
    """
    issues = []
    by_id = {task["fragment_id"]: task for task in outline["tasks"]}
    allowed = {
        str(s["public_value"]) for s in fragment["public_steps"]
        if s.get("public_value") is not None
    }
    for dep in fragment["depends_on"]:
        allowed.update(
            str(s["public_value"])
            for s in by_id[dep]["public_steps"]
            if s.get("public_value") is not None
        )
    present = set()
    for value in allowed:
        present.update(_numeric_tokens(value))
    # An unambiguous number unique to an unrelated branch is still a
    # concrete sign of scope contamination. No requirement that any local
    # number itself be repeated in the Goal.
    foreign = set()
    for other in outline["tasks"]:
        if (other["fragment_id"] == fragment["fragment_id"]
                or other["fragment_id"] in fragment["depends_on"]):
            continue
        for step in other["public_steps"]:
            if step.get("public_value") is not None:
                foreign.update(_numeric_tokens(step["public_value"]))
    for value in sorted((_numeric_tokens(goal) & foreign) - present):
        issues.append("unrelated_branch_numeric:" + value)

    # If a specific != threshold IS explicitly given, check that the goal
    # has not replaced != with an ordered comparison on that SAME literal.
    # If it omits the threshold entirely, that's valid summarization.
    for step in fragment["public_steps"]:
        value = step.get("public_value")
        comparison = str(step.get("comparison") or "").lower()
        if value is None or not any(
            key in comparison for key in ("not equal", "different", "unequal", "other than")
        ):
            continue
        for literal in _numeric_tokens(value):
            if literal not in _numeric_tokens(goal):
                continue
            escaped = re.escape(literal)
            ordered = (
                r"(?:after|before|later than|earlier than|greater than|less than)"
                r"\s+(?:the\s+year\s+)?"
            )
            if re.search(r"\b" + ordered + escaped + r"\b", goal, re.I):
                # Broader context may include multiple criteria. Preserve
                # conservative behaviour: flag the clear ordered-comparison
                # phrase when no explicit != wording occurs elsewhere.
                if not re.search(
                    r"\b(?:not equal|different from|other than|except|excluding|isn't|is not)\b",
                    goal, re.I,
                ):
                    issues.append("not_equal_rewritten_as_ordered_comparison:" + str(value))
    return list(dict.fromkeys(issues))


def validate_review(review, fragment, goal):
    """Verify the compact reviewer protocol, not brittle word/step alignment.

    LLM assessment is not semantic proof: explicit deterministic fidelity
    issues are independently checked before review and on certificate replay.
    """
    if not isinstance(review, dict) or set(review) != {
        "verdict", "issues", "summary",
    }:
        raise ValueError("invalid_goal_review_schema")
    if review["verdict"] not in ("pass", "revise", "uncertain"):
        raise ValueError("invalid_goal_review_verdict")
    if (not isinstance(review["issues"], list) or len(review["issues"]) > 3
            or any(not isinstance(x, str) or not x.strip()
                   for x in review["issues"])
            or not isinstance(review["summary"], str)
            or not review["summary"].strip()):
        raise ValueError("invalid_goal_review_details")
    if review["verdict"] == "pass" and review["issues"]:
        raise ValueError("goal_review_pass_with_issues")
    if review["verdict"] == "revise" and not review["issues"]:
        raise ValueError("goal_review_revision_without_issue")
    return review


def review_fragment_goal(backend, outline, fragment, reference, goal,
                         *, audit=None):
    from .backend import BackendError
    if not callable(getattr(backend, "complete_json", None)):
        raise ValueError("goal_critic_backend_missing")
    issues = deterministic_goal_issues(outline, fragment, goal)
    if issues:
        return {
            "verdict": "revise", "issues": issues[:3],
            "summary": "Fix explicit public-contract violations first.",
        }
    packet = _fragment_goal_packet(outline, fragment, reference)
    packet["candidate_goal"] = goal
    for attempt in range(2):
        try:
            if audit is not None:
                audit["critic_calls"] = audit.get("critic_calls", 0) + 1
            output = backend.complete_json(
                _REVIEW_SYSTEM + (
                    "\nYour previous reply failed the JSON protocol. Return "
                    "only verdict, issues and summary; no extra keys."
                    if attempt else ""
                ),
                json.dumps(packet, ensure_ascii=False),
                response_schema=_REVIEW_SCHEMA,
            )
            return validate_review(output, fragment, goal)
        except (BackendError, ValueError, TypeError, KeyError) as exc:
            if audit is not None:
                audit.setdefault("protocol_failures", []).append(str(exc)[:180])
            last_error = exc
    # Reviewer failure is not a semantic discrepancy and must not certify.
    # Both attempts reviewed the SAME Goal: never waste a Goal Author call
    # merely because the Critic was unable to serialize its decision.
    return {
        "verdict": "uncertain", "issues": [],
        "summary": "goal_review_protocol_error:" + str(last_error)[:160],
    }


def certify_goal_record(payload, world, canonical, outline, blueprint, reviews,
                        *, oracle_answer=None):
    """Commit only a fully reviewed, immutable task-level Blueprint."""
    if not isinstance(reviews, list) or len(reviews) != len(outline["tasks"]):
        raise ValueError("goal_review_count_mismatch")
    validate_delegation_blueprint(
        blueprint, outline, payload=payload, oracle_answer=oracle_answer,
    )
    for fragment, row, goal in zip(
        outline["tasks"], reviews, blueprint["goals"]
    ):
        if (not isinstance(row, dict) or set(row) != {"fragment_id", "review"}
                or row["fragment_id"] != fragment["fragment_id"]):
            raise ValueError("goal_review_fragment_binding")
        validate_review(row["review"], fragment, goal["goal"])
        if row["review"]["verdict"] != "pass":
            raise ValueError("uncertified_goal_review")
        if deterministic_goal_issues(outline, fragment, goal["goal"]):
            raise ValueError("goal_failed_deterministic_fidelity")
    cert = {
        "schema": GOAL_CERT_SCHEMA, "method": GOAL_REVIEW_METHOD,
        "task_id": payload["task_id"],
        "program_sha256": digest(world["program"]),
        "canonical_sha256": digest(canonical),
        "outline_sha256": digest(outline),
        "blueprint_sha256": digest(blueprint),
        "reviews_sha256": digest(reviews),
    }
    return {
        "task_id": payload["task_id"],
        "draft_question": canonical,
        "delegation_blueprint": blueprint,
        "goal_reviews": reviews,
        "goal_certificate": cert,
        "publication_status": "goals_certified",
    }


def validate_goal_record(record, payload, world, canonical, outline, *,
                         oracle_answer=None):
    """Stage 2B recompile + replay (no Goal LLM calls) before Question Author."""
    if (not isinstance(record, dict)
            or record.get("publication_status") != "goals_certified"
            or record.get("task_id") != payload.get("task_id")
            or record.get("draft_question") != canonical):
        raise ValueError("invalid_stage2a_record_binding")
    cert = record.get("goal_certificate")
    if not isinstance(cert, dict):
        raise ValueError("missing_stage2a_certificate")
    expected = certify_goal_record(
        payload, world, canonical, outline,
        record["delegation_blueprint"], record["goal_reviews"],
        oracle_answer=oracle_answer,
    )["goal_certificate"]
    if cert != expected:
        raise ValueError("stage2a_goal_certificate_mismatch")
    return digest(cert)


__all__ = [
    "GOAL_CERT_SCHEMA", "GOAL_REVIEW_METHOD",
    "deterministic_goal_issues",
    "validate_review", "review_fragment_goal", "certify_goal_record",
    "validate_goal_record",
]
