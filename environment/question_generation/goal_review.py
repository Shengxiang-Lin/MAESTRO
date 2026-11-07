"""Stage 2A: independent local Goal fidelity reviews and replayable certificates.

A pass is an audited (fallible) semantic judgement, NOT a formal proof.
The deterministic checks protect obvious public literals, numeric operators,
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


GOAL_CERT_SCHEMA = "maestro.stage2a.certified_blueprint.v2"
GOAL_REVIEW_METHOD = "local_contract_goal_critic_v2"

_REVIEW_SYSTEM = """Review ONE natural-language Sub-Agent goal for semantic
faithfulness to its LOCAL public fragment contract and direct dependencies.

Decide whether executing the candidate goal would return the SAME answer
as executing the immutable local public steps, in the same candidate pool.
Inspect relation direction and subjects/objects, attribute owner, exact
literals and comparisons (!= is NOT > or <), AND vs OR, upstream input
binding, ranking/counting, final output and extra unsupported constraints.

The natural goal is a compact ASSIGNMENT, not an operator-by-operator
execution trace. One clause may faithfully express several filters or
INTERSECT operations. DO NOT require the model to explicitly name step IDs,
state INTERSECT as an operation, or preserve commutative filter ordering.
"Entities satisfying A and B" already correctly represents intersection.
"Different from X" means != X (not simply after/before X).
"Entity type": null in a public step means that the exact type label
is WITHHELD from the public projection, NOT that the FILTER_TYPE step
does not exist. Do not demand a private entity type or invent it.
Check whether the goal retains the type restriction in public wording.

Public dependency contracts are PLANNING requirements, not completed
observations. Never assume their answers or introduce constraints from
other sibling branches. Unspecified public anchor values must not be
invented or replaced by precise facts. Source evidence must be discovered
if not already provided.

Return ONLY a compact JSON object:
{"verdict":"pass|revise|uncertain","issues":[],"summary":"..."}
For PASS use issues=[] and briefly explain that the answer-defining
constraints are preserved. For REVISE give one to three concrete
answer-changing discrepancies in issues and a minimal semantic repair.
For UNCERTAIN briefly explain which specific meaning cannot be confirmed.
Do not reject a faithful natural paraphrase merely for merging steps,
changing wording, or omitting internal tool mechanics.
Do not claim PASS unless ALL answer-defining conditions are faithful.
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
    """High-confidence violations only. Unclear equivalence goes to critic."""
    issues = []
    own = {
        str(s["public_value"]) for s in fragment["public_steps"]
        if s.get("public_value") is not None
    }
    by_id = {task["fragment_id"]: task for task in outline["tasks"]}
    allowed = set(own)
    for dep in fragment["depends_on"]:
        allowed.update(
            str(s["public_value"])
            for s in by_id[dep]["public_steps"]
            if s.get("public_value") is not None
        )
    available = _numeric_tokens(goal)
    present = set()
    for value in allowed:
        present.update(_numeric_tokens(value))
    # Local numerical comparisons are answer-defining. If the model omits a
    # value, this is a repair request, not a license to hallucinate it.
    for value in own:
        digits = _numeric_tokens(value)
        if digits and not digits.issubset(available):
            # ISO dates may be rendered in natural prose; a matching year is
            # sufficient for this cheap check, the critic judges full date.
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                if value[:4] in available:
                    continue
            issues.append("missing_local_numeric:" + value)
    # Only foreign explicit public numerals are suspicious; avoid rejecting
    # text merely because it mentions 'one' or 'two' in natural language.
    foreign = set()
    for other in outline["tasks"]:
        if other["fragment_id"] == fragment["fragment_id"] or other["fragment_id"] in fragment["depends_on"]:
            continue
        for s in other["public_steps"]:
            if s.get("public_value") is not None:
                foreign.update(_numeric_tokens(s["public_value"]))
    for val in sorted((available & foreign) - present):
        # A shared literal may be valid in both sibling branches. The
        # explicit 'present' set above deliberately permits that case.
        issues.append("unrelated_branch_numeric:" + val)
    # Preserve clear comparisons, especially the != -> > regression.
    for step in fragment["public_steps"]:
        value = step.get("public_value")
        cmp = str(step.get("comparison") or "").lower()
        if not value or not _numeric_tokens(value):
            continue
        if any(k in cmp for k in ("not equal", "different", "unequal", "other than")):
            if re.search(r"\b(?:after|before|later than|earlier than|greater than|less than)\b",
                         goal, re.I) and not re.search(
                r"\b(?:not equal|different from|other than|except|excluding|isn't|is not)\b",
                goal, re.I
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
