"""Stage 2A: independent local Goal fidelity reviews and replayable certificates.

A pass is an audited (fallible) semantic judgement, NOT a formal proof.
The deterministic checks protect obvious public literals, numeric operators,
branch leakage and review evidence. No private Gold facts are sent to models.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping

from environment.delegation_blueprint import (
    _fragment_goal_packet, _validate_goal, validate_delegation_blueprint,
)
from environment.trajectory_simulation.eligibility import digest


GOAL_CERT_SCHEMA = "maestro.stage2a.certified_blueprint.v1"
GOAL_REVIEW_METHOD = "local_contract_goal_critic_v1"
_SEMANTIC_OPS = {
    "RELATE", "INTERSECT", "UNION", "COUNT", "QUERY_NAME",
    "QUERY_ATTR", "QUERY_RELATION", "SELECT_AMONG", "SELECT_BETWEEN",
    "FILTER_TYPE", "FILTER_YEAR", "FILTER_DATE", "FILTER_NUMBER",
    "FILTER_STRING", "VERIFY_STRING", "VERIFY_YEAR", "VERIFY_DATE",
    "VERIFY_NUMBER",
}
_REVIEW_SYSTEM = """Independently review ONE Sub-Agent goal against its LOCAL
authoritative public semantic steps and EXACT direct dependency contracts.
Do not use other fragments' generated text or assume knowledge of Gold answers.

Check EVERY substantive operation. For RELATE check the edge direction and
which entities are subjects vs objects. For FILTER/VERIFY check owner, exact
date/value, comparator (not equal != after/before), conjunction, and scope.
For INTERSECT require same entity in ALL branches; for UNION require ANY
branch. Check named singleton vs groups, separate candidate pools and final
output type. A goal must not impose any condition from a sibling branch,
silently narrow a pool, substitute a parent for a child or invent source
information. The upstream dependency contracts are INPUT descriptions:
do not claim their answers are already known or received.

The goal is allowed to combine consecutive semantics in natural prose.
Do not require literal operator names, steps or internal fragment IDs.

Return compact JSON with fields:
- verdict: pass, revise or uncertain;
- issues: 0-3 concrete local errors, empty for pass;
- summary: concise reason;
- coverage: one item for EACH substantive local step with step_id and an
  ACTUAL short verbatim quote from the candidate goal that expresses the step.
Do not invent quote spans; the caller checks quotes against the goal.
If a local semantic step is unexpressed, verdict=revise, explain which step
and give a minimal repair. If unclear, verdict=uncertain; do not guess.
PASS only with no issues and coverage of EVERY substantive local step.
"""
_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"enum": ["pass", "revise", "uncertain"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "coverage": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "string"},
                    "goal_quote": {"type": "string"},
                },
                "required": ["step_id", "goal_quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["verdict", "issues", "summary", "coverage"],
    "additionalProperties": False,
}


def substantive_step_ids(fragment):
    return [
        step["id"] for step in fragment["public_steps"]
        if step["operation"] in _SEMANTIC_OPS
    ]


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
    if not isinstance(review, dict) or set(review) != {
        "verdict", "issues", "summary", "coverage",
    }:
        raise ValueError("invalid_goal_review_schema")
    if review["verdict"] not in ("pass", "revise", "uncertain"):
        raise ValueError("invalid_goal_review_verdict")
    if (not isinstance(review["issues"], list) or len(review["issues"]) > 3
            or any(not isinstance(x, str) or not x.strip() for x in review["issues"])
            or not isinstance(review["summary"], str)
            or not review["summary"].strip()
            or not isinstance(review["coverage"], list)):
        raise ValueError("invalid_goal_review_details")
    ids = substantive_step_ids(fragment)
    seen = []
    for row in review["coverage"]:
        if (not isinstance(row, dict) or set(row) != {"step_id", "goal_quote"}
                or not isinstance(row["step_id"], str)
                or not isinstance(row["goal_quote"], str)
                or not row["goal_quote"].strip()
                or row["goal_quote"].casefold() not in goal.casefold()):
            raise ValueError("invalid_goal_review_quote")
        seen.append(row["step_id"])
    if len(seen) != len(set(seen)) or any(s not in ids for s in seen):
        raise ValueError("invalid_goal_review_coverage_ids")
    if review["verdict"] == "pass":
        if review["issues"] or set(seen) != set(ids):
            raise ValueError("goal_review_pass_without_full_coverage")
    else:
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
            "coverage": [],
        }
    packet = _fragment_goal_packet(outline, fragment, reference)
    packet["candidate_goal"] = goal
    try:
        if audit is not None:
            audit["critic_calls"] = audit.get("critic_calls", 0) + 1
        output = backend.complete_json(
            _REVIEW_SYSTEM, json.dumps(packet, ensure_ascii=False),
            response_schema=_REVIEW_SCHEMA,
        )
        return validate_review(output, fragment, goal)
    except (BackendError, ValueError, TypeError, KeyError) as exc:
        if audit is not None:
            audit.setdefault("protocol_failures", []).append(str(exc)[:180])
        # Invalid reviewer JSON or fake quotes are NEVER a PASS.
        return {
            "verdict": "uncertain", "issues": [],
            "summary": "goal_review_protocol_error:" + str(exc)[:160],
            "coverage": [],
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
    "substantive_step_ids", "deterministic_goal_issues",
    "validate_review", "review_fragment_goal", "certify_goal_record",
    "validate_goal_record",
]
