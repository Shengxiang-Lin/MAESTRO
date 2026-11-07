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
    _fragment_goal_packet, _terminal_result_contract,
    validate_delegation_blueprint,
)
from environment.trajectory_simulation.eligibility import digest


GOAL_CERT_SCHEMA = "maestro.stage2a.certified_blueprint.v4"
GOAL_REVIEW_METHOD = "subtask_intent_target_and_owner_critic_v4"

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
2) OUTPUT: Does it aim at the correct kind of result (entity names,
   entity set, NUMBER/COUNT, or attribute VALUE), and correct PUBLIC
   entity target when specified? For a terminal COUNT, a Goal that only
   says 'identify the single entity' FAILS: it must say count/how many/
   total number. For QUERY_NAME, returning names of PEOPLE instead of
   EVENTS FAILS when the terminal_result_contract says Event. For
   QUERY_ATTR, returning names instead of attribute values FAILS.
   terminal_result_contract, when provided, is the authoritative small
   output hint; it does not contain the full question or other branches.
3) NO CONTRADICTION: Does anything it explicitly CLAIM reverse a
   relation, misattribute a property, alter AND/OR, alter a comparison,
   add an unrelated branch condition, invent an answer, or narrow/expand
   the candidate pool in a way incompatible with the contract? In particular
   'male population < 727' means an area with THAT population ATTRIBUTE,
   NOT an entity that is male. A place's end date must not become the
   end date of an event found later along a relation hop. Track which
   input entities own attributes at EACH relevant step.

ALLOWED abstraction: omitted thresholds, dates, names, intermediate
steps, attribute predicates and explicit operator names. A phrase like
'people meeting the specified criteria' may stand for multiple accurate
filters whose details remain in the separate contract. Do not require
EVERY public step to be mentioned or 'INTERSECT' to be spelled out.
Do not reject faithful high-level paraphrases for brevity or style.

Not allowed: 'people born after 1840' when the contract requires before
1840 (if explicitly stated); 'A OR B' when the fragment requires A AND B;
returning people when the fragment must return their events; replacing
'count matching entities' with 'find one entity'; turning an entity's
male population into that entity's gender; pretending a sibling's output
is already known; or a generic Goal such as 'find relevant information'
that does not identify an actual subtask target.

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


_COUNT_INTENT = re.compile(
    r"\b(?:count|counting|tally|tallying|how many|number of|total number|"
    r"total count|quantity of|cardinality)\b", re.I
)
_NAME_INTENT = re.compile(
    r"\b(?:name|names|named|identity|identities|which|who)\b", re.I
)
_ATTRIBUTE_INTENT = re.compile(
    r"\b(?:attribute|value|values|date|year|population|height|mass|"
    r"weight|address|age|measurement|number|score|rank|ranking|"
    r"largest|smallest)\b", re.I
)
_OUTPUT_ENTITY_ALIASES = {
    "event": r"\bevents?\b",
    "person": r"\b(?:person|people|persons|individuals?|humans?)\b",
    "individual": r"\b(?:person|people|persons|individuals?|humans?)\b",
    "human": r"\b(?:person|people|persons|individuals?|humans?)\b",
    "organization": r"\b(?:organizations?|organisations?|companies|company|firms?)\b",
    "place": r"\b(?:places?|locations?|sites?)\b",
    "administrative area": r"\b(?:administrative areas?|regions?|districts?)\b",
    "taxon": r"\b(?:taxa|taxons?|species)\b",
    "creative work": r"\b(?:creative works?|works?)\b",
}


def _terminal_goal_issues(outline, fragment, goal):
    """High-confidence final answer-type mismatches, NOT detail coverage."""
    output = _terminal_result_contract(outline, fragment)
    if not output:
        return []
    action = output["action"]
    issues = []
    count = bool(_COUNT_INTENT.search(goal))
    if action == "COUNT":
        # Unlike numeric thresholds, a COUNT result is the very purpose
        # of the terminal subtask. A Goal that just finds one entity is
        # not an adequate intent hint for a counting task.
        if not count:
            issues.append("terminal_count_intent_missing")
    elif action == "QUERY_NAME":
        if count and not _NAME_INTENT.search(goal):
            issues.append("terminal_names_replaced_by_count")
        # Natural "identify matching entities" can mean finding names,
        # so do not require a literal QUERY_NAME/names token.
    elif action == "QUERY_ATTR":
        if re.search(
            r"\b(?:return|report|list|give|provide)\s+"
            r"(?:the\s+)?(?:names?|identities)\s+of\b", goal, re.I
        ) and not _ATTRIBUTE_INTENT.search(goal):
            issues.append("terminal_attribute_values_replaced_by_names")

    # Check *explicit* result noun only. The Goal may mention Persons
    # as input conditions while its actual output is Events.
    target = str(output.get("target_entity_type", "")).lower()
    expected = _OUTPUT_ENTITY_ALIASES.get(target)
    if expected:
        clauses = re.findall(
            r"\b(?:return|report|give|provide|list|count)\b[^.;]{0,115}",
            goal, flags=re.I,
        )
        for clause in clauses:
            if re.search(expected, clause, re.I):
                continue
            for other, alias in _OUTPUT_ENTITY_ALIASES.items():
                if other == target:
                    continue
                # Overlap in entity synonyms is intentional.
                if (target in {"person", "individual", "human"}
                        and other in {"person", "individual", "human"}):
                    continue
                if re.search(alias, clause, re.I):
                    issues.append(
                        "terminal_wrong_result_entity_type:" + target
                    )
                    return list(dict.fromkeys(issues))
    return list(dict.fromkeys(issues))


def _explicit_attribute_owner_issues(fragment, goal):
    """Flag explicit gender/aggregate confusion, not omitted attributes."""
    attributes = {
        str(s.get("attribute", "")).lower()
        for s in fragment.get("public_steps", ())
    }
    issues = []
    for sex in ("male", "female"):
        if sex + " population" not in attributes:
            continue
        if re.search(
            r"\b(?:entities|areas|regions|places|organizations|events)\b"
            r"[^.;]{0,85}\b(?:are|were|is)\s+" + sex + r"\b",
            goal, re.I,
        ) or re.search(
            r"\b" + sex +
            r"\s+(?:entities|areas|regions|places|organizations|events)\b",
            goal, re.I,
        ):
            issues.append("population_attribute_misread_as_gender:" + sex)
    return issues


def _explicit_relation_owner_issues(fragment, goal):
    """Only unambiguous place-hop/date shifts; leave vague prose to Critic."""
    steps = fragment.get("public_steps", ())
    for index, step in enumerate(steps):
        if (step.get("operation") != "RELATE"
                or step.get("direction") != "forward"
                or step.get("relation") not in ("birth place", "death place")):
            continue
        # The immediately followed public steps still act on the PLACE
        # reached by the relation, up to the next relation hop.
        later = []
        for current in steps[index + 1:]:
            if current.get("operation") == "RELATE":
                break
            later.append(current)
        if not any(
            s.get("operation") in ("FILTER_DATE", "FILTER_YEAR")
            and s.get("attribute") == "end date"
            for s in later
        ):
            continue
        if re.search(
            r"\b(?:events?|competitions?|matches?)\b"
            r"[^.;]{0,70}\b(?:ended|ending|end date)\b", goal, re.I,
        ):
            return ["place_end_date_attributed_to_event"]
    return []


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
    issues.extend(_terminal_goal_issues(outline, fragment, goal))
    issues.extend(_explicit_attribute_owner_issues(fragment, goal))
    issues.extend(_explicit_relation_owner_issues(fragment, goal))
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
