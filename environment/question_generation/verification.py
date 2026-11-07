"""Stage-2 blueprint/question certificate and independent Stage-3 replay.

The task DAG topology is recompiled, then the PUBLIC blueprint is checked
against that graph and its bound question. This checks integrity, not a
mathematical proof of natural-language semantic equivalence.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from environment.trajectory_simulation.eligibility import (
    controlled_question_reference, digest,
)
from environment.trajectory_simulation.compiler import compile_gold_solution

from environment.delegation_blueprint import (
    compile_delegation_outline, validate_delegation_blueprint,
)
from .goal_review import validate_goal_record
from .question_review import passing_judgment
from .semantic_checks import (
    explicit_join_contradictions, missing_typed_literals,
    relation_semantic_contradictions,
)

SCHEMA = "maestro.stage2_surface_cert.blueprint.v2"
REVIEW_CONTRACT = 31
METHOD = "fragment_blueprint_guided_author_critic_v2"
CERTIFIED_SCHEMA = "maestro.stage2_surface_cert.certified_goals.v3"
CERTIFIED_REVIEW_CONTRACT = 32
CERTIFIED_METHOD = "stage2a_certified_goals_question_critic_v3"


def _certificate(task_id, program, canonical, question, review):
    certified = "goal_certificate_sha256" in review
    return {
        "schema": CERTIFIED_SCHEMA if certified else SCHEMA,
        "review_contract": (
            CERTIFIED_REVIEW_CONTRACT if certified else REVIEW_CONTRACT
        ),
        "task_id": task_id,
        "question_sha256": digest(question),
        "program_sha256": digest(program),
        "renderer_question_sha256": digest(canonical),
        "method": CERTIFIED_METHOD if certified else METHOD,
        "review_sha256": digest(review),
    }


def _meaningful_edit(reference, candidate):
    normalize = lambda s: re.sub(r"[^a-z0-9]+", "", s.casefold())
    return (bool(candidate.strip())
            and normalize(reference) != normalize(candidate))


def _simple_guards(plan, scoped, candidate):
    return {
        "missing_typed_literals": missing_typed_literals(plan, candidate),
        "explicit_join_contradictions": explicit_join_contradictions(
            scoped, candidate,
        ),
        "relation_semantic_contradictions": relation_semantic_contradictions(
            scoped, candidate,
        ),
    }


def _check_public_candidate(plan, canonical, scoped, candidate):
    if (not isinstance(candidate, str) or not _meaningful_edit(scoped, candidate)
            or not _meaningful_edit(canonical, candidate)):
        raise ValueError("natural_question_not_rewritten")
    guards = _simple_guards(plan, scoped, candidate)
    if any(guards.values()):
        raise ValueError("natural_question_failed_public_guards")
    return guards


def certify_approved_question(
    task_id: str, world: Mapping, canonical: str, question: str,
    scoped_reference: str, critic_review: Mapping, *,
    revision_rounds: int, plan, blueprint, gold, resources=(),
    goal_record=None, question_brief=None,
):
    """Issue a small certificate, only after Critic PASS and public checks."""
    if (not isinstance(critic_review, Mapping)
            or not passing_judgment(critic_review)
            or type(revision_rounds) is not int or revision_rounds < 0):
        raise ValueError("natural_question_not_approved")
    guards = _check_public_candidate(
        plan, canonical, scoped_reference, question,
    )
    expected_outline = compile_delegation_outline(
        gold, world, canonical, plan, resources,
    )
    blueprint_hash = validate_delegation_blueprint(
        blueprint, expected_outline,
        payload={"resources": resources},
        oracle_answer=world.get("_display_answer"),
    )
    if goal_record is not None:
        validate_goal_record(
            goal_record,
            {"task_id": task_id, "resources": resources},
            world, canonical, expected_outline,
            oracle_answer=world.get("_display_answer"),
        )
    review = {
        "eligible": True, "reviewed": True, "decision": "eligible",
        "method": CERTIFIED_METHOD if goal_record is not None else METHOD,
        "issues": [],
        "question_sha256": digest(question),
        "program_sha256": digest(world["program"]),
        "scoped_reference_sha256": digest(scoped_reference),
        "critic_review": dict(critic_review),
        "revision_rounds": revision_rounds,
        "public_guards_sha256": digest(guards),
        "blueprint_sha256": blueprint_hash,
        **({
            "question_brief_sha256": digest(question_brief),
        } if question_brief is not None else {}),
        **({
            "goal_certificate_sha256": digest(goal_record["goal_certificate"]),
        } if goal_record is not None else {}),
    }
    return _certificate(
        task_id, world["program"], canonical, question, review,
    ), review


def _rebuild_scoped_reference(realized, world, canonical, gold, rendered):
    """Never trust the reference text stored by Stage 2."""
    from .scoped_surface import build_scoped_plan, render_scoped
    if rendered is None or rendered.question != canonical:
        raise ValueError("missing_matching_stage1_renderer")
    if gold is None:
        gold = compile_gold_solution(
            {**realized, "question": canonical},
            world, rendered, schedule="compact",
        )
    controlled = controlled_question_reference(gold, world, rendered)
    if controlled is None or controlled.get("question") != canonical:
        raise ValueError("unverified_stage1_reference")
    plan = build_scoped_plan(world, canonical, gold, controlled)
    return render_scoped(plan, [0] * len(plan)), plan


def validate_surface_certificate(
    realized, world, canonical, *, gold=None, rendered=None,
):
    """Replay the exact Stage-2 approval record without another model call.

    This does not recheck an LLM's semantic correctness. Other Stage-3
    public-evidence checks run independently as before.
    """
    certificate = realized.get("surface_certificate")
    blueprint = realized.get("delegation_blueprint")
    authoring = realized.get("authoring") or {}
    review = authoring.get("surface_review")
    question = realized.get("question")
    if (not isinstance(certificate, dict) or not isinstance(review, dict)
            or not isinstance(question, str) or not isinstance(blueprint, dict)):
        raise ValueError("missing_stage2_surface_certificate")
    required = {
        "schema", "review_contract", "task_id", "question_sha256",
        "program_sha256", "renderer_question_sha256", "method",
        "review_sha256",
    }
    if set(certificate) != required:
        raise ValueError("invalid_stage2_surface_certificate")
    certified = certificate.get("schema") == CERTIFIED_SCHEMA
    if certified:
        if (certificate.get("review_contract") != CERTIFIED_REVIEW_CONTRACT
                or certificate.get("method") != CERTIFIED_METHOD):
            raise ValueError("unsupported_stage2_review_contract")
    elif (certificate.get("schema") != SCHEMA
          or certificate.get("review_contract") != REVIEW_CONTRACT
          or certificate.get("method") != METHOD):
        raise ValueError("unsupported_stage2_review_contract")
    expected = _certificate(
        realized.get("task_id"), world["program"], canonical, question, review,
    )
    if certificate != expected:
        raise ValueError("stage2_surface_certificate_input_mismatch")
    if (review.get("method") != (CERTIFIED_METHOD if certified else METHOD)
            or review.get("eligible") is not True
            or review.get("reviewed") is not True
            or review.get("decision") != "eligible"
            or review.get("issues") != []
            or review.get("question_sha256") != digest(question)
            or review.get("program_sha256") != digest(world["program"])
            or type(review.get("revision_rounds")) is not int
            or review["revision_rounds"] < 0
            or not isinstance(review.get("critic_review"), dict)
            or not passing_judgment(review["critic_review"])):
        raise ValueError("stage2_review_not_approved")
    if realized.get("publication_status") != "naturalized_certified":
        raise ValueError("stage2_publication_status_mismatch")

    rendered = rendered or world.get("_rendered")
    if gold is None:
        gold = compile_gold_solution(
            {**realized, "question": canonical}, world, rendered,
            schedule="compact",
        )
    scoped, plan = _rebuild_scoped_reference(
        realized, world, canonical, gold, rendered,
    )
    if review.get("scoped_reference_sha256") != digest(scoped):
        raise ValueError("stage2_scoped_reference_changed")
    guards = _check_public_candidate(
        plan, canonical, scoped, question,
    )
    if review.get("public_guards_sha256") != digest(guards):
        raise ValueError("stage2_public_guard_record_changed")
    outline = compile_delegation_outline(
        gold, world, canonical, plan, realized.get("resources", ()),
    )
    valid_hash = validate_delegation_blueprint(
        blueprint, outline, payload=realized,
        oracle_answer=world.get("_display_answer", getattr(rendered, "answer", None)),
    )
    if review.get("blueprint_sha256") != valid_hash:
        raise ValueError("stage2_blueprint_binding_mismatch")
    if "question_brief_sha256" in review:
        from .question_brief import compile_question_brief
        expected_brief = compile_question_brief(
            blueprint, world, canonical, gold,
        )
        if review["question_brief_sha256"] != digest(expected_brief):
            raise ValueError("stage2_question_brief_binding_mismatch")
    if certified:
        record = {
            "task_id": realized["task_id"],
            "draft_question": canonical,
            "delegation_blueprint": blueprint,
            "goal_certificate": realized.get("goal_certificate"),
            "goal_reviews": realized.get("goal_reviews"),
            "publication_status": "goals_certified",
        }
        validate_goal_record(
            record, realized, world, canonical, outline,
            oracle_answer=world.get("_display_answer",
                                    getattr(rendered, "answer", None)),
        )
        if review.get("goal_certificate_sha256") != digest(
            record["goal_certificate"]
        ):
            raise ValueError("stage2_goal_certificate_binding_mismatch")
    return {
        "eligible": True, "reviewed": True, "decision": "eligible",
        "method": "stage2_verified_editorial_certificate",
        "issues": [],
        "question_sha256": digest(question),
        "program_sha256": digest(world["program"]),
        "stage2_method": CERTIFIED_METHOD if certified else METHOD,
        "stage2_review_contract": (
            CERTIFIED_REVIEW_CONTRACT if certified else REVIEW_CONTRACT
        ),
    }


__all__ = [
    "SCHEMA", "REVIEW_CONTRACT", "METHOD",
    "CERTIFIED_SCHEMA", "CERTIFIED_REVIEW_CONTRACT", "CERTIFIED_METHOD",
    "certify_approved_question", "validate_surface_certificate",
]
