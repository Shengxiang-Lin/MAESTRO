"""Lightweight Stage-2 editorial certificate and Stage-3 deterministic replay.

A checked reference and bounded public guards are independently rebuilt.
The Critic's PASS is an auditable model assessment, NOT a formal proof.
No final judge, evidence tree, or probe coverage is required.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from environment.trajectory_simulation.eligibility import (
    controlled_question_reference, digest,
)
from environment.trajectory_simulation.compiler import compile_gold_solution

from .question_review import passing_judgment
from .semantic_checks import (
    explicit_join_contradictions, missing_typed_literals,
    relation_semantic_contradictions,
)

SCHEMA = "maestro.stage2_surface_cert.simple.v1"
REVIEW_CONTRACT = 29
METHOD = "simple_author_critic_local_repair_v1"


def _certificate(task_id, program, canonical, question, review):
    return {
        "schema": SCHEMA,
        "review_contract": REVIEW_CONTRACT,
        "task_id": task_id,
        "question_sha256": digest(question),
        "program_sha256": digest(program),
        "renderer_question_sha256": digest(canonical),
        "method": METHOD,
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
    revision_rounds: int, plan,
):
    """Issue a small certificate, only after Critic PASS and public checks."""
    if (not isinstance(critic_review, Mapping)
            or not passing_judgment(critic_review)
            or type(revision_rounds) is not int or revision_rounds < 0):
        raise ValueError("natural_question_not_approved")
    guards = _check_public_candidate(
        plan, canonical, scoped_reference, question,
    )
    review = {
        "eligible": True, "reviewed": True, "decision": "eligible",
        "method": METHOD, "issues": [],
        "question_sha256": digest(question),
        "program_sha256": digest(world["program"]),
        "scoped_reference_sha256": digest(scoped_reference),
        "critic_review": dict(critic_review),
        "revision_rounds": revision_rounds,
        "public_guards_sha256": digest(guards),
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
    authoring = realized.get("authoring") or {}
    review = authoring.get("surface_review")
    question = realized.get("question")
    if (not isinstance(certificate, dict) or not isinstance(review, dict)
            or not isinstance(question, str)):
        raise ValueError("missing_stage2_surface_certificate")
    required = {
        "schema", "review_contract", "task_id", "question_sha256",
        "program_sha256", "renderer_question_sha256", "method",
        "review_sha256",
    }
    if set(certificate) != required:
        raise ValueError("invalid_stage2_surface_certificate")
    if (certificate.get("schema") != SCHEMA
            or certificate.get("review_contract") != REVIEW_CONTRACT
            or certificate.get("method") != METHOD):
        raise ValueError("unsupported_stage2_review_contract")
    expected = _certificate(
        realized.get("task_id"), world["program"], canonical, question, review,
    )
    if certificate != expected:
        raise ValueError("stage2_surface_certificate_input_mismatch")
    if (review.get("method") != METHOD
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
    return {
        "eligible": True, "reviewed": True, "decision": "eligible",
        "method": "stage2_verified_editorial_certificate",
        "issues": [],
        "question_sha256": digest(question),
        "program_sha256": digest(world["program"]),
        "stage2_method": METHOD,
        "stage2_review_contract": REVIEW_CONTRACT,
    }


__all__ = [
    "SCHEMA", "REVIEW_CONTRACT", "METHOD",
    "certify_approved_question", "validate_surface_certificate",
]
