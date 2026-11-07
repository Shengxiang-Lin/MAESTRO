"""Publish only fully reviewed natural questions.

A certificate binds the actual Surface text, frozen ProgramIR and a
separately reconstructed public Scoped reference to an independent final
editorial judgment. The decision is an auditable LLM assessment, *not* a
formal proof of semantic equivalence. Stage 3 verifies the exact record
without another model call; it still performs its own public-executability
and evidence checks.
"""
from __future__ import annotations

from collections.abc import Mapping

from environment.trajectory_simulation.eligibility import (
    controlled_question_reference, digest,
)
from environment.trajectory_simulation.compiler import compile_gold_solution

from .question_review import passing_judgment
from .semantic_probes import (
    compile_candidate_probes, validate_candidate_probe_records,
)
from .semantic_checks import (
    validate_group_decisions, explicit_join_contradictions, naturalness_gate,
    grouped_semantic_checks, missing_typed_literals,
    relation_semantic_contradictions, CheckFormatError,
)

SCHEMA = "maestro.stage2_surface_cert.v6"
REVIEW_CONTRACT = 23
METHOD = "candidate_only_semantic_probe_review_v5"


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


def certify_approved_question(
    task_id: str, world: Mapping, canonical: str, question: str,
    scoped_reference: str, final_judgment: Mapping, *,
    critic_verdict: str, revision_rounds: int,
    semantic_checklist: list[dict],
    candidate_probes: list[dict] | None = None,
):
    """Mint a certificate only after the Critic and independent judge pass."""
    if (not question or question == canonical or question == scoped_reference
            or critic_verdict != "pass"
            or not passing_judgment(final_judgment)
            or revision_rounds < 0):
        raise ValueError("natural_question_not_approved")
    if (not naturalness_gate(scoped_reference, question)["improved_enough_to_review"]
            or explicit_join_contradictions(scoped_reference, question)
            or relation_semantic_contradictions(scoped_reference, question)):
        raise ValueError("natural_question_failed_deterministic_risk_guards")
    validate_group_decisions(semantic_checklist, final_judgment.get("checks"))
    if candidate_probes is None:
        raise ValueError("candidate_probe_records_required")
    # Actual public-plan comparison runs at issuance and Stage-3 replay.
    review = {
        "eligible": True, "reviewed": True, "decision": "eligible",
        "method": METHOD, "issues": [],
        "question_sha256": digest(question),
        "program_sha256": digest(world["program"]),
        "scoped_reference_sha256": digest(scoped_reference),
        "critic_verdict": critic_verdict,
        "revision_rounds": revision_rounds,
        "final_judgment": dict(final_judgment),
        "semantic_checklist_sha256": digest(semantic_checklist),
        "candidate_probes": list(candidate_probes),
    }
    return _certificate(task_id, world["program"], canonical, question, review), review


def _rebuild_scoped_reference(realized, world, canonical, gold, rendered):
    """Reconstruct the reference independently of Stage-2 saved prose."""
    from .scoped_surface import build_scoped_plan, render_scoped
    if rendered is None or rendered.question != canonical:
        raise ValueError("missing_matching_stage1_renderer")
    if gold is None:
        gold = compile_gold_solution(
            {**realized, "question": canonical}, world, rendered,
            schedule="compact",
        )
    verified = controlled_question_reference(gold, world, rendered)
    if verified is None or verified.get("question") != canonical:
        raise ValueError("unverified_stage1_reference")
    plan = build_scoped_plan(world, canonical, gold, verified)
    return render_scoped(plan, [0] * len(plan)), plan


def validate_surface_certificate(realized, world, canonical, *, gold=None, rendered=None):
    """Verify editorial certificate, bindings and reference; fail closed.

    No automatic acceptance of Canonical or Scoped fallback is supported.
    An LLM's semantic judgment is not mechanically provable by this check.
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
            or review.get("critic_verdict") != "pass"
            or review.get("question_sha256") != digest(question)
            or review.get("program_sha256") != digest(world["program"])
            or type(review.get("revision_rounds")) is not int
            or review["revision_rounds"] < 0
            or not isinstance(review.get("final_judgment"), dict)
            or not passing_judgment(review["final_judgment"])):
        raise ValueError("stage2_review_not_approved")
    if realized.get("publication_status") != "naturalized_certified":
        raise ValueError("stage2_publication_status_mismatch")
    rendered = rendered or world.get("_rendered")
    scoped, plan = _rebuild_scoped_reference(
        realized, world, canonical, gold, rendered,
    )
    if (question in (canonical, scoped)
            or review.get("scoped_reference_sha256") != digest(scoped)):
        raise ValueError("stage2_reference_or_naturalization_mismatch")
    checks = grouped_semantic_checks(plan)
    if (review.get("semantic_checklist_sha256") != digest(checks)
            or not naturalness_gate(scoped, question)["improved_enough_to_review"]
            or explicit_join_contradictions(scoped, question)
            or relation_semantic_contradictions(scoped, question)
            or missing_typed_literals(plan, question)):

        raise ValueError("stage2_semantic_checklist_or_style_mismatch")
    try:
        validate_group_decisions(checks, review["final_judgment"].get("checks"))
    except CheckFormatError as exc:
        raise ValueError("stage2_final_judge_risk_decision_invalid") from exc
    expected_probes = compile_candidate_probes(plan, scoped, question)
    validate_candidate_probe_records(
        expected_probes, review.get("candidate_probes"),
    )
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
