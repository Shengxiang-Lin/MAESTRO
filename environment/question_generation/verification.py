"""Teacher-private Stage-2 ProgramIR semantic certification.

The public Author/Critic never receive private World facts. The final
graph-aware reviewer sees only the question, public renderer text, and
the authoritative structured ProgramIR meanings. A certificate is an
auditable pipeline assertion, not a formal equivalence proof or a signature.
"""
from __future__ import annotations

from dataclasses import replace

from environment.trajectory_simulation.eligibility import (
    check_question, controlled_question_reference, digest,
)
from environment.trajectory_simulation.compiler import compile_gold_solution
from environment.trajectory_simulation.backend import LLMPlanner

SCHEMA = "maestro.stage2_surface_cert.v1"
REVIEW_CONTRACT = 13


def certify_surface(payload, world, capability, question, critic):
    """Return (certificate or None, graph-review decision).

    Never certify an LLM rewrite on the old public-only Critic's verdict
    alone. A reviewer error or ambiguous result is not an approval.
    """
    rendered = world.get("_rendered")
    if rendered is None or rendered.task_id != payload["task_id"]:
        raise ValueError("missing_matching_controlled_renderer")
    canonical = rendered.question
    realized = {
        "task_id": payload["task_id"],
        "question": canonical,
        "capability_world": capability,
    }
    gold = compile_gold_solution(realized, world, rendered, schedule="compact")
    reference = controlled_question_reference(gold, world, rendered)
    if reference is None or reference["question"] != canonical:
        raise ValueError("controlled_renderer_certificate_unavailable")

    if question == canonical:
        result = {
            "eligible": True, "reviewed": True, "decision": "eligible",
            "method": "verified_controlled_renderer", "issues": [],
            "question_sha256": digest(question),
            "program_sha256": digest(world["program"]),
        }
    else:
        from .backend import LLMSemanticCritic
        if not isinstance(critic, LLMSemanticCritic):
            result = {
                "eligible": False, "reviewed": False,
                "decision": "review_error",
                "review_failure_kind": "graph_reviewer_unavailable",
                "issues": ["noncanonical rewrite needs a graph-aware reviewer"],
            }
        else:
            # Ordinary Stage-2 Critic responses are capped at 768 tokens by
            # default. The fuller graph reviewer needs its own budget.
            reviewer = LLMPlanner(replace(
                critic.backend, temperature=0.0,
                max_tokens=max(2048, critic.backend.max_tokens),
            ))
            result = check_question(
                {**gold, "question": question},
                world, reviewer, rendered=rendered,
            )
    certificate = None
    if (result.get("eligible") is True and result.get("reviewed") is True
            and result.get("decision") == "eligible"
            and result.get("issues") == []
            and result.get("method") in (
                "verified_controlled_renderer", "llm_semantic_screen",
            )):
        content = {
            "schema": SCHEMA,
            "review_contract": REVIEW_CONTRACT,
            "task_id": payload["task_id"],
            "question_sha256": digest(question),
            "program_sha256": digest(world["program"]),
            "renderer_question_sha256": digest(canonical),
            "method": result["method"],
            "review_sha256": digest(result),
        }
        certificate = content
    return certificate, result


def validate_surface_certificate(realized, world, canonical):
    """Verify Stage-2 publication linkage; do not make an additional LLM call."""
    cert = realized.get("surface_certificate")
    review = (realized.get("authoring") or {}).get("surface_review")
    question = realized.get("question")
    if not isinstance(cert, dict) or not isinstance(review, dict):
        raise ValueError("missing_stage2_surface_certificate")
    if set(cert) != {
        "schema", "review_contract", "task_id", "question_sha256",
        "program_sha256", "renderer_question_sha256",
        "method", "review_sha256",
    }:
        raise ValueError("invalid_stage2_surface_certificate")
    expected = {
        "schema": SCHEMA, "review_contract": REVIEW_CONTRACT,
        "task_id": realized.get("task_id"),
        "question_sha256": digest(question),
        "program_sha256": digest(world["program"]),
        "renderer_question_sha256": digest(canonical),
        "method": review.get("method"),
        "review_sha256": digest(review),
    }
    if cert != expected:
        raise ValueError("stage2_surface_certificate_input_mismatch")
    if (review.get("eligible") is not True
            or review.get("reviewed") is not True
            or review.get("decision") != "eligible"
            or review.get("issues") != []):
        raise ValueError("stage2_surface_review_not_eligible")
    if review.get("question_sha256") != digest(question) or review.get(
        "program_sha256"
    ) != digest(world["program"]):
        raise ValueError("stage2_surface_review_fingerprint_mismatch")
    if review.get("method") not in (
        "verified_controlled_renderer", "llm_semantic_screen",
    ):
        raise ValueError("stage2_surface_review_method_unsupported")
    if question == canonical:
        if review["method"] != "verified_controlled_renderer":
            raise ValueError("canonical_surface_certificate_method_mismatch")
    elif review["method"] != "llm_semantic_screen":
        raise ValueError("rewritten_surface_requires_graph_semantic_review")
    return {
        "eligible": True, "reviewed": True, "decision": "eligible",
        "method": "stage2_verified_programir_certificate",
        "issues": [],
        "question_sha256": digest(question),
        "program_sha256": digest(world["program"]),
        "stage2_method": review["method"],
        "stage2_review_contract": REVIEW_CONTRACT,
    }
