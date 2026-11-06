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
REVIEW_CONTRACT = 17

from .scoped_surface import METHOD as SCOPED_METHOD


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
            from .alignment import check_alignment
            result = check_alignment(
                gold, world, canonical, question, critic.backend,
            )
            # Small-model alignment can miss a consistent-looking owner or
            # endpoint substitution. Keep a differently represented full
            # executor-semantics audit as an independent publication gate.
            if result["eligible"]:
                alignment = result
                full = check_alignment(
                    gold, world, canonical, question, critic.backend,
                    executor_view=True,
                )
                executor_alignment = full
                if full["eligible"]:
                    full = check_question(
                        {**gold, "question": question}, world,
                        LLMPlanner(replace(
                            critic.backend, temperature=0.0,
                            max_tokens=max(2048, critic.backend.max_tokens),
                        )), rendered=rendered,
                    )
                result = {
                    **alignment, "eligible": full.get("eligible") is True
                    and full.get("reviewed") is True
                    and full.get("decision") == "eligible"
                    and full.get("issues") == [],
                    "decision": full.get("decision", "review_error"),
                    "issues": full.get("issues", ["full_graph_review_incomplete"]),
                    "full_graph_review": full,
                    "executor_alignment": executor_alignment,
                    **({"review_failure_kind": full["review_failure_kind"]}
                       if full.get("review_failure_kind") else {}),
                }
    certificate = None
    if (result.get("eligible") is True and result.get("reviewed") is True
            and result.get("decision") == "eligible"
            and result.get("issues") == []
            and result.get("method") in (
                "verified_controlled_renderer", "llm_contract_alignment",
            )):
        certificate = _mint_certificate(payload['task_id'], world, canonical, question, result)
    return certificate, result


def _mint_certificate(task_id, world, canonical, question, review):
    return {
        'schema': SCHEMA, 'review_contract': REVIEW_CONTRACT, 'task_id': task_id,
        'question_sha256': digest(question), 'program_sha256': digest(world['program']),
        'renderer_question_sha256': digest(canonical), 'method': review['method'],
        'review_sha256': digest(review),
    }

def certify_scoped_surface(task_id, world, canonical, question, groups):
    from .scoped_surface import read_scoped_semantics
    readback = read_scoped_semantics(question, groups)
    if readback is None or readback['query'] != groups.query or question == canonical:
        raise ValueError('scoped question did not round-trip')
    review = {
        'eligible': True, 'reviewed': True, 'decision': 'eligible', 'issues': [],
        'method': SCOPED_METHOD, 'question_sha256': digest(question),
        'program_sha256': digest(world['program']), 'grammar_sha256': digest(groups),
        'semantic_readback': readback, 'query_contract_sha256': digest(groups.query),
    }
    return _mint_certificate(task_id, world, canonical, question, review), review

def validate_surface_certificate(realized, world, canonical, *, gold=None, rendered=None):
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
        "schema": SCHEMA, "review_contract": cert.get("review_contract"),
        "task_id": realized.get("task_id"),
        "question_sha256": digest(question),
        "program_sha256": digest(world["program"]),
        "renderer_question_sha256": digest(canonical),
        "method": review.get("method"),
        "review_sha256": digest(review),
    }
    if cert != expected:
        raise ValueError("stage2_surface_certificate_input_mismatch")
    if cert["review_contract"] not in (13, 14, 15, 16, REVIEW_CONTRACT):
        raise ValueError("stage2_surface_review_contract_unsupported")
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
        "verified_controlled_renderer", "llm_semantic_screen", "llm_contract_alignment", SCOPED_METHOD,
    ):
        raise ValueError("stage2_surface_review_method_unsupported")
    if question == canonical:
        if review["method"] != "verified_controlled_renderer":
            raise ValueError("canonical_surface_certificate_method_mismatch")
    elif review['method'] == SCOPED_METHOD:
        from .scoped_surface import build_scoped_plan, read_scoped_semantics
        if cert['review_contract'] != REVIEW_CONTRACT:
            raise ValueError('unsupported scoped grammar review contract')
        rendered = rendered or world.get('_rendered')
        if rendered is None or rendered.question != canonical:
            raise ValueError('missing scoped grammar renderer')
        if gold is None:
            gold = compile_gold_solution({**realized, 'question': canonical}, world, rendered, schedule='compact')
        reference = controlled_question_reference(gold, world, rendered)
        if reference is None or reference['question'] != canonical:
            raise ValueError('unverified scoped grammar reference')
        groups = build_scoped_plan(world, canonical, gold, reference)
        readback = read_scoped_semantics(question, groups)
        if (review.get('grammar_sha256') != digest(groups) or readback is None
                or readback['query'] != groups.query
                or review.get('semantic_readback') != readback
                or review.get('query_contract_sha256') != digest(groups.query)):
            raise ValueError('scoped grammar round-trip mismatch')
    elif review["method"] != (
        "llm_semantic_screen" if cert["review_contract"] == 13 else "llm_contract_alignment"
    ):
        raise ValueError("rewritten_surface_requires_graph_semantic_review")
    if review["method"] == "llm_contract_alignment":
        full = review.get("full_graph_review") or {}
        if (review.get("alignment_version") != 1
                or full.get("eligible") is not True
                or full.get("reviewed") is not True
                or full.get("decision") != "eligible"
                or full.get("issues") != []
                or full.get("method") != "llm_semantic_screen"
                or full.get("question_sha256") != digest(question)
                or full.get("program_sha256") != digest(world["program"])):
            raise ValueError("stage2_independent_graph_gate_missing")
    return {
        "eligible": True, "reviewed": True, "decision": "eligible",
        "method": "stage2_verified_programir_certificate",
        "issues": [],
        "question_sha256": digest(question),
        "program_sha256": digest(world["program"]),
        "stage2_method": review["method"],
        "stage2_review_contract": cert["review_contract"],
    }
