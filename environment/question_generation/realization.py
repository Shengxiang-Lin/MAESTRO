"""One task: checked Scoped reference → author/critic revisions → final judge.

The reference is a PRIVATE generation artifact, never published as a fallback.
Only a genuinely natural final rewrite that passes both editorial reviews and
public-boundary checks enters the accepted questions JSONL.
"""
from __future__ import annotations

from collections.abc import Mapping

from environment.trajectory_simulation.compiler import (
    GoldCompilationError, compile_gold_solution,
)
from environment.trajectory_simulation.eligibility import (
    controlled_question_reference, digest,
)

from .backend import BackendError
from .question_review import (
    ReviewFormatError, author_question, critique_question, judge_question,
    passing_judgment, rejection_reason,
)
from .scoped_surface import build_scoped_plan, render_scoped
from .semantic_checks import (
    semantic_checks, naturalness_gate, explicit_join_contradictions,
    CheckFormatError,
)
from .validators import postflight
from .verification import certify_approved_question


def _oracle(world):
    return (world.get("_display_answer", world.get("answer"))
            if isinstance(world, Mapping) else None)


def _model(value):
    return getattr(value, "backend", value)


def _reject(payload, reason, audit, *, detail=None):
    result = {
        "task_id": payload.get("task_id"),
        "publication_status": "rejected",
        "reason": reason,
        "revision_attempts": sum(
            item.get("stage") == "candidate" for item in audit
        ),
    }
    if detail is not None:
        result["detail"] = str(detail)[:240]
    return False, result, audit


def _realize_one(payload, world, capability, author, critic, *,
                 max_revisions=2):
    """No legacy modes, no grammar-only publication and no fallback.

    max_revisions=N permits one initial draft followed by at most N targeted
    revisions, each based on a real candidate and critic/judge feedback.
    """
    canonical = str(payload.get("draft_question") or "")
    rendered = world.get("_rendered")
    if not canonical or getattr(rendered, "question", None) != canonical:
        return _reject(payload, "rejected_reference",
                       [], detail="frozen_canonical_draft_mismatch")
    audit = []
    try:
        gold = compile_gold_solution(
            {
                "task_id": payload["task_id"],
                "question": canonical,
                "capability_world": capability,
            }, world, rendered, schedule="compact",
        )
        controlled = controlled_question_reference(gold, world, rendered)
        if controlled is None or controlled.get("question") != canonical:
            raise ValueError("unverified_controlled_reference")
        plan = build_scoped_plan(world, canonical, gold, controlled)
        scoped = render_scoped(plan, [0] * len(plan))
        checkpoints = semantic_checks(plan)
        if not checkpoints:
            raise ValueError("missing_public_semantic_checkpoints")
        # No model is needed to select between nearly equivalent templates.
        check = postflight(
            payload, scoped, oracle_answer=_oracle(world),
            check_contract=False,
        )
        if not check.accepted:
            raise ValueError("scoped_reference_invalid: " + ";".join(check.issues))
    except (GoldCompilationError, KeyError, ValueError, TypeError, IndexError) as exc:
        return _reject(payload, "rejected_reference", audit, detail=exc)

    audit.append({
        "stage": "reference",
        "kind": "deterministic_scoped",
        "scoped_question": scoped,
        "scoped_question_sha256": digest(scoped),
        "semantic_checks": checkpoints,
    })

    author_backend = _model(author)
    critic_backend = _model(critic)
    if (not callable(getattr(author_backend, "complete_json", None))
            or not callable(getattr(critic_backend, "complete_json", None))):
        return _reject(payload, "rejected_validation_error",
                       audit, detail="author_and_critic_backends_required")

    previous = None
    feedback = None
    last_failure = "rejected_validation_error"
    last_detail = None

    for attempt in range(max_revisions + 1):
        entry = {
            "stage": "candidate", "attempt": attempt + 1,
            "author_calls": 0, "critic_calls": 0, "final_judge_calls": 0,
        }
        audit.append(entry)
        try:
            entry["author_calls"] = 1
            candidate = author_question(
                author_backend, scoped, previous=previous, feedback=feedback,
            )
        except (BackendError, ReviewFormatError, CheckFormatError, ValueError, TypeError) as exc:
            entry["failure"] = "author_model_error"
            entry["detail"] = str(exc)[:240]
            last_failure, last_detail = "rejected_validation_error", str(exc)
            # A model/protocol failure does not license a surrogate reference.
            continue

        entry["question"] = candidate
        previous = candidate
        if candidate in (canonical, scoped):
            entry["failure"] = "not_naturalized"
            feedback = {
                "verdict": "revise",
                "issues": [{
                    "kind": "style", "severity": "minor",
                    "explanation": "The draft still reproduces the mechanical reference.",
                    "suggestion": "Express the same conditions as a real user question.",
                }],
                "summary": "Keep every necessary clue but change the actual phrasing.",
            }
            last_failure = "rejected_quality"
            continue
        style = naturalness_gate(scoped, candidate)
        entry["naturalness_gate"] = style
        if not style["improved_enough_to_review"]:
            entry["failure"] = "cosmetic_rephrase"
            last_failure, last_detail = "rejected_quality", "cosmetic_rephrase"
            feedback = {
                "verdict": "revise",
                "issues": [{
                    "kind": "naturalness", "severity": "minor",
                    "explanation": "The draft only makes cosmetic word changes.",
                    "suggestion": "Restructure the question substantially while preserving the exact information need; reduce mechanical group definitions.",
                }],
                "summary": "Make a genuine natural-language rewrite, not a near-copy.",
            }
            continue
        contradictions = explicit_join_contradictions(scoped, candidate)
        entry["explicit_join_contradictions"] = contradictions
        if contradictions:
            entry["failure"] = "explicit_set_logic_contradiction"
            last_failure, last_detail = "rejected_semantic", ";".join(contradictions)
            feedback = {
                "verdict": "revise",
                "issues": [{
                    "kind": "set_logic", "severity": "major",
                    "explanation": reason,
                    "suggestion": "Restore the original intersection (ALL) versus union (ANY) meaning for this exact group.",
                } for reason in contradictions],
                "summary": "Do not reverse set operations while rephrasing.",
            }
            continue
        boundary = postflight(
            payload, candidate, oracle_answer=_oracle(world),
            check_contract=False,
        )
        entry["public_boundary"] = boundary.to_dict()
        if not boundary.accepted:
            entry["failure"] = "public_boundary"
            feedback = {
                "verdict": "revise", "issues": [{
                    "kind": "public_boundary", "severity": "major",
                    "explanation": "The draft violates a public question rule: " +
                                   "; ".join(boundary.issues),
                    "suggestion": "Restore public wording and remove any leaked answer or internal identifier.",
                }],
                "summary": "Repair the listed public-boundary problem without changing the reference meaning.",
            }
            last_failure = "rejected_validation_error"
            last_detail = ";".join(boundary.issues)
            continue

        try:
            entry["critic_calls"] = 1
            critique = critique_question(critic_backend, scoped, candidate, checkpoints)
        except (BackendError, ReviewFormatError, ValueError, TypeError) as exc:
            entry["failure"] = "critic_protocol_error"
            entry["detail"] = str(exc)[:240]
            last_failure, last_detail = "rejected_validation_error", str(exc)
            feedback = {
                "verdict": "revise", "issues": [],
                "summary": "Review parsing failed. Preserve the full semantics and return clear JSON.",
            }
            continue

        entry["critic_feedback"] = critique
        if (critique["verdict"] != "pass"
                or any(item["severity"] == "major" for item in critique["issues"])):
            entry["failure"] = "critic_requests_revision"
            feedback = critique
            last_failure = (
                "rejected_uncertain" if critique["verdict"] == "uncertain"
                else "rejected_semantic"
                if any(item["severity"] == "major" for item in critique["issues"])
                else "rejected_quality"
            )
            last_detail = critique["summary"]
            continue

        # The final judge sees the reference, final text and *no previous*
        # review history or previous answers. It is a different prompt/role,
        # not necessarily a different checkpoint.
        try:
            entry["final_judge_calls"] = 1
            final = judge_question(critic_backend, scoped, candidate, checkpoints)
        except (BackendError, ReviewFormatError, ValueError, TypeError) as exc:
            entry["failure"] = "final_judge_protocol_error"
            entry["detail"] = str(exc)[:240]
            last_failure, last_detail = "rejected_validation_error", str(exc)
            continue

        entry["final_judgment"] = final
        if not passing_judgment(final):
            entry["failure"] = "final_judge_rejected"
            last_failure = rejection_reason(final)
            last_detail = final["explanation"]
            feedback = {
                "verdict": "revise",
                "issues": [{
                    "kind": "final_judge", "severity": "major",
                    "explanation": issue,
                    "suggestion": "Repair this specific issue while retaining correct phrasing.",
                } for issue in final["issues"][:5]],
                "summary": final["explanation"],
            }
            continue

        try:
            certificate, review = certify_approved_question(
                payload["task_id"], world, canonical, candidate, scoped, final,
                critic_verdict="pass", revision_rounds=attempt,
                semantic_checklist=checkpoints,
            )
        except (ValueError, TypeError, KeyError) as exc:
            entry["failure"] = "certificate_error"
            last_failure, last_detail = "rejected_validation_error", str(exc)
            continue

        row = {
            "task_id": payload["task_id"],
            "question": candidate,
            "draft_question": payload.get("draft_question"),
            "task_design": payload.get("task_design", {}),
            "resources": payload.get("resources", ()),
            "available_agents": payload.get("available_agents", ()),
            "capability_world": capability,
            "surface_certificate": certificate,
            "publication_status": "naturalized_certified",
            "authoring": {
                "strategy": "iterative_author_critic_final_judge",
                "revision_rounds": attempt,
                "surface_review": review,
            },
        }
        entry["accepted"] = True
        return True, row, audit

    return _reject(payload, last_failure, audit, detail=last_detail)


__all__ = ["_realize_one"]
