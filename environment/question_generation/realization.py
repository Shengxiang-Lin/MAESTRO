"""Gold DAG → public delegation blueprint → benchmark Author → Critic → repair.

The immutable graph controls future Sub-Agent tasks; only reusable PUBLIC
task goals are authored before the natural user question. Stage 3 uses the
same blueprint without leaking intermediate or final answers to the models.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from environment.trajectory_simulation.compiler import (
    GoldCompilationError, compile_gold_solution,
)
from environment.trajectory_simulation.eligibility import (
    controlled_question_reference, digest,
)

from environment.delegation_blueprint import compile_delegation_outline
from .goal_review import validate_goal_record
from .backend import BackendError
from .question_review import (
    ReviewFormatError, author_question, critique_question, passing_judgment,
)
from .scoped_surface import build_scoped_plan, render_scoped
from .semantic_checks import (
    explicit_join_contradictions, relation_semantic_contradictions,
    missing_typed_literals,
)
from .validators import postflight
from .verification import certify_approved_question


def _oracle(world):
    return (world.get("_display_answer", world.get("answer"))
            if isinstance(world, Mapping) else None)


def _model(value):
    return getattr(value, "backend", value)


def _reject(payload, reason, audit, *, detail=None):
    row = {
        "task_id": payload.get("task_id"),
        "publication_status": "rejected",
        "reason": reason,
        "revision_attempts": sum(
            a.get("stage") == "candidate" for a in audit
        ),
    }
    if detail is not None:
        row["detail"] = str(detail)[:240]
    return False, row, audit


def _meaningful_edit(reference: str, candidate: str) -> bool:
    """Require a real lexical edit, NOT a high edit-distance threshold.

    A modest but clear multi-sentence rewrite is permitted. Cosmetic
    changes in punctuation/spacing/case alone do not qualify.
    """
    norm = lambda s: re.sub(r"[^a-z0-9]+", "", s.casefold())
    return bool(candidate.strip() and norm(reference) != norm(candidate))


def _feedback(problem, fix, *, reference_excerpt="", candidate_excerpt=""):
    return {
        "verdict": "revise", "naturalness": "acceptable",
        "summary": problem,
        "differences": [{
            "reference_excerpt": reference_excerpt,
            "candidate_excerpt": candidate_excerpt,
            "problem": problem,
            "minimal_fix": fix,
        }],
    }


def _review_attempt(backend, reference, candidate, entry, *, blueprint=None):
    """Only one schema-repair retry against the SAME draft."""
    for use_schema in (True, False):
        entry["critic_calls"] += 1
        try:
            return critique_question(
                backend, reference, candidate, use_schema=use_schema,
                blueprint=blueprint,
            )
        except (BackendError, ReviewFormatError, TypeError, ValueError) as exc:
            entry.setdefault("critic_protocol_failures", []).append({
                "schema_mode": use_schema,
                "reason": str(exc)[:200],
            })
    return None


def _realize_one(payload, world, capability, author, critic, *,
                 max_revisions=2, goal_record=None):
    """Generate up to 1+max_revisions drafts; Critic returns local patches."""
    canonical = str(payload.get("draft_question") or "")
    rendered = world.get("_rendered")
    if not canonical or getattr(rendered, "question", None) != canonical:
        return _reject(
            payload, "rejected_reference", [],
            detail="frozen_canonical_draft_mismatch",
        )
    audit = []
    try:
        gold = compile_gold_solution({
            "task_id": payload["task_id"],
            "question": canonical,
            "capability_world": capability,
        }, world, rendered, schedule="compact")
        controlled = controlled_question_reference(gold, world, rendered)
        if controlled is None or controlled.get("question") != canonical:
            raise ValueError("unverified_controlled_reference")
        plan = build_scoped_plan(world, canonical, gold, controlled)
        scoped = render_scoped(plan, [0] * len(plan))
        reference_check = postflight(
            payload, scoped, oracle_answer=_oracle(world),
            check_contract=False,
        )
        if not reference_check.accepted:
            raise ValueError(
                "scoped_reference_invalid: " + ";".join(reference_check.issues)
            )
    except (GoldCompilationError, KeyError, ValueError, TypeError, IndexError) as exc:
        return _reject(payload, "rejected_reference", audit, detail=exc)

    audit.append({
        "stage": "reference",
        "kind": "deterministic_scoped",
        "scoped_question": scoped,
        "scoped_question_sha256": digest(scoped),
    })
    author_backend, critic_backend = _model(author), _model(critic)
    if (not callable(getattr(author_backend, "complete_json", None))
            or not callable(getattr(critic_backend, "complete_json", None))):
        return _reject(
            payload, "rejected_validation_error", audit,
            detail="author_and_critic_backends_required",
        )

    # The Gold fragment graph exists before language generation. Its public
    # projection is the *one shared* planning intent used by Question,
    # Sub-Agent reporting and later Main delegation. Gold answers and witness
    # data are excluded from the LLM input.
    try:
        outline = compile_delegation_outline(
            gold, world, canonical, plan, payload.get("resources", ()),
        )
    except (KeyError, ValueError, TypeError) as exc:
        return _reject(payload, "rejected_reference", audit, detail=exc)
    # Stage 2B is strictly downstream of Stage 2A. Neither Goal writing
    # nor local Goal review can occur here. Direct users of this function
    # cannot silently bypass Goal certification.
    try:
        if goal_record is None:
            raise ValueError("missing_stage2a_certified_blueprint")
        validate_goal_record(
            goal_record, payload, world, canonical, outline,
            oracle_answer=_oracle(world),
        )
        blueprint = goal_record["delegation_blueprint"]
    except (ValueError, KeyError, TypeError) as exc:
        return _reject(payload, "rejected_goal_certificate", audit, detail=exc)
    audit.append({
        "stage": "certified_blueprint_reused", "author_calls": 0,
        "critic_calls": 0,
        "goal_certificate_sha256": digest(goal_record["goal_certificate"]),
    })

    previous, feedback = None, None
    last_reason, last_detail = "rejected_quality", "not_naturalized"

    for attempt in range(max_revisions + 1):
        entry = {
            "stage": "candidate",
            "attempt": attempt + 1,
            "author_calls": 1,
            "critic_calls": 0,
        }
        audit.append(entry)
        try:
            candidate = author_question(
                author_backend, scoped, previous=previous, feedback=feedback,
                blueprint=blueprint,
            )
        except (BackendError, ReviewFormatError, TypeError, ValueError) as exc:
            entry["failure"] = "author_model_error"
            last_reason, last_detail = "rejected_validation_error", str(exc)
            continue

        entry["question"] = candidate
        previous = candidate
        if not _meaningful_edit(scoped, candidate) or not _meaningful_edit(canonical, candidate):
            entry["failure"] = "not_naturalized"
            feedback = _feedback(
                "The draft is essentially unchanged.",
                "Make a user-facing lexical rewrite, not punctuation-only changes; "
                "keep the original condition logic and entity scopes.",
            )
            last_reason, last_detail = "rejected_quality", "unchanged_question"
            continue

        # A literal operator-list reproduction is not a benchmark-style user
        # question. This is a narrow surface check, not semantic parsing.
        if re.search(
            r"\b(?:let the (?:first|second|third|fourth|fifth) group consist "
            r"of|define the (?:first|second|third|fourth|fifth) group as)\b",
            candidate, re.I,
        ):
            entry["failure"] = "mechanical_group_template"
            feedback = _feedback(
                "The question still reads like a compiler's group definition.",
                "Rewrite it as a coherent information-seeking request using "
                "the same constraints; keep all branches and entity owners.",
            )
            last_reason, last_detail = "rejected_quality", "mechanical_group_template"
            continue

        # These guards are intentionally few and *high precision*. They
        # catch concrete omissions/explicit substitutions, not arbitrary
        # paraphrase syntax. Run before Critic to save model calls.
        missing = missing_typed_literals(plan, candidate)
        joins = explicit_join_contradictions(scoped, candidate)
        relations = relation_semantic_contradictions(scoped, candidate)
        entry["deterministic_guards"] = {
            "missing_typed_literals": missing,
            "explicit_join_contradictions": joins,
            "relation_semantic_contradictions": relations,
        }
        if missing or joins or relations:
            issue = (
                f"Missing required public value(s): {', '.join(missing[:5])}"
                if missing else
                f"Explicit group operation changed: {', '.join(joins[:3])}"
                if joins else
                f"Explicit relation meaning changed: {', '.join(relations[:3])}"
            )
            fix = (
                "Restore the listed exact values in their original entity "
                "scope; do not change any other clause."
                if missing else
                "Correct only the changed AND/OR or relation wording so the "
                "same entities qualify."
            )
            entry["failure"] = "deterministic_semantic_guard"
            feedback = _feedback(issue, fix)
            last_reason, last_detail = "rejected_semantic", issue
            continue

        boundary = postflight(
            payload, candidate, oracle_answer=_oracle(world),
            check_contract=False,
        )
        entry["public_boundary"] = boundary.to_dict()
        if not boundary.accepted:
            issue = "; ".join(boundary.issues)
            entry["failure"] = "public_boundary"
            feedback = _feedback(
                "Public question boundary failure: " + issue,
                "Fix only the reported public-boundary issue; preserve meaning.",
            )
            last_reason, last_detail = "rejected_validation_error", issue
            continue

        critique = _review_attempt(
            critic_backend, scoped, candidate, entry, blueprint=blueprint,
        )
        if critique is None:
            entry["failure"] = "critic_protocol_error"
            return _reject(
                payload, "rejected_validation_error", audit,
                detail="critic_exhausted_same_draft_format_retry",
            )

        entry["critic_feedback"] = critique
        if critique["verdict"] == "uncertain":
            entry["failure"] = "critic_uncertain"
            return _reject(
                payload, "rejected_uncertain", audit,
                detail=critique["summary"],
            )
        if not passing_judgment(critique):
            entry["failure"] = "critic_requests_revision"
            feedback = critique
            last_reason = (
                "rejected_quality" if critique["naturalness"] == "needs_polish"
                and not critique["differences"] else "rejected_semantic"
            )
            last_detail = critique["summary"]
            continue

        try:
            certificate, review = certify_approved_question(
                payload["task_id"], world, canonical, candidate, scoped,
                critique, revision_rounds=attempt, plan=plan,
                blueprint=blueprint, gold=gold,
                resources=payload.get("resources", ()),
                goal_record=goal_record,
            )
        except (ValueError, TypeError, KeyError) as exc:
            entry["failure"] = "certificate_error"
            last_reason, last_detail = "rejected_validation_error", str(exc)
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
            "delegation_blueprint": blueprint,
            "goal_certificate": goal_record["goal_certificate"],
            "goal_reviews": goal_record["goal_reviews"],
            "publication_status": "naturalized_certified",
            "authoring": {
                "strategy": "certified_goal_blueprint_question_author_critic",
                "revision_rounds": attempt,
                "surface_review": review,
            },
        }
        entry["accepted"] = True
        return True, row, audit

    return _reject(payload, last_reason, audit, detail=last_detail)


__all__ = ["_realize_one", "_meaningful_edit"]
