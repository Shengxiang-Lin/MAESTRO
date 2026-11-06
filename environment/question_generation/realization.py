"""Per-task question realization; the pipeline module owns only streaming I/O.

The active scoped/free paths keep gold facts private and make Stage 3
consume exactly the same versioned publication certificate contract.
"""
from __future__ import annotations

from collections.abc import Mapping

from .backend import (
    BackendError, LLMQuestionAuthor, LLMSemanticCritic,
)
from .validators import postflight

def _oracle(world):return world.get("_display_answer",world.get("answer")) if isinstance(world,Mapping) else None
def _realize_one(payload,world,capability,author,critic,*,question_max_attempts,use_critic,surface_mode="graph"):
    """Naturalize from a PUBLIC semantic scaffold, then certify against the graph.

    The ordinary Stage-2 Critic is repair-only: it cannot veto a paraphrase
    the authoritative ProgramIR reviewer certifies, or approve one it rejects.
    """
    from .verification import certify_surface
    from .public_semantics import public_semantic_scaffold, public_question_contract, naturalization_metrics
    from environment.trajectory_simulation.compiler import GoldCompilationError, compile_gold_solution

    canonical=str(payload.get("draft_question") or "")
    rendered=world.get("_rendered")
    if not canonical or getattr(rendered,"question",None)!=canonical:
        return False,{"task_id":payload.get("task_id"),
                      "reason":"frozen_canonical_draft_mismatch"},[]
    try:
        scaffold=public_semantic_scaffold(world,canonical)
        gold=compile_gold_solution(
            {"task_id":payload["task_id"],"question":canonical,
             "capability_world":capability},world,rendered,schedule="compact",
        )
        query_contract=public_question_contract(world,canonical,gold)
    except (KeyError,TypeError,ValueError,GoldCompilationError) as exc:
        # No partial or private graph information is passed to either model.
        return False,{"task_id":payload.get("task_id"),
                      "reason":"public_scaffold_unavailable",
                      "error":str(exc)[:200]},[]

    public_payload={**payload,"public_semantic_scaffold":scaffold,
                    "question_contract":query_contract}
    audits=[];feedback=None;best=None

    def verify_proposal(text,entry,label):
        local=postflight(payload,text,oracle_answer=_oracle(world))
        entry[label+"_postflight"]=local.to_dict()
        if not local.accepted:
            return None,"; ".join(local.issues)
        try:
            certificate,review=certify_surface(
                payload,world,capability,text,critic,
            )
        except (GoldCompilationError,ValueError,KeyError,TypeError,BackendError) as exc:
            entry[label+"_graph_error"]=str(exc)[:240]
            return None,"graph_verification_unavailable"
        entry[label+"_programir_review"]=review
        if certificate is None:
            return None,"graph_review:"+str(
                review.get("review_failure_kind") or
                review.get("decision") or "not_confirmed"
            )
        return (text,certificate,review),None

    scoped = surface_mode in ("scoped", "free") and isinstance(author, LLMQuestionAuthor)
    if scoped:
        from .scoped_surface import build_scoped_plan, choose_scoped, render_scoped
        from .verification import certify_scoped_surface, controlled_question_reference
        entry = {"stage": "scoped_surface", "scoped_style_model_calls": 0}
        audits.append(entry)
        try:
            reference = controlled_question_reference(gold, world, rendered)
            if reference is None or reference["question"] != canonical:
                raise ValueError("unverified scoped grammar reference")
            groups = build_scoped_plan(world, canonical, gold, reference)
            entry["scoped_style_model_calls"] = 1
            try:
                question = render_scoped(groups, choose_scoped(author, groups))
                entry["scoped_style"] = "llm_selected"
            except (BackendError, ValueError) as exc:
                entry["scoped_style_error"] = str(exc)[:240]
                question = render_scoped(groups, [0] * len(groups))
                entry["scoped_style"] = "deterministic_fallback"
            # The grammar protects typed slots. Keep the ordinary public-boundary checks.
            local = postflight(payload, question, oracle_answer=_oracle(world), check_contract=False)
            entry["scoped_postflight"] = local.to_dict()
            if local.accepted:
                cert, review = certify_scoped_surface(payload["task_id"], world, canonical, question, groups)
                metrics = naturalization_metrics(canonical, question)
                entry["scoped_review"] = review
                best = (question, cert, review, False, metrics)
            else:
                feedback = "; ".join(local.issues)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            entry["scoped_unsupported"] = str(exc)[:240]
            feedback = "scoped_grammar_unavailable"
    # Stage 2 experimental free-language path: start from the already checked
    # Scoped sentence graph, then let Author rewrite prose without slot or
    # template constraints. Crucially, the blind reader sees ONLY the new
    # question, while reference semantics stay inside deterministic validation.
    free_verified = False
    if surface_mode == "free" and best is not None:
        from dataclasses import replace
        from .free_surface import assess_free_style, call_free_rewriter
        from .semantic_readback import read_free_question
        from .verification import certify_free_surface
        reader = (
            replace(
                critic.backend, temperature=0.0,
                max_tokens=max(4096, critic.backend.max_tokens),
            ) if isinstance(critic, LLMSemanticCritic) else None
        )
        scoped_question = best[0]
        guidance = None
        for index in range(1, max(1, question_max_attempts) + 1):
            entry = {
                "stage": "free_rewrite",
                "attempt": index,
                "free_author_calls": 0,
                "blind_readback_calls": 0,
            }
            audits.append(entry)
            if reader is None:
                entry["failure"] = "independent_reader_not_configured"
                break
            try:
                entry["free_author_calls"] = 1
                candidate = call_free_rewriter(
                    author, scoped_question, feedback=guidance,
                )
            except BackendError as exc:
                entry["failure"] = "free_author_error: " + str(exc)[:180]
                guidance = "Return a single natural-language question as valid JSON."
                continue
            entry["candidate"] = candidate
            boundary = postflight(
                payload, candidate, oracle_answer=_oracle(world),
                check_contract=False,
            )
            entry["postflight"] = boundary.to_dict()
            if not boundary.accepted:
                entry["failure"] = "free_rewrite_public_boundary"
                guidance = (
                    "Keep all public identifying clues and do not reveal the "
                    "answer or imply any provided evidence that was not supplied."
                )
                continue
            style = assess_free_style(scoped_question, candidate)
            entry["style"] = style.to_dict()
            if not style.improved:
                entry["failure"] = style.reason
                guidance = (
                    "Rewrite more substantially: reduce procedural group "
                    "labels and simplify sentence structure without losing "
                    "any required relationship or filter."
                )
                continue
            # No reference contract, gold, program or earlier parser output
            # is sent to the parsing model; verification remains fail-closed.
            result = read_free_question(candidate, groups, reader)
            entry["blind_readback_calls"] = result.calls
            entry["blind_readback_summary"] = {
                "eligible": result.eligible,
                "reason": result.reason,
                "passes": list(result.audits),
            }
            if not result.eligible:
                entry["failure"] = "independent_semantic_readback_failed"
                guidance = (
                    "Some semantic dependency was ambiguous or changed. "
                    "Preserve all relation directions, filters and owners "
                    "while writing a natural question."
                )
                continue
            try:
                certificate, review = certify_free_surface(
                    payload["task_id"], world, canonical, candidate,
                    groups, result,
                )
            except (ValueError, TypeError, KeyError) as exc:
                entry["failure"] = "free_certificate_failed:" + str(exc)[:160]
                continue
            entry["free_semantic_certificate_method"] = review["method"]
            best = (
                candidate, certificate, review, False,
                naturalization_metrics(canonical, candidate),
            )
            free_verified = True
            break

    if not scoped:
        from .graph_realization import realize_graph_candidate
        best, feedback = realize_graph_candidate(
            payload, public_payload, author, critic,
            canonical=canonical, audits=audits,
            question_max_attempts=question_max_attempts,
            use_critic=use_critic, verify_proposal=verify_proposal,
        )

    if best is None:
        canonical_entry={"stage":"certified_renderer_fallback",
                         "reason":feedback}
        best,issue=verify_proposal(canonical,canonical_entry,"canonical")
        audits.append(canonical_entry)
        if best is None:
            return False,{
                "task_id":payload.get("task_id"),
                "reason":"question_realization_failed",
                "last_feedback":issue,"audit":audits,
            },audits
        text,certificate,review=best
        best=(text,certificate,review,True,
              naturalization_metrics(canonical,text))

    question,certificate,graph_review,fallback,metrics=best
    realized={
        "task_id":payload["task_id"],"question":question,
        "draft_question":payload.get("draft_question"),
        "task_design":payload.get("task_design",{}),
        "resources":payload.get("resources",()),
        "available_agents":payload.get("available_agents",()),
        "capability_world":capability,
        "surface_certificate":certificate,
        "publication_status":(
            "naturalized_certified" if question!=canonical else "canonical_only"
        ),
        "authoring":{
            "attempts":len([a for a in audits if "attempt" in a])+sum(a.get("scoped_style_model_calls",0) for a in audits),
            "semantic_critic_enabled":use_critic and not scoped,
            "blind_readback_enabled":surface_mode=="free",
            "free_rewrite_verified":free_verified,
            "strategy":(
                "independent_free_semantic_readback" if free_verified
                else "scoped_grammar_realization" if scoped
                else "public_question_contract_realization_v2"
            ),
            "surface_review":graph_review,
            "surface_fallback":fallback,
            "naturalization_metrics":metrics,
            "audit":audits,
        },
    }
    return True,realized,audits

