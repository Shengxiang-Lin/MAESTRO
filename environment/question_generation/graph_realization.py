"""Optional graph-review research baseline (not the default Stage-2 path).

The active scoped/free pipeline uses certified grammar and blind text-only
readback. This module isolates the historical Author→ProgramIR→Critic repair
baseline so changes to that experiment do not complicate the primary flow.
"""
from __future__ import annotations

from .backend import BackendError
from .prompts import author_prompt, critic_prompt
from .validators import preflight
from .public_semantics import naturalization_metrics

def _critic_review(critic,payload,question,*,max_tries=2):
    last=None
    for _ in range(max_tries):
        try:return critic.review(payload,question,critic_prompt(payload,question))
        except BackendError as exc:last=exc
    raise last or BackendError("critic failed")

def realize_graph_candidate(
    payload, public_payload, author, critic, *, canonical, audits,
    question_max_attempts, use_critic, verify_proposal,
):
    feedback = None
    best = None
    for attempt in range(1,max(1,question_max_attempts)+1):
        entry={"attempt":attempt,"strategy":"public_program_grounded_realization"}
        try:
            response=author.author(
                public_payload,
                author_prompt(public_payload,retry_feedback=feedback),
            )
            question=" ".join(str(response.get("question") or "").split())
            entry["author_response"]=response
        except BackendError as exc:
            entry["author_error"]=str(exc)[:240]
            audits.append(entry);feedback="author_response_failed"
            continue
        local=preflight(payload,question)
        entry["preflight"]=local.to_dict()
        if not local.accepted:
            audits.append(entry);feedback="; ".join(local.issues)
            continue

        # Give every structurally valid candidate a chance with the FULL
        # semantic graph. This prevents false negatives from the smaller
        # public-only Critic's speculative hop/owner objections.
        accepted,issue=verify_proposal(question,entry,"author")
        if accepted is None and use_critic and question!=canonical:
            # Ask the public-only Critic for a targeted wording repair, but
            # never send it private World data or arbitrary graph-review text.
            try:
                critic_response=_critic_review(
                    critic,public_payload,question,max_tries=1,
                )
                entry["semantic_critic_repair"]=critic_response
                repair=" ".join(str(
                    critic_response.get("repair_question") or ""
                ).split())
                if repair and repair!=question:
                    repair_pre=preflight(payload,repair)
                    entry["repair_preflight"]=repair_pre.to_dict()
                    if repair_pre.accepted:
                        accepted,repair_issue=verify_proposal(
                            repair,entry,"repair",
                        )
                        if accepted is None:
                            entry["repair_failure"]=repair_issue
            except BackendError as exc:
                entry["critic_repair_error"]=str(exc)[:200]
        audits.append(entry)
        if accepted is None:
            # Only bounded, public, non-specific diagnostic cues are sent to
            # Author. Never echo teacher-private audit facts into its prompt.
            feedback=(
                "The wording did not pass complete dependency/ownership "
                "verification. Preserve EVERY edge direction, filter owner, "
                "branch join and requested output from the public scaffold. "
                "Do not assume two connected entities share attributes."
            )
            continue

        text,certificate,review=accepted
        metrics=naturalization_metrics(canonical,text)
        entry["naturalization_metrics"]=metrics
        if text!=canonical and metrics["meaningful_rewrite_proxy"]:
            best=(text,certificate,review,False,metrics)
            break
        # A nearly unchanged but correct paraphrase is retained as a SAFE
        # candidate while giving the next attempt a chance to improve style.
        if best is None or text!=canonical:
            best=(text,certificate,review,False,metrics)
        feedback=(
            "The previous wording passed semantic checks but remained too "
            "close to the mechanical canonical template. Use shorter natural "
            "clauses or explicit 'both/either' branches without omitting "
            "any public condition in the scaffold."
        )

    return best, feedback


__all__ = ["realize_graph_candidate"]
