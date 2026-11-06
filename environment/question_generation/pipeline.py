"""Frozen-boundary LLM question generation with deterministic and semantic review."""
from __future__ import annotations
import argparse,json
from concurrent.futures import FIRST_COMPLETED,ThreadPoolExecutor,wait
from pathlib import Path
from typing import Any,Mapping
from tqdm.auto import tqdm
from environment.config import EnvironmentConfigError,parse_with_config,resolve_model
from .backend import BackendError,LLMQuestionAuthor,LLMSemanticCritic,OpenAICompatibleBackend,PassthroughAuthorBackend,PassthroughCriticBackend
from .boundary import BoundaryError,validate_frozen_inputs
from .prompts import author_prompt,critic_prompt
from .validators import postflight,preflight
from .streaming import (QuestionCheckpointError,QuestionProgress,
    question_output_lock,signature_for_question_run)

def _rows(path):
    with Path(path).open("r",encoding="utf-8") as handle:return [json.loads(line) for line in handle if line.strip()]
def _world_by_task(path,seed=17):
    from environment.task_construction.rendering.pipeline import render_task_record
    result={}
    for index,row in enumerate(_rows(path)):
        task=render_task_record(row,seed=seed+index)
        if task.task_id in result:raise BoundaryError(f"duplicate world task_id: {task.task_id}")
        result[task.task_id]={**row,"_display_answer":task.answer,"_rendered":task}
    return result
def _capability_by_task(path):return {str(row.get("task_id")):row for row in _rows(path) if row.get("task_id")}
def _backends(args):
    author_spec=resolve_model(args,"question_author")
    critic_spec=resolve_model(args,"question_critic") if not args.no_semantic_critic else {"provider":"passthrough","backend":"passthrough","model":None,"base_url":None,"api_key":None,"json_mode":False}
    # Keep resolved metadata free from API keys.
    args._resolved_models={
        "question_author":{k:v for k,v in author_spec.items() if k!="api_key"},
        "question_critic":{k:v for k,v in critic_spec.items() if k!="api_key"},
    }
    if author_spec["provider"]=="passthrough":
        author=PassthroughAuthorBackend()
    else:
        author=LLMQuestionAuthor(OpenAICompatibleBackend(
            author_spec["base_url"],author_spec["model"],author_spec["api_key"],
            args.question_temperature,args.max_tokens,args.timeout,
            author_spec["json_mode"] and not args.no_json_mode))
    if critic_spec["provider"]=="passthrough":
        critic=PassthroughCriticBackend()
    else:
        critic=LLMSemanticCritic(OpenAICompatibleBackend(
            critic_spec["base_url"],critic_spec["model"],critic_spec["api_key"],
            args.critic_temperature,args.critic_max_tokens,args.timeout,
            critic_spec["json_mode"] and not args.no_json_mode))
    return author,critic
def _oracle(world):return world.get("_display_answer",world.get("answer")) if isinstance(world,Mapping) else None
def _critic_review(critic,payload,question,*,max_tries=2):
    last=None
    for _ in range(max_tries):
        try:return critic.review(payload,question,critic_prompt(payload,question))
        except BackendError as exc:last=exc
    raise last or BackendError("critic failed")
def _realize_one(payload,world,capability,author,critic,*,question_max_attempts,use_critic):
    """Naturalize from a PUBLIC semantic scaffold, then certify against the graph.

    The ordinary Stage-2 Critic is repair-only: it cannot veto a paraphrase
    the authoritative ProgramIR reviewer certifies, or approve one it rejects.
    """
    from .verification import certify_surface
    from .public_semantics import public_semantic_scaffold, naturalization_metrics
    from environment.trajectory_simulation.compiler import GoldCompilationError

    canonical=str(payload.get("draft_question") or "")
    rendered=world.get("_rendered")
    if not canonical or getattr(rendered,"question",None)!=canonical:
        return False,{"task_id":payload.get("task_id"),
                      "reason":"frozen_canonical_draft_mismatch"},[]
    try:
        scaffold=public_semantic_scaffold(world,canonical)
    except (KeyError,TypeError,ValueError) as exc:
        # No partial or private graph information is passed to either model.
        return False,{"task_id":payload.get("task_id"),
                      "reason":"public_scaffold_unavailable",
                      "error":str(exc)[:200]},[]

    public_payload={**payload,"public_semantic_scaffold":scaffold}
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
        "authoring":{
            "attempts":len([a for a in audits if "attempt" in a]),
            "semantic_critic_enabled":use_critic,
            "strategy":"public_program_grounded_realization_v1",
            "surface_review":graph_review,
            "surface_fallback":fallback,
            "naturalization_metrics":metrics,
            "audit":audits,
        },
    }
    return True,realized,audits

def _safe_future_result(future,index,authoring):
    """Keep failures isolated to the corresponding frozen input task."""
    try:
        return future.result()
    except Exception as exc:
        return False,{
            "task_id":authoring[index].get("task_id"),
            "reason":"unhandled_exception",
            "error":str(exc),
        },[]


def _stream_parallel(authoring,worlds,capabilities,author,critic,args,progress):
    """Bound in-flight requests and append completed results in input order.

    The Author/Critic calls within an individual task remain sequential.
    Scheduling and completion may be out of order; writing never is.
    """
    workers=args.question_workers
    total=len(authoring)
    next_submit=progress.completed
    pending={}
    completed_this_run=0
    bar=tqdm(total=total,initial=progress.completed,desc="Question generation",
             unit="task",disable=args.no_progress,dynamic_ncols=True)
    try:
        with ThreadPoolExecutor(max_workers=workers,thread_name_prefix="question") as pool:
            active={}
            def submit_ready():
                nonlocal next_submit
                while len(active)<2*workers and next_submit<total:
                    index=next_submit
                    payload=authoring[index]
                    future=pool.submit(
                        _realize_one,payload,
                        worlds.get(str(payload.get("task_id")),{}),
                        capabilities.get(str(payload.get("task_id")),{}),
                        author,critic,
                        question_max_attempts=args.question_max_attempts,
                        use_critic=not args.no_semantic_critic)
                    active[future]=index
                    next_submit+=1

            submit_ready()
            while active:
                ready,_=wait(tuple(active),return_when=FIRST_COMPLETED)
                for future in ready:
                    index=active.pop(future)
                    pending[index]=_safe_future_result(future,index,authoring)
                    completed_this_run+=1
                    bar.update(1)
                # Write only the contiguous completed prefix. A slow early
                # task cannot reorder the questions or their critic audits.
                while progress.completed in pending:
                    index=progress.completed
                    ok,row,audit=pending.pop(index)
                    progress.append(index,ok,row,audit)
                bar.set_postfix({
                    "written":progress.completed,
                    "accepted":progress.accepted,
                    "workers":workers,
                })
                submit_ready()
    finally:
        bar.close()
    return completed_this_run


def run(args):
    try:
        frozen=validate_frozen_inputs(
            args.manifest,args.authoring_inputs,args.capabilities,worlds_path=args.worlds)
    except (BoundaryError,FileNotFoundError,json.JSONDecodeError) as exc:
        raise SystemExit(f"Frozen-input validation failed: {exc}") from exc
    authoring=_rows(args.authoring_inputs)
    if args.max_records is not None:
        authoring=authoring[:args.max_records]
    worlds=_world_by_task(args.worlds,seed=args.seed)
    capabilities=_capability_by_task(args.capabilities)
    if any(str(row.get("task_id")) not in worlds for row in authoring):
        raise SystemExit("authoring tasks are missing corresponding executable worlds")
    if args.flush_every<1:
        raise SystemExit("--flush-every must be >= 1")
    if args.question_workers<1:
        raise SystemExit("--question-workers must be >= 1")
    author,critic=_backends(args)
    output_dir=Path(args.output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    task_ids=[str(row.get("task_id","")) for row in authoring]
    source_files=[
        Path(args.manifest),Path(args.authoring_inputs),
        Path(args.worlds),Path(args.capabilities),
        Path(__file__),Path(__file__).with_name("streaming.py"),
        Path(__file__).with_name("backend.py"),
        Path(__file__).with_name("prompts.py"),
        Path(__file__).with_name("validators.py"),
        Path(__file__).with_name("boundary.py"),
        Path(__file__).with_name("verification.py"),
        Path(__file__).with_name("public_semantics.py"),
        Path("environment/trajectory_simulation/eligibility.py"),
        Path("environment/trajectory_simulation/compiler.py"),
        Path("environment/trajectory_simulation/skeleton.py"),
        Path("environment/trajectory_simulation/backend.py"),
    ]
    signature=signature_for_question_run(
        frozen,args,task_ids,source_files=source_files)
    with question_output_lock(output_dir):
        with QuestionProgress(
            output_dir,signature,task_ids,
            flush_every=args.flush_every,restart=args.restart_questions
        ) as progress:
            resumed_count=progress.completed
            _stream_parallel(authoring,worlds,capabilities,author,critic,args,progress)
            accepted=progress.accepted
            rejected=progress.rejected
            if progress.completed!=len(authoring):
                raise QuestionCheckpointError(
                    f"question stream incomplete: {progress.completed}/{len(authoring)}")
        # Acceptance now includes safely certified renderer fallbacks. Report
        # naturalized and fallback counts separately to avoid inflated claims.
        certified_paraphrases=canonical_fallbacks=canonical_selected=0
        meaningful_paraphrases=0
        total_template_reduction=0
        with (output_dir/"realized_capability_worlds.jsonl").open(
            "r",encoding="utf-8",
        ) as finalized:
            for line in finalized:
                if not line.strip():
                    continue
                row=json.loads(line)
                author_metadata=row.get("authoring") or {}
                gain=author_metadata.get("naturalization_metrics") or {}
                if gain.get("meaningful_rewrite_proxy") is True:
                    meaningful_paraphrases+=1
                    total_template_reduction+=max(
                        0, int(gain.get("canonical_template_phrases",0))
                         - int(gain.get("surface_template_phrases",0)),
                    )
                if author_metadata.get("surface_fallback") is True:
                    canonical_fallbacks+=1
                elif ((row.get("surface_certificate") or {}).get("method")
                      =="llm_semantic_screen"):
                    certified_paraphrases+=1
                else:
                    canonical_selected+=1
        summary={
            "pipeline":"public_program_grounded_author_then_full_programir_certificate_with_repair",
            "frozen_batch_validation":frozen,
            "provider":args._resolved_models["question_author"]["backend"],
            "model":args._resolved_models["question_author"]["model"],
            "critic_model":args._resolved_models["question_critic"]["model"] if not args.no_semantic_critic else None,
            "model_roles":args._resolved_models,
            "requested":len(authoring),
            "accepted":accepted,
            "rejected":rejected,
            "acceptance_rate":accepted/len(authoring) if authoring else 0.,
            "semantic_critic":{"enabled":not args.no_semantic_critic},
            "surface_certification":{
                "contract":"maestro.stage2_surface_cert.v1",
                "certified_paraphrases":certified_paraphrases,
                "meaningful_certified_paraphrases":meaningful_paraphrases,
                "meaningful_rewrite_rate":meaningful_paraphrases/len(authoring)
                    if authoring else 0.0,
                "template_phrases_removed_from_meaningful":total_template_reduction,
                "canonical_fallbacks":canonical_fallbacks,
                "canonical_selected":canonical_selected,
                "published_certified":accepted,
            },
            "streaming":{
                "enabled":True,
                "ordered_writes":True,
                "flush_every":args.flush_every,
                "question_workers":args.question_workers,
                "resumed_records":resumed_count,
                "checkpoint":str(output_dir/"question_progress_state.json"),
            },
            "artifacts":{
                "realized":str(output_dir/"realized_capability_worlds.jsonl"),
                "rejected":str(output_dir/"question_rejected.jsonl"),
                "critic_audit":str(output_dir/"question_critic_audit.jsonl"),
                "summary":str(output_dir/"question_generation_summary.json"),
            },
        }
        # Write completion summary only when all ordered records are committed.
        summary_path=output_dir/"question_generation_summary.json"
        summary_tmp=summary_path.with_suffix(".json.tmp")
        summary_tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
        summary_tmp.replace(summary_path)
    return summary

def build_parser():
    p=argparse.ArgumentParser(description="Generate natural questions over a frozen MAESTRO task batch");p.add_argument("--manifest");p.add_argument("--authoring-inputs");p.add_argument("--worlds");p.add_argument("--capabilities");p.add_argument("--output-dir");p.add_argument("--max-records",type=int);p.add_argument("--seed",type=int,default=17);p.add_argument("--provider",choices=("passthrough","openai-compatible"),default="passthrough");p.add_argument("--base-url");p.add_argument("--model");p.add_argument("--critic-model");p.add_argument("--api-key-env",default="DASHSCOPE_API_KEY");p.add_argument("--question-temperature",type=float,default=.4);p.add_argument("--critic-temperature",type=float,default=0.);p.add_argument("--max-tokens",type=int,default=2048);p.add_argument("--critic-max-tokens",type=int,default=768);p.add_argument("--timeout",type=float,default=180.);p.add_argument("--question-max-attempts",type=int,default=2);p.add_argument("--question-workers",type=int,default=4);p.add_argument("--flush-every",type=int,default=5);p.add_argument("--restart-questions",action="store_true");p.add_argument("--no-semantic-critic",action="store_true");p.add_argument("--no-json-mode",action="store_true");p.add_argument("--no-progress",action="store_true");return p
def main():
    try:
        args=parse_with_config(build_parser(),"question_generation")
    except EnvironmentConfigError as exc:
        raise SystemExit(f"Environment configuration error: {exc}") from exc
    missing=[key for key in ("manifest","authoring_inputs","worlds","capabilities","output_dir") if not getattr(args,key,None)]
    if missing:raise SystemExit("Missing question-generation inputs (CLI or YAML): "+", ".join(missing))
    if args.question_workers<1:raise SystemExit("--question-workers must be >= 1")
    if args.flush_every<1:raise SystemExit("--flush-every must be >= 1")
    try:
        result=run(args)
    except EnvironmentConfigError as exc:
        raise SystemExit(f"Environment model error: {exc}") from exc
    except QuestionCheckpointError as exc:
        raise SystemExit(f"Question-generation checkpoint error: {exc}") from exc
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
