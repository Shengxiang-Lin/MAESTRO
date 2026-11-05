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
        result[task.task_id]={**row,"_display_answer":task.answer}
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
    feedback=None;audits=[]
    for attempt in range(1,max(1,question_max_attempts)+1):
        try:response=author.author(payload,author_prompt(payload,retry_feedback=feedback))
        except BackendError as exc:
            feedback=f"author_backend_error: {exc}";audits.append({"attempt":attempt,"author_error":str(exc)});continue
        question=" ".join(str(response.get("question") or "").split());pre=preflight(payload,question);audit={"attempt":attempt,"author_response":response,"preflight":pre.to_dict()}
        if not pre.accepted:
            feedback="; ".join(pre.issues);audits.append(audit);continue
        candidate=question
        if use_critic:
            try:review=_critic_review(critic,payload,candidate)
            except BackendError as exc:
                audit["critic_error"]=str(exc);audits.append(audit);feedback=f"semantic_critic_error: {exc}";continue
            audit["semantic_critic"]=review
            if (review.get("accept") is not True
                    or str(review.get("semantic_fidelity","")).lower()!="pass"
                    or review.get("issues") != []):
                repair=" ".join(str(review.get("repair_question") or "").split())
                if repair:
                    repair_pre=preflight(payload,repair);audit["repair_preflight"]=repair_pre.to_dict()
                    if repair_pre.accepted:
                        try:repair_review=_critic_review(critic,payload,repair,max_tries=1)
                        except BackendError as exc:
                            audit["repair_critic_error"]=str(exc);audits.append(audit);feedback="repair critic failed";continue
                        audit["repair_semantic_critic"]=repair_review
                        if (repair_review.get("accept") is not True
                                or str(repair_review.get("semantic_fidelity","")).lower()!="pass"
                                or repair_review.get("issues") != []):
                            audits.append(audit);feedback="repair still violates frozen semantics";continue
                        candidate=repair
                    else:
                        audits.append(audit);feedback="critic repair failed deterministic preflight: "+"; ".join(repair_pre.issues);continue
                else:
                    issues=review.get("issues",());feedback="semantic critic: "+"; ".join(map(str,issues));audits.append(audit);continue
        post=postflight(payload,candidate,oracle_answer=_oracle(world));audit["postflight"]=post.to_dict();audits.append(audit)
        if not post.accepted:
            feedback="; ".join(post.issues);continue
        realized={"task_id":payload["task_id"],"question":candidate,"draft_question":payload.get("draft_question"),"task_design":payload.get("task_design",{}),"resources":payload.get("resources",()),"available_agents":payload.get("available_agents",()),"capability_world":capability,"authoring":{"attempts":attempt,"semantic_critic_enabled":use_critic,"audit":audits}}
        return True,realized,audits
    return False,{"task_id":payload.get("task_id"),"reason":"question_realization_failed","last_feedback":feedback,"audit":audits},audits
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
        summary={
            "pipeline":"frozen_public_author_then_deterministic_preflight_then_semantic_critic_repair_then_postflight",
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
