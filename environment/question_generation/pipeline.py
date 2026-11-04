"""Frozen-boundary LLM question generation with deterministic and semantic review."""
from __future__ import annotations
import argparse,json,os
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from typing import Any,Mapping
from tqdm.auto import tqdm
from environment.config import EnvironmentConfigError,parse_with_config,resolve_model
from .backend import BackendError,LLMQuestionAuthor,LLMSemanticCritic,OpenAICompatibleBackend,PassthroughAuthorBackend,PassthroughCriticBackend
from .boundary import BoundaryError,validate_frozen_inputs
from .prompts import author_prompt,critic_prompt
from .validators import postflight,preflight

def _rows(path):
    with Path(path).open("r",encoding="utf-8") as handle:return [json.loads(line) for line in handle if line.strip()]
def _world_by_task(path,seed=17):
    from environment.task_construction.rendering.pipeline import render_task_record
    result={}
    for index,row in enumerate(_rows(path)):
        task=render_task_record(row,seed=seed+index);result[task.task_id]=row
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
def _oracle(world):return world.get("answer") if isinstance(world,Mapping) else None
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
            if not bool(review.get("accept")) or str(review.get("semantic_fidelity","pass")).lower()=="fail":
                repair=" ".join(str(review.get("repair_question") or "").split())
                if repair:
                    repair_pre=preflight(payload,repair);audit["repair_preflight"]=repair_pre.to_dict()
                    if repair_pre.accepted:candidate=repair
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
def run(args):
    try:frozen=validate_frozen_inputs(args.manifest,args.authoring_inputs,args.capabilities,worlds_path=args.worlds)
    except (BoundaryError,FileNotFoundError,json.JSONDecodeError) as exc:raise SystemExit(f"Frozen-input validation failed: {exc}") from exc
    authoring=_rows(args.authoring_inputs)
    if args.max_records is not None:authoring=authoring[:args.max_records]
    worlds=_world_by_task(args.worlds,seed=args.seed);capabilities=_capability_by_task(args.capabilities);output_dir=Path(args.output_dir);output_dir.mkdir(parents=True,exist_ok=True);realized_path=output_dir/"realized_capability_worlds.jsonl";rejected_path=output_dir/"question_rejected.jsonl";critic_path=output_dir/"question_critic_audit.jsonl";summary_path=output_dir/"question_generation_summary.json";author,critic=_backends(args);results=[];workers=max(1,args.question_workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(_realize_one,payload,worlds.get(str(payload.get("task_id")),{}),capabilities.get(str(payload.get("task_id")),{}),author,critic,question_max_attempts=args.question_max_attempts,use_critic=not args.no_semantic_critic):index for index,payload in enumerate(authoring)}
        iterator=as_completed(futures);bar=tqdm(total=len(futures),desc="Question generation",unit="task",disable=args.no_progress)
        for future in iterator:
            index=futures[future]
            try:ok,row,audit=future.result()
            except Exception as exc:ok=False;row={"task_id":authoring[index].get("task_id"),"reason":"unhandled_exception","error":str(exc)};audit=[]
            results.append((index,ok,row,audit));bar.update(1)
        bar.close()
    results.sort(key=lambda item:item[0]);accepted=sum(ok for _,ok,_,_ in results)
    with realized_path.open("w",encoding="utf-8") as good,rejected_path.open("w",encoding="utf-8") as bad,critic_path.open("w",encoding="utf-8") as critic_out:
        for _,ok,row,audit in results:
            (good if ok else bad).write(json.dumps(row,ensure_ascii=False)+"\n");critic_out.write(json.dumps({"task_id":row.get("task_id"),"accepted":ok,"audit":audit},ensure_ascii=False)+"\n")
    summary={"pipeline":"frozen_public_author_then_deterministic_preflight_then_semantic_critic_repair_then_postflight","frozen_batch_validation":frozen,"provider":args._resolved_models["question_author"]["backend"],"model":args._resolved_models["question_author"]["model"],"critic_model":args._resolved_models["question_critic"]["model"] if not args.no_semantic_critic else None,"model_roles":args._resolved_models,"requested":len(authoring),"accepted":accepted,"rejected":len(authoring)-accepted,"acceptance_rate":accepted/len(authoring) if authoring else 0.,"semantic_critic":{"enabled":not args.no_semantic_critic},"artifacts":{"realized":str(realized_path),"rejected":str(rejected_path),"critic_audit":str(critic_path),"summary":str(summary_path)}};summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");return summary
def build_parser():
    p=argparse.ArgumentParser(description="Generate natural questions over a frozen MAESTRO task batch");p.add_argument("--manifest");p.add_argument("--authoring-inputs");p.add_argument("--worlds");p.add_argument("--capabilities");p.add_argument("--output-dir");p.add_argument("--max-records",type=int);p.add_argument("--seed",type=int,default=17);p.add_argument("--provider",choices=("passthrough","openai-compatible"),default="passthrough");p.add_argument("--base-url");p.add_argument("--model");p.add_argument("--critic-model");p.add_argument("--api-key-env",default="DASHSCOPE_API_KEY");p.add_argument("--question-temperature",type=float,default=.4);p.add_argument("--critic-temperature",type=float,default=0.);p.add_argument("--max-tokens",type=int,default=2048);p.add_argument("--critic-max-tokens",type=int,default=768);p.add_argument("--timeout",type=float,default=180.);p.add_argument("--question-max-attempts",type=int,default=2);p.add_argument("--question-workers",type=int,default=4);p.add_argument("--no-semantic-critic",action="store_true");p.add_argument("--no-json-mode",action="store_true");p.add_argument("--no-progress",action="store_true");return p
def main():
    try:
        args=parse_with_config(build_parser(),"question_generation")
    except EnvironmentConfigError as exc:
        raise SystemExit(f"Environment configuration error: {exc}") from exc
    missing=[key for key in ("manifest","authoring_inputs","worlds","capabilities","output_dir") if not getattr(args,key,None)]
    if missing:raise SystemExit("Missing question-generation inputs (CLI or YAML): "+", ".join(missing))
    if args.question_workers<1:raise SystemExit("--question-workers must be >= 1")
    try:
        result=run(args)
    except EnvironmentConfigError as exc:
        raise SystemExit(f"Environment model error: {exc}") from exc
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
