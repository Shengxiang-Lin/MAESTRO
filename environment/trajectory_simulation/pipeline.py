"""Plan and synthesize dependency-aware clean orchestration trajectories."""
from __future__ import annotations
import json,os
from pathlib import Path
from typing import Any,Mapping
from tqdm.auto import tqdm
from environment.question_generation.backend import BackendError,OpenAICompatibleBackend
from environment.task_construction.rendering.pipeline import render_task_record
from .backend import LLMTeacher,PassthroughTeacher
from .compiler import compile_frontiers
from .prompts import teacher_prompt
from .validator import validate_step,validate_trajectory

def _rows(path):
    with Path(path).open("r",encoding="utf-8") as handle:return [json.loads(line) for line in handle if line.strip()]
def _world_map(path,seed=17):
    result={}
    for index,row in enumerate(_rows(path)):
        rendered=render_task_record(row,seed=seed+index);result[rendered.task_id]=(row,rendered)
    return result
def plan_trajectories_jsonl(worlds_path,realized_path,output_path,*,max_records=None):
    realized=_rows(realized_path)
    if max_records is not None:realized=realized[:max_records]
    target=Path(output_path);target.parent.mkdir(parents=True,exist_ok=True);accepted=0;unreachable=0
    with target.open("w",encoding="utf-8") as writer:
        for row in realized:
            plan=compile_frontiers(row);accepted+=int(bool(plan.get("complete")));unreachable+=len(plan.get("unreachable_resource_ids",()));writer.write(json.dumps(plan,ensure_ascii=False)+"\n")
    return {"planned":len(realized),"complete":accepted,"unreachable_resource_count":unreachable,"output":str(target)}
def _resource_payload(rendered,resource_ids):
    wanted=set(map(str,resource_ids));return [{"resource_id":source.source_id,"source_type":source.source_type,"title":source.title,"content":source.content} for source in rendered.sources if source.source_id in wanted]
def _feedback(resources):
    return {"result":"\n".join(f"[{item['resource_id']}] {item['content']}" for item in resources),"summary":f"Observed {len(resources)} environment-authoritative resource(s).","trace_summary":{"resource_ids":[item["resource_id"] for item in resources],"source_types":[item["source_type"] for item in resources],"environment_authoritative":True}}
def _teacher(provider,*,base_url=None,model=None,api_key=None,temperature=.2,max_tokens=2048,timeout=90.,json_mode=True):
    if provider=="passthrough":return PassthroughTeacher()
    if not base_url or not model:raise ValueError("base_url/model required for openai-compatible trajectory teacher")
    return LLMTeacher(OpenAICompatibleBackend(base_url,model,api_key,temperature,max_tokens,timeout,json_mode))
def synthesize_trajectories_jsonl(worlds_path,realized_path,output_path,*,provider="passthrough",base_url=None,model=None,api_key=None,temperature=.2,max_tokens=2048,timeout=90.,max_records=None,max_attempts=3,rejected_output_path=None,progress=True,flush_every=5,json_mode=True,seed=17):
    realized=_rows(realized_path)
    if max_records is not None:realized=realized[:max_records]
    worlds=_world_map(worlds_path,seed=seed);teacher=_teacher(provider,base_url=base_url,model=model,api_key=api_key,temperature=temperature,max_tokens=max_tokens,timeout=timeout,json_mode=json_mode);target=Path(output_path);target.parent.mkdir(parents=True,exist_ok=True);rejected_path=Path(rejected_output_path) if rejected_output_path else target.with_suffix(target.suffix+".rejected.jsonl");accepted=0;rejected=0;out=target.open("w",encoding="utf-8");bad=rejected_path.open("w",encoding="utf-8");bar=tqdm(realized,desc="Trajectory simulation",unit="task",disable=not progress)
    try:
        for task_index,row in enumerate(bar):
            task_id=str(row.get("task_id"));pair=worlds.get(task_id);plan=compile_frontiers(row)
            if pair is None or not plan.get("complete"):
                rejected+=1;bad.write(json.dumps({"task_id":task_id,"reason":"missing_world_or_incomplete_frontier","plan":plan},ensure_ascii=False)+"\n");continue
            _,rendered=pair;history=[];steps=[];failed=None
            for frontier in plan.get("rounds",()):
                resources=_resource_payload(rendered,frontier.get("available_resource_ids",()));payload={"task_id":task_id,"question":row.get("question"),"frontier":frontier,"resources":resources,"history":history};step=None;issues=[]
                for attempt in range(1,max(1,max_attempts)+1):
                    try:candidate=teacher.generate(payload,teacher_prompt(payload))
                    except BackendError as exc:issues=["teacher_backend_error:"+str(exc)];continue
                    ok,issues=validate_step(candidate,frontier,is_final_frontier=False)
                    if ok:step=dict(candidate);break
                if step is None:failed={"round":frontier.get("round"),"issues":issues};break
                feedback=_feedback([item for item in resources if item["resource_id"] in set(map(str,step.get("resource_ids",()))) ]);record={"round":frontier.get("round"),"frontier":frontier,"main_action":step,"subagent_feedback":feedback};steps.append(record);history.append(record)
            if failed is None:
                final={"action":"complete","resource_ids":[],"subtask":"","summary":"All frozen evidence frontiers have been observed; answer from accumulated evidence.","completion_claim":True};steps.append({"round":len(plan.get("rounds",()))+1,"frontier":{"available_resource_ids":[]},"main_action":final,"subagent_feedback":None})
                flat=[step["main_action"] for step in steps];validation=validate_trajectory(plan,flat)
                if not validation.get("accepted"):failed={"issues":validation.get("issues")}
            if failed is not None:
                rejected+=1;bad.write(json.dumps({"task_id":task_id,"reason":"trajectory_validation_failed","failure":failed,"steps":steps},ensure_ascii=False)+"\n");continue
            payload={"task_id":task_id,"question":row.get("question"),"plan":plan,"steps":steps,"validation":validation,"metadata":{"provider":provider,"model":model if provider=="openai-compatible" else "passthrough","clean_trajectory":True,"privileged_error_labels_exposed_to_main":False}};out.write(json.dumps(payload,ensure_ascii=False)+"\n");accepted+=1
            if flush_every and accepted%flush_every==0:out.flush();bad.flush()
    finally:out.close();bad.close()
    return {"requested":len(realized),"accepted":accepted,"rejected":rejected,"acceptance_rate":accepted/len(realized) if realized else 0.,"output":str(target),"rejected_output":str(rejected_path),"provider":provider,"model":model if provider=="openai-compatible" else "passthrough"}
__all__=["plan_trajectories_jsonl","synthesize_trajectories_jsonl"]
