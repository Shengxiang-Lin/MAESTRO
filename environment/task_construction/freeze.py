"""Strict P1 freeze and public projection for LLM question authoring."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any,Mapping
from .rendering.pipeline import render_task_record
class FreezeError(RuntimeError):pass

def _rows(path):
    with Path(path).open("r",encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()
def _by_task(rows):return {str(row.get("task_id")):row for row in rows if row.get("task_id") is not None}
def _task_id_from_world(record):return render_task_record(record,seed=17).task_id
def _public_authoring_projection(world_record,design,capability,*,seed):
    from environment.question_generation.contracts import terminal_contract
    draft=render_task_record(world_record,seed=seed);resources=[]
    for resource in capability.get("resources",()):
        if not isinstance(resource,Mapping):continue
        resources.append({"resource_id":resource.get("resource_id"),"source_type":resource.get("source_type"),"visibility":resource.get("visibility"),"role":resource.get("role"),"required_capability_ids":resource.get("required_capability_ids",()),"available_tool_families":resource.get("available_tool_families",())})
    agents=[]
    for item in capability.get("capabilities",()):
        if not isinstance(item,Mapping):continue
        agents.append({"capability_id":item.get("capability_id"),"specialist":item.get("specialist"),"source_types":item.get("source_types",()),"tool_families":item.get("tool_families",())})
    return {"task_id":design["task_id"],"draft_question":draft.question,"task_design":dict(design.get("public_contract",{})),"resources":resources,"available_agents":agents,"freeze_version":4,"terminal_contract":terminal_contract(world_record,draft.question)}
def freeze_batch(summary:Mapping[str,Any],*,accepted_count:int,seed:int,worlds_path,task_design_path,scenario_path,capability_path,authoring_input_path,manifest_path):
    worlds=_rows(worlds_path);designs=_rows(task_design_path);scenarios=_rows(scenario_path);capabilities=_rows(capability_path)
    if accepted_count<=0:raise FreezeError("cannot freeze an empty batch")
    if len(worlds)!=accepted_count:raise FreezeError(f"world batch has {len(worlds)} records, need {accepted_count}")
    for label,rows in (("task_designs",designs),("scenarios",scenarios),("capabilities",capabilities)):
        if len(rows)!=accepted_count:raise FreezeError(f"{label} count mismatch: {len(rows)} != {accepted_count}")
    for section in ("semantic_plausibility","execution_contract"):
        audit=summary.get(section,{})
        if audit.get("audited")!=accepted_count or audit.get("accepted")!=accepted_count:
            raise FreezeError(f"{section} audit incomplete or rejected")
    if not summary.get("task_design",{}).get("taxonomy_aware_clue_validation"):
        raise FreezeError("taxonomy-aware task-design audit required")
    design_by=_by_task(designs);scenario_by=_by_task(scenarios);capability_by=_by_task(capabilities);failures=[];projections=[]
    for index,world in enumerate(worlds[:accepted_count]):
        task_id=_task_id_from_world(world);design=design_by.get(task_id);scenario=scenario_by.get(task_id);capability=capability_by.get(task_id)
        if design is None:failures.append(f"{task_id}:missing_task_design");continue
        if not bool(design.get("quality",{}).get("accepted")):failures.append(f"{task_id}:task_design_rejected")
        if scenario is None or not bool(scenario.get("accepted")):failures.append(f"{task_id}:scenario_invalid")
        if capability is None:failures.append(f"{task_id}:missing_capability");continue
        audit=capability.get("binding_audit",{});profile=audit.get("p1_orchestration_profile",{}) if isinstance(audit,Mapping) else {}
        if not bool(profile.get("specialist_required")):failures.append(f"{task_id}:specialist_not_required")
        if bool(profile.get("direct_only_shortcut")):failures.append(f"{task_id}:direct_only_shortcut")
        if not bool(profile.get("orchestration_ready")):failures.append(f"{task_id}:not_orchestration_ready")
        discovery=profile.get("progressive_discovery",{}) if isinstance(profile,Mapping) else {}
        if int(discovery.get("progressive_discovery_depth",0) or 0)<2:failures.append(f"{task_id}:discovery_depth_lt_2")
        if discovery.get("unreachable_resource_ids"):failures.append(f"{task_id}:unreachable_resources")
        projections.append(_public_authoring_projection(world,design,capability,seed=seed+index))
    semantic=summary.get("semantic_plausibility",{}) if isinstance(summary,Mapping) else {};execution=summary.get("execution_contract",{}) if isinstance(summary,Mapping) else {}
    if semantic and int(semantic.get("accepted",0) or 0)<accepted_count:failures.append("batch:semantic_plausibility_incomplete")
    if execution and int(execution.get("accepted",0) or 0)<accepted_count:failures.append("batch:execution_contract_incomplete")
    authoring=Path(authoring_input_path);authoring.parent.mkdir(parents=True,exist_ok=True)
    with authoring.open("w",encoding="utf-8") as writer:
        for row in projections:writer.write(json.dumps(row,ensure_ascii=False)+"\n")
    manifest={"version":4,"status":"frozen" if not failures and len(projections)>=accepted_count else "rejected",
        "candidate_pool_size":summary.get("pipeline",{}).get("candidate_pool_size"),
        "candidate_pool_fully_evaluated":summary.get("worlds",{}).get("pool_fully_evaluated"),
        "seed":seed,"frozen_count":min(len(projections),accepted_count),"failures":failures,"semantic_world_policy_version":2,"artifacts":{"worlds":{"path":str(worlds_path),"sha256":_sha(worlds_path)},"task_designs":{"path":str(task_design_path),"sha256":_sha(task_design_path)},"scenarios":{"path":str(scenario_path),"sha256":_sha(scenario_path)},"capabilities":{"path":str(capability_path),"sha256":_sha(capability_path)},"authoring_inputs":{"path":str(authoring),"sha256":_sha(authoring)}},"invariants":{"p15_wording_only":True,"private_world_not_in_author_prompt":True,"direct_only_shortcuts":0,"minimum_progressive_discovery_depth":2}}
    target_path=Path(manifest_path);target_path.parent.mkdir(parents=True,exist_ok=True);target_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    if manifest["status"]!="frozen":raise FreezeError("P1 freeze rejected: "+"; ".join(failures[:20]))
    return manifest
__all__=["FreezeError","freeze_batch"]
