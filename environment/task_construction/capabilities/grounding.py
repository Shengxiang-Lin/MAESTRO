"""P1 capability grounding over frozen TaskDesign resources."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Any,Mapping
from ..rendering.pipeline import program_from_dict,world_from_dict
from ..scenario.validity import assess_scenario,scenario_required_fact_ids
from ..validation import minimal_source_proof
from .discovery import bind_resource_discovery
from .hints import available_tool_families,capability_hint,capability_id_for_source,specialist_for_source
from .schema import AcquisitionResource,AcquisitionTopology,CapabilityRequirement

def _read_jsonl(path):
    rows=[]
    with Path(path).open("r",encoding="utf-8") as handle:
        for line in handle:
            if line.strip():rows.append(json.loads(line))
    return rows

def _design_by_task(path):
    result={}
    for row in _read_jsonl(path):
        task_id=str(row.get("task_id") or "")
        if task_id:result[task_id]=row
    return result

def _task_id(record):
    import hashlib
    basis=str(record.get("source_program_signature") or program_from_dict(record["program"]).canonical_signature());return f"world-{hashlib.sha1(basis.encode()).hexdigest()[:12]}"

def _resource_rows(design,world):
    visibility={str(item.get("source_id")):item for item in design.get("resource_visibility",()) if isinstance(item,Mapping)}
    resources=[];capabilities={}
    for source_id,source in sorted(world.sources.items()):
        spec=visibility.get(source_id,{});source_type=str(source.source_type);cid=capability_id_for_source(source_type);hint=capability_hint(source_type);capabilities[cid]=CapabilityRequirement(cid,specialist_for_source(source_type),tuple(hint["source_types"]),tuple(hint["tool_families"]),str(hint["rationale"]));visibility_label=str(spec.get("visibility","discoverable"));families=available_tool_families(source_type);families=tuple(dict.fromkeys(("search",*families))) if visibility_label=="discoverable" else families;resources.append(AcquisitionResource(source_id,source_type,visibility_label,str(spec.get("role",source.metadata.get("material_role","discovery"))),(cid,),families,{"material_role":source.metadata.get("material_role"),"surface_style":source.metadata.get("surface_style"),"scenario_prior_origin":source.metadata.get("scenario_prior_origin"),"retrieval_policy":design.get("metadata",{}).get("retrieval_policy"),"access_mode":"search_then_inspect" if visibility_label=="discoverable" else "provided_input"}))
    return tuple(resources),tuple(capabilities[key] for key in sorted(capabilities))

def _decision_fact_sources(world,required_fact_ids):
    required=set(map(str,required_fact_ids));result=set()
    for source_id,source in world.sources.items():
        if required.intersection(map(str,source.fact_ids)):result.add(source_id)
    return result

def ground_capability_world(record,design):
    program=program_from_dict(record["program"]);world=world_from_dict(record["world"]);resources,capabilities=_resource_rows(design,world);public_clues=design.get("public_clues",())
    question_anchors=[]
    for clue in public_clues:
        if not isinstance(clue,Mapping):continue
        graph=clue.get("clue_graph",{})
        if not isinstance(graph,Mapping):continue
        for node in graph.get("nodes",()):
            if not isinstance(node,Mapping):continue
            for key in ("value","predicate"):
                value=node.get(key)
                if value is not None:question_anchors.append(str(value))
    bound,discovery=bind_resource_discovery(world,resources,question_anchor_values=question_anchors,web_query_bootstrap=design.get("entry_strategy")=="web_search");scenario=assess_scenario(program,world);required=scenario_required_fact_ids(scenario);decision_sources=_decision_fact_sources(world,required);proof=minimal_source_proof(program,world,max_sources=8,max_witnesses=16);proof_sources={sid for witness in proof.get("witnesses",()) for sid in witness};direct={r.resource_id for r in bound if r.visibility in {"attached","provided","initial"}};direct_decision_ok=bool(required) and decision_sources.issubset(direct);direct_answer_ok=bool(proof.get("witnesses")) and any(set(witness).issubset(direct) for witness in proof.get("witnesses",()));specialist_required=any(r.source_type not in {"text"} or r.visibility=="discoverable" for r in bound);depth=int(discovery.get("progressive_discovery_depth",0) or 0);orchestration_ready=bool(scenario.accepted and specialist_required and not (direct_decision_ok and direct_answer_ok) and discovery.get("all_resources_reachable") and depth>=2 and not discovery.get("premature_anchor_leakage"));audit={"p1_orchestration_profile":{"specialist_required":specialist_required,"direct_only_shortcut":bool(direct_decision_ok and direct_answer_ok),"decision_support_source_ids":sorted(decision_sources),"answer_proof_source_ids":sorted(proof_sources),"progressive_discovery":discovery,"orchestration_ready":orchestration_ready},"progressive_discovery":discovery,"scenario_valid":scenario.accepted,"direct_only_shortcut":bool(direct_decision_ok and direct_answer_ok),"orchestration_ready":orchestration_ready};topology=AcquisitionTopology(str(design.get("task_id") or _task_id(record)),str(design.get("entry_strategy") or "hybrid"),bound,capabilities,tuple(r.resource_id for r in bound if r.visibility=="discoverable"),audit);return topology

def ground_capabilities_jsonl(worlds_path,output_path,*,max_records=None,seed=17,task_design_path=None):
    if task_design_path is None:raise ValueError("task_design_path is required")
    designs=_design_by_task(task_design_path);source=Path(worlds_path);target=Path(output_path);target.parent.mkdir(parents=True,exist_ok=True);total=ready=specialist=direct_shortcuts=0;depth_hist=Counter();unreachable=0
    with source.open("r",encoding="utf-8") as reader,target.open("w",encoding="utf-8") as writer:
        for line in reader:
            if max_records is not None and total>=max_records:break
            if not line.strip():continue
            record=json.loads(line);task_id=_task_id(record);design=designs.get(task_id)
            if design is None:raise ValueError(f"missing TaskDesign for {task_id}")
            topology=ground_capability_world(record,design);writer.write(json.dumps(topology.to_dict(),ensure_ascii=False)+"\n");total+=1;audit=topology.binding_audit;profile=audit.get("p1_orchestration_profile",{});ready+=int(bool(profile.get("orchestration_ready")));specialist+=int(bool(profile.get("specialist_required")));direct_shortcuts+=int(bool(profile.get("direct_only_shortcut")));discovery=profile.get("progressive_discovery",{});depth_hist[int(discovery.get("progressive_discovery_depth",0) or 0)]+=1;unreachable+=len(discovery.get("unreachable_resource_ids",()))
    return {"input":str(source),"output":str(target),"task_design":str(task_design_path),"grounded":total,"specialist_required":specialist,"orchestration_ready":ready,"direct_only_shortcuts":direct_shortcuts,"unreachable_resources":unreachable,"progressive_discovery_depth_histogram":{str(k):v for k,v in sorted(depth_hist.items())},"seed":seed}
