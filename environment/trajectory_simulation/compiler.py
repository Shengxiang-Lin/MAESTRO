"""Compile frozen acquisition topology into dependency-aware trajectory frontiers."""
from __future__ import annotations
from typing import Any,Mapping

def compile_frontiers(realized:Mapping[str,Any])->dict[str,Any]:
    capability=realized.get("capability_world",{}) if isinstance(realized.get("capability_world"),Mapping) else {};resources=[dict(item) for item in capability.get("resources",()) if isinstance(item,Mapping)];pending={str(item.get("resource_id")):item for item in resources if item.get("resource_id")};observed=set();rounds=[];wave=0
    while pending:
        ready=[]
        for resource_id,item in pending.items():
            visibility=str(item.get("visibility","discoverable"));parents=set(map(str,item.get("anchor_dependency_resource_ids",())))
            if visibility in {"attached","provided","initial"} or parents.issubset(observed):ready.append(resource_id)
        if not ready:break
        wave+=1;ready=sorted(ready);rounds.append({"round":wave,"available_resource_ids":ready,"already_observed_resource_ids":sorted(observed)})
        for resource_id in ready:observed.add(resource_id);pending.pop(resource_id,None)
    return {"task_id":realized.get("task_id"),"question":realized.get("question"),"rounds":rounds,"resource_count":len(resources),"reachable_resource_ids":sorted(observed),"unreachable_resource_ids":sorted(pending),"complete":not pending,"policy":"environment_authoritative_progressive_frontier"}
__all__=["compile_frontiers"]
