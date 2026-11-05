"""Deterministic progressive-discovery analysis for frozen task resources.

Anchors are derived from evidence that is actually present in a resource.  They
are private acquisition metadata: P1.5 may use the public topology but never the
private concrete anchor registry.
"""
from __future__ import annotations
import re
from collections import defaultdict
from typing import Any,Iterable,Mapping,Sequence
from ..world import SyntheticWorld
from .schema import AcquisitionResource
_GENERIC_TYPES={"entity","person","organization","thing","event","place","number","date","year","value"}
_PADDING_PREFIXES=("context-","record ")
def _distinctive(value:Any)->bool:
    text=" ".join(str(value).split()).strip()
    if len(text)<3:return False
    lower=text.casefold()
    if lower in _GENERIC_TYPES or lower.startswith(_PADDING_PREFIXES):return False
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)",text):return False
    return bool(re.search(r"[A-Za-z]",text) or len(text)>=6)
def _fact_anchor_values(world:SyntheticWorld,source_id:str)->list[tuple[str,str]]:
    source=world.sources[source_id];fact_map={fact.fact_id:fact for fact in world.facts};result=[]
    for fact_id in source.fact_ids:
        fact=fact_map.get(fact_id)
        if fact is None:continue
        for role,value in (("subject",fact.subject),("object",fact.object)):
            if _distinctive(value):result.append((role,str(value)))
    seen=set();dedup=[]
    for role,value in result:
        key=value.casefold()
        if key in seen:continue
        seen.add(key);dedup.append((role,value))
    return dedup
def build_anchor_registry(world:SyntheticWorld,resources:Sequence[AcquisitionResource],*,question_anchor_values:Iterable[str]=())->dict[str,Any]:
    anchors={};by_resource=defaultdict(list);providers=defaultdict(set);counter=0
    for resource in resources:
        for role,value in _fact_anchor_values(world,resource.resource_id):
            key=value.casefold();existing=next((aid for aid,item in anchors.items() if str(item["surface"]).casefold()==key),None)
            if existing is None:
                counter+=1;existing=f"anchor_{counter:03d}";anchors[existing]={"surface":value,"kind":role,"providers":[]}
            providers[existing].add(resource.resource_id);by_resource[resource.resource_id].append(existing)
    question_ids=[]
    for raw in question_anchor_values:
        if not _distinctive(raw):continue
        key=str(raw).casefold();aid=next((aid for aid,item in anchors.items() if str(item["surface"]).casefold()==key),None)
        if aid is None:
            counter+=1;aid=f"anchor_{counter:03d}";anchors[aid]={"surface":str(raw),"kind":"question","providers":[]}
        question_ids.append(aid)
    for aid,item in anchors.items():item["providers"]=sorted(providers.get(aid,()))
    return {"anchors":anchors,"resource_anchor_ids":{rid:sorted(set(ids)) for rid,ids in by_resource.items()},"question_anchor_ids":sorted(set(question_ids)),"policy":"concrete_observed_values_only_no_generic_type_or_bare_numeric_anchor"}
def bind_resource_discovery(
    world:SyntheticWorld, resources:Sequence[AcquisitionResource], *,
    question_anchor_values:Iterable[str]=(),
    web_query_bootstrap:bool=False,
)->tuple[tuple[AcquisitionResource,...],dict[str,Any]]:
    registry=build_anchor_registry(world,resources,question_anchor_values=question_anchor_values)
    if web_query_bootstrap:
        if any(r.visibility!="discoverable" for r in resources):
            raise ValueError("web_search_bootstrap_requires_all_sources_discoverable")
        # The user question is itself a valid *search starting point*, not a
        # file containing candidate names. This stage certifies only the
        # potential acquisition topology. Actual query-dependent retrieval
        # is independently tested in Stage 3 against ALL public documents.
        bound=tuple(AcquisitionResource(
            r.resource_id,r.source_type,r.visibility,r.role,
            r.required_capability_ids,r.available_tool_families,
            {**dict(r.metadata),"search_entry":"public_question"},
            (),r.produces_anchors,(),
        ) for r in resources)
        ids=sorted(r.resource_id for r in resources)
        return bound,{
            "progressive_discovery_depth":2,
            "resource_observation_depths":{sid:2 for sid in ids},
            "frontiers":[
                {"wave":1,"resource_ids":[],"available_anchor_ids":
                 registry["question_anchor_ids"]},
                {"wave":2,"resource_ids":ids,"available_anchor_ids":
                 sorted(registry["anchors"])},
            ],
            "unreachable_resource_ids":[],
            "all_resources_reachable":True,
            "anchor_registry":registry,
            "premature_anchor_leakage":False,
            "bootstrap_policy":"public_question_search_not_preprovided_files",
            "retrieval_verified_here":False,
        }
    available=set(registry["question_anchor_ids"]);updated=[];depths={};frontiers=[];pending={resource.resource_id:resource for resource in resources}
    wave=1
    # Initial/provided resources are observed at wave one.
    initial=[r for r in resources if r.visibility in {"attached","provided","initial"}]
    for resource in initial:
        available.update(registry["resource_anchor_ids"].get(resource.resource_id,()));depths[resource.resource_id]=1;pending.pop(resource.resource_id,None)
    frontiers.append({"wave":1,"resource_ids":sorted(depths),"available_anchor_ids":sorted(available)})
    while pending:
        ready=[]
        for resource in pending.values():
            own=set(registry["resource_anchor_ids"].get(resource.resource_id,()))
            candidates=[aid for aid,item in registry["anchors"].items() if aid in available and resource.resource_id not in item.get("providers",())]
            if candidates:ready.append((resource,candidates))
        if not ready:break
        wave+=1;observed=[]
        for resource,candidates in sorted(ready,key=lambda item:item[0].resource_id):
            selected=sorted(candidates,key=lambda aid:(-len(str(registry["anchors"][aid]["surface"])),aid))[0];parents=tuple(sorted(registry["anchors"][selected].get("providers",())))
            updated.append((resource.resource_id,selected,parents));depths[resource.resource_id]=wave;available.update(registry["resource_anchor_ids"].get(resource.resource_id,()));pending.pop(resource.resource_id,None);observed.append(resource.resource_id)
        frontiers.append({"wave":wave,"resource_ids":observed,"available_anchor_ids":sorted(available)})
    choice={rid:(anchor,parents) for rid,anchor,parents in updated};bound=[]
    for resource in resources:
        produced=tuple(registry["resource_anchor_ids"].get(resource.resource_id,()))
        if resource.resource_id in choice:
            anchor,parents=choice[resource.resource_id];bound.append(AcquisitionResource(resource.resource_id,resource.source_type,resource.visibility,resource.role,resource.required_capability_ids,resource.available_tool_families,resource.metadata,(anchor,),produced,parents))
        else:bound.append(AcquisitionResource(resource.resource_id,resource.source_type,resource.visibility,resource.role,resource.required_capability_ids,resource.available_tool_families,resource.metadata,(),produced,()))
    unreachable=sorted(pending);report={"progressive_discovery_depth":max(depths.values(),default=0),"resource_observation_depths":depths,"frontiers":frontiers,"unreachable_resource_ids":unreachable,"all_resources_reachable":not unreachable,"anchor_registry":registry,"premature_anchor_leakage":False}
    return tuple(bound),report
__all__=["bind_resource_discovery","build_anchor_registry"]
