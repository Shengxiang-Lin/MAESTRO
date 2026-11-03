"""Capacity-aware protected-fact placement for public evidence."""
from __future__ import annotations
from dataclasses import replace
from typing import Any,Mapping,Sequence
from ..executor import ExecutionError,SymbolicExecutor
from ..ir import ProgramIR
from ..scenario.information import assess_goal_information_completeness
from ..validation import indispensable_fact_ids,indispensable_source_ids,minimal_source_proof,normalize_answer
from ..world import EvidenceSource,SyntheticWorld
def _capacity_for_source(source,targets_by_slot):
    slot=int(source.metadata.get("layout_position",-1))
    if slot<0 or slot>=len(targets_by_slot):raise ValueError(f"source {source.source_id} has invalid layout position {slot}")
    return max(1,int(targets_by_slot[slot]))
def _overflow(source_fact_ids,capacities):return sum(max(0,len(source_fact_ids[sid])-capacities[sid]) for sid in capacities)
def _record_attempt(world,*,status,protected_fact_count,movable_fact_count,moved_fact_count,overflow_before,overflow_after,capacities,reason=None):
    allocation=dict(world.metadata.get("evidence_allocation",{}));payload={"status":status,"protected_fact_count":int(protected_fact_count),"movable_fact_count":int(movable_fact_count),"moved_fact_count":int(moved_fact_count),"critical_capacity_total":int(sum(capacities.values())),"overflow_before":int(overflow_before),"overflow_after":int(overflow_after),"overflow_reduction":int(overflow_before-overflow_after),"capacities":{str(k):int(v) for k,v in capacities.items()}}
    if reason:payload["reason"]=reason
    allocation["capacity_rebalancing"]=payload;policy=dict(allocation.get("policy",{}));policy["capacity_aware_protected_fact_allocation"]=True;allocation["policy"]=policy;world.metadata={**world.metadata,"evidence_allocation":allocation};return world
def rebalance_protected_facts(program:ProgramIR,world:SyntheticWorld,*,targets_by_slot:Sequence[int],min_critical_sources:int,min_noncritical_sources:int,max_sources:int):
    if len(targets_by_slot)!=len(world.sources):raise ValueError("capacity plan does not match realized source count")
    executor=SymbolicExecutor()
    try:baseline_answer=normalize_answer(executor.execute(program,world))
    except (ExecutionError,ValueError,TypeError):return world,[],[]
    allocation=dict(world.metadata.get("evidence_allocation",{}));critical_sources=sorted(sid for sid,source in world.sources.items() if str(source.metadata.get("role",""))=="critical");capacities={sid:_capacity_for_source(world.sources[sid],targets_by_slot) for sid in critical_sources};original_fact_ids={sid:list(source.fact_ids) for sid,source in world.sources.items()};before_overflow=_overflow(original_fact_ids,capacities) if capacities else 0
    if len(critical_sources)<min_critical_sources:return _record_attempt(world,status="skipped",protected_fact_count=0,movable_fact_count=0,moved_fact_count=0,overflow_before=before_overflow,overflow_after=before_overflow,capacities=capacities,reason="too_few_critical_sources"),[],[]
    baseline_information=assess_goal_information_completeness(program,world,executor=executor)
    if not baseline_information.accepted:return _record_attempt(world,status="skipped",protected_fact_count=0,movable_fact_count=0,moved_fact_count=0,overflow_before=before_overflow,overflow_after=before_overflow,capacities=capacities,reason="baseline_information_incomplete"),[],[]
    baseline_indispensable_facts=set(map(str,indispensable_fact_ids(program,world,executor=executor)));baseline_indispensable_sources=set(map(str,indispensable_source_ids(program,world,executor=executor)));protected_fact_ids=baseline_indispensable_facts|set(map(str,baseline_information.required_fact_ids))
    for source in world.sources.values():
        if bool(source.metadata.get("proof_corroboration",False)):protected_fact_ids.update(map(str,source.fact_ids))
    fact_to_source={str(fid):sid for sid,source in world.sources.items() for fid in source.fact_ids};movable=sorted(fid for fid in protected_fact_ids if fact_to_source.get(fid) in capacities)
    if not movable or before_overflow<=0:
        world=_record_attempt(world,status="not_needed",protected_fact_count=len(protected_fact_ids),movable_fact_count=len(movable),moved_fact_count=0,overflow_before=before_overflow,overflow_after=before_overflow,capacities=capacities);return world,sorted(baseline_indispensable_sources),sorted(baseline_indispensable_facts)
    candidate_fact_ids={sid:[str(fid) for fid in source.fact_ids if str(fid) not in movable] for sid,source in world.sources.items()};assigned=set()
    for sid in critical_sources:
        if sid not in baseline_indispensable_sources:continue
        choices=[fid for fid in movable if fact_to_source.get(fid)==sid and fid in baseline_indispensable_facts and fid not in assigned] or [fid for fid in movable if fact_to_source.get(fid)==sid and fid not in assigned]
        if choices:anchor=sorted(choices)[0];candidate_fact_ids[sid].append(anchor);assigned.add(anchor)
    remaining=[fid for fid in movable if fid not in assigned];remaining.sort(key=lambda fid:(0 if fid in baseline_indispensable_facts else 1,fact_to_source.get(fid,""),fid))
    for fid in remaining:
        original=fact_to_source.get(fid);ranked=[]
        for sid in critical_sources:
            cap=capacities[sid];current=len(candidate_fact_ids[sid]);increment=max(0,current+1-cap)-max(0,current-cap);ranked.append((float(increment),(current+1)/max(1,cap),0 if sid==original else 1,current,sid))
        candidate_fact_ids[min(ranked)[-1]].append(fid)
    after_overflow=_overflow(candidate_fact_ids,capacities);moved=sum(fact_to_source.get(fid)!=next(sid for sid in critical_sources if fid in candidate_fact_ids[sid]) for fid in movable)
    if after_overflow>=before_overflow:
        world=_record_attempt(world,status="no_improvement",protected_fact_count=len(protected_fact_ids),movable_fact_count=len(movable),moved_fact_count=moved,overflow_before=before_overflow,overflow_after=after_overflow,capacities=capacities);return world,sorted(baseline_indispensable_sources),sorted(baseline_indispensable_facts)
    new_source={fid:sid for sid,fids in candidate_fact_ids.items() for fid in fids};candidate_facts=[replace(fact,source_id=new_source.get(fact.fact_id,fact.source_id)) for fact in world.facts];candidate_sources={sid:EvidenceSource(source.source_id,source.source_type,tuple(candidate_fact_ids[sid]),dict(source.metadata)) for sid,source in world.sources.items()};candidate=SyntheticWorld(dict(world.entities),candidate_facts,candidate_sources,dict(world.metadata))
    try:answer=normalize_answer(executor.execute(program,candidate))
    except (ExecutionError,ValueError,TypeError):answer=None
    def revert(reason):return _record_attempt(world,status="reverted",protected_fact_count=len(protected_fact_ids),movable_fact_count=len(movable),moved_fact_count=moved,overflow_before=before_overflow,overflow_after=after_overflow,capacities=capacities,reason=reason),sorted(baseline_indispensable_sources),sorted(baseline_indispensable_facts)
    if answer!=baseline_answer:return revert("oracle_changed")
    information=assess_goal_information_completeness(program,candidate,executor=executor)
    if not information.accepted:return revert("information_completeness_changed")
    proof=minimal_source_proof(program,candidate,executor=executor,max_sources=max_sources,max_witnesses=16);witnesses=[list(map(str,w)) for w in proof.get("witnesses",())]
    if not proof.get("exhaustive") or not witnesses:return revert("minimal_proof_invalid")
    proof_relevant={sid for witness in witnesses for sid in witness};candidate_indispensable_sources=list(map(str,indispensable_source_ids(program,candidate,executor=executor)));candidate_indispensable_facts=list(map(str,indispensable_fact_ids(program,candidate,executor=executor)));nonproof=[sid for sid in candidate.sources if sid not in proof_relevant];baseline_proof=allocation.get("minimal_proof",{});baseline_min=baseline_proof.get("min_size");baseline_witness_count=int(allocation.get("proof_witness_count",0) or 0);failure=None
    if len(candidate_indispensable_sources)<min_critical_sources:failure="too_few_indispensable_sources"
    elif len(nonproof)<min_noncritical_sources:failure="too_few_nonproof_sources"
    elif any(sid not in proof_relevant for sid in critical_sources):failure="critical_source_became_nonproof"
    elif isinstance(baseline_min,int) and int(proof.get("min_size",0))<baseline_min:failure="minimum_proof_size_decreased"
    elif baseline_witness_count>1 and len(witnesses)<2:failure="alternate_proof_witness_lost"
    if failure:return revert(failure)
    indispensable_set=set(candidate_indispensable_sources);ind_fact_set=set(candidate_indispensable_facts);updated={}
    for sid,source in candidate.sources.items():
        old=set(original_fact_ids[sid]);new=set(source.fact_ids);metadata={**dict(source.metadata),"proof_relevant":sid in proof_relevant,"indispensable":sid in indispensable_set,"contains_indispensable_fact":any(fid in ind_fact_set for fid in source.fact_ids),"bundle_size":len(source.fact_ids),"capacity_rebalance_moved_in":len(new-old),"capacity_rebalance_moved_out":len(old-new)};updated[sid]=EvidenceSource(source.source_id,source.source_type,source.fact_ids,metadata)
    candidate.sources=updated;ca=dict(candidate.metadata.get("evidence_allocation",{}));ca.update({"indispensable_source_ids":sorted(candidate_indispensable_sources),"critical_source_ids":sorted(candidate_indispensable_sources),"proof_relevant_source_ids":sorted(proof_relevant),"nonproof_source_ids":sorted(nonproof),"minimal_proof":proof,"proof_witness_count":len(witnesses),"multiple_proof_witnesses":len(witnesses)>1,"indispensable_fact_count":len(candidate_indispensable_facts)});candidate.metadata={**candidate.metadata,"evidence_allocation":ca};candidate=_record_attempt(candidate,status="accepted",protected_fact_count=len(protected_fact_ids),movable_fact_count=len(movable),moved_fact_count=moved,overflow_before=before_overflow,overflow_after=after_overflow,capacities=capacities);return candidate,sorted(candidate_indispensable_sources),sorted(candidate_indispensable_facts)
