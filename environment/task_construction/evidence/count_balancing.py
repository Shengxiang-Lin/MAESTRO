"""Answer-independent source-count balancing for final evidence worlds."""
from __future__ import annotations
import random
from collections import Counter
from dataclasses import replace
from typing import Any,Mapping,Sequence
from ..executor import SymbolicExecutor
from ..ir import ProgramIR
from ..source_motifs import SourceMotifPrior
from ..validation import indispensable_fact_ids,indispensable_source_ids,minimal_source_proof,normalize_answer
from ..world import EvidenceSource,Fact,SyntheticWorld
from .allocation import DIAGNOSTIC_ROLES,STANDARD_MODALITIES,SYNTHETIC_MODALITIES,bundle_stats
from .balancing import DEFAULT_BUNDLE_SIZE_WEIGHTS,allocate_evidence_sources as _allocate_bundle_balanced_sources,sample_target_bundle_sizes
from .semantics import EvidenceAllocationError
from .source_count import DEFAULT_SOURCE_COUNT_WEIGHTS,sample_target_source_count
def _copy_world(world):
    result=SyntheticWorld(dict(world.entities),list(world.facts),{},dict(world.metadata))
    for source in world.sources.values():result.add_source(EvidenceSource(source.source_id,source.source_type,tuple(source.fact_ids),dict(source.metadata)))
    return result
def _next_source_id(world):
    index=1
    while f"source_{index:02d}" in world.sources:index+=1
    return f"source_{index:02d}"
def _next_fact_id(known,counter):
    while True:
        counter[0]+=1;fact_id=f"cf{counter[0]:05d}"
        if fact_id not in known:known.add(fact_id);return fact_id
def _sample_added_modality(rng,*,synthetic_modality_probability):return (rng.choice(SYNTHETIC_MODALITIES),"source_count_synthetic_extension") if rng.random()<synthetic_modality_probability else (rng.choice(STANDARD_MODALITIES),"source_count_standard_extension")
def _clone_fact(template,*,source_id,known_ids,counter):return replace(template,fact_id=_next_fact_id(known_ids,counter),source_id=source_id)
def _append_source(world,*,templates,source_type,source_type_origin,role,proof_corroboration,rng,bundle_size,known_fact_ids,clone_counter,corroborates_source_id=None):
    if not templates:raise EvidenceAllocationError("cannot add an evidence source without fact templates")
    source_id=_next_source_id(world);selected=list(templates) if proof_corroboration else []
    if not proof_corroboration:
        pool=list(templates);rng.shuffle(pool)
        while len(selected)<max(1,bundle_size):selected.append(pool[len(selected)%len(pool)])
    clones=[_clone_fact(template,source_id=source_id,known_ids=known_fact_ids,counter=clone_counter) for template in selected]
    for clone in clones:world.add_fact(clone)
    metadata={"role":role,"source_type_origin":source_type_origin,"layout_position":len(world.sources),"bundle_origin":"independent_source_corroboration" if proof_corroboration else "source_count_nonproof_padding","semantic_conflict":False,"proof_corroboration":proof_corroboration,"evidence_version":"balanced","target_bundle_size":len(clones),"bundle_size":len(clones),"size_balance_padding":len(clones) if not proof_corroboration else 0,"size_balance_trimmed":0,"source_count_padding":not proof_corroboration}
    if corroborates_source_id is not None:metadata["corroborates_source_id"]=corroborates_source_id
    world.add_source(EvidenceSource(source_id,source_type,tuple(clone.fact_id for clone in clones),metadata));return source_id
def _background_templates(world):
    allocation=world.metadata.get("evidence_allocation",{});nonproof_ids=set(map(str,allocation.get("nonproof_source_ids",())));fact_map={fact.fact_id:fact for fact in world.facts};templates=[fact_map[fid] for sid in sorted(nonproof_ids) if sid in world.sources for fid in world.sources[sid].fact_ids if fid in fact_map];return templates
def _refresh_metadata(program,world,*,executor,min_critical_sources,min_noncritical_sources,target_source_count,source_count_weights,reserved_corroboration,realized_corroboration_source,source_count_padding_sources):
    proof=minimal_source_proof(program,world,executor=executor,max_sources=target_source_count,max_witnesses=16)
    if not proof.get("exhaustive") or not proof.get("witnesses"):raise EvidenceAllocationError("no exhaustive minimal proof after source-count balancing")
    witnesses=[list(map(str,w)) for w in proof.get("witnesses",())];proof_relevant={sid for witness in witnesses for sid in witness};indispensable_sources=indispensable_source_ids(program,world,executor=executor);indispensable_set=set(indispensable_sources);indispensable_facts=indispensable_fact_ids(program,world,executor=executor);indispensable_fact_set=set(indispensable_facts)
    if len(indispensable_sources)<min_critical_sources:raise EvidenceAllocationError(f"only {len(indispensable_sources)} indispensable sources after source-count balancing; need {min_critical_sources}")
    nonproof=[sid for sid in world.sources if sid not in proof_relevant]
    if len(nonproof)<min_noncritical_sources:raise EvidenceAllocationError(f"only {len(nonproof)} non-proof sources after source-count balancing; need {min_noncritical_sources}")
    bad=[sid for sid in proof_relevant if str(world.sources[sid].metadata.get("role","")) in DIAGNOSTIC_ROLES]
    if bad:raise EvidenceAllocationError(f"diagnostic sources entered a minimum proof after source-count balancing: {bad}")
    if len(world.sources)!=target_source_count:raise EvidenceAllocationError(f"source-count target mismatch: target={target_source_count}, realized={len(world.sources)}")
    role_counts=Counter();updated={}
    for sid,source in world.sources.items():role=str(source.metadata.get("role","unspecified"));role_counts[role]+=1;updated[sid]=EvidenceSource(source.source_id,source.source_type,source.fact_ids,{**dict(source.metadata),"proof_relevant":sid in proof_relevant,"indispensable":sid in indispensable_set,"contains_indispensable_fact":any(fid in indispensable_fact_set for fid in source.fact_ids),"bundle_size":len(source.fact_ids)})
    world.sources=updated;allocation=dict(world.metadata.get("evidence_allocation",{}));layout=dict(allocation.get("layout",{}));ordered=sorted(world.sources.values(),key=lambda source:(int(source.metadata.get("layout_position",10000)),source.source_id));layout["source_types"]=[source.source_type for source in ordered];layout["source_type_origins"]=[source.metadata.get("source_type_origin","unknown") for source in ordered];layout["target_source_count"]=target_source_count;layout["realized_source_count"]=len(world.sources);primary=sorted(sid for sid,source in world.sources.items() if str(source.metadata.get("role",""))=="critical");world.metadata={**world.metadata,"evidence_allocation":{**allocation,"layout":layout,"available_source_count":len(world.sources),"indispensable_source_ids":list(indispensable_sources),"critical_source_ids":list(indispensable_sources),"primary_proof_source_ids":primary,"proof_relevant_source_ids":sorted(proof_relevant),"nonproof_source_ids":sorted(nonproof),"source_roles":{sid:source.metadata.get("role","unspecified") for sid,source in world.sources.items()},"role_counts":dict(role_counts),"bundle_size_by_role":bundle_stats(world.sources),"minimal_proof":proof,"proof_witness_count":len(witnesses),"multiple_proof_witnesses":len(witnesses)>1,"indispensable_fact_count":len(indispensable_facts),"target_source_count":target_source_count,"realized_source_count":len(world.sources),"source_count_target_match":len(world.sources)==target_source_count,"source_count_padding_source_ids":list(source_count_padding_sources),"source_count_corroboration_reserved":reserved_corroboration,"source_count_corroboration_source_id":realized_corroboration_source,"policy":{**dict(allocation.get("policy",{})),"source_count_weights":[list(item) for item in source_count_weights],"answer_independent_source_count":True,"corroboration_slot_reservation":True}}};return world,list(indispensable_sources),list(indispensable_facts)
def _try_add_corroboration(program,world,*,rng,executor,min_critical_sources,min_noncritical_sources,target_source_count,synthetic_modality_probability,known_fact_ids,clone_counter):
    if len(world.sources)>=target_source_count:return world,None
    indispensable_sources=indispensable_source_ids(program,world,executor=executor)
    if len(indispensable_sources)<=min_critical_sources:return world,None
    indispensable_facts=set(indispensable_fact_ids(program,world,executor=executor));fact_map={fact.fact_id:fact for fact in world.facts};donors=[]
    for sid in indispensable_sources:
        templates=[fact_map[fid] for fid in world.sources[sid].fact_ids if fid in indispensable_facts and fid in fact_map]
        if templates:donors.append((len(templates),sid,templates))
    if not donors:return world,None
    donors.sort(key=lambda item:(item[0],item[1]));minimum=donors[0][0];_,donor_id,templates=rng.choice([item for item in donors if item[0]==minimum]);source_type,origin=_sample_added_modality(rng,synthetic_modality_probability=synthetic_modality_probability);candidate=_copy_world(world);corroborating=_append_source(candidate,templates=templates,source_type=source_type,source_type_origin=origin,role="corroborating",proof_corroboration=True,rng=rng,bundle_size=len(templates),known_fact_ids=known_fact_ids,clone_counter=clone_counter,corroborates_source_id=donor_id);proof=minimal_source_proof(program,candidate,executor=executor,max_sources=target_source_count,max_witnesses=16)
    if len(proof.get("witnesses",()))<2:return world,None
    if len(indispensable_source_ids(program,candidate,executor=executor))<min_critical_sources:return world,None
    proof_relevant={str(sid) for witness in proof.get("witnesses",()) for sid in witness};nonproof_count=sum(sid not in proof_relevant for sid in candidate.sources)
    if nonproof_count<min_noncritical_sources:return world,None
    return candidate,corroborating
def allocate_evidence_sources(program:ProgramIR,world:SyntheticWorld,*,min_critical_sources:int=2,min_noncritical_sources:int=1,max_sources:int=6,motif_prior:SourceMotifPrior|None=None,predicate_catalog:Mapping[str,Any]|None=None,mutation_records:Sequence[Mapping[str,Any]]|None=None,role_bundle_max_facts:int=3,critical_padding_max_facts:int=3,synthetic_modality_probability:float=.35,proof_corroboration_probability:float=.30,bundle_size_weights:Sequence[float]=DEFAULT_BUNDLE_SIZE_WEIGHTS,source_count_weights:Sequence[tuple[int,float]]=DEFAULT_SOURCE_COUNT_WEIGHTS,seed:int=17):
    if max_sources<min_critical_sources+min_noncritical_sources:raise ValueError("max_sources is too small for requested source roles")
    if not 0<=proof_corroboration_probability<=1:raise ValueError("proof_corroboration_probability must be in [0, 1]")
    rng=random.Random(seed^0x50A2CE);target_source_count=sample_target_source_count(min_sources=min_critical_sources+min_noncritical_sources,max_sources=max_sources,rng=rng,weights=source_count_weights);reserve=target_source_count>=min_critical_sources+min_noncritical_sources+2 and rng.random()<proof_corroboration_probability;base_max=target_source_count-1 if reserve else target_source_count;common=dict(min_critical_sources=min_critical_sources,min_noncritical_sources=min_noncritical_sources,motif_prior=motif_prior,predicate_catalog=predicate_catalog,mutation_records=mutation_records,role_bundle_max_facts=role_bundle_max_facts,critical_padding_max_facts=critical_padding_max_facts,synthetic_modality_probability=synthetic_modality_probability,bundle_size_weights=bundle_size_weights,seed=seed)
    try:base_world,_,_=_allocate_bundle_balanced_sources(program,world,max_sources=base_max,proof_corroboration_probability=0.,**common)
    except EvidenceAllocationError:
        if not reserve:raise
        reserve=False;base_world,_,_=_allocate_bundle_balanced_sources(program,world,max_sources=target_source_count,proof_corroboration_probability=0.,**common)
    executor=SymbolicExecutor();baseline_answer=normalize_answer(executor.execute(program,base_world));balanced=_copy_world(base_world);known_fact_ids={fact.fact_id for fact in balanced.facts};clone_counter=[0];realized_corroboration=None
    if reserve:balanced,realized_corroboration=_try_add_corroboration(program,balanced,rng=rng,executor=executor,min_critical_sources=min_critical_sources,min_noncritical_sources=min_noncritical_sources,target_source_count=target_source_count,synthetic_modality_probability=synthetic_modality_probability,known_fact_ids=known_fact_ids,clone_counter=clone_counter)
    padding_sources=[]
    while len(balanced.sources)<target_source_count:
        templates=_background_templates(balanced)
        if not templates:raise EvidenceAllocationError("no non-proof templates available for source-count padding")
        source_id=_next_source_id(balanced);target_size=sample_target_bundle_sizes([source_id],max_bundle_size=critical_padding_max_facts,rng=rng,weights=bundle_size_weights)[source_id];source_type,origin=_sample_added_modality(rng,synthetic_modality_probability=synthetic_modality_probability);padding_sources.append(_append_source(balanced,templates=templates,source_type=source_type,source_type_origin=origin,role="redundant",proof_corroboration=False,rng=rng,bundle_size=target_size,known_fact_ids=known_fact_ids,clone_counter=clone_counter))
    if normalize_answer(executor.execute(program,balanced))!=baseline_answer:raise EvidenceAllocationError("source-count balancing changed the oracle answer")
    return _refresh_metadata(program,balanced,executor=executor,min_critical_sources=min_critical_sources,min_noncritical_sources=min_noncritical_sources,target_source_count=target_source_count,source_count_weights=source_count_weights,reserved_corroboration=reserve,realized_corroboration_source=realized_corroboration,source_count_padding_sources=padding_sources)
