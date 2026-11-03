"""Role-independent source-size balancing for the evidence environment."""
from __future__ import annotations
import random
from collections import Counter
from dataclasses import replace
from typing import Any,Mapping,Sequence
from ..executor import ExecutionError,SymbolicExecutor
from ..ir import ProgramIR
from ..scenario.information import assess_goal_information_completeness
from ..source_motifs import SourceMotifPrior
from ..validation import indispensable_fact_ids,indispensable_source_ids,minimal_source_proof,normalize_answer
from ..world import EvidenceSource,Fact,SyntheticWorld
from .allocation import DIAGNOSTIC_ROLES,allocate_structured_evidence_sources,bundle_stats
from .semantics import EvidenceAllocationError
DEFAULT_BUNDLE_SIZE_WEIGHTS=(.30,.35,.35)
def sample_target_bundle_sizes(source_ids:Sequence[str],*,max_bundle_size:int,rng:random.Random,weights:Sequence[float]=DEFAULT_BUNDLE_SIZE_WEIGHTS)->dict[str,int]:
    if max_bundle_size<1:raise ValueError("max_bundle_size must be >= 1")
    sizes=list(range(1,max_bundle_size+1));raw=[max(0.,float(v)) for v in weights[:max_bundle_size]] if len(weights)>=max_bundle_size else [1. for _ in sizes]
    if not any(raw):raw=[1. for _ in sizes]
    return {source_id:int(rng.choices(sizes,weights=raw,k=1)[0]) for source_id in source_ids}
_sample_target_bundle_sizes=sample_target_bundle_sizes
def _next_fact_id(known,counter):
    while True:
        counter[0]+=1;fact_id=f"bf{counter[0]:05d}"
        if fact_id not in known:known.add(fact_id);return fact_id
def _clone_background_fact(template,*,source_id,known,counter):return replace(template,fact_id=_next_fact_id(known,counter),source_id=source_id)
def _rebuild_world(base_world,*,kept_fact_ids,added_facts,source_fact_ids,source_metadata):
    facts=[fact for fact in base_world.facts if fact.fact_id in kept_fact_ids];facts.extend(added_facts);by_id={fact.fact_id:fact for fact in facts};result=SyntheticWorld(dict(base_world.entities),list(facts),{},dict(base_world.metadata))
    for source_id,base_source in base_world.sources.items():result.add_source(EvidenceSource(source_id,base_source.source_type,tuple(fid for fid in source_fact_ids[source_id] if fid in by_id),dict(source_metadata[source_id])))
    return result
def allocate_evidence_sources(program:ProgramIR,world:SyntheticWorld,*,min_critical_sources:int=2,min_noncritical_sources:int=1,max_sources:int=6,motif_prior:SourceMotifPrior|None=None,predicate_catalog:Mapping[str,Any]|None=None,mutation_records:Sequence[Mapping[str,Any]]|None=None,role_bundle_max_facts:int=3,critical_padding_max_facts:int=3,synthetic_modality_probability:float=.35,proof_corroboration_probability:float=.30,bundle_size_weights:Sequence[float]=DEFAULT_BUNDLE_SIZE_WEIGHTS,seed:int=17):
    if critical_padding_max_facts<1:raise ValueError("critical_padding_max_facts must be >= 1")
    base_world,_,base_indispensable_facts=allocate_structured_evidence_sources(program,world,min_critical_sources=min_critical_sources,min_noncritical_sources=min_noncritical_sources,max_sources=max_sources,motif_prior=motif_prior,predicate_catalog=predicate_catalog,mutation_records=mutation_records,role_bundle_max_facts=role_bundle_max_facts,critical_padding_max_facts=1,synthetic_modality_probability=synthetic_modality_probability,proof_corroboration_probability=proof_corroboration_probability,seed=seed)
    executor=SymbolicExecutor();baseline_answer=normalize_answer(executor.execute(program,base_world));baseline_information=assess_goal_information_completeness(program,base_world,executor=executor)
    if not baseline_information.accepted:raise EvidenceAllocationError("structured evidence is information-incomplete before size balancing")
    decision_support_fact_ids=set(baseline_information.required_fact_ids);allocation=dict(base_world.metadata.get("evidence_allocation",{}));base_nonproof=set(allocation.get("nonproof_source_ids",()));source_ids=sorted(base_world.sources);rng=random.Random(seed^0x26A61);target_sizes=sample_target_bundle_sizes(source_ids,max_bundle_size=critical_padding_max_facts,rng=rng,weights=bundle_size_weights);protected_fact_ids=set(base_indispensable_facts)|decision_support_fact_ids;corroborated=allocation.get("proof_corroborated_fact_id")
    if corroborated:protected_fact_ids.add(str(corroborated))
    for source in base_world.sources.values():
        if bool(source.metadata.get("proof_corroboration",False)):protected_fact_ids.update(fid for fid in source.fact_ids if str(fid).startswith("pf"))
    for source_id,source in base_world.sources.items():
        if str(source.metadata.get("role",""))=="conflicting":target_sizes[source_id]=max(2,target_sizes[source_id])
    fact_map={fact.fact_id:fact for fact in base_world.facts};source_fact_ids={sid:list(source.fact_ids) for sid,source in base_world.sources.items()};source_metadata={sid:dict(source.metadata) for sid,source in base_world.sources.items()};kept_fact_ids=set(fact_map);dropped=[]
    for source_id in source_ids:
        current=list(source_fact_ids[source_id]);protected=[fid for fid in current if fid in protected_fact_ids];optional=[fid for fid in current if fid not in protected_fact_ids];target=max(target_sizes[source_id],len(protected),1);keep_optional=max(0,target-len(protected));rng.shuffle(optional);kept_optional=optional[:keep_optional];removed=optional[keep_optional:];source_fact_ids[source_id]=protected+kept_optional
        for fact_id in removed:kept_fact_ids.discard(fact_id);dropped.append(fact_id)
    background_templates=[fact_map[fid] for sid in sorted(base_nonproof) for fid in base_world.sources[sid].fact_ids if fid in fact_map and fid not in protected_fact_ids]
    if not background_templates:background_templates=[fact_map[fid] for fid in sorted(set(fact_map)-protected_fact_ids)]
    added=[];known_ids=set(fact_map);clone_counter=[0];padding_by_source=Counter()
    if background_templates:
        padding_order=list(source_ids);rng.shuffle(padding_order)
        for source_id in padding_order:
            while len(source_fact_ids[source_id])<target_sizes[source_id]:template=rng.choice(background_templates);clone=_clone_background_fact(template,source_id=source_id,known=known_ids,counter=clone_counter);added.append(clone);source_fact_ids[source_id].append(clone.fact_id);padding_by_source[source_id]+=1
    for source_id in source_ids:source_metadata[source_id].update({"evidence_version":"balanced","target_bundle_size":target_sizes[source_id],"bundle_size":len(source_fact_ids[source_id]),"size_balance_padding":padding_by_source[source_id],"size_balance_trimmed":len(base_world.sources[source_id].fact_ids)-len([fid for fid in source_fact_ids[source_id] if fid in fact_map])})
    balanced=_rebuild_world(base_world,kept_fact_ids=kept_fact_ids,added_facts=added,source_fact_ids=source_fact_ids,source_metadata=source_metadata)
    try:answer=normalize_answer(executor.execute(program,balanced))
    except (ExecutionError,ValueError,TypeError) as exc:raise EvidenceAllocationError(f"size-balanced world is not executable: {exc}") from exc
    if answer!=baseline_answer:raise EvidenceAllocationError("source-size balancing changed the oracle answer")
    balanced_information=assess_goal_information_completeness(program,balanced,executor=executor)
    if not balanced_information.accepted:raise EvidenceAllocationError("source-size balancing violated decision-information completeness")
    proof=minimal_source_proof(program,balanced,executor=executor,max_sources=max_sources,max_witnesses=16)
    if not proof.get("exhaustive") or not proof.get("witnesses"):raise EvidenceAllocationError("no exhaustive minimal proof after size balancing")
    witnesses=[list(map(str,w)) for w in proof.get("witnesses",())];proof_relevant={sid for witness in witnesses for sid in witness};indispensable_sources=indispensable_source_ids(program,balanced,executor=executor);indispensable_set=set(indispensable_sources);post_indispensable_facts=indispensable_fact_ids(program,balanced,executor=executor);post_set=set(post_indispensable_facts)
    if len(indispensable_sources)<min_critical_sources:raise EvidenceAllocationError(f"only {len(indispensable_sources)} indispensable sources after size balancing; need {min_critical_sources}")
    nonproof_sources=[sid for sid in balanced.sources if sid not in proof_relevant]
    if len(nonproof_sources)<min_noncritical_sources:raise EvidenceAllocationError(f"only {len(nonproof_sources)} non-proof sources after size balancing; need {min_noncritical_sources}")
    bad=[sid for sid in proof_relevant if str(balanced.sources[sid].metadata.get("role","")) in DIAGNOSTIC_ROLES]
    if bad:raise EvidenceAllocationError(f"diagnostic sources entered a minimum proof after balancing: {bad}")
    base_witness_count=int(allocation.get("proof_witness_count",0) or 0)
    if base_witness_count>1 and len(witnesses)<2:raise EvidenceAllocationError("size balancing destroyed alternate proof witnesses")
    updated={};role_counts=Counter()
    for sid,source in balanced.sources.items():
        role=str(source.metadata.get("role","unspecified"));role_counts[role]+=1;metadata={**dict(source.metadata),"proof_relevant":sid in proof_relevant,"indispensable":sid in indispensable_set,"contains_indispensable_fact":any(fid in post_set for fid in source.fact_ids),"bundle_size":len(source.fact_ids)};updated[sid]=EvidenceSource(source.source_id,source.source_type,source.fact_ids,metadata)
    balanced.sources=updated;primary=sorted(sid for sid,source in balanced.sources.items() if str(source.metadata.get("role",""))=="critical");realized={sid:len(source.fact_ids) for sid,source in balanced.sources.items()};match=sum(realized[sid]==target_sizes[sid] for sid in source_ids);balanced.metadata={**balanced.metadata,"evidence_allocation":{**allocation,"evidence_version":"balanced","indispensable_source_ids":list(indispensable_sources),"critical_source_ids":list(indispensable_sources),"primary_proof_source_ids":primary,"proof_relevant_source_ids":sorted(proof_relevant),"nonproof_source_ids":sorted(nonproof_sources),"source_roles":{sid:source.metadata.get("role","unspecified") for sid,source in balanced.sources.items()},"role_counts":dict(role_counts),"bundle_size_by_role":bundle_stats(balanced.sources),"minimal_proof":proof,"proof_witness_count":len(witnesses),"multiple_proof_witnesses":len(witnesses)>1,"indispensable_fact_count":len(post_indispensable_facts),"decision_support_fact_count":len(decision_support_fact_ids),"size_balance_protected_fact_count":len(protected_fact_ids),"target_bundle_sizes":target_sizes,"realized_bundle_sizes":realized,"bundle_size_target_match_rate":match/len(source_ids) if source_ids else 0.,"size_balance_padding_fact_count":len(added),"size_balance_trimmed_fact_count":len(dropped),"policy":{**dict(allocation.get("policy",{})),"bundle_size_weights":list(bundle_size_weights),"bundle_size_max":critical_padding_max_facts,"role_independent_bundle_targets":True,"preserve_decision_support_facts":True}}};return balanced,list(indispensable_sources),list(post_indispensable_facts)
