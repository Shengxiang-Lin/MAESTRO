"""Heterogeneous evidence allocation with proof/distractor separation."""
from __future__ import annotations
import random,re
from collections import Counter,defaultdict
from dataclasses import replace
from datetime import date,datetime
from typing import Any,Mapping,Sequence
from ..executor import SymbolicExecutor
from ..ir import ProgramIR
from ..source_motifs import SourceMotifPrior
from ..validation import indispensable_fact_ids,indispensable_source_ids,minimal_source_proof
from ..world import EvidenceSource,Fact,SyntheticWorld
from .semantics import EvidenceAllocationError,classify_noncritical_facts,select_role_bundles
STANDARD_MODALITIES=("text","table","image_report");SYNTHETIC_MODALITIES=("video_report","audio_transcript","database");DIAGNOSTIC_ROLES={"conflicting","competing","near_miss","irrelevant","redundant"}
def _value_year(value):
    if isinstance(value,(date,datetime)):return int(value.year)
    if isinstance(value,bool):return None
    if isinstance(value,int):return value if 1000<=value<=9999 else None
    match=re.match(r"^(\d{4})(?:\D|$)",str(value).strip())
    if not match:return None
    try:return int(match.group(1))
    except ValueError:return None
def temporal_granularity_compatible(left:Fact,right:Fact)->bool:
    if {left.object_kind,right.object_kind}-{"year","date"}:return False
    ly=_value_year(left.object);ry=_value_year(right.object);return ly is not None and ly==ry
_temporal_granularity_compatible=temporal_granularity_compatible
def normalize_temporal_conflicts(buckets:Mapping[str,Sequence[Fact]],facts:Sequence[Fact],critical_fact_ids:Sequence[str])->dict[str,list[Fact]]:
    result={role:list(items) for role,items in buckets.items()};conflicting=list(result.get("conflicting",()))
    if not conflicting:return result
    critical_set=set(critical_fact_ids);by=defaultdict(list)
    for fact in facts:
        if fact.fact_id in critical_set:by[(fact.subject,fact.predicate)].append(fact)
    keep=[]
    for fact in conflicting:
        refs=by.get((fact.subject,fact.predicate),())
        if refs and any(temporal_granularity_compatible(fact,ref) for ref in refs):result.setdefault("corroborating",[]).append(fact)
        else:keep.append(fact)
    if keep:result["conflicting"]=keep
    else:result.pop("conflicting",None)
    return result
_normalize_temporal_conflicts=normalize_temporal_conflicts
def _next_augmented_fact_id(prefix,known,counter):
    while True:
        counter[0]+=1;fact_id=f"{prefix}{counter[0]:05d}"
        if fact_id not in known:known.add(fact_id);return fact_id
def _clone_facts(facts,*,prefix,known_ids,counter):return [replace(fact,fact_id=_next_augmented_fact_id(prefix,known_ids,counter),source_id=None) for fact in facts]
def _sample_modality_layout(source_count,*,motif_prior,rng,synthetic_modality_probability):
    motif=motif_prior.sample(source_count,rng=rng,prefer_cross_modal=True);modalities=list(motif.source_types);origins=[motif.origin for _ in modalities];synthetic_cap=max(1,source_count//2);synthetic_count=0;order=list(range(source_count));rng.shuffle(order)
    for index in order:
        if synthetic_count>=synthetic_cap:break
        if rng.random()<synthetic_modality_probability:modalities[index]=rng.choice(SYNTHETIC_MODALITIES);origins[index]="synthetic_modality_augmentation";synthetic_count+=1
    if source_count>=4 and synthetic_count==0 and synthetic_modality_probability>0 and rng.random()<.75:index=rng.randrange(source_count);modalities[index]=rng.choice(SYNTHETIC_MODALITIES);origins[index]="synthetic_modality_augmentation";synthetic_count=1
    return modalities,origins,{"base_motif":motif.to_dict(),"source_types":list(modalities),"source_type_origins":list(origins),"synthetic_modality_count":synthetic_count}
def _balance_modality_overlap(modalities,origins,*,proof_indexes,role_indexes,rng):
    if not proof_indexes or not role_indexes:return
    def is_synthetic(index):return modalities[index] in SYNTHETIC_MODALITIES
    proof_synth=[i for i in proof_indexes if is_synthetic(i)];role_synth=[i for i in role_indexes if is_synthetic(i)];all_synth=proof_synth+role_synth
    if len(all_synth)>=2:
        if not proof_synth:donor=rng.choice(role_synth);receiver=rng.choice(list(proof_indexes));modalities[donor],modalities[receiver]=modalities[receiver],modalities[donor];origins[donor],origins[receiver]=origins[receiver],origins[donor]
        elif not role_synth:donor=rng.choice(proof_synth);receiver=rng.choice(list(role_indexes));modalities[donor],modalities[receiver]=modalities[receiver],modalities[donor];origins[donor],origins[receiver]=origins[receiver],origins[donor]
def bundle_stats(sources):
    sizes=defaultdict(list)
    for source in sources.values():sizes[str(source.metadata.get("role","unspecified"))].append(len(source.fact_ids))
    return {role:{"sources":len(v),"total_facts":sum(v),"min":min(v),"max":max(v),"mean":sum(v)/len(v)} for role,v in sizes.items()}
_bundle_stats=bundle_stats
def allocate_structured_evidence_sources(program:ProgramIR,world:SyntheticWorld,*,min_critical_sources:int=2,min_noncritical_sources:int=1,max_sources:int=6,motif_prior:SourceMotifPrior|None=None,predicate_catalog:Mapping[str,Any]|None=None,mutation_records:Sequence[Mapping[str,Any]]|None=None,role_bundle_max_facts:int=3,critical_padding_max_facts:int=3,synthetic_modality_probability:float=.35,proof_corroboration_probability:float=.30,seed:int=17):
    if min_critical_sources<1:raise ValueError("min_critical_sources must be >= 1")
    if min_noncritical_sources<0:raise ValueError("min_noncritical_sources must be >= 0")
    if max_sources<min_critical_sources+min_noncritical_sources:raise ValueError("max_sources is too small for requested source roles")
    if role_bundle_max_facts<1 or critical_padding_max_facts<1:raise ValueError("source bundle limits must be >= 1")
    if not 0<=synthetic_modality_probability<=1 or not 0<=proof_corroboration_probability<=1:raise ValueError("probabilities must be in [0,1]")
    executor=SymbolicExecutor();rng=random.Random(seed);motif_prior=motif_prior or SourceMotifPrior.fallback();baseline_indispensable_facts=indispensable_fact_ids(program,world,executor=executor)
    if len(baseline_indispensable_facts)<min_critical_sources:raise EvidenceAllocationError(f"only {len(baseline_indispensable_facts)} indispensable facts; need {min_critical_sources}")
    baseline_critical_set=set(baseline_indispensable_facts);noncritical_facts=[fact for fact in world.facts if fact.fact_id not in baseline_critical_set]
    if len(noncritical_facts)<min_noncritical_sources:raise EvidenceAllocationError(f"only {len(noncritical_facts)} non-critical facts; need {min_noncritical_sources}")
    reserved_role_sources=min(min_noncritical_sources,len(noncritical_facts));core_capacity=max_sources-reserved_role_sources;core_count=max(min_critical_sources,min(4,len(baseline_indispensable_facts),core_capacity))
    if core_count>core_capacity:raise EvidenceAllocationError("cannot reserve non-proof sources within max_sources")
    available_role_slots=max_sources-core_count;desired_role_sources=min(available_role_slots,max(min_noncritical_sources,min(3,max(1,(len(noncritical_facts)+1)//2))));total_source_count=core_count+desired_role_sources;modalities,modality_origins,layout_meta=_sample_modality_layout(total_source_count,motif_prior=motif_prior,rng=rng,synthetic_modality_probability=synthetic_modality_probability);slot_indexes=list(range(total_source_count));proof_indexes=sorted(rng.sample(slot_indexes,core_count));proof_index_set=set(proof_indexes);role_indexes=[i for i in slot_indexes if i not in proof_index_set];_balance_modality_overlap(modalities,modality_origins,proof_indexes=proof_indexes,role_indexes=role_indexes,rng=rng);source_ids=[f"source_{i+1:02d}" for i in slot_indexes];core_source_ids=[source_ids[i] for i in proof_indexes];role_source_ids=[source_ids[i] for i in role_indexes]
    assignments={};grouped=defaultdict(list);critical=list(baseline_indispensable_facts);rng.shuffle(critical)
    for index,fact_id in enumerate(critical):source_id=core_source_ids[index%core_count];assignments[fact_id]=source_id;grouped[source_id].append(fact_id)
    buckets=normalize_temporal_conflicts(classify_noncritical_facts(world.facts,baseline_indispensable_facts,predicate_catalog=predicate_catalog),world.facts,baseline_indispensable_facts);proof_corroboration_source=proof_corroborated_fact=None;single_fact_core=[sid for sid in core_source_ids if len(grouped[sid])==1];can_corroborate=len(role_source_ids)>min_noncritical_sources and core_count>=3 and len(baseline_indispensable_facts)>=3 and bool(single_fact_core)
    if can_corroborate and rng.random()<proof_corroboration_probability:proof_corroboration_source=rng.choice(role_source_ids);source_with_claim=rng.choice(single_fact_core);proof_corroborated_fact=grouped[source_with_claim][0]
    nonproof_role_ids=[sid for sid in role_source_ids if sid!=proof_corroboration_source];bundles=select_role_bundles(buckets,mutation_records=mutation_records,limit=len(nonproof_role_ids),rng=rng,max_bundle_size=role_bundle_max_facts)
    if len(bundles)<min_noncritical_sources:raise EvidenceAllocationError(f"only {len(bundles)} explicit distractor bundles; need {min_noncritical_sources}")
    augmented_facts=[];known_fact_ids={fact.fact_id for fact in world.facts};augmented_counter=[0]
    if len(bundles)>=2 and rng.random()<.30:
        template=max(bundles[:-1] or bundles,key=lambda item:len(item["facts"]));clones=_clone_facts(template["facts"],prefix="rf",known_ids=known_fact_ids,counter=augmented_counter);augmented_facts.extend(clones);bundles[-1]={"role":"corroborating" if rng.random()<.5 else "redundant","facts":clones,"origin":"repeated_noncritical_bundle"}
    role_meta={};role_fact_ids=set()
    for source_id,bundle in zip(nonproof_role_ids,bundles):
        role=str(bundle["role"])
        for fact in bundle["facts"]:role_fact_ids.add(fact.fact_id);assignments[fact.fact_id]=source_id;grouped[source_id].append(fact.fact_id)
        role_meta[source_id]={"role":role,"bundle_origin":bundle["origin"],"semantic_conflict":role=="conflicting","proof_corroboration":False}
    if proof_corroboration_source is not None and proof_corroborated_fact is not None:
        original=next(fact for fact in world.facts if fact.fact_id==proof_corroborated_fact);clone=_clone_facts([original],prefix="pf",known_ids=known_fact_ids,counter=augmented_counter)[0];augmented_facts.append(clone);assignments[clone.fact_id]=proof_corroboration_source;grouped[proof_corroboration_source].append(clone.fact_id);role_meta[proof_corroboration_source]={"role":"corroborating","bundle_origin":"independent_proof_corroboration","semantic_conflict":False,"proof_corroboration":True,"corroborates_fact_id":proof_corroborated_fact}
    original_noncritical_ids={f.fact_id for f in noncritical_facts};used={fid for fid in role_fact_ids if fid in original_noncritical_ids};remaining=[f.fact_id for f in noncritical_facts if f.fact_id not in used];rng.shuffle(remaining);active_source_ids=list(core_source_ids)+[sid for sid in role_source_ids if sid in role_meta]
    if not active_source_ids:raise EvidenceAllocationError("evidence allocation produced no sources")
    target_sizes={sid:rng.choice(list(range(1,critical_padding_max_facts+1))+list(range(2,critical_padding_max_facts+1))) for sid in active_source_ids};padding_order=list(active_source_ids);rng.shuffle(padding_order)
    for sid in padding_order:
        while remaining and len(grouped[sid])<target_sizes[sid]:fid=remaining.pop();assignments[fid]=sid;grouped[sid].append(fid)
    while remaining:sid=min(active_source_ids,key=lambda s:len(grouped[s]));fid=remaining.pop();assignments[fid]=sid;grouped[sid].append(fid)
    all_facts=list(world.facts)+augmented_facts;result=SyntheticWorld(dict(world.entities),[replace(fact,source_id=assignments.get(fact.fact_id)) for fact in all_facts],{}, {**world.metadata,"evidence_allocated":True});slot_for_source={source_ids[i]:i for i in slot_indexes};core_source_set=set(core_source_ids)
    for sid in active_source_ids:
        slot=slot_for_source[sid];is_core=sid in core_source_set;metadata={"role":"critical" if is_core else role_meta[sid]["role"],"source_type_origin":modality_origins[slot],"layout_position":slot,"bundle_size":len(grouped[sid]),"proof_corroboration":bool(role_meta.get(sid,{}).get("proof_corroboration",False))}
        if not is_core:
            metadata.update({"bundle_origin":role_meta[sid]["bundle_origin"],"semantic_conflict":role_meta[sid]["semantic_conflict"]})
            if role_meta[sid].get("corroborates_fact_id"):metadata["corroborates_fact_id"]=role_meta[sid]["corroborates_fact_id"]
        result.add_source(EvidenceSource(sid,modalities[slot],tuple(grouped[sid]),metadata))
    proof=minimal_source_proof(program,result,executor=executor,max_sources=max_sources,max_witnesses=16)
    if not proof.get("exhaustive") or not proof.get("witnesses"):raise EvidenceAllocationError("no exhaustive minimal source proof found")
    witnesses=[list(map(str,w)) for w in proof.get("witnesses",[])];proof_relevant={sid for witness in witnesses for sid in witness};indispensable_sources=indispensable_source_ids(program,result,executor=executor);indispensable_set=set(indispensable_sources);post_indispensable_facts=indispensable_fact_ids(program,result,executor=executor)
    if len(indispensable_sources)<min_critical_sources:raise EvidenceAllocationError(f"only {len(indispensable_sources)} indispensable sources after corroboration; need {min_critical_sources}")
    nonproof_sources=[sid for sid in result.sources if sid not in proof_relevant]
    if len(nonproof_sources)<min_noncritical_sources:raise EvidenceAllocationError(f"only {len(nonproof_sources)} non-proof sources; need {min_noncritical_sources}")
    if any(str(result.sources[sid].metadata.get("role",""))=="critical" for sid in nonproof_sources):raise EvidenceAllocationError("a core proof source became non-proof")
    bad=[sid for sid in proof_relevant if str(result.sources[sid].metadata.get("role","")) in DIAGNOSTIC_ROLES]
    if bad:raise EvidenceAllocationError(f"diagnostic distractor sources entered a minimal proof: {bad}")
    if proof_corroboration_source is not None and (proof_corroboration_source not in proof_relevant or len(witnesses)<2):raise EvidenceAllocationError("requested proof corroboration did not create alternate minimum witnesses")
    post_set=set(post_indispensable_facts);updated={};role_counts=Counter()
    for sid,source in result.sources.items():
        role=str(source.metadata.get("role","unspecified"));role_counts[role]+=1;metadata={**dict(source.metadata),"proof_relevant":sid in proof_relevant,"indispensable":sid in indispensable_set,"contains_indispensable_fact":any(fid in post_set for fid in source.fact_ids),"bundle_size":len(source.fact_ids)};updated[sid]=EvidenceSource(source.source_id,source.source_type,source.fact_ids,metadata)
    result.sources=updated;result.metadata={**result.metadata,"evidence_allocation":{"layout":layout_meta,"motif_prior":motif_prior.summary(),"indispensable_source_ids":list(indispensable_sources),"critical_source_ids":list(indispensable_sources),"proof_relevant_source_ids":sorted(proof_relevant),"nonproof_source_ids":sorted(nonproof_sources),"source_roles":{sid:source.metadata.get("role","unspecified") for sid,source in result.sources.items()},"role_counts":dict(role_counts),"bundle_size_by_role":bundle_stats(result.sources),"available_source_count":len(result.sources),"baseline_indispensable_fact_count":len(baseline_indispensable_facts),"indispensable_fact_count":len(post_indispensable_facts),"proof_witness_count":len(witnesses),"multiple_proof_witnesses":len(witnesses)>1,"minimal_proof":proof,"proof_corroboration_source_id":proof_corroboration_source,"proof_corroborated_fact_id":proof_corroborated_fact,"policy":{"synthetic_modality_probability":synthetic_modality_probability,"proof_corroboration_probability":proof_corroboration_probability,"role_bundle_max_facts":role_bundle_max_facts,"critical_padding_max_facts":critical_padding_max_facts}}};return result,list(indispensable_sources),list(post_indispensable_facts)
