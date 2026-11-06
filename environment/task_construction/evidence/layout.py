"""Role-independent visible layout planning for public evidence."""
from __future__ import annotations
import math,random
from typing import Any,Mapping,Sequence
from ..executor import SymbolicExecutor
from ..ir import ProgramIR
from ..source_motifs import SourceMotifPrior
from ..validation import normalize_answer
from ..world import EvidenceSource,Fact,SyntheticWorld
from .allocation import SYNTHETIC_MODALITIES
from .balancing import DEFAULT_BUNDLE_SIZE_WEIGHTS
from .capacity_balancing import rebalance_protected_facts
from .count_balancing import allocate_evidence_sources as _allocate_source_count_balanced
from .semantics import EvidenceAllocationError
from .source_count import DEFAULT_SOURCE_COUNT_WEIGHTS,sample_target_source_count
DEFAULT_SOURCE_COUNT_CORROBORATION_PROBABILITY=.40;DEFAULT_BUNDLE_TARGET_MAX_FACTS=9;_PADDING_PREDICATE_PREFIX="o3:contextDescriptor";_DEFAULT_SURFACE_STYLE={"text":"document_excerpt","table":"structured_table","image_report":"visual_observation","video_report":"temporal_inspection_log","audio_transcript":"speaker_transcript","database":"structured_record"}
def _plan_bundle_targets(source_count,*,fact_count,max_bundle_size,seed,weights=DEFAULT_BUNDLE_SIZE_WEIGHTS):
    if source_count<1 or max_bundle_size<1:raise ValueError("source_count/max_bundle_size must be >= 1")
    nominal=max(1,min(max_bundle_size,int(math.ceil(max(1,fact_count)/source_count))));active_max=min(max_bundle_size,max(nominal+2,min(4,max_bundle_size)));sizes=list(range(1,active_max+1));raw=[max(0.,float(v)) for v in weights[:active_max]] if len(weights)>=active_max else [1./(1.+abs(size-nominal)) for size in sizes]
    if not any(raw):raw=[1. for _ in sizes]
    rng=random.Random(seed^0xB61D1E);targets=[int(rng.choices(sizes,weights=raw,k=1)[0]) for _ in range(source_count)]
    if source_count>=4 and active_max>=4 and max(targets)<4:targets[rng.randrange(source_count)]=4
    return targets
_MEDIA_SOURCE_TYPES = frozenset(("image", "image_report", "video",
                                "video_report", "audio", "audio_transcript"))
_ACQUISITION_SCENARIOS = ("web_search", "hybrid", "provided_media")
_DEFAULT_SCENARIO_WEIGHTS = (45, 40, 15)


def _plan_acquisition_scenario(seed, weights=_DEFAULT_SCENARIO_WEIGHTS):
    """Choose how information is acquired BEFORE sampling source modalities.

    Modality and initial availability are independent: web search may return
    videos/images, not only HTML pages. The weights affect *proposals*, not
    the final quality-filtered distribution.
    """
    if len(weights) != 3 or any(float(w) < 0 for w in weights) or not sum(weights):
        raise ValueError("acquisition_scenario_weights must contain three nonnegative values with positive sum")
    rng = random.Random(seed ^ 0x5EAC70)
    return rng.choices(_ACQUISITION_SCENARIOS, weights=weights, k=1)[0]


def _plan_modality_layout(
    source_count, *, motif_prior, synthetic_modality_probability, seed,
    acquisition_scenario_weights=_DEFAULT_SCENARIO_WEIGHTS,
):
    scenario = _plan_acquisition_scenario(seed, acquisition_scenario_weights)
    rng = random.Random(seed ^ 0xF1A14E)
    motif = motif_prior.sample(source_count, rng=rng, prefer_cross_modal=scenario != "provided_media")
    modalities = list(motif.source_types)
    if len(modalities) != source_count:
        raise ValueError("source motif slot count mismatch")
    origins = [motif.origin for _ in modalities]
    synthetic_cap = max(1, source_count // 2)
    synthetic_count = sum(v in SYNTHETIC_MODALITIES for v in modalities)
    order = list(range(source_count))
    rng.shuffle(order)
    for index in order:
        if synthetic_count >= synthetic_cap:
            break
        if modalities[index] in SYNTHETIC_MODALITIES:
            continue
        if rng.random() < synthetic_modality_probability:
            modalities[index] = rng.choice(SYNTHETIC_MODALITIES)
            origins[index] = "final_layout_synthetic_augmentation"
            synthetic_count += 1
    if (source_count >= 4 and synthetic_count == 0
            and synthetic_modality_probability > 0 and rng.random() < .75):
        index = rng.randrange(source_count)
        modalities[index] = rng.choice(SYNTHETIC_MODALITIES)
        origins[index] = "final_layout_synthetic_augmentation"
        synthetic_count = 1

    # Scenario-specific constraints are applied at this *same upstream stage*,
    # before the world has materialized source roles, facts or visibility.
    # Web-search worlds may contain every media type: the media is DISCOVERED,
    # not assumed to be an initial attachment.
    if scenario in {"hybrid", "provided_media"} and not any(
        typ in _MEDIA_SOURCE_TYPES for typ in modalities
    ):
        index = rng.randrange(source_count)
        modalities[index] = "image_report"
        origins[index] = "acquisition_scenario_media_entry"
    if scenario == "hybrid" and source_count > 1 and all(
        typ in _MEDIA_SOURCE_TYPES for typ in modalities
    ):
        index = rng.randrange(source_count)
        modalities[index] = "text"
        origins[index] = "acquisition_scenario_search_document"
    if scenario == "provided_media":
        # A media-centric task begins with a real media input. Other media
        # sources may still require discovery/inspection across later waves.
        for index, typ in enumerate(modalities):
            if typ not in _MEDIA_SOURCE_TYPES:
                modalities[index] = "image_report"
                origins[index] = "acquisition_scenario_media_only"

    slot_order = list(range(source_count))
    rng.shuffle(slot_order)
    return modalities, origins, slot_order, {
        "base_motif": motif.to_dict(),
        "synthetic_modality_count": synthetic_count,
        "planned_before_roles": True,
        "acquisition_scenario": scenario,
        "acquisition_scenario_weights": list(acquisition_scenario_weights),
        "scenario_planned_before_source_modalities": True,
    }


def _final_material_metadata(*,slot,source_type,plan_meta):
    motif=plan_meta.get("base_motif",{})
    if not isinstance(motif,Mapping):return {}
    roles=motif.get("artifact_roles",());styles=motif.get("surface_styles",());base_types=motif.get("source_types",());roles=roles if isinstance(roles,(list,tuple)) else ();styles=styles if isinstance(styles,(list,tuple)) else ();base_types=base_types if isinstance(base_types,(list,tuple)) else ();result={}
    if 0<=slot<len(roles) and str(roles[slot]).strip():result["material_role"]=str(roles[slot])
    base_type=str(base_types[slot]) if 0<=slot<len(base_types) else None
    if base_type is not None:result["surface_style"]=str(styles[slot]) if base_type==source_type and slot<len(styles) and str(styles[slot]).strip() else _DEFAULT_SURFACE_STYLE.get(source_type,"document_excerpt")
    origin=motif.get("origin")
    if origin is not None and str(origin).strip():result["scenario_prior_origin"]=str(origin)
    return result
def _apply_planned_layout(world,*,modalities,origins,slot_order,plan_meta):
    source_ids=sorted(world.sources)
    if len(source_ids)!=len(modalities) or len(source_ids)!=len(slot_order):raise ValueError("planned modality layout does not match realized source count")
    updated={}
    for source_id,slot in zip(source_ids,slot_order):
        source=world.sources[source_id];final_type=str(modalities[slot]);metadata={**dict(source.metadata),**_final_material_metadata(slot=int(slot),source_type=final_type,plan_meta=plan_meta),"layout_position":int(slot),"source_type_origin":str(origins[slot]),"final_modality_planned_before_roles":True};updated[source_id]=EvidenceSource(source.source_id,final_type,source.fact_ids,metadata)
    world.sources=updated;allocation=dict(world.metadata.get("evidence_allocation",{}));layout=dict(allocation.get("layout",{}));layout.update({"source_types":list(modalities),"source_type_origins":list(origins),"final_slot_assignment":{source_id:int(slot) for source_id,slot in zip(source_ids,slot_order)},"final_layout_plan":dict(plan_meta),"modality_layout_planned_before_roles":True,"material_grammar_written_to_sources":True});policy=dict(allocation.get("policy",{}));policy["modality_layout_planned_before_roles"]=True;policy["material_grammar_written_to_sources"]=True;allocation["layout"]=layout;allocation["policy"]=policy;world.metadata={**world.metadata,"evidence_allocation":allocation};return world
def _next_padding_fact_id(known,counter):
    while True:
        counter[0]+=1;fid=f"lf{counter[0]:05d}"
        if fid not in known:known.add(fid);return fid
def _padding_predicate(program,seed):
    used={str(value) for node in program.nodes for value in node.inputs};index=0
    while True:
        candidate=f"{_PADDING_PREDICATE_PREFIX}{(seed+index)%997}"
        if candidate not in used:return candidate
        index+=1
def _apply_planned_bundle_targets(program,world,*,targets_by_slot,seed):
    if len(targets_by_slot)!=len(world.sources):raise ValueError("planned bundle targets do not match realized source count")
    executor=SymbolicExecutor();baseline_answer=normalize_answer(executor.execute(program,world));predicate=_padding_predicate(program,seed);known={fact.fact_id for fact in world.facts};fact_map={fact.fact_id:fact for fact in world.facts};entity_fallback=sorted(world.entities);counter=[0];added=[];updated={};planned={};realized={};overflow_sources=[];padding_total=0
    for source_id,source in world.sources.items():
        slot=int(source.metadata.get("layout_position",-1))
        if slot<0 or slot>=len(targets_by_slot):raise ValueError(f"source {source_id} has invalid planned layout position {slot}")
        target=max(1,int(targets_by_slot[slot]));fact_ids=list(source.fact_ids);padding_count=max(0,target-len(fact_ids));subject=next((fact_map[fid].subject for fid in fact_ids if fid in fact_map),None)
        if subject is None and entity_fallback:subject=entity_fallback[slot%len(entity_fallback)]
        if subject is None and padding_count:raise EvidenceAllocationError("cannot pad a source in an entity-free world")
        for offset in range(padding_count):
            fid=_next_padding_fact_id(known,counter);padding=Fact(fid,str(subject),predicate,f"context-{slot+1}-{offset+1}","string",source_id);added.append(padding);fact_ids.append(fid)
        real=len(fact_ids);overflow=max(0,real-target)
        if overflow:overflow_sources.append(source_id)
        padding_total+=padding_count;planned[source_id]=target;realized[source_id]=real;updated[source_id]=EvidenceSource(source.source_id,source.source_type,tuple(fact_ids),{**dict(source.metadata),"planned_bundle_size":target,"bundle_size":real,"bundle_size_plan_padding":padding_count,"bundle_size_plan_overflow":overflow,"bundle_size_planned_before_roles":True})
    world.facts.extend(added);world.sources=updated
    if normalize_answer(executor.execute(program,world))!=baseline_answer:raise EvidenceAllocationError("role-independent bundle planning changed the oracle answer")
    allocation=dict(world.metadata.get("evidence_allocation",{}));previous=allocation.get("target_bundle_sizes")
    if previous is not None:allocation["prelayout_target_bundle_sizes"]=previous
    allocation.update({"target_bundle_sizes":planned,"realized_bundle_sizes":realized,"bundle_size_target_match_rate":sum(realized[sid]==planned[sid] for sid in planned)/len(planned) if planned else 0.,"bundle_size_plan_padding_fact_count":padding_total,"bundle_size_plan_overflow_source_ids":sorted(overflow_sources),"bundle_size_plan_overflow_source_count":len(overflow_sources)});layout=dict(allocation.get("layout",{}));layout["bundle_size_targets"]=list(map(int,targets_by_slot));layout["bundle_size_planned_before_roles"]=True;allocation["layout"]=layout;policy=dict(allocation.get("policy",{}));policy["bundle_size_planned_before_roles"]=True;policy["bundle_size_plan_is_minimum_visible_target"]=True;allocation["policy"]=policy;world.metadata={**world.metadata,"evidence_allocation":allocation};return world
def allocate_evidence_sources(program:ProgramIR,world:SyntheticWorld,*,min_critical_sources:int=2,min_noncritical_sources:int=1,max_sources:int=6,motif_prior:SourceMotifPrior|None=None,predicate_catalog:Mapping[str,Any]|None=None,mutation_records:Sequence[Mapping[str,Any]]|None=None,role_bundle_max_facts:int=3,critical_padding_max_facts:int=3,synthetic_modality_probability:float=.35,proof_corroboration_probability:float=.30,source_count_corroboration_probability:float|None=None,bundle_size_weights:Sequence[float]=DEFAULT_BUNDLE_SIZE_WEIGHTS,bundle_target_max_facts:int=DEFAULT_BUNDLE_TARGET_MAX_FACTS,source_count_weights:Sequence[tuple[int,float]]=DEFAULT_SOURCE_COUNT_WEIGHTS,acquisition_scenario_weights:Sequence[float]=_DEFAULT_SCENARIO_WEIGHTS,seed:int=17):
    motif_prior=motif_prior or SourceMotifPrior.fallback();planning_rng=random.Random(seed^0x50A2CE);target_source_count=sample_target_source_count(min_sources=min_critical_sources+min_noncritical_sources,max_sources=max_sources,rng=planning_rng,weights=source_count_weights);modalities,origins,slot_order,plan_meta=_plan_modality_layout(target_source_count,motif_prior=motif_prior,synthetic_modality_probability=synthetic_modality_probability,seed=seed,acquisition_scenario_weights=acquisition_scenario_weights);bundle_targets=_plan_bundle_targets(target_source_count,fact_count=len(world.facts),max_bundle_size=bundle_target_max_facts,seed=seed,weights=bundle_size_weights);plan_meta={**plan_meta,"bundle_size_targets":list(bundle_targets),"bundle_target_max_facts":int(bundle_target_max_facts),"bundle_size_planned_before_roles":True,"world_fact_count_at_planning":len(world.facts)};effective=proof_corroboration_probability if source_count_corroboration_probability is None else source_count_corroboration_probability
    if not 0<=effective<=1:raise ValueError("source_count_corroboration_probability must be in [0, 1]")
    result,ind_sources,ind_facts=_allocate_source_count_balanced(program,world,min_critical_sources=min_critical_sources,min_noncritical_sources=min_noncritical_sources,max_sources=max_sources,motif_prior=motif_prior,predicate_catalog=predicate_catalog,mutation_records=mutation_records,role_bundle_max_facts=role_bundle_max_facts,critical_padding_max_facts=critical_padding_max_facts,synthetic_modality_probability=synthetic_modality_probability,proof_corroboration_probability=effective,bundle_size_weights=bundle_size_weights,source_count_weights=source_count_weights,seed=seed);realized=int(result.metadata.get("evidence_allocation",{}).get("realized_source_count",len(result.sources)))
    if realized!=target_source_count:raise ValueError(f"planned source count {target_source_count} != realized source count {realized}")
    result=_apply_planned_layout(result,modalities=modalities,origins=origins,slot_order=slot_order,plan_meta=plan_meta);result,ind_sources,ind_facts=rebalance_protected_facts(program,result,targets_by_slot=bundle_targets,min_critical_sources=min_critical_sources,min_noncritical_sources=min_noncritical_sources,max_sources=max_sources);result=_apply_planned_bundle_targets(program,result,targets_by_slot=bundle_targets,seed=seed);allocation=dict(result.metadata.get("evidence_allocation",{}));policy=dict(allocation.get("policy",{}));policy.update({"source_count_corroboration_probability":effective,"modality_layout_planned_before_roles":True,"bundle_size_planned_before_roles":True,"bundle_target_max_facts":int(bundle_target_max_facts),"capacity_aware_protected_fact_allocation":True,"material_grammar_written_to_sources":True});allocation["policy"]=policy;result.metadata={**result.metadata,"evidence_allocation":allocation};return result,ind_sources,ind_facts
