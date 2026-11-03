"""Public evidence API for MAESTRO synthetic environments."""
from __future__ import annotations
from typing import Any, Mapping
from ..world import EvidenceSource
from .allocation import DIAGNOSTIC_ROLES,SYNTHETIC_MODALITIES,allocate_structured_evidence_sources,bundle_stats,normalize_temporal_conflicts,temporal_granularity_compatible
from .balancing import DEFAULT_BUNDLE_SIZE_WEIGHTS,sample_target_bundle_sizes
from .layout import DEFAULT_BUNDLE_TARGET_MAX_FACTS,DEFAULT_SOURCE_COUNT_CORROBORATION_PROBABILITY,_apply_planned_bundle_targets,allocate_evidence_sources as _allocate_evidence_sources
from .scenario_prior import ScenarioEvidencePrior
from .semantics import EvidenceAllocationError,classify_noncritical_facts,select_role_bundles
from .source_count import DEFAULT_SOURCE_COUNT_WEIGHTS,SourceCountLeakageAudit,answer_shape_signature,sample_target_source_count
DEFAULT_MIN_VISIBLE_BUNDLE_SIZE=2
_DEFAULT_SURFACE_STYLE={"text":"document_excerpt","table":"structured_table","image_report":"visual_observation","video_report":"temporal_inspection_log","audio_transcript":"speaker_transcript","database":"structured_record"}
def _scenario_material_motif(layout:Mapping[str,Any])->Mapping[str,Any]:
    final_plan=layout.get("final_layout_plan")
    if isinstance(final_plan,Mapping):
        final_base=final_plan.get("base_motif")
        if isinstance(final_base,Mapping):return final_base
    legacy=layout.get("base_motif",{});return legacy if isinstance(legacy,Mapping) else {}
def _apply_scenario_artifact_metadata(world):
    allocation=world.metadata.get("evidence_allocation",{});layout=allocation.get("layout",{}) if isinstance(allocation,Mapping) else {}
    if not isinstance(layout,Mapping):return world
    base=_scenario_material_motif(layout);roles=list(base.get("artifact_roles",()) or ());styles=list(base.get("surface_styles",()) or ());base_types=list(base.get("source_types",()) or ())
    if not roles:return world
    updated={}
    for source_id,source in world.sources.items():
        try:slot=int(source.metadata.get("layout_position",-1))
        except (TypeError,ValueError):slot=-1
        role=roles[slot] if 0<=slot<len(roles) else "supporting_lookup";style=styles[slot] if 0<=slot<len(styles) else _DEFAULT_SURFACE_STYLE.get(source.source_type,"document_excerpt")
        if 0<=slot<len(base_types) and str(base_types[slot])!=source.source_type:style=_DEFAULT_SURFACE_STYLE.get(source.source_type,style)
        metadata={**dict(source.metadata),"material_role":str(role),"surface_style":str(style),"scenario_prior_origin":str(base.get("origin","unknown")),"discovery_position":slot,"discovery_requires_previous":bool(slot>0),"perception_abstracted":source.source_type in {"image_report","video_report","audio_transcript"}}
        updated[source_id]=EvidenceSource(source.source_id,source.source_type,source.fact_ids,metadata)
    world.sources=updated;allocation=dict(allocation);policy=dict(allocation.get("policy",{}));policy.update({"scenario_semantic_bundle_roles":True,"perception_abstraction":"modality_conditioned_textual_observation","dataset_truth_used_as_oracle":False});allocation["policy"]=policy;world.metadata={**world.metadata,"evidence_allocation":allocation};return world
def allocate_evidence_sources(program,world,*,seed:int=17,minimum_visible_bundle_size:int=DEFAULT_MIN_VISIBLE_BUNDLE_SIZE,**kwargs:Any):
    if minimum_visible_bundle_size<1:raise ValueError("minimum_visible_bundle_size must be >= 1")
    result,critical_sources,critical_facts=_allocate_evidence_sources(program,world,seed=seed,**kwargs);allocation=result.metadata.get("evidence_allocation",{});layout=allocation.get("layout",{}) if isinstance(allocation,dict) else {};raw_targets=layout.get("bundle_size_targets",()) if isinstance(layout,dict) else ()
    if raw_targets and minimum_visible_bundle_size>1:
        targets=[max(minimum_visible_bundle_size,int(value)) for value in raw_targets]
        if list(map(int,raw_targets))!=targets or any(len(source.fact_ids)<minimum_visible_bundle_size for source in result.sources.values()):result=_apply_planned_bundle_targets(program,result,targets_by_slot=targets,seed=seed^0x2B17)
    result=_apply_scenario_artifact_metadata(result);allocation=dict(result.metadata.get("evidence_allocation",{}));policy=dict(allocation.get("policy",{}));policy["minimum_visible_bundle_size"]=int(minimum_visible_bundle_size);policy["minimum_visible_bundle_size_role_independent"]=True;allocation["policy"]=policy;result.metadata={**result.metadata,"evidence_allocation":allocation};return result,critical_sources,critical_facts
__all__=["DEFAULT_BUNDLE_SIZE_WEIGHTS","DEFAULT_BUNDLE_TARGET_MAX_FACTS","DEFAULT_MIN_VISIBLE_BUNDLE_SIZE","DEFAULT_SOURCE_COUNT_CORROBORATION_PROBABILITY","DEFAULT_SOURCE_COUNT_WEIGHTS","DIAGNOSTIC_ROLES","EvidenceAllocationError","SYNTHETIC_MODALITIES","ScenarioEvidencePrior","SourceCountLeakageAudit","allocate_evidence_sources","allocate_structured_evidence_sources","answer_shape_signature","bundle_stats","classify_noncritical_facts","normalize_temporal_conflicts","sample_target_bundle_sizes","sample_target_source_count","select_role_bundles","temporal_granularity_compatible"]
