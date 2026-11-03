"""Ground selected semantic candidates into executable heterogeneous worlds."""
from __future__ import annotations
import json,random
from collections import Counter
from pathlib import Path
from typing import Any,Mapping
from tqdm.auto import tqdm
from ..binding import AttributeBinding,BindingPlan,RelationBinding
from ..data_sources.taxonomy import TaxonomyIndex
from ..evidence import EvidenceAllocationError,ScenarioEvidencePrior,allocate_evidence_sources
from ..executor import SymbolicExecutor
from ..ir import ProgramIR,ProgramNode
from ..quality.difficulty import candidate_elimination_profile
from ..quality.observed_support import ObservedSupport
from ..quality.semantic_policy import normalize_program_world
from ..validation import normalize_answer,validate_task
from ..synthesis.hardener import HardeningError,harden_world
from ..synthesis.world_sampler import WorldSampler,WorldSynthesisError

def _program(payload):return ProgramIR([ProgramNode(int(n["node_id"]),str(n["op"]),tuple(int(v) for v in n.get("dependencies",())),tuple(str(v) for v in n.get("inputs",())),n.get("raw_function")) for n in payload.get("nodes",())],metadata=dict(payload.get("metadata",{})))
def _binding(row):
    raw=row.get("binding",{});program=_program(row["program"]);node_types={int(k):str(v) for k,v in raw.get("node_types",{}).items()};relations={}
    for key,item in raw.get("relation_bindings",{}).items():relations[int(key)]=RelationBinding(str(item["predicate"]),str(item["subject_type"]),str(item["object_type"]),str(item.get("direction","forward")))
    attributes={}
    for key,item in raw.get("attribute_bindings",{}).items():attributes[int(key)]=AttributeBinding(str(item["predicate"]),str(item["entity_type"]),str(item["value_kind"]),item.get("datatype"),str(item.get("semantic_family","generic")),item.get("canonical_unit"),item.get("value_min"),item.get("value_max"),bool(item.get("integer_values",False)),float(item.get("profile_confidence",0.) or 0.),str(item.get("profile_evidence","")))
    return BindingPlan(program,node_types,relations,attributes)
def _read_rows(path):
    with Path(path).open("r",encoding="utf-8") as handle:
        for line in handle:
            if line.strip():yield json.loads(line)
def _serialize_world(world):return world.to_dict() if hasattr(world,"to_dict") else {"entities":world.entities,"facts":[fact.__dict__ for fact in world.facts],"sources":{sid:source.__dict__ for sid,source in world.sources.items()},"metadata":world.metadata}
def generate_world_batch(candidates_path:str|Path,yago_catalog_path:str|Path,taxonomy_path:str|Path,output_path:str|Path,*,target:int,min_depth:int=5,seed:int=17,min_critical_sources:int=2,min_noncritical_sources:int=1,min_proof_sources:int=2,max_sources:int=6,mmqa_source_motifs_path:str|Path|None=None,progress:bool=False)->dict[str,Any]:
    catalog=json.loads(Path(yago_catalog_path).read_text(encoding="utf-8"));taxonomy=TaxonomyIndex.from_json(taxonomy_path);observed=ObservedSupport(catalog,taxonomy);motif_prior=ScenarioEvidencePrior.from_path(mmqa_source_motifs_path) if mmqa_source_motifs_path is not None and Path(mmqa_source_motifs_path).is_file() else ScenarioEvidencePrior.fallback();rng=random.Random(seed);accepted=[];rejected=[];reasons=Counter();iterator=list(_read_rows(candidates_path))
    if progress:iterator=tqdm(iterator,desc="Executable worlds",unit="candidate")
    for index,row in enumerate(iterator):
        if len(accepted)>=target:break
        try:
            plan=_binding(row);sampler=WorldSampler(seed=seed+index*101);world=sampler.synthesize(plan);program,world=normalize_program_world(plan.program,world);plan=BindingPlan(program,plan.node_types,plan.relation_bindings,plan.attribute_bindings);world,mutations=harden_world(program,world,seed=seed+index*211,near_miss_count=3,background_fact_count=4);world,critical_sources,critical_facts=allocate_evidence_sources(program,world,min_critical_sources=min_critical_sources,min_noncritical_sources=min_noncritical_sources,max_sources=max_sources,motif_prior=motif_prior,predicate_catalog=catalog,mutation_records=mutations,seed=seed+index*307);semantic=observed.world(world)
            if not semantic.get("accepted",False):raise ValueError("observed_semantic_support_rejected")
            report=validate_task(program,world,critical_sources=critical_sources,min_depth=min_depth,min_nodes=min_depth,require_hard_composition=True,min_entities=2,min_facts=3,min_indispensable_facts=2,min_critical_sources=min_critical_sources,min_noncritical_sources=min_noncritical_sources,min_proof_sources=min_proof_sources,max_proof_search_sources=max_sources,allow_multi_answer=True)
            if not report.accepted:raise ValueError("validation:"+",".join(report.reasons))
            answer=SymbolicExecutor().execute(program,world);difficulty=candidate_elimination_profile(mutations,answer);evidence=dict(world.metadata.get("evidence_allocation",{}));payload={"candidate_id":row.get("candidate_id"),"source_program_signature":row.get("source_program_signature") or program.canonical_signature(),"reasoning_motif":row.get("reasoning_motif"),"program":program.to_dict(),"binding":row.get("binding",{}),"world":_serialize_world(world),"answer":answer,"candidate_elimination":difficulty.to_dict(),"evidence":{**evidence,"critical_source_ids":critical_sources,"critical_fact_ids":critical_facts},"validation":report.__dict__,"semantic_plausibility":semantic};accepted.append(payload)
        except (WorldSynthesisError,HardeningError,EvidenceAllocationError,ValueError,RuntimeError) as exc:
            reason=str(exc).split(":",1)[0] or type(exc).__name__;reasons[reason]+=1;rejected.append({"candidate_id":row.get("candidate_id"),"reason":str(exc)})
    target_path=Path(output_path);target_path.parent.mkdir(parents=True,exist_ok=True)
    with target_path.open("w",encoding="utf-8") as writer:
        for row in accepted:writer.write(json.dumps(row,ensure_ascii=False,default=str)+"\n")
    rejected_path=target_path.with_suffix(target_path.suffix+".quality_rejected.jsonl")
    with rejected_path.open("w",encoding="utf-8") as writer:
        for row in rejected:writer.write(json.dumps(row,ensure_ascii=False)+"\n")
    return {"output":str(target_path),"rejected_output":str(rejected_path),"target":target,"accepted":len(accepted),"target_filled":len(accepted)>=target,"fill_rate":len(accepted)/target if target else 0.,"rejected":len(rejected),"rejection_reasons":dict(reasons.most_common()),"policy":"graph_grounded_binding_then_executable_world_then_hardening_then_balanced_evidence_then_symbolic_validation"}
__all__=["generate_world_batch"]
