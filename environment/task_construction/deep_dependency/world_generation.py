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
def _json_default(value):
    if isinstance(value,set):return sorted(value,key=str)
    raise TypeError(f"Unsupported artifact value: {type(value).__name__}")

def _serialize_world(world):return world.to_dict() if hasattr(world,"to_dict") else {"entities":world.entities,"facts":[fact.__dict__ for fact in world.facts],"sources":{sid:source.__dict__ for sid,source in world.sources.items()},"metadata":world.metadata}
def generate_world_batch(candidates_path,yago_catalog_path,taxonomy_path,output_path,*,
    min_depth=5,seed=17,min_critical_sources=2,min_noncritical_sources=1,
    min_proof_sources=2,max_sources=6,mmqa_source_motifs_path=None,progress=False,
    world_attempts_per_candidate=3,acquisition_scenario_weights=(45,40,15)):
    from ..task_design.planner import plan_task_design
    from ..task_design.quality import assess_task_design
    from ..scenario.validity import assess_scenario
    from ..capabilities.grounding import ground_capability_world
    from ..rendering.pipeline import render_task_record
    from ..quality.diversity import BatchDiversityTracker
    from .candidate_selection import _binding_predicates,_binding_type_pairs
    from ..quality.hardness import pure_topology_signature
    from .terminal_families import terminal_family
    catalog=json.loads(Path(yago_catalog_path).read_text(encoding='utf-8'))
    taxonomy=TaxonomyIndex.from_json(taxonomy_path)
    observed=ObservedSupport(catalog,taxonomy)
    motif_prior=ScenarioEvidencePrior.from_path(mmqa_source_motifs_path) if mmqa_source_motifs_path and Path(mmqa_source_motifs_path).is_file() else ScenarioEvidencePrior.fallback()
    accepted,rejected,reasons=[],[],Counter()
    # Preserve the existing relative diversity caps at every accepted count.
    tracker=BatchDiversityTracker(target=None)
    rows=list(_read_rows(candidates_path))
    iterator=tqdm(rows,desc='Executable worlds',disable=not progress)
    candidates_tested=0
    for index,row in enumerate(iterator):
        candidates_tested+=1
        for attempt in range(world_attempts_per_candidate):
            run_seed=seed+index*1009+attempt*104729
            try:
                plan=_binding(row)
                dimensions=dict(topology=pure_topology_signature(plan.program),
                    predicates=_binding_predicates(row['binding']),
                    type_pairs=_binding_type_pairs(row['binding']),terminal_family=terminal_family(plan.program))
                ok,diversity_reasons=tracker.would_accept(**dimensions)
                if not ok:
                    raise ValueError('diversity:'+','.join(diversity_reasons))
                sampler=WorldSampler(seed=run_seed,catalog=catalog,taxonomy=taxonomy)
                world=sampler.synthesize(plan)
                program=sampler.concrete_program
                world,mutations=harden_world(program,world,seed=run_seed+211,
                    near_miss_count=3,background_fact_count=4)
                world,critical_sources,critical_facts=allocate_evidence_sources(program,world,
                    min_critical_sources=min_critical_sources,min_noncritical_sources=min_noncritical_sources,
                    max_sources=max_sources,motif_prior=motif_prior,predicate_catalog=catalog,
                    mutation_records=mutations,seed=run_seed+307,
                     acquisition_scenario_weights=acquisition_scenario_weights)
                semantic=observed.world(world)
                if not semantic.get('accepted',False):
                    raise ValueError('observed_semantic_support_rejected')
                report=validate_task(program,world,critical_sources=critical_sources,
                    min_depth=min_depth,min_nodes=min_depth,require_hard_composition=True,
                    min_entities=2,min_facts=3,min_indispensable_facts=2,
                    min_critical_sources=min_critical_sources,min_noncritical_sources=min_noncritical_sources,
                    min_proof_sources=min_proof_sources,max_proof_search_sources=max_sources,allow_multi_answer=True)
                if not report.accepted:
                    raise ValueError('validation:'+','.join(report.reasons))
                answer=SymbolicExecutor().execute(program,world)
                if isinstance(answer,set):answer=sorted(answer)
                evidence=dict(world.metadata.get('evidence_allocation',{}))
                payload={'candidate_id':row.get('candidate_id'),
                    'source_program_signature':row.get('source_program_signature'),
                    'reasoning_motif':row.get('reasoning_motif'),'program':program.to_dict(),
                    'binding':{**row['binding'],'program':program.to_dict()},'world':_serialize_world(world),
                    'answer':answer,'candidate_elimination':candidate_elimination_profile(mutations,answer).to_dict(),
                    'evidence':{**evidence,'critical_source_ids':critical_sources,'critical_fact_ids':critical_facts},
                    'validation':report.__dict__,'semantic_plausibility':semantic}
                from ..quality.execution_contract import assess_execution_contract
                execution=assess_execution_contract(payload)
                if not execution["accepted"]:raise ValueError("execution_contract:"+",".join(execution["issues"]))
                design=plan_task_design(payload)
                quality=assess_task_design(design,taxonomy=taxonomy)
                if not quality.accepted:
                    raise ValueError('task_design:'+','.join(quality.reasons))
                scenario=assess_scenario(program,world)
                if not scenario.accepted:
                    raise ValueError('scenario:'+','.join(scenario.reasons))
                profile=ground_capability_world(payload,design.to_dict()).binding_audit['p1_orchestration_profile']
                if not profile.get('orchestration_ready'):
                    raise ValueError('orchestration_not_ready')
                render_task_record(payload,seed=seed+len(accepted))
                accepted.append(payload)
                tracker.add(**dimensions)
                break
            except (WorldSynthesisError,HardeningError,EvidenceAllocationError,ValueError,RuntimeError) as exc:
                reason=str(exc).split(':',1)[0] or type(exc).__name__
                reasons[reason]+=1
                rejected.append({'candidate_id':row.get('candidate_id'),'attempt':attempt+1,'reason':str(exc)})
                if reason=='diversity':
                    break
    target_path=Path(output_path)
    target_path.parent.mkdir(parents=True,exist_ok=True)
    with target_path.open('w',encoding='utf-8') as writer:
        for row in accepted:
            writer.write(json.dumps(row,ensure_ascii=False,default=_json_default)+'\n')
    rejected_path=target_path.with_suffix(target_path.suffix+'.quality_rejected.jsonl')
    with rejected_path.open('w',encoding='utf-8') as writer:
        for row in rejected:
            writer.write(json.dumps(row,ensure_ascii=False)+'\n')
    scenario_histogram=Counter(
        row["world"].get("metadata", {})
        .get("evidence_allocation", {})
        .get("layout", {})
        .get("final_layout_plan", {})
        .get("acquisition_scenario", "legacy")
        for row in accepted
    )
    return {
        "acquisition_scenario_histogram":dict(scenario_histogram),
        "acquisition_scenario_weights":list(acquisition_scenario_weights),
        "output":str(target_path),
        "rejected_output":str(rejected_path),
        "candidate_pool_size":len(rows),
        "candidates_tested":candidates_tested,
        "pool_fully_evaluated":candidates_tested==len(rows),
        "accepted":len(accepted),
        "candidate_acceptance_rate":len(accepted)/candidates_tested if candidates_tested else 0.0,
        "rejected_attempts":len(rejected),
        "rejection_reasons":dict(reasons),
        "diversity":tracker.summary(),
        "world_attempts_per_candidate":world_attempts_per_candidate,
    }


__all__=['generate_world_batch']
