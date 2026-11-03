"""Quality gates for executable synthetic tasks."""
from __future__ import annotations
from dataclasses import dataclass,field
from itertools import combinations
from typing import Any,Dict,List,Mapping,Sequence
from .executor import ExecutionError,SymbolicExecutor
from .quality.hardness import complexity_profile
from .ir import ProgramIR
from .quality.semantic_policy import SemanticWorldPolicy
from .world import SyntheticWorld
@dataclass
class ValidationReport:
    accepted:bool;checks:Dict[str,bool]=field(default_factory=dict);details:Dict[str,Any]=field(default_factory=dict);reasons:List[str]=field(default_factory=list)
    def add(self,name:str,passed:bool,detail:Any=None)->None:
        self.checks[name]=bool(passed)
        if detail is not None:self.details[name]=detail
        if not passed:self.reasons.append(name)
        self.accepted=all(self.checks.values())
def normalize_answer(answer:Any)->Any:
    if isinstance(answer,set):return tuple(sorted(str(x) for x in answer))
    if isinstance(answer,list):return tuple(sorted(str(x) for x in answer))
    if isinstance(answer,tuple):return tuple(sorted(str(x) for x in answer))
    return answer
def answer_is_unique(answer):return len(answer)==1 if isinstance(answer,(set,list,tuple)) else answer is not None
def answer_is_nonempty(answer):return len(answer)>0 if isinstance(answer,(set,list,tuple,dict)) else answer is not None
def _is_deep_dependency_program(program):
    deep=program.metadata.get("deep_dependency",{});return isinstance(deep,Mapping) and isinstance(deep.get("composition_links"),(list,tuple)) and bool(deep.get("composition_links"))
def critical_fact_ablation(program,world,*,executor=None):
    executor=executor or SymbolicExecutor();baseline=normalize_answer(executor.execute(program,world));result={}
    for fact in world.facts:
        ablated=world.without_facts([fact.fact_id])
        try:answer=normalize_answer(executor.execute(program,ablated));result[fact.fact_id]={"changed":answer!=baseline,"answer":answer}
        except (ExecutionError,ValueError,TypeError) as exc:result[fact.fact_id]={"changed":True,"error":str(exc)}
    return result
def minimal_fact_support(program,world,*,executor=None):
    executor=executor or SymbolicExecutor();baseline=normalize_answer(executor.execute(program,world));working=world
    for fact_id in sorted(f.fact_id for f in world.facts):
        if not any(f.fact_id==fact_id for f in working.facts):continue
        candidate=working.without_facts([fact_id])
        try:answer=normalize_answer(executor.execute(program,candidate))
        except (ExecutionError,ValueError,TypeError):continue
        if answer==baseline:working=candidate
    return sorted(f.fact_id for f in working.facts)
def indispensable_fact_ids(program,world,*,executor=None):
    executor=executor or SymbolicExecutor();ablations=critical_fact_ablation(program,world,executor=executor);global_ids=[fid for fid,info in ablations.items() if info.get("changed")]
    if not _is_deep_dependency_program(program):return global_ids
    support=minimal_fact_support(program,world,executor=executor);return support if len(support)>len(global_ids) else global_ids
def source_ablation(program,world,*,executor=None):
    executor=executor or SymbolicExecutor();baseline=normalize_answer(executor.execute(program,world));result={}
    for source_id in sorted(world.sources):
        candidate=world.without_sources([source_id])
        try:answer=normalize_answer(executor.execute(program,candidate));result[source_id]={"changed":answer!=baseline,"answer":answer}
        except (ExecutionError,ValueError,TypeError) as exc:result[source_id]={"changed":True,"error":str(exc)}
    return result
def indispensable_source_ids(program,world,*,executor=None):return [sid for sid,info in source_ablation(program,world,executor=executor).items() if info.get("changed")]
def minimal_source_proof(program,world,*,executor=None,max_sources:int=8,max_witnesses:int=8):
    executor=executor or SymbolicExecutor();source_ids=sorted(world.sources);baseline=normalize_answer(executor.execute(program,world))
    if len(source_ids)>max_sources:return {"min_size":None,"witnesses":[],"searched_subsets":0,"source_count":len(source_ids),"exhaustive":False,"reason":f"source_count>{max_sources}"}
    searched=0
    for size in range(1,len(source_ids)+1):
        witnesses=[]
        for subset in combinations(source_ids,size):
            searched+=1;removed=[sid for sid in source_ids if sid not in set(subset)];candidate=world.without_sources(removed)
            try:answer=normalize_answer(executor.execute(program,candidate))
            except (ExecutionError,ValueError,TypeError):continue
            if answer==baseline:
                witnesses.append(list(subset))
                if len(witnesses)>=max_witnesses:break
        if witnesses:return {"min_size":size,"witnesses":witnesses,"searched_subsets":searched,"source_count":len(source_ids),"exhaustive":True}
    return {"min_size":None,"witnesses":[],"searched_subsets":searched,"source_count":len(source_ids),"exhaustive":True}
def validate_task(program,world,*,critical_sources:Sequence[str]=(),min_depth:int=1,min_nodes:int=1,require_branch_or_merge:bool=False,require_hard_composition:bool=False,min_entities:int=0,min_facts:int=0,min_indispensable_facts:int=0,min_critical_sources:int=0,min_noncritical_sources:int=0,min_proof_sources:int=0,max_proof_search_sources:int=8,allow_multi_answer:bool=True,executor=None):
    executor=executor or SymbolicExecutor();report=ValidationReport(True)
    try:answer=executor.execute(program,world)
    except (ExecutionError,ValueError,TypeError) as exc:report.add("executable",False,str(exc));return report
    normalized=normalize_answer(answer);answer_ok=answer_is_nonempty(answer) if allow_multi_answer else answer_is_unique(answer);cardinality=len(answer) if isinstance(answer,(set,list,tuple,dict)) else 1;report.add("executable",True,normalized);report.add("unique_answer",answer_ok,{"answer":normalized,"cardinality":cardinality,"multi_answer_allowed":allow_multi_answer});metrics=program.metrics();complexity_ok=metrics["depth"]>=min_depth and metrics["node_count"]>=min_nodes;profile=complexity_profile(program,min_depth=min_depth)
    if require_hard_composition:complexity_ok=complexity_ok and profile.qualifies
    elif require_branch_or_merge:complexity_ok=complexity_ok and (metrics["fanout_branch_nodes"]>0 or metrics["independent_branches"]>0 or metrics["merge_nodes"]>0)
    report.add("complexity",complexity_ok,{**metrics,"hardness_profile":profile.to_dict()});report.add("world_size",len(world.entities)>=min_entities and len(world.facts)>=min_facts,{"entities":len(world.entities),"facts":len(world.facts)});semantic_report=SemanticWorldPolicy().assess(world);report.add("semantic_world_plausibility",semantic_report.accepted,semantic_report.to_dict())
    if min_indispensable_facts>0:
        fact_ablations=critical_fact_ablation(program,world,executor=executor);globally=[fid for fid,info in fact_ablations.items() if info.get("changed")];effective=indispensable_fact_ids(program,world,executor=executor);mode="minimal_support_witness" if set(effective)!=set(globally) else "global_single_fact_ablation";report.add("fact_indispensability",len(effective)>=min_indispensable_facts,{"count":len(effective),"fact_ids":effective,"support_mode":mode,"global_indispensable_fact_ids":globally,"ablations":fact_ablations})
    missing_sources=[sid for sid in critical_sources if sid not in world.sources];report.add("critical_sources_exist",not missing_sources,missing_sources);report.add("critical_source_count",len(critical_sources)>=min_critical_sources,{"count":len(critical_sources),"required":min_critical_sources});ablations={};source_indispensable=True
    for source_id in critical_sources:
        if source_id not in world.sources:continue
        try:ablated=normalize_answer(executor.execute(program,world.without_sources([source_id])));changed=ablated!=normalized;ablations[source_id]={"answer":ablated,"changed":changed};source_indispensable=source_indispensable and changed
        except (ExecutionError,ValueError,TypeError) as exc:ablations[source_id]={"error":str(exc),"changed":True}
    report.add("critical_source_indispensability",source_indispensable,ablations)
    if min_noncritical_sources>0:
        critical_set=set(critical_sources);noncritical_ids=[sid for sid in sorted(world.sources) if sid not in critical_set];role_counts={}
        for source in world.sources.values():role=str(source.metadata.get("role","unspecified"));role_counts[role]=role_counts.get(role,0)+1
        report.add("noncritical_source_count",len(noncritical_ids)>=min_noncritical_sources,{"count":len(noncritical_ids),"required":min_noncritical_sources,"source_ids":noncritical_ids,"role_counts":role_counts})
    if min_proof_sources>0:
        proof=minimal_source_proof(program,world,executor=executor,max_sources=max_proof_search_sources);min_size=proof.get("min_size");report.add("minimal_source_proof",bool(proof.get("exhaustive")) and isinstance(min_size,int) and min_size>=min_proof_sources,{**proof,"required":min_proof_sources})
    return report
