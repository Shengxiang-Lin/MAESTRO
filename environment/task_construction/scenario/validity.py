"""P0: scenario validity for latent orchestration tasks."""
from __future__ import annotations
from dataclasses import dataclass,field
from itertools import combinations
from typing import Any,Dict,Mapping
from ..executor import ExecutionError,SymbolicExecutor
from ..ir import ProgramIR
from ..quality.semantic_policy import SemanticWorldPolicy
from ..validation import answer_is_nonempty,indispensable_fact_ids,minimal_source_proof,normalize_answer
from ..world import SyntheticWorld
from .information import assess_goal_information_completeness
@dataclass
class ScenarioValidityReport:
    accepted:bool=True;checks:Dict[str,bool]=field(default_factory=dict);details:Dict[str,Any]=field(default_factory=dict);reasons:list[str]=field(default_factory=list)
    def add(self,name,passed,detail=None):
        self.checks[name]=bool(passed)
        if detail is not None:self.details[name]=detail
        if not passed:self.reasons.append(name)
        self.accepted=all(self.checks.values())
    def to_dict(self):return {"accepted":self.accepted,"checks":dict(self.checks),"details":dict(self.details),"reasons":list(self.reasons)}
def _fact_source_coverage(world):
    coverage={fact.fact_id:set() for fact in world.facts}
    for fact in world.facts:
        if fact.source_id is not None:coverage.setdefault(fact.fact_id,set()).add(str(fact.source_id))
    for source_id,source in world.sources.items():
        for fact_id in source.fact_ids:coverage.setdefault(fact_id,set()).add(source_id)
    return coverage
def _minimum_recovery_source_cover(required_fact_ids,coverage,world,*,max_sources=8,max_witnesses=8):
    source_ids=sorted(world.sources)
    if not required_fact_ids:return {"min_size":0,"witnesses":[[]],"searched_subsets":0,"source_count":len(source_ids),"exhaustive":True}
    if len(source_ids)>max_sources:return {"min_size":None,"witnesses":[],"searched_subsets":0,"source_count":len(source_ids),"exhaustive":False,"reason":f"source_count>{max_sources}"}
    searched=0
    for size in range(1,len(source_ids)+1):
        witnesses=[]
        for subset in combinations(source_ids,size):
            searched+=1;chosen=set(subset)
            if all(coverage.get(fact_id,set())&chosen for fact_id in required_fact_ids):witnesses.append(list(subset))
            if len(witnesses)>=max_witnesses:break
        if witnesses:return {"min_size":size,"witnesses":witnesses,"searched_subsets":searched,"source_count":len(source_ids),"exhaustive":True}
    return {"min_size":None,"witnesses":[],"searched_subsets":searched,"source_count":len(source_ids),"exhaustive":True}
def assess_scenario(program,world,*,max_proof_search_sources=8,executor=None):
    executor=executor or SymbolicExecutor();report=ScenarioValidityReport();semantic=SemanticWorldPolicy().assess(world);report.add("world_coherence",semantic.accepted,semantic.to_dict())
    try:latent_result=executor.execute(program,world);report.add("goal_reachability",answer_is_nonempty(latent_result),{"latent_result":normalize_answer(latent_result)})
    except (ExecutionError,ValueError,TypeError) as exc:
        report.add("goal_reachability",False,{"error":str(exc)});report.add("goal_information_completeness",False,{"reason":"goal_unreachable"});report.add("evidence_recoverability",False,{"reason":"goal_unreachable"});report.add("evidence_dependence",False,{"reason":"goal_unreachable"});return report
    information=assess_goal_information_completeness(program,world,executor=executor);report.add("goal_information_completeness",information.accepted,information.to_dict());indispensable=indispensable_fact_ids(program,world,executor=executor);required_fact_ids=set(indispensable)|set(information.required_fact_ids);coverage=_fact_source_coverage(world);uncovered=[fid for fid in sorted(required_fact_ids) if not coverage.get(fid)];answer_proof=minimal_source_proof(program,world,executor=executor,max_sources=max_proof_search_sources);answer_exists=isinstance(answer_proof.get("min_size"),int) and bool(answer_proof.get("witnesses"));decision=_minimum_recovery_source_cover(required_fact_ids,coverage,world,max_sources=max_proof_search_sources);decision_exists=isinstance(decision.get("min_size"),int) and bool(decision.get("witnesses"));recoverable=information.accepted and not uncovered and answer_exists and decision_exists;report.add("evidence_recoverability",recoverable,{"indispensable_fact_ids":list(indispensable),"decision_support_fact_ids":list(information.required_fact_ids),"required_fact_ids":sorted(required_fact_ids),"uncovered_required_fact_ids":uncovered,"uncovered_indispensable_fact_ids":[fid for fid in indispensable if not coverage.get(fid)],"fact_source_coverage":{fid:sorted(sids) for fid,sids in sorted(coverage.items()) if fid in required_fact_ids},"minimum_answer_source_proof":answer_proof,"minimum_decision_recovery_source_cover":decision});report.add("evidence_dependence",bool(required_fact_ids) and answer_exists,{"indispensable_fact_count":len(indispensable),"decision_support_fact_count":len(information.required_fact_ids),"required_fact_count":len(required_fact_ids),"minimum_answer_proof_source_count":answer_proof.get("min_size"),"minimum_decision_recovery_source_count":decision.get("min_size")});return report
def scenario_required_fact_ids(report):
    payload=report.to_dict() if isinstance(report,ScenarioValidityReport) else dict(report);detail=dict(payload.get("details",{})).get("evidence_recoverability",{});required=detail.get("required_fact_ids")
    if required is None:required=detail.get("indispensable_fact_ids",())
    return tuple(str(v) for v in required)
