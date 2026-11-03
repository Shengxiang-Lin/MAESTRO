"""Decision-information completeness checks for latent orchestration tasks."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from ..executor import ExecutionError,SymbolicExecutor
from ..ir import ProgramIR
from ..world import SyntheticWorld
_ATTRIBUTE_FILTER_OPS={"FILTER_STRING","FILTER_NUMBER","FILTER_YEAR","FILTER_DATE"};_VALUE_COMPLETE_OPS={"QUERY_ATTR","SELECT_AMONG","SELECT_BETWEEN"};_ATTRIBUTE_DECISION_OPS=_ATTRIBUTE_FILTER_OPS|_VALUE_COMPLETE_OPS
def _as_entities(value:Any)->set[str]:
    if isinstance(value,set):return {str(item) for item in value}
    if isinstance(value,(list,tuple)):return {str(item) for item in value}
    if value is None:return set()
    return {str(value)}
@dataclass(frozen=True)
class InformationCompletenessReport:
    accepted:bool;checked_node_count:int=0;required_fact_ids:tuple[str,...]=();missing_requirements:tuple[dict[str,Any],...]=();node_details:tuple[dict[str,Any],...]=()
    def to_dict(self):return {"accepted":self.accepted,"checked_node_count":self.checked_node_count,"required_fact_ids":list(self.required_fact_ids),"missing_requirements":[dict(item) for item in self.missing_requirements],"node_details":[dict(item) for item in self.node_details]}
def assess_goal_information_completeness(program:ProgramIR,world:SyntheticWorld,*,executor:SymbolicExecutor|None=None)->InformationCompletenessReport:
    executor=executor or SymbolicExecutor()
    try:trace=executor.execute_trace(program,world)
    except (ExecutionError,ValueError,TypeError) as exc:return InformationCompletenessReport(False,missing_requirements=({"reason":"trace_unavailable","error":str(exc)},))
    fact_index={}
    for fact in world.facts:fact_index.setdefault((str(fact.subject),str(fact.predicate)),[]).append(str(fact.fact_id))
    required=set();missing=[];details=[];node_map=program.node_map()
    for node_id in program.topological_order():
        node=node_map[node_id]
        if node.op not in _ATTRIBUTE_DECISION_OPS or not node.inputs:continue
        attribute=str(node.inputs[0])
        if node.op=="SELECT_BETWEEN":
            candidates=set()
            for dep in node.dependencies[:2]:candidates|=_as_entities(trace.get(dep))
        else:
            if not node.dependencies:continue
            candidates=_as_entities(trace.get(node.dependencies[0]))
        if node.op in _ATTRIBUTE_FILTER_OPS:
            matched=_as_entities(trace.get(node.node_id));node_fact_ids=[];matched_without=[];absent=[]
            for entity in sorted(candidates):
                fact_ids=fact_index.get((entity,attribute),[])
                if not fact_ids:absent.append(entity)
                if entity not in matched:continue
                if not fact_ids:matched_without.append(entity);missing.append({"reason":"matched_candidate_missing_support","node_id":node.node_id,"operator":node.op,"attribute":attribute,"entity_id":entity});continue
                required.update(fact_ids);node_fact_ids.extend(fact_ids)
            details.append({"node_id":node.node_id,"operator":node.op,"semantics":"closed_world_membership","attribute":attribute,"candidate_count":len(candidates),"candidate_ids":sorted(candidates),"matched_candidate_count":len(matched),"matched_candidate_ids":sorted(matched),"attribute_absent_candidate_ids":absent,"missing_candidate_ids":matched_without,"support_fact_ids":sorted(set(node_fact_ids))});continue
        node_missing=[];node_fact_ids=[]
        for entity in sorted(candidates):
            fact_ids=fact_index.get((entity,attribute),[])
            if not fact_ids:node_missing.append(entity);missing.append({"reason":"candidate_missing_attribute","node_id":node.node_id,"operator":node.op,"attribute":attribute,"entity_id":entity});continue
            required.update(fact_ids);node_fact_ids.extend(fact_ids)
        details.append({"node_id":node.node_id,"operator":node.op,"semantics":"value_complete","attribute":attribute,"candidate_count":len(candidates),"candidate_ids":sorted(candidates),"covered_candidate_count":len(candidates)-len(node_missing),"missing_candidate_ids":node_missing,"support_fact_ids":sorted(set(node_fact_ids))})
    return InformationCompletenessReport(not missing,len(details),tuple(sorted(required)),tuple(missing),tuple(details))
