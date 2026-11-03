"""Proof-aware task fuzzification and intermediate-entity concealment."""
from __future__ import annotations
from dataclasses import dataclass
from statistics import median
from typing import Any,Iterable,Mapping
from ..executor import ExecutionError,SymbolicExecutor
from ..ir import ProgramIR,ProgramNode
from ..world import SyntheticWorld
_VALUE_FILTER_OPS={"FILTER_STRING","FILTER_NUMBER","FILTER_YEAR","FILTER_DATE"}
def _as_set(value):
    if isinstance(value,set):return {str(item) for item in value}
    if isinstance(value,(list,tuple)):return {str(item) for item in value}
    return {str(value)}
@dataclass(frozen=True)
class LatentEntityPlan:
    entity_id:str;entity_type:str;find_node_ids:tuple[int,...];rule:str;identifying_node_ids:tuple[int,...];support_fact_ids:tuple[str,...]
    def to_dict(self):return {"entity_id":self.entity_id,"entity_type":self.entity_type,"find_node_ids":list(self.find_node_ids),"rule":self.rule,"identifying_node_ids":list(self.identifying_node_ids),"support_fact_ids":list(self.support_fact_ids)}
@dataclass(frozen=True)
class PublicQueryPlan:
    latent_entities:tuple[LatentEntityPlan,...]=();explicit_entity_ids:tuple[str,...]=();proof_entity_ids:tuple[str,...]=();minimum_required_source_count:int|None=None
    @property
    def latent_entity_ids(self):return tuple(item.entity_id for item in self.latent_entities)
    @property
    def concealed_find_node_ids(self):return frozenset(node_id for item in self.latent_entities for node_id in item.find_node_ids)
    def concealment_for_node(self,node_id):
        for item in self.latent_entities:
            if int(node_id) in item.find_node_ids:return item
        return None
    def to_dict(self):
        explicit=set(self.explicit_entity_ids);latent=set(self.latent_entity_ids);proof=set(self.proof_entity_ids);return {"latent_entities":[item.to_dict() for item in self.latent_entities],"latent_entity_ids":sorted(latent),"explicit_entity_ids":sorted(explicit),"proof_entity_ids":sorted(proof),"latent_intermediate_count":len(latent),"explicit_entity_concealment_rate":len(latent&explicit)/len(explicit) if explicit else 0.,"minimum_required_source_count":self.minimum_required_source_count,"single_source_answerable":self.minimum_required_source_count is not None and self.minimum_required_source_count<=1,"grounding_clue_count":len(self.latent_entities),"grounding_clue_support_fact_count":sum(len(item.support_fact_ids) for item in self.latent_entities),"grounding_clue_indispensability_rate":sum(bool(item.support_fact_ids) for item in self.latent_entities)/len(self.latent_entities) if self.latent_entities else 1.}
def _critical_fact_ids(record):
    evidence=record.get("evidence",{});return {str(value) for value in evidence.get("critical_fact_ids",())} if isinstance(evidence,Mapping) else set()
def _minimum_required_source_count(record):
    evidence=record.get("evidence",{});proof=evidence.get("minimal_proof",{}) if isinstance(evidence,Mapping) else {};value=proof.get("min_size") if isinstance(proof,Mapping) else None;return int(value) if isinstance(value,int) else None
def _proof_entity_ids(world,critical_fact_ids):
    result=set()
    for fact in world.facts:
        if fact.fact_id not in critical_fact_ids:continue
        result.add(str(fact.subject))
        if fact.object_kind=="entity":result.add(str(fact.object))
    return tuple(sorted(result))
def _parents(program):
    result={node.node_id:[] for node in program.nodes}
    for node in program.nodes:
        for dep in node.dependencies:result.setdefault(dep,[]).append(node.node_id)
    return result
def _explicit_find_nodes(program):
    result={}
    for node in program.nodes:
        if node.op!="FIND" or not node.inputs:continue
        token=str(node.inputs[0])
        if token.startswith("entity:"):result.setdefault(token.split(":",1)[1],[]).append(node.node_id)
    return result
def _subgraph_nodes(program,root_id):
    nodes=program.node_map();visited=set();stack=[int(root_id)]
    while stack:
        node_id=stack.pop()
        if node_id in visited or node_id not in nodes:continue
        visited.add(node_id);stack.extend(int(dep) for dep in nodes[node_id].dependencies)
    return visited
def _branch_predicates(program,root_id):
    nodes=program.node_map();result=set()
    for node_id in _subgraph_nodes(program,root_id):
        node=nodes[node_id]
        if node.op=="RELATE" and node.inputs:result.add(str(node.inputs[0]))
        elif node.op in _VALUE_FILTER_OPS and node.inputs:result.add(str(node.inputs[0]))
    return result
def _support_facts_for_entity(world,*,entity_id,predicates,critical_fact_ids):
    predicate_set={str(v) for v in predicates};result=[]
    for fact in world.facts:
        if critical_fact_ids and fact.fact_id not in critical_fact_ids:continue
        if predicate_set and fact.predicate not in predicate_set:continue
        if fact.subject==entity_id or (fact.object_kind=="entity" and str(fact.object)==entity_id):result.append(fact.fact_id)
    return tuple(sorted(set(result)))
def _filter_uniquely_identifies(entity_id,filter_node,entity_type,world):
    if filter_node.op not in _VALUE_FILTER_OPS or len(filter_node.inputs)<3:return False
    probe=ProgramIR([ProgramNode(0,"FIND",(),(f"type:{entity_type}",)),ProgramNode(1,filter_node.op,(0,),tuple(filter_node.inputs))],metadata={"public_query_probe":True})
    try:matched=SymbolicExecutor().execute(probe,world)
    except (ExecutionError,ValueError,TypeError):return False
    return _as_set(matched)=={str(entity_id)}
def _candidate_filter_concealment(entity_id,find_nodes,program,world,critical_fact_ids,parent_map):
    nodes=program.node_map();entity_type=world.entities.get(entity_id)
    if not entity_type:return None
    for find_id in find_nodes:
        for parent_id in parent_map.get(find_id,()):
            parent=nodes[parent_id]
            if parent.op not in _VALUE_FILTER_OPS or not _filter_uniquely_identifies(entity_id,parent,entity_type,world):continue
            predicate=str(parent.inputs[0]) if parent.inputs else "";support=_support_facts_for_entity(world,entity_id=entity_id,predicates=(predicate,),critical_fact_ids=critical_fact_ids)
            if critical_fact_ids and not support:continue
            return LatentEntityPlan(entity_id,entity_type,find_nodes,"unique_attribute_clue",(parent_id,),support)
    return None
def _candidate_intersection_concealment(entity_id,find_nodes,program,world,critical_fact_ids,parent_map,trace):
    nodes=program.node_map();entity_type=world.entities.get(entity_id)
    if not entity_type:return None
    find_set=set(find_nodes)
    for find_id in find_nodes:
        for parent_id in parent_map.get(find_id,()):
            parent=nodes[parent_id]
            if parent.op!="INTERSECT" or len(parent.dependencies)<2:continue
            other=[dep for dep in parent.dependencies if dep not in find_set]
            if not other:continue
            derived=None;valid=True
            for dep in other:
                if dep not in trace:valid=False;break
                values=_as_set(trace[dep]);derived=values if derived is None else derived&values
            if not valid or derived!={str(entity_id)}:continue
            predicates=set()
            for dep in other:predicates.update(_branch_predicates(program,dep))
            support=_support_facts_for_entity(world,entity_id=entity_id,predicates=predicates,critical_fact_ids=critical_fact_ids)
            if critical_fact_ids and not support:continue
            return LatentEntityPlan(entity_id,entity_type,find_nodes,"unique_intersection_clue",tuple(sorted(other)),support)
    return None
def build_public_query_plan(record,program,world,*,max_latent_entities=2):
    explicit=_explicit_find_nodes(program);critical=_critical_fact_ids(record);parent_map=_parents(program)
    try:trace=SymbolicExecutor().execute_trace(program,world)
    except (ExecutionError,ValueError,TypeError):trace={}
    candidates=[]
    for entity_id,raw in sorted(explicit.items()):
        find_nodes=tuple(sorted(raw));intersection=_candidate_intersection_concealment(entity_id,find_nodes,program,world,critical,parent_map,trace)
        if intersection is not None:candidates.append((0,entity_id,intersection));continue
        filtered=_candidate_filter_concealment(entity_id,find_nodes,program,world,critical,parent_map)
        if filtered is not None:candidates.append((1,entity_id,filtered))
    candidates.sort(key=lambda item:(item[0],item[1]));chosen=tuple(item[2] for item in candidates[:max(0,int(max_latent_entities))]);return PublicQueryPlan(chosen,tuple(sorted(explicit)),_proof_entity_ids(world,critical),_minimum_required_source_count(record))
def query_grounding_batch_summary(plans:Iterable[Mapping[str,Any]]):
    rows=[dict(plan) for plan in plans]
    if not rows:return {"task_count":0,"tasks_with_latent_intermediate":0,"latent_intermediate_rate":0.}
    latent=[int(row.get("latent_intermediate_count",0) or 0) for row in rows];ratios=[float(row.get("exposed_proof_entity_ratio",0.) or 0.) for row in rows];source_counts=[int(v) for v in (row.get("minimum_required_source_count") for row in rows) if isinstance(v,int)];clue_count=sum(int(row.get("grounding_clue_count",0) or 0) for row in rows);supported=sum(int(row.get("grounding_clue_supported_count",0) or 0) for row in rows);return {"task_count":len(rows),"tasks_with_latent_intermediate":sum(v>0 for v in latent),"latent_intermediate_rate":sum(v>0 for v in latent)/len(rows),"latent_intermediate_count":{"mean":sum(latent)/len(latent),"median":median(latent),"max":max(latent,default=0)},"exposed_proof_entity_ratio_mean":sum(ratios)/len(ratios) if ratios else 0.,"minimum_required_source_count":{"mean":sum(source_counts)/len(source_counts) if source_counts else 0.,"median":median(source_counts) if source_counts else 0.,"min":min(source_counts,default=0),"max":max(source_counts,default=0)},"single_source_answerable_tasks":sum(bool(row.get("single_source_answerable")) for row in rows),"grounding_clue_count":clue_count,"grounding_clue_indispensability_rate":supported/clue_count if clue_count else 1.,"grounding_clue_decorative_rate":1.-supported/clue_count if clue_count else 0.}
