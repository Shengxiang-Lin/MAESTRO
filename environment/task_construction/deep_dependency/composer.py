"""Programmatic composition of deeper ProgramIR candidates from KQA-Pro topology fragments.

The composer changes graph structure only.  Predicate/type semantics are assigned
later by Semantic Binding, so KQA values are never copied into synthetic truth.
"""
from __future__ import annotations
import random
from dataclasses import replace
from typing import Iterable,Sequence
from ..ir import ProgramIR,ProgramNode
_TERMINALS={"COUNT","QUERY_NAME","QUERY_ATTR","QUERY_RELATION","SELECT_BETWEEN","SELECT_AMONG","VERIFY_STRING","VERIFY_NUMBER","VERIFY_YEAR","VERIFY_DATE"}
_ENTITY_OPS={"FIND","FIND_ALL","RELATE","FILTER_TYPE","FILTER_STRING","FILTER_NUMBER","FILTER_YEAR","FILTER_DATE","INTERSECT","UNION","SELECT_BETWEEN","SELECT_AMONG"}
def _renumber(nodes):
    return [ProgramNode(index,node.op,tuple(node.dependencies),tuple(node.inputs),node.raw_function) for index,node in enumerate(nodes)]
def _fragment(program:ProgramIR,*,min_depth:int,max_depth:int,rng:random.Random)->list[ProgramNode]:
    order=program.topological_order();node_map=program.node_map();entity=[node_id for node_id in order if node_map[node_id].op in _ENTITY_OPS]
    if not entity:return []
    end=rng.choice(entity);ancestors=set();stack=[end]
    while stack and len(ancestors)<max_depth:
        current=stack.pop()
        if current in ancestors:continue
        ancestors.add(current);stack.extend(node_map[current].dependencies)
    selected=[node_id for node_id in order if node_id in ancestors]
    if len(selected)<min_depth:selected=entity[:min(len(entity),max_depth)]
    remap={old:index for index,old in enumerate(selected)};nodes=[]
    for old in selected:
        node=node_map[old];deps=tuple(remap[d] for d in node.dependencies if d in remap);nodes.append(ProgramNode(remap[old],node.op,deps,node.inputs,node.raw_function))
    return nodes
def _tail(nodes):return nodes[-1].node_id if nodes else None
def _attach_chain(base,fragment):
    if not base:return list(fragment)
    if not fragment:return list(base)
    offset=len(base);tail=_tail(base);result=list(base)
    for index,node in enumerate(fragment):
        deps=tuple(offset+d for d in node.dependencies)
        if not deps and index==0 and node.op in _ENTITY_OPS and node.op not in {"FIND","FIND_ALL"}:deps=(tail,)
        result.append(ProgramNode(offset+index,node.op,deps,node.inputs,node.raw_function))
    return result
def _add_branch_join(nodes,rng):
    entity_ids=[node.node_id for node in nodes if node.op in _ENTITY_OPS]
    if len(entity_ids)<2:return nodes
    left=entity_ids[-1];right=rng.choice(entity_ids[:-1]);op=rng.choice(("INTERSECT","UNION"));return [*nodes,ProgramNode(len(nodes),op,(left,right),(),op)]
def _terminal(nodes,template:ProgramIR,rng):
    terminals=[template.node_map()[node_id] for node_id in template.topological_order() if template.node_map()[node_id].op in _TERMINALS]
    terminal=rng.choice(terminals) if terminals else ProgramNode(-1,"QUERY_NAME",(),(),"QueryName");entity_ids=[node.node_id for node in nodes if node.op in _ENTITY_OPS]
    if not entity_ids:return nodes
    if terminal.op=="QUERY_RELATION" and len(entity_ids)>=2:deps=(entity_ids[-2],entity_ids[-1])
    elif terminal.op=="SELECT_BETWEEN" and len(entity_ids)>=2:deps=(entity_ids[-2],entity_ids[-1])
    else:deps=(entity_ids[-1],)
    return [*nodes,ProgramNode(len(nodes),terminal.op,deps,terminal.inputs,terminal.raw_function)]
def compose_deep_program(fragments:Sequence[ProgramIR],*,min_fragment_depth:int=2,max_fragment_depth:int=4,min_orchestration_depth:int=3,max_orchestration_depth:int=5,min_parallel_width:int=2,seed:int=17)->ProgramIR:
    if not fragments:raise ValueError("at least one ProgramIR fragment is required")
    rng=random.Random(seed);fragment_count=max(2,min(len(fragments),rng.randint(min_orchestration_depth,max_orchestration_depth)));chosen=rng.sample(list(fragments),fragment_count) if len(fragments)>=fragment_count else [rng.choice(fragments) for _ in range(fragment_count)];nodes=[];links=[]
    for index,program in enumerate(chosen):
        fragment=_fragment(program,min_depth=min_fragment_depth,max_depth=max_fragment_depth,rng=rng)
        before=len(nodes);nodes=_attach_chain(nodes,fragment);links.append({"fragment":index,"start":before,"end":len(nodes)-1})
    if min_parallel_width>=2:nodes=_add_branch_join(nodes,rng)
    nodes=_terminal(nodes,chosen[-1],rng);nodes=_renumber(nodes);metadata={"source":"kqa_pro_composition","deep_dependency":{"composition_links":links,"fragment_count":fragment_count,"min_fragment_depth":min_fragment_depth,"max_fragment_depth":max_fragment_depth,"orchestration_depth":fragment_count,"parallel_width":min_parallel_width,"seed":seed}};return ProgramIR(nodes,metadata=metadata)
def compose_candidate_stream(programs:Sequence[ProgramIR],*,count:int,seed:int=17,**kwargs)->Iterable[ProgramIR]:
    rng=random.Random(seed)
    for index in range(max(0,int(count))):yield compose_deep_program(programs,seed=rng.randrange(1<<31),**kwargs)
__all__=["compose_candidate_stream","compose_deep_program"]
