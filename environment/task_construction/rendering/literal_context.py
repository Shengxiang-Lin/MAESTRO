"""Shared predicate context for rendering facts, filters and terminal claims."""
from __future__ import annotations
import re
from typing import Any
def unitless_numeric_claim(node,predicate):return predicate is None and len(node.inputs)>1 and bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)",node.inputs[1]))
def value_predicate(program,node_id):
    node=program.node_map()[node_id];return node.inputs[0] if node.op=="QUERY_ATTR" and node.inputs else None
def surface_literal(draft,value:Any,*,predicate=None):
    scoped=draft.hidden_metadata.get("literal_surfaces_by_predicate",{})
    if predicate is not None and str(value) in scoped.get(predicate,{}):return str(scoped[predicate][str(value)])
    return str(draft.hidden_metadata.get("literal_surfaces",{}).get(str(value),value))
def collect_literal_surfaces(program,world,aliases):
    result={}
    def add(predicate,value,kind=None):
        if predicate is not None:result.setdefault(predicate,{})[str(value)]=aliases.literal(value,predicate=predicate,object_kind=kind)
    for fact in world.facts:
        if fact.object_kind!="entity":add(fact.predicate,fact.object,fact.object_kind)
    for node in program.nodes:
        if node.op.startswith("FILTER_") and node.op!="FILTER_TYPE" and len(node.inputs)>=3:add(node.inputs[0],node.inputs[2],"year" if node.op=="FILTER_YEAR" else "date" if node.op=="FILTER_DATE" else None)
        elif node.op.startswith("VERIFY_") and node.dependencies and len(node.inputs)>=2:add(value_predicate(program,node.dependencies[0]),node.inputs[1],"year" if node.op=="VERIFY_YEAR" else "date" if node.op=="VERIFY_DATE" else None)
    return result
