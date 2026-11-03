"""Render varied natural-language questions directly from executable ProgramIR.

The renderer is deterministic for a given seed and uses only the bound program
plus the shared surface lexicon. It never reads the oracle answer. A separate
proof-aware PublicQueryPlan may hide an explicit intermediate identity when an
indispensable clue uniquely identifies the same entity.
"""
from __future__ import annotations
import hashlib
from typing import Dict
from ..ir import ProgramIR,ProgramNode
from .naming import EntityAliasBook,comparison_phrase,predicate_label,select_mode_phrase,type_label
from .query_grounding import PublicQueryPlan
from .literal_context import unitless_numeric_claim,value_predicate
def _stable_variant(*parts:object,modulo:int)->int:
    digest=hashlib.sha1("|".join(map(str,parts)).encode("utf-8")).digest();return int.from_bytes(digest[:4],"big")%max(modulo,1)
def _input_value(value,aliases,*,predicate=None):
    token=str(value)
    if token.startswith("entity:"):return aliases.entity(token.split(":",1)[1])
    if token.startswith("type:"):return type_label(token.split(":",1)[1])
    return aliases.literal(token,predicate=predicate)
def _relation_description(relation,dep,direction):
    key=relation.strip().lower()
    if direction=="backward":mapping={"alumni of":f"entities that are alumni of {dep}","spouse":f"entities married to {dep}","part of":f"entities that are part of {dep}","award":f"entities that received {dep}","has cause":f"entities whose recorded cause is {dep}","sponsor":f"entities sponsored by {dep}","member of":f"entities that are members of {dep}","parent taxon":f"taxa whose parent taxon is {dep}"}
    else:mapping={"alumni of":f"organizations that {dep} is an alum of","spouse":f"the spouses of {dep}","part of":f"entities that {dep} is part of","award":f"awards received by {dep}","has cause":f"causes recorded for {dep}","sponsor":f"sponsors recorded for {dep}","member of":f"entities that {dep} belongs to","parent taxon":f"the parent taxon of {dep}"}
    return mapping.get(key)
class QuestionRenderer:
    def __init__(self,aliases:EntityAliasBook,*,seed:int=17,grounding_plan:PublicQueryPlan|None=None):self.aliases=aliases;self.seed=int(seed);self.grounding_plan=grounding_plan or PublicQueryPlan();self.template_id="unrendered";self._program_signature="";self._variants=[]
    def _variant(self,node_id,family,modulo):
        value=_stable_variant(self.seed,self._program_signature,node_id,family,modulo=modulo);self._variants.append(f"{family}:{value}");return value
    def _concealment(self,node_id):return self.grounding_plan.concealment_for_node(node_id)
    @staticmethod
    def _join_conditions(deps,connector):
        if not deps:return "the stated conditions"
        if len(deps)==1:return deps[0]
        if len(deps)==2:return f"{deps[0]} {connector} {deps[1]}"
        return "; ".join(deps[:-1])+f"; {connector} {deps[-1]}"
    def _describe(self,node:ProgramNode,values:Dict[int,str],program:ProgramIR|None=None):
        deps=[values[d] for d in node.dependencies];inputs=list(node.inputs);op=node.op
        if op=="FIND":
            concealment=self._concealment(node.node_id)
            return f"the {type_label(concealment.entity_type)} identified by the other stated clues" if concealment is not None else _input_value(inputs[0],self.aliases) if inputs else "the referenced entity"
        if op=="FIND_ALL":return ("the available entities","the entities in the records","all candidate entities")[self._variant(node.node_id,"find_all",3)]
        if op=="FILTER_TYPE":
            kind=type_label(inputs[0].split(":",1)[-1]) if inputs else "Entity";variant=self._variant(node.node_id,"filter_type",3)
            return (f"{kind} entities among {deps[0]}",f"{kind} entities drawn from {deps[0]}",f"{deps[0]} restricted to {kind} entities")[variant]
        if op=="RELATE":
            relation=predicate_label(inputs[0]) if inputs else "relation";direction=inputs[1] if len(inputs)>1 else "forward";natural=_relation_description(relation,deps[0],direction)
            if natural is not None:return natural
            variant=self._variant(node.node_id,"relate",4)
            options=(f"entities whose {relation} points to {deps[0]}",f"entities linked to {deps[0]} as the source of {relation}",f"entities for which {deps[0]} is the {relation} target",f"entities connected into {deps[0]} by {relation}") if direction=="backward" else (f"entities connected from {deps[0]} by {relation}",f"the {relation} targets of {deps[0]}",f"entities reached through {relation} from {deps[0]}",f"entities linked to {deps[0]} via {relation}")
            return options[variant]
        if op.startswith("FILTER_") and not op.startswith("FILTER_QUALIFIER"):
            attr=predicate_label(inputs[0]) if inputs else "attribute";comparator=comparison_phrase(inputs[1]) if len(inputs)>1 else "equal to";target=_input_value(inputs[2],self.aliases,predicate=inputs[0]) if len(inputs)>2 else "the target value";direct=self._concealment(node.dependencies[0]) if len(node.dependencies)==1 else None;variant=self._variant(node.node_id,"filter_value",3)
            if direct is not None:
                kind=type_label(direct.entity_type);return (f"the {kind} whose {attr} is {comparator} {target}",f"the {kind} identified by having {attr} {comparator} {target}",f"the {kind} with {attr} {comparator} {target}")[variant]
            return (f"{deps[0]} with {attr} {comparator} {target}",f"candidates in {deps[0]} whose {attr} is {comparator} {target}",f"{deps[0]} after requiring {attr} to be {comparator} {target}")[variant]
        if op=="INTERSECT":
            visible=[values[d] for d in node.dependencies if self._concealment(d) is None]
            if visible and len(visible)<len(node.dependencies):
                if len(visible)==1:return visible[0]
                deps=visible
            joined=self._join_conditions(deps,"and");variant=self._variant(node.node_id,"intersect",4);return (f"entities common to {joined}",f"entities meeting both sides of: {joined}" if len(deps)==2 else f"entities meeting every condition in: {joined}",f"the overlap between {joined}",f"entities that satisfy each of these constraints: {joined}")[variant]
        if op=="UNION":
            joined=self._join_conditions(deps,"or");return (f"entities appearing in at least one of {joined}",f"the union of {joined}",f"entities satisfying either side of {joined}")[self._variant(node.node_id,"union",3)]
        if op=="QUERY_ATTR":return f"the {predicate_label(inputs[0]) if inputs else 'attribute'} values recorded for {deps[0]}"
        if op=="QUERY_RELATION":return f"the relation linking {deps[0]} to {deps[1]}"
        if op in {"SELECT_AMONG","SELECT_BETWEEN"}:
            attr=predicate_label(inputs[0]) if inputs else "attribute";mode=select_mode_phrase(inputs[1]) if len(inputs)>1 else "largest";pool=deps[0] if len(deps)==1 else self._join_conditions(deps,"or");return f"the entity in {pool} with the {mode} {attr} value"
        if op=="COUNT":return f"the number of {deps[0]}"
        if op=="QUERY_NAME":return deps[0]
        if op.startswith("VERIFY_"):
            comparator=comparison_phrase(inputs[0]) if inputs else "equal to";predicate=value_predicate(program,node.dependencies[0]) if program and node.dependencies else None;target=(inputs[1] if unitless_numeric_claim(node,predicate) else _input_value(inputs[1],self.aliases,predicate=predicate)) if len(inputs)>1 else "the target value";return f"whether a value in {deps[0]} is {comparator} {target}"
        return f"the result of {op.lower()} over "+", ".join(deps or inputs)
    def render(self,program):
        if not program.nodes:raise ValueError("cannot render a question from an empty program")
        self._program_signature=program.canonical_signature();self._variants=[];node_map=program.node_map();order=program.topological_order();descriptions={}
        for node_id in order:descriptions[node_id]=self._describe(node_map[node_id],descriptions,program)
        final=node_map[order[-1]];deps=[descriptions[d] for d in final.dependencies];inputs=list(final.inputs);variant=self._variant("final",final.op.lower(),4)
        if final.op=="COUNT":options=(f"How many entities match this description: {deps[0]}?",f"What is the number of entities described by the following conditions: {deps[0]}?",f"Count the entities that fit these conditions: {deps[0]}.",f"How many candidates remain after applying these conditions: {deps[0]}?")
        elif final.op=="QUERY_NAME":options=(f"Which entity or entities match these conditions: {deps[0]}?",f"Identify the entity or entities described here: {deps[0]}.",f"Which candidates satisfy the following constraints: {deps[0]}?",f"What entity or entities are selected by these conditions: {deps[0]}?")
        elif final.op=="QUERY_ATTR":
            attr=predicate_label(inputs[0]) if inputs else "attribute";options=(f"What is the {attr} recorded for {deps[0]}?",f"Which {attr} value belongs to {deps[0]}?",f"Report the {attr} for {deps[0]}.",f"What {attr} value is associated with {deps[0]}?")
        elif final.op=="QUERY_RELATION":options=(f"What relation connects {deps[0]} to {deps[1]}?",f"Which relation links {deps[0]} with {deps[1]}?",f"Identify the relation from {deps[0]} to {deps[1]}.",f"How are {deps[0]} and {deps[1]} related?")
        elif final.op in {"SELECT_AMONG","SELECT_BETWEEN"}:
            attr=predicate_label(inputs[0]) if inputs else "attribute";mode=select_mode_phrase(inputs[1]) if len(inputs)>1 else "largest";pool=deps[0] if len(deps)==1 else self._join_conditions(deps,"or");options=(f"Among {pool}, which entity has the {mode} {attr} value?",f"Select the entity with the {mode} {attr} from {pool}.",f"Which candidate in {pool} ranks {mode} by {attr}?",f"From {pool}, identify the entity whose {attr} is {mode}.")
        elif final.op.startswith("VERIFY_"):
            comparator=comparison_phrase(inputs[0]) if inputs else "equal to";predicate=value_predicate(program,final.dependencies[0]);target=(inputs[1] if unitless_numeric_claim(final,predicate) else _input_value(inputs[1],self.aliases,predicate=predicate)) if len(inputs)>1 else "the target value";options=(f"Is any value in {deps[0]} {comparator} {target}?",f"Does {deps[0]} contain a value {comparator} {target}?",f"Determine whether a value in {deps[0]} is {comparator} {target}.",f"Can any value from {deps[0]} be verified as {comparator} {target}?")
        else:options=(f"What is {descriptions[final.node_id]}?",f"Determine {descriptions[final.node_id]}.",f"What result follows from {descriptions[final.node_id]}?",f"Report {descriptions[final.node_id]}.")
        question=options[variant];latent_rules=",".join(sorted(item.rule for item in self.grounding_plan.latent_entities)) or "none";self.template_id=f"{final.op.lower()}|ground:{latent_rules}|"+"|".join(self._variants);return question
