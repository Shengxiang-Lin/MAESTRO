"""Executable symbolic-world synthesis for a bound ProgramIR."""
from __future__ import annotations
import random
from typing import Any,Mapping
from ..binding.program_binder import BindingPlan
from ..executor import SymbolicExecutor,ExecutionError
from ..ir import ProgramIR
from ..semantic.literal_factory import SemanticLiteralFactory
from ..world import Fact,SyntheticWorld
class WorldSynthesisError(RuntimeError):pass
class WorldSampler:
    def __init__(self,*,seed:int=17):self.seed=int(seed);self.rng=random.Random(seed);self.literal_factory=SemanticLiteralFactory(seed=seed);self._entity_counter=0;self._fact_counter=0
    def _entity(self,world,entity_type):
        self._entity_counter+=1;entity_id=f"e{self._entity_counter:04d}";world.entities[entity_id]=str(entity_type or "schema:Thing");return entity_id
    def _fact(self,world,subject,predicate,obj,kind="entity"):
        self._fact_counter+=1;fact=Fact(f"f{self._fact_counter:05d}",str(subject),str(predicate),obj,str(kind),None);world.add_fact(fact);return fact
    @staticmethod
    def _entities(value):
        if isinstance(value,set):return set(map(str,value))
        if isinstance(value,(list,tuple)):return set(map(str,value))
        return {str(value)} if value is not None else set()
    def synthesize(self,plan:BindingPlan)->SyntheticWorld:
        program=plan.program;world=SyntheticWorld(metadata={"binding":plan.to_dict(),"synthesis_seed":self.seed});values={};nodes=program.node_map()
        for node_id in program.topological_order():
            node=nodes[node_id];op=node.op;deps=[values.get(dep) for dep in node.dependencies];binding=plan.attribute_bindings.get(node_id)
            if op in {"FIND","FIND_ALL"}:
                entity=self._entity(world,plan.node_types.get(node_id,"schema:Thing"));values[node_id]={entity};continue
            if op=="FILTER_TYPE":values[node_id]=set(self._entities(deps[0]));continue
            if op=="RELATE":
                source=next(iter(self._entities(deps[0])),None)
                if source is None:raise WorldSynthesisError("RELATE dependency has no entity")
                relation=plan.relation_bindings[node_id];target=self._entity(world,plan.node_types.get(node_id,relation.object_type));direction=node.inputs[1] if len(node.inputs)>1 else relation.direction
                if direction=="backward":self._fact(world,target,relation.predicate,source,"entity")
                else:self._fact(world,source,relation.predicate,target,"entity")
                values[node_id]={target};continue
            if op.startswith("FILTER_") and not op.startswith("FILTER_QUALIFIER"):
                candidates=self._entities(deps[0]);values[node_id]=set(candidates)
                if not candidates:continue
                subject=next(iter(candidates));attr=plan.attribute_bindings.get(node_id);predicate=str(node.inputs[0]);kind={"FILTER_STRING":"string","FILTER_NUMBER":"number","FILTER_YEAR":"year","FILTER_DATE":"date"}.get(op,"string");target=node.inputs[2] if len(node.inputs)>2 else None
                if target is None and attr is not None:target=self.literal_factory.value(attr,ordinal=4)
                if kind=="number":
                    try:target=float(target);target=int(target) if target.is_integer() else target
                    except Exception:target=self.literal_factory.value(attr,ordinal=4) if attr else 10
                elif kind=="year":
                    try:target=int(str(target)[:4])
                    except Exception:target=2018
                self._fact(world,subject,predicate,target,kind);continue
            if op=="INTERSECT":
                sets=[self._entities(dep) for dep in deps];common=set.intersection(*sets) if sets else set()
                if not common and sets:
                    anchor=next(iter(sets[0]),None)
                    if anchor is not None:
                        for dep_id,entities in zip(node.dependencies[1:],sets[1:]):
                            if anchor not in entities:
                                values[dep_id]={anchor}
                        common={anchor}
                values[node_id]=common;continue
            if op=="UNION":
                merged=set()
                for dep in deps:merged|=self._entities(dep)
                values[node_id]=merged;continue
            if op=="COUNT":values[node_id]=len(self._entities(deps[0]));continue
            if op=="QUERY_NAME":values[node_id]=sorted(self._entities(deps[0]));continue
            if op=="QUERY_ATTR":
                entities=self._entities(deps[0]);attr=plan.attribute_bindings.get(node_id);predicate=str(node.inputs[0]);observed=[]
                for index,entity in enumerate(sorted(entities)):
                    value=self.literal_factory.value(attr,ordinal=3+index) if attr else f"value-{index+1}";kind=attr.value_kind if attr else "string";self._fact(world,entity,predicate,value,kind);observed.append(value)
                values[node_id]=observed;continue
            if op=="QUERY_RELATION":
                left=self._entities(deps[0]);right=self._entities(deps[1]);predicate="o3:relatedTo"
                if left and right:self._fact(world,next(iter(left)),predicate,next(iter(right)),"entity")
                values[node_id]=[predicate];continue
            if op in {"SELECT_AMONG","SELECT_BETWEEN"}:
                entities=set()
                for dep in deps:entities|=self._entities(dep)
                attr=plan.attribute_bindings.get(node_id);predicate=str(node.inputs[0]);minimum=str(node.inputs[1] if len(node.inputs)>1 else "max").lower() in {"min","smallest","less","<","<cmp:less>"};ordered=sorted(entities)
                for index,entity in enumerate(ordered):
                    ordinal=(1+index*3) if minimum else (7-index*3);value=self.literal_factory.value(attr,ordinal=max(1,min(8,ordinal))) if attr else index+1;kind=attr.value_kind if attr else "number";self._fact(world,entity,predicate,value,kind)
                values[node_id]={ordered[0] if minimum else ordered[-1]} if ordered else set();continue
            if op.startswith("VERIFY_"):
                values[node_id]=True;continue
            raise WorldSynthesisError(f"unsupported synthesis operator: {op}")
        # Final authority is the real executor, not the construction shortcuts above.
        try:answer=SymbolicExecutor().execute(program,world)
        except (ExecutionError,ValueError,TypeError) as exc:raise WorldSynthesisError(f"synthesized world is not executable: {exc}") from exc
        world.metadata["oracle_answer"]=answer;world.metadata["attribute_semantics"]={binding.predicate:{"family":binding.semantic_family,"canonical_unit":binding.canonical_unit,"value_kind":binding.value_kind,"minimum":binding.value_min,"maximum":binding.value_max,"integer_values":binding.integer_values,"confidence":binding.profile_confidence,"evidence":binding.profile_evidence} for binding in plan.attribute_bindings.values()};return world
__all__=["WorldSampler","WorldSynthesisError"]
