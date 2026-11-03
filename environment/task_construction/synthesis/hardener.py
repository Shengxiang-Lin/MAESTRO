"""Program-preserving hardening with coherent near-miss and distractor facts."""
from __future__ import annotations
import random
from dataclasses import replace
from typing import Any
from ..executor import SymbolicExecutor
from ..validation import normalize_answer
from ..world import Fact,SyntheticWorld
class HardeningError(RuntimeError):pass
def _next_entity(world,counter,entity_type):
    while True:
        counter[0]+=1;entity_id=f"d{counter[0]:04d}"
        if entity_id not in world.entities:world.entities[entity_id]=entity_type;return entity_id
def _next_fact(world,counter,subject,predicate,obj,kind):
    existing={fact.fact_id for fact in world.facts}
    while True:
        counter[0]+=1;fact_id=f"df{counter[0]:05d}"
        if fact_id not in existing:return Fact(fact_id,subject,predicate,obj,kind,None)
def harden_world(program,world,*,seed=17,near_miss_count=3,background_fact_count=4):
    rng=random.Random(seed);executor=SymbolicExecutor();baseline=normalize_answer(executor.execute(program,world));result=world.copy();entity_counter=[0];fact_counter=[0];mutations=[];facts=list(world.facts);entity_facts=[fact for fact in facts if fact.object_kind=="entity"]
    for index in range(max(0,int(near_miss_count))):
        if not entity_facts:break
        template=rng.choice(entity_facts);source_type=result.entities.get(template.subject,"schema:Thing");object_type=result.entities.get(str(template.object),"schema:Thing");subject=_next_entity(result,entity_counter,source_type);obj=_next_entity(result,entity_counter,object_type);added=[]
        # Copy several local constraints, then deliberately break one edge/value.
        related=[fact for fact in facts if fact.subject==template.subject][:4]
        if not related:related=[template]
        broken=rng.randrange(len(related))
        for j,fact in enumerate(related):
            if fact.object_kind=="entity":value=obj if str(fact.object)==str(template.object) else _next_entity(result,entity_counter,result.entities.get(str(fact.object),"schema:Thing"))
            else:value=fact.object
            predicate=fact.predicate
            if j==broken:
                if fact.object_kind=="entity":value=_next_entity(result,entity_counter,result.entities.get(str(fact.object),"schema:Thing"))
                elif isinstance(value,(int,float)) and not isinstance(value,bool):value=value+1+(index%3)
                else:value=f"{value}-near"
            new=_next_fact(result,fact_counter,subject,predicate,value,fact.object_kind);result.add_fact(new);added.append(new.fact_id)
        mutations.append({"kind":"coherent_near_miss_subgraph","candidate_entity_id":subject,"added_fact_ids":added,"broken_from_fact":related[broken].fact_id,"matched_constraints":max(0,len(related)-1),"support_degree":len(related)})
    for index in range(max(0,int(background_fact_count))):
        if not facts:break
        template=rng.choice(facts);subject=_next_entity(result,entity_counter,result.entities.get(template.subject,"schema:Thing"));value=template.object
        if template.object_kind=="entity":value=_next_entity(result,entity_counter,result.entities.get(str(template.object),"schema:Thing"))
        elif isinstance(value,(int,float)) and not isinstance(value,bool):value=value+10+index
        else:value=f"background-{index+1}"
        new=_next_fact(result,fact_counter,subject,template.predicate,value,template.object_kind);result.add_fact(new);mutations.append({"kind":"background_distractor","added_fact_ids":[new.fact_id]})
    try:answer=normalize_answer(executor.execute(program,result))
    except Exception as exc:raise HardeningError(f"hardened world became non-executable: {exc}") from exc
    if answer!=baseline:raise HardeningError("hardening changed the oracle answer")
    result.metadata={**result.metadata,"hardening":{"seed":seed,"mutation_count":len(mutations),"mutations":mutations}};return result,mutations
__all__=["HardeningError","harden_world"]
