"""Agent-visible and hidden schemas for deterministic rendering."""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any,Mapping,Sequence
@dataclass(frozen=True)
class RenderedSource:
    source_id:str;source_type:str;title:str;content:Any;fact_ids:tuple[str,...]=();hidden_metadata:Mapping[str,Any]=field(default_factory=dict)
    def observation_dict(self):return {"source_id":self.source_id,"source_type":self.source_type,"title":self.title,"content":self.content}
    def debug_dict(self):return {**self.observation_dict(),"fact_ids":list(self.fact_ids),"hidden_metadata":dict(self.hidden_metadata)}
@dataclass(frozen=True)
class RenderedTask:
    task_id:str;question:str;sources:tuple[RenderedSource,...];answer:Any;raw_answer:Any;hidden_metadata:Mapping[str,Any]=field(default_factory=dict)
    def observation_dict(self):return {"task_id":self.task_id,"question":self.question,"sources":[s.observation_dict() for s in self.sources]}
    def training_dict(self,*,include_hidden=False):
        payload={**self.observation_dict(),"answer":self.answer}
        if include_hidden:payload["hidden"]={"raw_answer":self.raw_answer,"task":dict(self.hidden_metadata),"sources":{s.source_id:{"fact_ids":list(s.fact_ids),**dict(s.hidden_metadata)} for s in self.sources}}
        return payload
def visible_source_types(sources:Sequence[RenderedSource])->tuple[str,...]:return tuple(source.source_type for source in sources)
def latent_entity_ids(draft:RenderedTask)->set[str]:
    plan=draft.hidden_metadata.get("public_query_plan",{})
    return {str(v) for v in plan.get("latent_entity_ids",())} if isinstance(plan,Mapping) else set()
