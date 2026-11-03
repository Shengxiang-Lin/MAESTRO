"""Core world and evidence data structures for the synthetic environment."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Set

@dataclass(frozen=True)
class Fact:
    fact_id: str; subject: str; predicate: str; object: Any; object_kind: str="entity"; source_id: str|None=None
    def to_dict(self)->Dict[str,Any]: return {"fact_id":self.fact_id,"subject":self.subject,"predicate":self.predicate,"object":self.object,"object_kind":self.object_kind,"source_id":self.source_id}
@dataclass(frozen=True)
class EvidenceSource:
    source_id:str; source_type:str; fact_ids:tuple[str,...]=(); metadata:Mapping[str,Any]=field(default_factory=dict)
    def to_dict(self)->Dict[str,Any]: return {"source_id":self.source_id,"source_type":self.source_type,"fact_ids":list(self.fact_ids),"metadata":dict(self.metadata)}
    def sort_key(self)->tuple[int,str]:
        try: pos=int(self.metadata.get("layout_position"))
        except (TypeError,ValueError): pos=10000
        return pos,self.source_id
@dataclass
class SyntheticWorld:
    entities:Dict[str,str]=field(default_factory=dict); facts:List[Fact]=field(default_factory=list); sources:Dict[str,EvidenceSource]=field(default_factory=dict); metadata:Dict[str,Any]=field(default_factory=dict)
    def copy(self): return SyntheticWorld(dict(self.entities),list(self.facts),dict(self.sources),dict(self.metadata))
    def add_entity(self,entity_id:str,entity_type:str="Entity")->None: self.entities[entity_id]=entity_type
    def add_fact(self,fact:Fact)->None:
        if any(x.fact_id==fact.fact_id for x in self.facts): raise ValueError(f"duplicate fact id: {fact.fact_id}")
        self.facts.append(fact); self.entities.setdefault(fact.subject,"Entity")
        if fact.object_kind=="entity": self.entities.setdefault(str(fact.object),"Entity")
    def has_statement(self,subject,predicate,obj,object_kind)->bool: return any(f.subject==subject and f.predicate==predicate and f.object==obj and f.object_kind==object_kind for f in self.facts)
    def add_source(self,source:EvidenceSource)->None:
        if source.source_id in self.sources: raise ValueError(f"duplicate source id: {source.source_id}")
        known={f.fact_id for f in self.facts}; missing=[i for i in source.fact_ids if i not in known]
        if missing: raise ValueError(f"source {source.source_id} references unknown facts: {missing}")
        self.sources[source.source_id]=source
    def facts_for_subject(self,subject,predicate=None): return [f for f in self.facts if f.subject==subject and (predicate is None or f.predicate==predicate)]
    def related(self,entities:Iterable[str],predicate:str,direction:str="forward")->Set[str]:
        seeds=set(entities)
        if direction=="forward": return {str(f.object) for f in self.facts if f.subject in seeds and f.predicate==predicate and f.object_kind=="entity"}
        if direction=="backward": return {f.subject for f in self.facts if f.predicate==predicate and f.object_kind=="entity" and str(f.object) in seeds}
        raise ValueError(f"unsupported direction: {direction}")
    def attribute_values(self,entities:Iterable[str],predicate:str)->Dict[str,List[Any]]:
        seeds=set(entities); result={e:[] for e in seeds}
        for f in self.facts:
            if f.subject in seeds and f.predicate==predicate: result[f.subject].append(f.object)
        return result
    def entities_of_type(self,entity_type:str)->Set[str]: return {e for e,k in self.entities.items() if k==entity_type}
    def without_facts(self,fact_ids:Iterable[str]):
        removed=set(fact_ids); kept=[f for f in self.facts if f.fact_id not in removed]; ids={f.fact_id for f in kept}; sources={sid:EvidenceSource(s.source_id,s.source_type,tuple(fid for fid in s.fact_ids if fid in ids),s.metadata) for sid,s in self.sources.items()}; return SyntheticWorld(dict(self.entities),kept,sources,dict(self.metadata))
    def with_fact_sources(self,assignments:Mapping[str,str]): return SyntheticWorld(dict(self.entities),[replace(f,source_id=assignments.get(f.fact_id,f.source_id)) for f in self.facts],dict(self.sources),dict(self.metadata))
    def without_sources(self,source_ids:Iterable[str]):
        removed=set(source_ids); removed_fact_ids=set()
        for sid in removed:
            s=self.sources.get(sid)
            if s: removed_fact_ids.update(s.fact_ids)
        kept=[f for f in self.facts if f.source_id not in removed and f.fact_id not in removed_fact_ids]; ids={f.fact_id for f in kept}; sources={}
        for sid,s in self.sources.items():
            if sid not in removed: sources[sid]=EvidenceSource(s.source_id,s.source_type,tuple(fid for fid in s.fact_ids if fid in ids),s.metadata)
        return SyntheticWorld(dict(self.entities),kept,sources,dict(self.metadata))
    def to_dict(self)->Dict[str,Any]: return {"entities":self.entities,"facts":[f.to_dict() for f in self.facts],"sources":{k:s.to_dict() for k,s in self.sources.items()},"metadata":self.metadata}
