"""Frozen task-design schema."""
from __future__ import annotations
from dataclasses import asdict,dataclass,field
from typing import Any,Mapping
@dataclass(frozen=True)
class ResourceVisibility:
    source_id:str;source_type:str;visibility:str;role:str;rationale:str
    def to_dict(self):return asdict(self)
    def public_dict(self):return {"source_type":self.source_type,"visibility":self.visibility,"role":self.role}
@dataclass(frozen=True)
class ClueGraphNode:
    clue_id:str;kind:str;output_type:str|None;semantics:Mapping[str,Any]
    def to_dict(self):return {"clue_id":self.clue_id,"kind":self.kind,"output_type":self.output_type,**dict(self.semantics)}
    def public_dict(self):return {k:v for k,v in self.to_dict().items() if not k.startswith("schema_")}
@dataclass(frozen=True)
class ClueGraphEdge:
    source:str;target:str
    def public_dict(self):return {"source":self.source,"target":self.target}
@dataclass(frozen=True)
class ClueGraph:
    nodes:tuple[ClueGraphNode,...];edges:tuple[ClueGraphEdge,...];output_ids:tuple[str,...]
    def to_dict(self):return {"nodes":[item.to_dict() for item in self.nodes],"edges":[item.public_dict() for item in self.edges],"output_ids":list(self.output_ids)}
    def public_dict(self):return {"nodes":[item.public_dict() for item in self.nodes],"edges":[item.public_dict() for item in self.edges],"output_ids":list(self.output_ids)}
@dataclass(frozen=True)
class TaskDesign:
    task_id:str;motif:str;final_operation:str;answer_type:str;entry_strategy:str;latent_entity_ids:tuple[str,...];latent_find_node_ids:tuple[int,...];public_clues:tuple[Mapping[str,Any],...];required_reasoning_node_ids:tuple[int,...];resource_visibility:tuple[ResourceVisibility,...];minimum_required_source_count:int|None;metadata:Mapping[str,Any];source_semantics:Mapping[str,Any]=field(default_factory=dict)
    def to_dict(self):
        payload=asdict(self);payload["public_clues"]=[dict(item) for item in self.public_clues];payload["resource_visibility"]=[item.to_dict() for item in self.resource_visibility];payload["metadata"]=dict(self.metadata);payload.pop("source_semantics",None)
        if self.source_semantics:payload["source_semantics"]=dict(self.source_semantics)
        return payload
    @staticmethod
    def _public_clue(item):
        clue={k:v for k,v in item.items() if k not in {"entity_id","fact_ids","node_ids","source_ids"}};graph=clue.get("clue_graph")
        if isinstance(graph,Mapping):
            sanitized=dict(graph);raw_nodes=graph.get("nodes",())
            if isinstance(raw_nodes,(list,tuple)):sanitized["nodes"]=[{k:v for k,v in node.items() if not str(k).startswith("schema_")} for node in raw_nodes if isinstance(node,Mapping)]
            clue["clue_graph"]=sanitized
        return clue
    def public_contract(self):
        result={"task_motif":self.motif,"final_operation":self.final_operation,"answer_type":self.answer_type,"entry_strategy":self.entry_strategy,"public_clues":[self._public_clue(item) for item in self.public_clues],"resources":[item.public_dict() for item in self.resource_visibility]}
        if self.source_semantics:result["source_semantics"]=dict(self.source_semantics)
        return result
