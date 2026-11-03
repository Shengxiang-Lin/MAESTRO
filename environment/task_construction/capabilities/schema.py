"""Capability and acquisition-topology schemas for P1 grounding."""
from __future__ import annotations
from dataclasses import asdict,dataclass,field
from typing import Any,Mapping
@dataclass(frozen=True)
class CapabilityRequirement:
    capability_id:str;specialist:str;source_types:tuple[str,...];tool_families:tuple[str,...];rationale:str
    def to_dict(self):return {**asdict(self),"source_types":list(self.source_types),"tool_families":list(self.tool_families)}
@dataclass(frozen=True)
class AcquisitionResource:
    resource_id:str;source_type:str;visibility:str;role:str;required_capability_ids:tuple[str,...];available_tool_families:tuple[str,...];metadata:Mapping[str,Any]=field(default_factory=dict);requires_anchors:tuple[str,...]=();produces_anchors:tuple[str,...]=();anchor_dependency_resource_ids:tuple[str,...]=()
    def to_dict(self):return {"resource_id":self.resource_id,"source_type":self.source_type,"visibility":self.visibility,"role":self.role,"required_capability_ids":list(self.required_capability_ids),"available_tool_families":list(self.available_tool_families),"metadata":dict(self.metadata),"requires_anchors":list(self.requires_anchors),"produces_anchors":list(self.produces_anchors),"anchor_dependency_resource_ids":list(self.anchor_dependency_resource_ids)}
@dataclass(frozen=True)
class AcquisitionTopology:
    task_id:str;mode:str;resources:tuple[AcquisitionResource,...];capabilities:tuple[CapabilityRequirement,...]=();discoverable_resource_ids:tuple[str,...]=();binding_audit:Mapping[str,Any]=field(default_factory=dict)
    def to_dict(self):return {"task_id":self.task_id,"mode":self.mode,"resources":[item.to_dict() for item in self.resources],"capabilities":[item.to_dict() for item in self.capabilities],"discoverable_resource_ids":list(self.discoverable_resource_ids),"binding_audit":dict(self.binding_audit)}
