"""Shared hardness policy and diversity signatures for synthetic tasks."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any,Dict
from ..ir import ProgramIR
@dataclass(frozen=True)
class HardnessProfile:
    qualifies:bool;label:str;depth_ok:bool;structural:bool;comparison:bool;temporal:bool;aggregation:bool;metrics:Dict[str,int]
    def to_dict(self)->Dict[str,Any]:return {"qualifies":self.qualifies,"label":self.label,"depth_ok":self.depth_ok,"structural":self.structural,"comparison":self.comparison,"temporal":self.temporal,"aggregation":self.aggregation,"metrics":dict(self.metrics)}
def pure_topology_signature(program:ProgramIR)->str:
    order=program.topological_order();remap={node_id:index for index,node_id in enumerate(order)};node_map=program.node_map();parts=[]
    for node_id in order:
        node=node_map[node_id];deps=",".join(str(remap[dep]) for dep in node.dependencies);parts.append(f"{remap[node_id]}:{node.op}<-{deps}")
    return "|".join(parts)
def complexity_profile(program:ProgramIR,*,min_depth:int=5)->HardnessProfile:
    metrics=program.metrics();depth_ok=metrics.get("depth",0)>=min_depth;structural=metrics.get("independent_branches",0)>=1 or metrics.get("set_ops",0)>=1 or metrics.get("merge_nodes",0)>=1 or metrics.get("fanout_branch_nodes",0)>=1;comparison=metrics.get("compare_ops",0)>=1 and (metrics.get("relation_ops",0)+metrics.get("filter_ops",0)+metrics.get("set_ops",0))>=1;temporal=metrics.get("temporal_ops",0)>=1 and metrics.get("node_count",0)>=max(6,min_depth) and (metrics.get("relation_ops",0)>=1 or metrics.get("aggregation_ops",0)>=1 or metrics.get("set_ops",0)>=1);aggregation=metrics.get("aggregation_ops",0)>=1 and metrics.get("node_count",0)>=max(6,min_depth) and (metrics.get("relation_ops",0)+metrics.get("filter_ops",0)+metrics.get("set_ops",0))>=2;qualifies=depth_ok and (structural or comparison or temporal or aggregation);label="comparison" if comparison else "temporal" if temporal else "multi_branch" if structural else "aggregation" if aggregation else "relational";return HardnessProfile(qualifies,label,depth_ok,structural,comparison,temporal,aggregation,metrics)
