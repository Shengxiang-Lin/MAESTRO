"""Benchmark-independent task program IR for synthetic environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

SET_OPS = {"INTERSECT", "UNION"}
COMPARE_OPS = {"COMPARE", "SELECT_BETWEEN", "SELECT_AMONG"}
TEMPORAL_OPS = {"FILTER_YEAR", "FILTER_DATE", "VERIFY_YEAR", "VERIFY_DATE"}
QUALIFIER_OPS = {"QUERY_ATTR_QUALIFIER", "QUERY_RELATION_QUALIFIER", "FILTER_QUALIFIER_STRING", "FILTER_QUALIFIER_NUMBER", "FILTER_QUALIFIER_YEAR", "FILTER_QUALIFIER_DATE", "QUERY_ATTR_CONDITIONAL"}

@dataclass(frozen=True)
class ProgramNode:
    node_id: int
    op: str
    dependencies: Tuple[int, ...] = ()
    inputs: Tuple[str, ...] = ()
    raw_function: str | None = None
    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "op": self.op, "dependencies": list(self.dependencies), "inputs": list(self.inputs), "raw_function": self.raw_function}

@dataclass
class ProgramIR:
    nodes: List[ProgramNode]
    metadata: Dict[str, Any] = field(default_factory=dict)
    def __post_init__(self) -> None:
        ids=[n.node_id for n in self.nodes]
        if len(ids)!=len(set(ids)): raise ValueError("ProgramIR node ids must be unique")
        known=set(ids)
        for node in self.nodes:
            missing=[dep for dep in node.dependencies if dep not in known]
            if missing: raise ValueError(f"node {node.node_id} has unknown dependencies: {missing}")
            if any(dep==node.node_id for dep in node.dependencies): raise ValueError(f"node {node.node_id} depends on itself")
        self.topological_order()
    def node_map(self)->Dict[int,ProgramNode]: return {n.node_id:n for n in self.nodes}
    def topological_order(self)->List[int]:
        node_map=self.node_map(); indegree={i:0 for i in node_map}; children={i:[] for i in node_map}
        for node in self.nodes:
            indegree[node.node_id]=len(node.dependencies)
            for dep in node.dependencies: children[dep].append(node.node_id)
        queue=sorted(i for i,d in indegree.items() if d==0); order=[]
        while queue:
            cur=queue.pop(0); order.append(cur)
            for child in sorted(children[cur]):
                indegree[child]-=1
                if indegree[child]==0: queue.append(child); queue.sort()
        if len(order)!=len(self.nodes): raise ValueError("ProgramIR dependency graph contains a cycle")
        return order
    def canonical_signature(self)->str:
        node_map=self.node_map(); pieces=[]
        for node_id in self.topological_order():
            node=node_map[node_id]; deps=",".join(str(d) for d in sorted(node.dependencies)); inputs=",".join(node.inputs); pieces.append(f"{node_id}:{node.op}<-{deps}[{inputs}]")
        return "|".join(pieces)
    def metrics(self)->Dict[str,int]:
        order=self.topological_order(); node_map=self.node_map(); depth={}; outdegree={i:0 for i in node_map}
        for node in self.nodes:
            for dep in node.dependencies: outdegree[dep]+=1
        for node_id in order:
            deps=node_map[node_id].dependencies; depth[node_id]=1 if not deps else 1+max(depth[d] for d in deps)
        root_count=sum(1 for n in self.nodes if not n.dependencies); sink_count=sum(1 for v in outdegree.values() if v==0); fanout_nodes=sum(1 for v in outdegree.values() if v>1); merge_nodes=sum(1 for n in self.nodes if len(n.dependencies)>1); independent_branches=max(root_count-1,0); ops=[n.op for n in self.nodes]
        return {"node_count":len(self.nodes),"depth":max(depth.values(),default=0),"root_count":root_count,"sink_count":sink_count,"independent_branches":independent_branches,"branch_nodes":fanout_nodes,"fanout_branch_nodes":fanout_nodes,"merge_nodes":merge_nodes,"set_ops":sum(op in SET_OPS for op in ops),"compare_ops":sum(op in COMPARE_OPS for op in ops),"temporal_ops":sum(op in TEMPORAL_OPS for op in ops),"qualifier_ops":sum(op in QUALIFIER_OPS for op in ops),"relation_ops":sum(op=="RELATE" for op in ops),"filter_ops":sum(op.startswith("FILTER_") for op in ops),"aggregation_ops":sum(op in {"COUNT","QUERY_ATTR"} for op in ops)}
    def to_dict(self)->Dict[str,Any]: return {"nodes":[n.to_dict() for n in self.nodes],"metadata":self.metadata,"signature":self.canonical_signature(),"metrics":self.metrics()}

def abstract_scalar(value: Any)->str:
    if isinstance(value,bool): return "<BOOL>"
    if isinstance(value,(int,float)): return "<NUMBER>"
    text=str(value).strip(); lowered=text.lower()
    if lowered in {"forward","backward"}: return lowered
    if lowered in {"=","==","!=",">","<",">=","<=","greater","less"}: return f"<CMP:{lowered}>"
    if len(text)==4 and text.isdigit(): return "<YEAR>"
    try: float(text.replace(",",""))
    except ValueError: pass
    else: return "<NUMBER>"
    if any(t in text for t in ("-","/")) and any(ch.isdigit() for ch in text): return "<DATE_OR_ID>"
    return "<SYMBOL>"

def dependency_depth(nodes: Sequence[ProgramNode])->int: return ProgramIR(list(nodes)).metrics()["depth"]
