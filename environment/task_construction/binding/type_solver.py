"""Small union/find type constraint solver for ProgramIR entity-set nodes."""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Dict,Iterable
from ..ir import ProgramIR
from ..data_sources.taxonomy import TaxonomyIndex
ENTITY_PRESERVING_OPS={"FILTER_TYPE","FILTER_STRING","FILTER_NUMBER","FILTER_YEAR","FILTER_DATE","INTERSECT","UNION","SELECT_BETWEEN","SELECT_AMONG"}
@dataclass
class TypeConstraintSolver:
    taxonomy:TaxonomyIndex;parent:Dict[int,int]=field(default_factory=dict);assigned:Dict[int,str]=field(default_factory=dict)
    @classmethod
    def for_program(cls,program:ProgramIR,taxonomy:TaxonomyIndex):
        solver=cls(taxonomy=taxonomy,parent={node.node_id:node.node_id for node in program.nodes})
        for node in program.nodes:
            if node.op in ENTITY_PRESERVING_OPS:
                for dep in node.dependencies:solver.union(node.node_id,dep)
        return solver
    def clone(self):return TypeConstraintSolver(self.taxonomy,dict(self.parent),dict(self.assigned))
    def find(self,node_id):
        root=self.parent[node_id]
        if root!=node_id:self.parent[node_id]=self.find(root)
        return self.parent[node_id]
    def union(self,left,right):
        a,b=self.find(left),self.find(right)
        if a==b:return True
        a_type,b_type=self.assigned.get(a),self.assigned.get(b);merged=self.taxonomy.reconcile(a_type,b_type)
        if a_type is not None and b_type is not None and merged is None:return False
        self.parent[b]=a;self.assigned.pop(b,None)
        if merged is not None:self.assigned[a]=merged
        return True
    def constrain(self,node_id,entity_type):
        if entity_type is None:return True
        root=self.find(node_id);current=self.assigned.get(root);merged=self.taxonomy.reconcile(current,entity_type)
        if current is not None and merged is None:return False
        self.assigned[root]=merged or entity_type;return True
    def can_constrain(self,node_id,entity_type):trial=self.clone();return trial.constrain(node_id,entity_type)
    def get(self,node_id):return self.assigned.get(self.find(node_id))
    def materialize(self,node_ids:Iterable[int])->Dict[int,str]:return {node_id:value for node_id in node_ids if (value:=self.get(node_id)) is not None}
