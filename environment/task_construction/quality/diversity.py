"""Batch-diversity accounting and bounded concentration caps."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Any,Iterable,Mapping
@dataclass(frozen=True)
class DiversityCaps:
    topology:int;predicate:int;type_pair:int;terminal_family:int
    def to_dict(self):return {"topology":self.topology,"predicate":self.predicate,"type_pair":self.type_pair,"terminal_family":self.terminal_family}
def diversity_caps(accepted_count:int)->DiversityCaps:
    n=max(1,int(accepted_count));return DiversityCaps(max(2,int(.18*n)+1),max(3,int(.28*n)+1),max(3,int(.30*n)+1),max(4,int(.42*n)+1))
class BatchDiversityTracker:
    def __init__(self,target=None):self.target=target;self.topology=Counter();self.predicate=Counter();self.type_pair=Counter();self.terminal_family=Counter();self.accepted=0
    def caps(self):return diversity_caps(self.target if self.target is not None else self.accepted)
    def would_accept(self,*,topology:str,predicates:Iterable[str],type_pairs:Iterable[str],terminal_family:str)->tuple[bool,list[str]]:
        caps=self.caps();reasons=[]
        if self.topology[topology]+1>caps.topology:reasons.append("topology_cap")
        if any(self.predicate[str(p)]+1>caps.predicate for p in set(predicates)):reasons.append("predicate_cap")
        if any(self.type_pair[str(p)]+1>caps.type_pair for p in set(type_pairs)):reasons.append("type_pair_cap")
        if self.terminal_family[str(terminal_family)]+1>caps.terminal_family:reasons.append("terminal_family_cap")
        return not reasons,reasons
    def add(self,*,topology:str,predicates:Iterable[str],type_pairs:Iterable[str],terminal_family:str):
        self.accepted+=1;self.topology[str(topology)]+=1;self.predicate.update(set(map(str,predicates)));self.type_pair.update(set(map(str,type_pairs)));self.terminal_family[str(terminal_family)]+=1
    def summary(self)->dict[str,Any]:return {"accepted":self.accepted,"caps":self.caps().to_dict(),"topology":dict(self.topology.most_common()),"predicate":dict(self.predicate.most_common()),"type_pair":dict(self.type_pair.most_common()),"terminal_family":dict(self.terminal_family.most_common())}
__all__=["BatchDiversityTracker","DiversityCaps","diversity_caps"]
