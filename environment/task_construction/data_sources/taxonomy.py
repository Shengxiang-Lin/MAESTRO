"""Compact taxonomy index for YAGO class compatibility checks."""
from __future__ import annotations
import json
from collections import defaultdict,deque
from dataclasses import dataclass,field
from pathlib import Path
from typing import Dict,Set
from .yago import iter_turtle_triples
SUBCLASS_PREDICATES={"rdfs:subClassOf","schema:subClassOf"}
@dataclass
class TaxonomyIndex:
    parents:Dict[str,Set[str]]=field(default_factory=dict); _ancestor_cache:Dict[str,frozenset[str]]=field(default_factory=dict,init=False,repr=False)
    @classmethod
    def from_turtle(cls,path,*,progress=False):
        parents=defaultdict(set)
        for triple in iter_turtle_triples(path,progress=progress,description="YAGO taxonomy: subclass index"):
            if triple.predicate in SUBCLASS_PREDICATES and not triple.object_is_literal:parents[triple.subject].add(triple.object)
        return cls({k:set(v) for k,v in parents.items()})
    @classmethod
    def from_json(cls,path):
        payload=json.loads(Path(path).read_text(encoding="utf-8"));raw=payload.get("parents",payload);return cls({str(k):set(map(str,v)) for k,v in raw.items()})
    def to_dict(self):return {"class_count":len(set(self.parents)|{p for values in self.parents.values() for p in values}),"edge_count":sum(len(v) for v in self.parents.values()),"parents":{k:sorted(v) for k,v in sorted(self.parents.items())}}
    def save(self,path):
        output=Path(path);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(self.to_dict(),ensure_ascii=False),encoding="utf-8")
    def ancestors(self,child,*,include_self=True):
        if child not in self._ancestor_cache:
            seen=set();queue=deque(self.parents.get(child,()))
            while queue:
                current=queue.popleft()
                if current in seen:continue
                seen.add(current);queue.extend(self.parents.get(current,()))
            self._ancestor_cache[child]=frozenset(seen)
        result=set(self._ancestor_cache[child]);
        if include_self:result.add(child)
        return result
    def is_subtype(self,child,parent):return bool(child and parent and (child==parent or parent in self.ancestors(child,include_self=False)))
    def compatible(self,left,right):return True if left is None or right is None else self.is_subtype(left,right) or self.is_subtype(right,left)
    def reconcile(self,left,right):
        if left is None:return right
        if right is None:return left
        if self.is_subtype(left,right):return left
        if self.is_subtype(right,left):return right
        return None

def build_taxonomy_index(taxonomy_path,output_path,*,progress=False):
    index=TaxonomyIndex.from_turtle(taxonomy_path,progress=progress);payload=index.to_dict();output=Path(output_path);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8");return {"class_count":payload["class_count"],"edge_count":payload["edge_count"],"output":str(output)}
