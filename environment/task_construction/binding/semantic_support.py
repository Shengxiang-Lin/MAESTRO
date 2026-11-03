"""Empirical predicate/type support for semantic binding refinements."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any,Mapping
from ..data_sources.taxonomy import TaxonomyIndex
MIN_EMPIRICAL_TYPE_MASS=3;BROAD_SCHEMA_TYPES=frozenset({"schema:Thing","owl:Thing","rdfs:Resource"})
def empirical_type_counts(info:Mapping[str,Any],side:str)->tuple[tuple[str,int],...]:
    key="empirical_subject_types" if side=="subject" else "empirical_object_types";result=[]
    for item in info.get(key,[]):
        if not isinstance(item,(list,tuple)) or not item:continue
        entity_type=str(item[0])
        try:count=int(item[1]) if len(item)>1 else 1
        except (TypeError,ValueError):count=1
        if count>0:result.append((entity_type,count))
    return tuple(result)
def preferred_candidate_types(info:Mapping[str,Any],side:str)->list[str]:
    schema_key="domain" if side=="subject" else "range";schema_types=[str(item) for item in info.get("schema",{}).get(schema_key,[]) if not str(item).startswith(("xsd:","rdf:"))];empirical_types=[t for t,_ in empirical_type_counts(info,side)]
    if schema_types:
        informative=[t for t in schema_types if t not in BROAD_SCHEMA_TYPES]
        if informative:return informative
        if empirical_types:return list(dict.fromkeys(empirical_types))
        return schema_types
    return list(dict.fromkeys(empirical_types))
@dataclass(frozen=True)
class TypeSupportAssessment:
    accepted:bool;decision:str;side:str;actual_type:str|None;schema_type:str|None;empirical_mass:int;supporting_mass:int
    @property
    def support_share(self):return None if self.empirical_mass<=0 else self.supporting_mass/self.empirical_mass
    def to_dict(self):return {"accepted":self.accepted,"decision":self.decision,"side":self.side,"actual_type":self.actual_type,"schema_type":self.schema_type,"empirical_mass":self.empirical_mass,"supporting_mass":self.supporting_mass,"support_share":self.support_share}
def assess_type_refinement(info:Mapping[str,Any],taxonomy:TaxonomyIndex,*,side:str,actual_type:str|None,schema_type:str|None,min_empirical_mass:int=MIN_EMPIRICAL_TYPE_MASS)->TypeSupportAssessment:
    pairs=empirical_type_counts(info,side);mass=sum(c for _,c in pairs)
    if actual_type is None or schema_type is None:return TypeSupportAssessment(True,"unknown_type_fallback",side,actual_type,schema_type,mass,0)
    if actual_type==schema_type:return TypeSupportAssessment(True,"schema_exact",side,actual_type,schema_type,mass,mass)
    if not taxonomy.compatible(actual_type,schema_type):return TypeSupportAssessment(False,"taxonomy_incompatible",side,actual_type,schema_type,mass,0)
    supporting=sum(count for observed,count in pairs if taxonomy.is_subtype(observed,actual_type))
    if supporting>0:return TypeSupportAssessment(True,"empirical_refinement_supported",side,actual_type,schema_type,mass,supporting)
    if mass<max(1,int(min_empirical_mass)):return TypeSupportAssessment(True,"sparse_empirical_fallback",side,actual_type,schema_type,mass,0)
    return TypeSupportAssessment(False,"empirical_refinement_unsupported",side,actual_type,schema_type,mass,0)
