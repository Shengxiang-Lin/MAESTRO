"""Typed predicate selection from the compact YAGO predicate catalog."""
from __future__ import annotations
import math,random
from dataclasses import dataclass
from typing import Any,Dict,List,Mapping,Tuple
from ..literals import datatype_kind,literal_kind
from ..semantic.profile import AttributeSemanticProfile,compile_attribute_profile
from ..data_sources.taxonomy import TaxonomyIndex
from .diversity_guidance import BindingDiversityGuidance
from .semantic_support import preferred_candidate_types
from .type_solver import TypeConstraintSolver
@dataclass(frozen=True)
class RelationBinding: predicate:str;subject_type:str;object_type:str;direction:str
@dataclass(frozen=True)
class AttributeBinding:
    predicate:str;entity_type:str;value_kind:str;datatype:str|None=None;semantic_family:str="generic";canonical_unit:str|None=None;value_min:float|None=None;value_max:float|None=None;integer_values:bool=False;profile_confidence:float=0.;profile_evidence:str=""
def _attribute_kinds(info:Mapping[str,Any])->Dict[str,str|None]:
    kinds={}
    for datatype in info.get("schema",{}).get("range",[]):
        kind=datatype_kind(str(datatype))
        if kind:kinds[kind]=str(datatype)
    for sample in info.get("samples",[]):
        raw=sample.get("object")
        if raw is None:continue
        kind=literal_kind(str(raw))
        if kind:kinds.setdefault(kind,None)
    return kinds
class PredicateBinder:
    def __init__(self,catalog:Mapping[str,Any],taxonomy:TaxonomyIndex,*,seed:int=17,min_count:int=25,diversity_guidance:BindingDiversityGuidance|None=None)->None:
        self.catalog=catalog;self.taxonomy=taxonomy;self.rng=random.Random(seed);self.min_count=min_count;self.diversity_guidance=diversity_guidance;self._cached_relation_options=None;self._attribute_profile_cache={};self._chosen_attribute_predicates={};self.profile_guidance_events=0;self.profile_guidance_supported_events=0
        from ..quality.observed_support import ObservedSupport
        self.observed_support=ObservedSupport(catalog,taxonomy)
    def _relation_options(self):
        if self._cached_relation_options is not None:return self._cached_relation_options
        options=[]
        for predicate,info in self.catalog.get("predicates",{}).items():
            count=int(info.get("count",0))
            if count<self.min_count or info.get("kind")!="relation":continue
            subjects=preferred_candidate_types(info,"subject");objects=preferred_candidate_types(info,"object")
            for subject_type in subjects[:12]:
                for object_type in objects[:12]:
                    if not self.observed_support.edge(str(predicate),subject_type,object_type)["accepted"]:continue
                    options.append((count,RelationBinding(str(predicate),subject_type,object_type,"forward")))
        self._cached_relation_options=options;return options
    @staticmethod
    def _relation_type_pair(binding):return f"relation:{binding.subject_type}->{binding.object_type}"
    @staticmethod
    def _attribute_type_pair(binding):return f"attribute:{binding.entity_type}->{binding.value_kind}"
    def _guided_cost(self,predicate,type_pair):
        g=self.diversity_guidance
        if g is None:return 0.
        return 2.*((g.predicate_usage.get(predicate,0)+1)/g.predicate_cap)+1.25*((g.type_pair_usage.get(type_pair,0)+1)/g.type_pair_cap)
    def _within_guidance_cap(self,predicate,type_pair):
        g=self.diversity_guidance
        return True if g is None else g.predicate_usage.get(predicate,0)+1<=g.predicate_cap and g.type_pair_usage.get(type_pair,0)+1<=g.type_pair_cap
    def _guided_relation_pool(self,feasible):
        if self.diversity_guidance is None:feasible.sort(key=lambda item:item[0],reverse=True);return feasible[:min(32,len(feasible))]
        ranked=[]
        for item in feasible:
            count,binding,_,_=item;type_pair=self._relation_type_pair(binding)
            if self._within_guidance_cap(binding.predicate,type_pair):ranked.append((self._guided_cost(binding.predicate,type_pair),-count,binding.predicate,type_pair,item))
        ranked.sort(key=lambda row:row[:4]);return [row[-1] for row in ranked[:min(32,len(ranked))]]
    def _guided_attribute_pool(self,feasible):
        if self.diversity_guidance is None:feasible.sort(key=lambda item:item[0],reverse=True);return feasible[:min(32,len(feasible))]
        ranked=[]
        for item in feasible:
            count,binding=item;type_pair=self._attribute_type_pair(binding)
            if self._within_guidance_cap(binding.predicate,type_pair):ranked.append((self._guided_cost(binding.predicate,type_pair),-count,binding.predicate,type_pair,item))
        ranked.sort(key=lambda row:row[:4]);return [row[-1] for row in ranked[:min(32,len(ranked))]]
    def _profile_support_for_pair(self,entity_type,left_predicate,right_predicate):
        support=self.catalog.get("entity_profile_support",{})
        if not isinstance(support,Mapping) or support.get("version")!=1:return 0
        joint=support.get("joint_attribute_support",{})
        if not isinstance(joint,Mapping):return 0
        pair="|".join(sorted((str(left_predicate),str(right_predicate))));best=0
        for profile_type,counts in joint.items():
            if not isinstance(counts,Mapping):continue
            compatible=profile_type=="*" or self.taxonomy.compatible(entity_type,str(profile_type)) or self.taxonomy.compatible(str(profile_type),entity_type)
            if compatible:best=max(best,int(counts.get(pair,0) or 0))
        return best
    def _profile_weight(self,binding):
        previous=[]
        for known_type,predicates in self._chosen_attribute_predicates.items():
            if known_type==binding.entity_type or self.taxonomy.compatible(binding.entity_type,known_type) or self.taxonomy.compatible(known_type,binding.entity_type):previous.extend(predicates)
        if not previous:return 1.
        self.profile_guidance_events+=1;best=max((self._profile_support_for_pair(binding.entity_type,p,binding.predicate) for p in previous),default=0)
        if best>0:self.profile_guidance_supported_events+=1;return min(4.,1.4+math.log1p(best))
        return .65
    def _attribute_profile(self,predicate,info,*,entity_type,value_kind):
        key=(str(predicate),str(entity_type),str(value_kind));profile=self._attribute_profile_cache.get(key)
        if profile is None:profile=compile_attribute_profile(str(predicate),info,entity_type=str(entity_type),value_kind=str(value_kind));self._attribute_profile_cache[key]=profile
        return profile
    def choose_relation(self,solver,input_node,output_node,*,direction="forward"):
        feasible=[]
        for count,option in self._relation_options():
            in_type=option.subject_type if direction=="forward" else option.object_type;out_type=option.object_type if direction=="forward" else option.subject_type;trial=solver.clone()
            if trial.constrain(input_node,in_type) and trial.constrain(output_node,out_type):feasible.append((count,option,in_type,out_type))
        if not feasible:return None
        pool=self._guided_relation_pool(feasible)
        if not pool:return None
        _,chosen,in_type,out_type=self.rng.choice(pool)
        if not solver.constrain(input_node,in_type) or not solver.constrain(output_node,out_type):return None
        return RelationBinding(chosen.predicate,chosen.subject_type,chosen.object_type,direction)
    def _attribute_specs(self,value_kind):
        for predicate,info in self.catalog.get("predicates",{}).items():
            count=int(info.get("count",0))
            if count<self.min_count or info.get("kind")!="attribute":continue
            kinds=_attribute_kinds(info)
            if value_kind and value_kind not in kinds:continue
            selected=value_kind or next(iter(kinds),None)
            if not selected:continue
            for domain in preferred_candidate_types(info,"subject")[:12]:yield count,predicate,info,domain,selected,kinds.get(selected)
    def choose_attribute(self,solver,entity_node,*,value_kind=None):
        feasible=[]
        for count,predicate,info,domain,selected_kind,datatype in self._attribute_specs(value_kind):
            if not solver.can_constrain(entity_node,domain):continue
            profile_type=domain
            if selected_kind=="number" and info.get("typed_literal_support",{}).get("version")==1:
                trial=solver.clone();trial.constrain(entity_node,domain);profile_type=trial.get(entity_node)
            profile=self._attribute_profile(str(predicate),info,entity_type=profile_type,value_kind=selected_kind)
            if not profile.safe:continue
            feasible.append((count,AttributeBinding(str(predicate),profile_type,selected_kind,datatype,profile.family,profile.canonical_unit,profile.minimum,profile.maximum,profile.integer_values,profile.confidence,profile.evidence)))
        if not feasible:return None
        pool=self._guided_attribute_pool(feasible)
        if not pool:return None
        weights=[max(1.,math.sqrt(max(1,count)))*self._profile_weight(binding) for count,binding in pool];_,chosen=self.rng.choices(pool,weights=weights,k=1)[0]
        if not solver.constrain(entity_node,chosen.entity_type):return None
        self._chosen_attribute_predicates.setdefault(chosen.entity_type,[]).append(chosen.predicate);return chosen
    def choose_common_entity_type(self):
        weighted=[]
        for _,info in self.catalog.get("predicates",{}).items():
            if info.get("kind") not in {"relation","attribute"}:continue
            count=int(info.get("count",0))
            for entity_type in preferred_candidate_types(info,"subject")[:4]:weighted.append((count,entity_type))
        if not weighted:return None
        weighted.sort(key=lambda item:item[0],reverse=True);return self.rng.choice(weighted[:min(64,len(weighted))])[1]
    def relation_for_types(self,left_type,right_type):
        options=[]
        for count,option in self._relation_options():
            if left_type and not self.taxonomy.compatible(left_type,option.subject_type):continue
            if right_type and not self.taxonomy.compatible(right_type,option.object_type):continue
            options.append((count,option.predicate))
        if not options:return None
        options.sort(reverse=True);return self.rng.choice(options[:min(32,len(options))])[1]
