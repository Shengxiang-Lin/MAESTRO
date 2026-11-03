"""Scenario-aware priors for heterogeneous evidence layout."""
from __future__ import annotations
import json,math,random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any,Iterable,Mapping
FALLBACK_PATTERNS=(("text","table"),("table","image_report"),("table","text"),("text","image_report"),("text","table","image_report"),("table","text","image_report"),("text","text","table"))
_DEFAULT_STYLE={"text":"document_excerpt","table":"structured_table","image_report":"visual_observation","video_report":"temporal_inspection_log","audio_transcript":"speaker_transcript","database":"structured_record"}
def _fallback_roles(count:int)->tuple[str,...]:
    if count<=1:return ("primary_lookup",) if count else ()
    return tuple("identification" if i==0 else "detail_lookup" if i==count-1 else "bridge_lookup" for i in range(count))
@dataclass(frozen=True)
class EvidenceLayoutPattern:
    source_types:tuple[str,...];count:int;artifact_roles:tuple[str,...]=();surface_styles:tuple[str,...]=();origin:str="mmqa_semantic_bundle";discovery_edge_count:int=0;intermediate_observation_count:int=0
    @property
    def cross_modal(self)->bool:return len(set(self.source_types))>1
    def normalized(self):
        roles=self.artifact_roles if len(self.artifact_roles)==len(self.source_types) else _fallback_roles(len(self.source_types));styles=self.surface_styles if len(self.surface_styles)==len(self.source_types) else tuple(_DEFAULT_STYLE.get(v,"document_excerpt") for v in self.source_types);return EvidenceLayoutPattern(self.source_types,max(1,int(self.count)),roles,styles,self.origin,int(self.discovery_edge_count),int(self.intermediate_observation_count))
    def to_dict(self):
        item=self.normalized();return {"source_types":list(item.source_types),"artifact_roles":list(item.artifact_roles),"surface_styles":list(item.surface_styles),"count":item.count,"cross_modal":item.cross_modal,"origin":item.origin,"discovery_edge_count":item.discovery_edge_count,"intermediate_observation_count":item.intermediate_observation_count}
class ScenarioEvidencePrior:
    def __init__(self,patterns:Iterable[EvidenceLayoutPattern],*,origin:str="mmqa_semantic_bundle")->None:
        cleaned=[item.normalized() for item in patterns if item.source_types and all(item.source_types)]
        if not cleaned:
            cleaned=[EvidenceLayoutPattern(pattern,1,_fallback_roles(len(pattern)),tuple(_DEFAULT_STYLE.get(v,"document_excerpt") for v in pattern),"fallback",max(0,len(pattern)-1)) for pattern in FALLBACK_PATTERNS];origin="fallback"
        self.patterns=tuple(cleaned);self.origin=origin
    @classmethod
    def fallback(cls):return cls((),origin="fallback")
    @staticmethod
    def _pattern_from_mapping(item:Mapping[str,Any],*,origin:str):
        types=tuple(str(v) for v in item.get("source_types",()) if str(v))
        if not types:return None
        return EvidenceLayoutPattern(types,max(1,int(item.get("count",1) or 1)),tuple(str(v) for v in item.get("artifact_roles",()) if str(v)),tuple(str(v) for v in item.get("surface_styles",()) if str(v)),origin,int(item.get("discovery_edge_count",0) or 0),int(item.get("intermediate_observation_count",0) or 0))
    @classmethod
    def from_bundle_bank(cls,path:str|Path):
        source=Path(path);counts=Counter();maxima={}
        with source.open("r",encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():continue
                payload=json.loads(line)
                if not isinstance(payload,Mapping):continue
                artifacts=payload.get("artifacts",())
                if not isinstance(artifacts,list) or not artifacts:continue
                types=tuple(str(item.get("modality","text")) for item in artifacts if isinstance(item,Mapping));roles=tuple(str(item.get("role","bridge_lookup")) for item in artifacts if isinstance(item,Mapping));styles=tuple(str(item.get("surface_style",_DEFAULT_STYLE.get(types[index] if index<len(types) else "text","document_excerpt"))) for index,item in enumerate(artifacts) if isinstance(item,Mapping))
                if not types or len(types)!=len(roles) or len(types)!=len(styles):continue
                key=(types,roles,styles);counts[key]+=1;intermediates=payload.get("intermediate_observations",());old_edges,old_intermediates=maxima.get(key,(0,0));maxima[key]=(max(old_edges,len(payload.get("transitions",()) or ())),max(old_intermediates,len(intermediates) if isinstance(intermediates,list) else 0))
        patterns=[]
        for (types,roles,styles),count in counts.items():
            edges,intermediates=maxima.get((types,roles,styles),(0,0));patterns.append(EvidenceLayoutPattern(types,count,roles,styles,f"mmqa_bundle:{source.name}",edges,intermediates))
        return cls(patterns,origin=f"mmqa_bundle:{source.name}")
    @classmethod
    def from_legacy_json(cls,path):
        source=Path(path);payload=json.loads(source.read_text(encoding="utf-8"));patterns=[]
        for item in payload.get("scenario_patterns",()):
            if isinstance(item,Mapping):
                parsed=cls._pattern_from_mapping(item,origin=f"mmqa_scenario_summary:{source.name}")
                if parsed is not None:patterns.append(parsed)
        if patterns:return cls(patterns,origin=f"mmqa_scenario_summary:{source.name}")
        for item in payload.get("top_source_patterns",()):
            if not isinstance(item,Mapping):continue
            raw=str(item.get("pattern") or "").strip()
            if not raw or raw=="unknown":continue
            types=tuple(part.strip() for part in raw.split("->") if part.strip())
            if types:patterns.append(EvidenceLayoutPattern(types,max(1,int(item.get("count",1) or 1)),origin=f"legacy_mmqa:{source.name}"))
        return cls(patterns,origin=f"legacy_mmqa:{source.name}")
    @classmethod
    def from_path(cls,path):
        source=Path(path);return cls.from_bundle_bank(source) if source.suffix.lower()==".jsonl" else cls.from_legacy_json(source)
    @classmethod
    def from_json(cls,path):return cls.from_path(path)
    @staticmethod
    def _resize(pattern,source_count):
        pattern=pattern.normalized();types=list(pattern.source_types);roles=list(pattern.artifact_roles);styles=list(pattern.surface_styles)
        while len(types)<source_count:
            index=len(types)%len(pattern.source_types);types.append(pattern.source_types[index]);roles.append("bridge_lookup" if len(types)<source_count else "detail_lookup");styles.append(pattern.surface_styles[index])
        types=types[:source_count];roles=roles[:source_count];styles=styles[:source_count]
        if source_count:roles[0]="identification" if source_count>1 else "primary_lookup";roles[-1]="detail_lookup" if source_count>1 else roles[-1]
        return EvidenceLayoutPattern(tuple(types),pattern.count,tuple(roles),tuple(styles),pattern.origin+":resized",max(pattern.discovery_edge_count,max(0,source_count-1)),pattern.intermediate_observation_count)
    def sample(self,source_count:int,*,rng:random.Random,prefer_cross_modal:bool=True):
        if source_count<1:raise ValueError("source_count must be >= 1")
        candidates=[item for item in self.patterns if len(item.source_types)==source_count]
        if not candidates:return self._resize(min(self.patterns,key=lambda item:abs(len(item.source_types)-source_count)),source_count)
        weights=[]
        for item in candidates:
            weight=math.sqrt(max(1,item.count));weight*=1.75 if prefer_cross_modal and item.cross_modal else 1.;weight*=1.20 if source_count>1 and item.discovery_edge_count else 1.;weight*=1.10 if item.intermediate_observation_count else 1.;weights.append(weight)
        return rng.choices(candidates,weights=weights,k=1)[0].normalized()
    def summary(self):return {"origin":self.origin,"pattern_count":len(self.patterns),"lengths":sorted({len(item.source_types) for item in self.patterns}),"cross_modal_patterns":sum(item.cross_modal for item in self.patterns),"patterns_with_discovery_edges":sum(item.discovery_edge_count>0 for item in self.patterns),"patterns_with_intermediate_observations":sum(item.intermediate_observation_count>0 for item in self.patterns),"policy":"scenario_roles_and_surface_grammar_not_dataset_truth"}
SourcePattern=EvidenceLayoutPattern;SourceMotifPrior=ScenarioEvidencePrior
__all__=["EvidenceLayoutPattern","FALLBACK_PATTERNS","ScenarioEvidencePrior","SourceMotifPrior","SourcePattern"]
