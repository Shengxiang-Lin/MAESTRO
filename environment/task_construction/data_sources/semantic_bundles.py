"""Ground heterogeneous dataset records into reusable scenario/evidence bundles.

Public datasets contribute object context and evidence-discovery structure. Concrete
answers remain private provenance and are never copied into the synthetic world's
oracle.
"""
from __future__ import annotations
import gzip,hashlib,json
from collections import Counter
from dataclasses import dataclass,field
from pathlib import Path
from typing import Any,Iterable,Iterator,Mapping,Sequence
from tqdm.auto import tqdm

_BUNDLE_SCHEMA="scenario_semantic_bundle.v1"
_MODALITY_ALIASES={"text":"text","txt":"text","document":"text","doc":"text","table":"table","tabular":"table","image":"image_report","img":"image_report","photo":"image_report","visual":"image_report","video":"video_report","audio":"audio_transcript","database":"database","db":"database"}
_SURFACE_STYLE={"text":"document_excerpt","table":"structured_table","image_report":"visual_observation","video_report":"temporal_inspection_log","audio_transcript":"speaker_transcript","database":"structured_record"}
_CUE_FIELDS=("title","caption","captions","text","passage","snippet","content","description","image_caption","image_captions","table","rows","header","headers")
_SOURCE_ID_FIELDS=("source_id","doc_id","document_id","image_id","table_id","id")
_ANSWER_FIELDS=("answer","answers","intermediate_answer","intermediate_answers")

def _stable_id(prefix:str,payload:Any)->str:
    text=json.dumps(payload,ensure_ascii=False,sort_keys=True,default=str);return f"{prefix}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]}"
def _normalize_modality(value:Any)->str:
    token=str(value or "text").strip().lower().replace("-","_").replace(" ","_");return _MODALITY_ALIASES.get(token,token if token in _SURFACE_STYLE else "text")
def _string_leaves(value:Any)->Iterator[str]:
    if value is None or isinstance(value,bool):return
    if isinstance(value,(str,int,float)):
        text=" ".join(str(value).split())
        if text:yield text
        return
    if isinstance(value,Mapping):
        for key,child in value.items():
            if str(key).lower() in {"url","path","image_path","file","filename"}:continue
            yield from _string_leaves(child)
        return
    if isinstance(value,Sequence) and not isinstance(value,(str,bytes)):
        for child in value:yield from _string_leaves(child)
def _dedupe_cues(values:Iterable[str],*,limit:int=10,width:int=260)->tuple[str,...]:
    result=[];seen=set()
    for raw in values:
        text=" ".join(str(raw).split())
        if not text:continue
        if len(text)>width:text=text[:width-1].rstrip()+"…"
        key=text.casefold()
        if key in seen:continue
        seen.add(key);result.append(text)
        if len(result)>=limit:break
    return tuple(result)
def _source_id(item:Mapping[str,Any],fallback:str)->str:
    for key in _SOURCE_ID_FIELDS:
        value=item.get(key)
        if value is not None and str(value).strip():return str(value).strip()
    return fallback
def _source_type(item:Mapping[str,Any])->str:
    for key in ("source_type","type","modality","media_type"):
        if item.get(key) is not None:return _normalize_modality(item.get(key))
    return "text"
def _artifact_cues(item:Mapping[str,Any])->tuple[str,...]:
    values=[]
    for key in _CUE_FIELDS:
        if key in item:values.extend(_string_leaves(item.get(key)))
    return _dedupe_cues(values)
def _question_text(record:Mapping[str,Any])->str|None:
    for value in (record.get("question"),record.get("query"),record.get("qa",{}).get("question") if isinstance(record.get("qa"),Mapping) else None):
        if value is not None and str(value).strip():return " ".join(str(value).split())
    return None
def _supporting_context(record:Mapping[str,Any])->list[Mapping[str,Any]]:
    candidates=[];metadata=record.get("metadata")
    if isinstance(metadata,Mapping):candidates.extend(metadata.get(key) for key in ("supporting_context","supporting_contexts","context") if key in metadata)
    candidates.extend(record.get(key) for key in ("supporting_context","supporting_contexts","context") if key in record)
    for raw in candidates:
        if isinstance(raw,Sequence) and not isinstance(raw,(str,bytes)):
            items=[item for item in raw if isinstance(item,Mapping)]
            if items:return items
    return []
def _intermediate_raw(record:Mapping[str,Any])->Any:
    metadata=record.get("metadata")
    for holder in [metadata,record]:
        if not isinstance(holder,Mapping):continue
        for key in ("intermediate_answers","intermediate_answer","intermediates"):
            if key in holder:return holder.get(key)
    return None
@dataclass(frozen=True)
class IntermediateObservation:
    value:str;modality:str|None=None;source_id:str|None=None;path:tuple[int|str,...]=()
    def to_dict(self)->dict[str,Any]:return {"value":self.value,"modality":self.modality,"source_id":self.source_id,"path":list(self.path)}
def _flatten_intermediate(value:Any,*,inherited_modality:str|None=None,inherited_source_id:str|None=None,path:tuple[int|str,...]=())->list[IntermediateObservation]:
    result=[]
    if value is None or isinstance(value,bool):return result
    if isinstance(value,Mapping):
        modality=inherited_modality
        for key in ("source_type","type","modality","media_type"):
            if value.get(key) is not None:modality=_normalize_modality(value.get(key));break
        source_id=inherited_source_id
        for key in _SOURCE_ID_FIELDS:
            if value.get(key) is not None and str(value.get(key)).strip():source_id=str(value.get(key)).strip();break
        answered=False
        for key in _ANSWER_FIELDS:
            if key in value:
                answered=True;result.extend(_flatten_intermediate(value.get(key),inherited_modality=modality,inherited_source_id=source_id,path=path+(key,)))
        if answered:return result
        skip=set(_SOURCE_ID_FIELDS)|{"source_type","type","modality","media_type"}
        for key,child in value.items():
            if key not in skip and isinstance(child,(Mapping,list,tuple)):result.extend(_flatten_intermediate(child,inherited_modality=modality,inherited_source_id=source_id,path=path+(str(key),)))
        return result
    if isinstance(value,Sequence) and not isinstance(value,(str,bytes)):
        for index,child in enumerate(value):result.extend(_flatten_intermediate(child,inherited_modality=inherited_modality,inherited_source_id=inherited_source_id,path=path+(index,)))
        return result
    text=" ".join(str(value).split())
    if text:result.append(IntermediateObservation(text,inherited_modality,inherited_source_id,path))
    return result
@dataclass(frozen=True)
class ArtifactPrototype:
    artifact_id:str;source_id:str;modality:str;role:str;surface_style:str;semantic_cues:tuple[str,...]=();local_observations:tuple[str,...]=();metadata:Mapping[str,Any]=field(default_factory=dict)
    def to_dict(self):return {"artifact_id":self.artifact_id,"source_id":self.source_id,"modality":self.modality,"role":self.role,"surface_style":self.surface_style,"semantic_cues":list(self.semantic_cues),"local_observations":list(self.local_observations),"metadata":dict(self.metadata)}
@dataclass(frozen=True)
class DiscoveryTransition:
    from_artifact:str;to_artifact:str;bridge_values:tuple[str,...]=();reason:str="progressive_discovery"
    def to_dict(self):return {"from_artifact":self.from_artifact,"to_artifact":self.to_artifact,"bridge_values":list(self.bridge_values),"reason":self.reason}
@dataclass(frozen=True)
class ScenarioSemanticBundle:
    bundle_id:str;dataset:str;source_record_id:str|None;question_hint:str|None;artifacts:tuple[ArtifactPrototype,...];transitions:tuple[DiscoveryTransition,...];intermediate_observations:tuple[IntermediateObservation,...];metadata:Mapping[str,Any]=field(default_factory=dict)
    def to_dict(self):return {"schema_id":_BUNDLE_SCHEMA,"bundle_id":self.bundle_id,"dataset":self.dataset,"source_record_id":self.source_record_id,"question_hint":self.question_hint,"artifacts":[i.to_dict() for i in self.artifacts],"transitions":[i.to_dict() for i in self.transitions],"intermediate_observations":[i.to_dict() for i in self.intermediate_observations],"metadata":dict(self.metadata)}
def _role_for(modality:str,index:int,count:int)->str:
    if count==1:return "primary_lookup"
    if index==0:return "identification"
    if index==count-1:return "detail_lookup"
    if modality=="table":return "comparison"
    return "bridge_lookup"
def extract_mmqa_semantic_bundle(record:Mapping[str,Any],*,record_index:int|None=None)->ScenarioSemanticBundle|None:
    contexts=_supporting_context(record)
    if not contexts:return None
    intermediates=tuple(_flatten_intermediate(_intermediate_raw(record)));by_source={};by_modality={}
    for item in intermediates:
        if item.source_id:by_source.setdefault(item.source_id,[]).append(item.value)
        if item.modality:by_modality.setdefault(item.modality,[]).append(item.value)
    artifacts=[]
    for index,item in enumerate(contexts):
        modality=_source_type(item);source_id=_source_id(item,f"context-{index+1}");artifact_id=f"artifact-{index+1:02d}";local=list(by_source.get(source_id,()))
        if not local:
            modality_values=by_modality.get(modality,())
            if modality_values and not any(value in observation for prior in artifacts for observation in prior.local_observations for value in modality_values):local=list(modality_values[:4])
        artifacts.append(ArtifactPrototype(artifact_id,source_id,modality,_role_for(modality,index,len(contexts)),_SURFACE_STYLE.get(modality,"document_excerpt"),_artifact_cues(item),_dedupe_cues(local,limit=6,width=160),{"context_index":index,"has_real_media_file":False,"perception_abstracted":modality in {"image_report","video_report","audio_transcript"}}))
    transitions=[]
    for index in range(len(artifacts)-1):
        current=artifacts[index];nxt=artifacts[index+1];bridge=current.local_observations;transitions.append(DiscoveryTransition(current.artifact_id,nxt.artifact_id,tuple(bridge[:3]),"intermediate_entity_or_value_determines_next_artifact" if bridge else "source_order_progressive_discovery"))
    record_id=None
    for key in ("qid","question_id","id"):
        if record.get(key) is not None:record_id=str(record.get(key));break
    basis={"record_id":record_id,"record_index":record_index,"modalities":[i.modality for i in artifacts],"question":_question_text(record)}
    return ScenarioSemanticBundle(_stable_id("mmqa",basis),"MultiModalQA",record_id,_question_text(record),tuple(artifacts),tuple(transitions),intermediates,{"perception_policy":"textual_semantic_surrogate_not_raw_media","final_answer_copied_to_synthetic_truth":False,"artifact_count":len(artifacts),"transition_count":len(transitions)})
def _iter_jsonl(path:str|Path)->Iterator[Mapping[str,Any]]:
    source=Path(path);opener=gzip.open if source.suffix==".gz" else open
    with opener(source,"rt",encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload=json.loads(line)
                if isinstance(payload,Mapping):yield payload
def build_mmqa_semantic_bundle_bank(paths:Sequence[str|Path],output_path:str|Path,*,summary_path:str|Path|None=None,max_records:int|None=None,progress:bool=False)->dict[str,Any]:
    target=Path(output_path);target.parent.mkdir(parents=True,exist_ok=True);scanned=written=nested_intermediate_records=intermediate_records=intermediate_leaf_count=0;modality_counts=Counter();role_counts=Counter();transition_patterns=Counter();artifact_count_hist=Counter()
    iterator:Iterable[tuple[Path,Mapping[str,Any]]]=((Path(path),record) for path in paths for record in _iter_jsonl(path))
    if progress:iterator=tqdm(iterator,desc="MMQA semantic bundles",unit="record")
    with target.open("w",encoding="utf-8") as writer:
        for _path,record in iterator:
            if max_records is not None and scanned>=max_records:break
            scanned+=1;raw_intermediate=_intermediate_raw(record)
            if isinstance(raw_intermediate,Sequence) and not isinstance(raw_intermediate,(str,bytes)) and any(isinstance(item,(list,tuple)) for item in raw_intermediate):nested_intermediate_records+=1
            bundle=extract_mmqa_semantic_bundle(record,record_index=scanned-1)
            if bundle is None:continue
            writer.write(json.dumps(bundle.to_dict(),ensure_ascii=False)+"\n");written+=1;leaves=len(bundle.intermediate_observations);intermediate_leaf_count+=leaves;intermediate_records+=int(leaves>0);artifact_count_hist[str(len(bundle.artifacts))]+=1;modality_counts.update(i.modality for i in bundle.artifacts);role_counts.update(i.role for i in bundle.artifacts)
            if bundle.artifacts:transition_patterns[" -> ".join(i.modality for i in bundle.artifacts)]+=1
    summary={"version":1,"schema_id":_BUNDLE_SCHEMA,"dataset":"MultiModalQA","inputs":[str(Path(p)) for p in paths],"output":str(target),"scanned":scanned,"bundles":written,"records_with_intermediate_observations":intermediate_records,"intermediate_observation_count":intermediate_leaf_count,"nested_intermediate_records_seen":nested_intermediate_records,"modality_counts":dict(modality_counts.most_common()),"artifact_role_counts":dict(role_counts.most_common()),"artifact_count_histogram":dict(sorted(artifact_count_hist.items())),"top_transition_patterns":[{"pattern":p,"count":c} for p,c in transition_patterns.most_common(100)],"perception_policy":"textual_semantic_surrogate_not_raw_media"}
    if summary_path is not None:
        destination=Path(summary_path);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary
__all__=["ArtifactPrototype","DiscoveryTransition","IntermediateObservation","ScenarioSemanticBundle","build_mmqa_semantic_bundle_bank","extract_mmqa_semantic_bundle"]
