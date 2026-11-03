"""Question-type-grounded material roles for MultiModalQA scenario bundles."""
from __future__ import annotations
import json
from collections import Counter,defaultdict
from contextlib import contextmanager
from dataclasses import dataclass,replace
from pathlib import Path
from typing import Any,Iterator,Mapping,Sequence
from . import semantic_bundles as _sb
from .mmqa_context_enrichment import build_mmqa_semantic_bundle_bank as _build_enriched_bundle_bank, extract_mmqa_semantic_bundle as _extract_enriched_bundle
from .semantic_bundles import DiscoveryTransition
QUESTION_TYPE_ROLE_POLICY="mmqa_official_question_type_material_grammar_v4"
QUESTION_TYPE_TRANSITION_POLICY="mmqa_official_question_type_dependency_grammar_v4"
_LEAF_MODALITY={"TextQ":"text","TableQ":"table","ImageQ":"image_report","ImageListQ":"image_report"}
_OFFICIAL_TYPES={"TextQ","TableQ","ImageQ","ImageListQ","Compose(TableQ,ImageListQ)","Compose(TextQ,ImageListQ)","Compose(ImageQ,TableQ)","Compose(ImageQ,TextQ)","Compose(TextQ,TableQ)","Compose(TableQ,TextQ)","Intersect(TableQ,TextQ)","Intersect(ImageListQ,TableQ)","Intersect(ImageListQ,TextQ)","Compare(Compose(TableQ,ImageQ),TableQ)","Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ))","Compare(TableQ,Compose(TableQ,TextQ))"}
_INTERSECT_FIRST_CHILD={"Intersect(TableQ,TextQ)":1,"Intersect(ImageListQ,TableQ)":0,"Intersect(ImageListQ,TextQ)":1}
@dataclass(frozen=True)
class _QNode: op:str;value:str|None=None;children:tuple["_QNode",...]=()
@dataclass(frozen=True)
class _Occurrence: occurrence_id:int;leaf:str;modality:str;role:str;basis:str;path:str
@dataclass(frozen=True)
class _OccurrenceEdge: source:int;target:int;reason:str
def _record_question_type(record:Mapping[str,Any])->str|None:
    metadata=record.get("metadata")
    if not isinstance(metadata,Mapping):return None
    value=metadata.get("type")
    if value is None:return None
    text="".join(str(value).split());return text or None
def _record_modalities(record:Mapping[str,Any])->list[str]:
    metadata=record.get("metadata")
    if not isinstance(metadata,Mapping):return []
    raw=metadata.get("modalities")
    if not isinstance(raw,Sequence) or isinstance(raw,(str,bytes)):return []
    result=[]
    for item in raw:
        token=str(item or "").strip().lower()
        if token and token not in result:result.append(token)
    return result
@contextmanager
def _tag_question_structure()->Iterator[None]:
    original=_sb.extract_mmqa_semantic_bundle
    def tagged(record:Mapping[str,Any],*,record_index:int|None=None):
        bundle=original(record,record_index=record_index)
        if bundle is None:return None
        metadata=dict(bundle.metadata);question_type=_record_question_type(record)
        if question_type:metadata["mmqa_question_type"]=question_type
        modalities=_record_modalities(record)
        if modalities:metadata["mmqa_declared_modalities"]=modalities
        metadata["mmqa_question_structure_only"]=True
        return replace(bundle,metadata=metadata)
    _sb.extract_mmqa_semantic_bundle=tagged
    try:yield
    finally:_sb.extract_mmqa_semantic_bundle=original
def _split_top_level_args(body:str)->tuple[str,str]|None:
    depth=0
    for index,char in enumerate(body):
        if char=="(":depth+=1
        elif char==")":
            depth-=1
            if depth<0:return None
        elif char=="," and depth==0:
            left=body[:index].strip();right=body[index+1:].strip();return (left,right) if left and right else None
    return None
def _parse_question_type(value:str|None)->_QNode|None:
    text="".join(str(value or "").split())
    if text in _LEAF_MODALITY:return _QNode(op="leaf",value=text)
    for op in ("Compose","Intersect","Compare"):
        prefix=op+"("
        if not text.startswith(prefix) or not text.endswith(")"):continue
        pair=_split_top_level_args(text[len(prefix):-1])
        if pair is None:return None
        left=_parse_question_type(pair[0]);right=_parse_question_type(pair[1])
        if left is None or right is None:return None
        return _QNode(op=op,children=(left,right))
    return None
def _bridge_role(leaf:str)->str:return "identification" if _LEAF_MODALITY.get(leaf)=="image_report" else "bridge_lookup"
def _single_role(leaf:str)->str:return "identification" if _LEAF_MODALITY.get(leaf)=="image_report" else "primary_lookup"
def _plan_question_type(question_type:str)->tuple[list[_Occurrence],list[_OccurrenceEdge]]:
    node=_parse_question_type(question_type)
    if node is None or question_type not in _OFFICIAL_TYPES:return [],[]
    occurrences=[];edges=[]
    def add_leaf(leaf,role,basis,path):
        occurrence_id=len(occurrences);occurrences.append(_Occurrence(occurrence_id,leaf,_LEAF_MODALITY[leaf],role,basis,path));return [occurrence_id]
    def plan(current,terminal_role,path):
        if current.op=="leaf":
            assert current.value is not None;role=terminal_role;basis=f"question_type_{terminal_role}"
            if terminal_role=="bridge_lookup":role=_bridge_role(current.value);basis="question_type_first_hop_bridge_material"
            elif terminal_role=="primary_lookup":role=_single_role(current.value);basis="question_type_single_hop_material"
            return add_leaf(current.value,role,basis,path)
        left,right=current.children
        if current.op=="Compose":
            inner=plan(right,"bridge_lookup",path+"/inner");outer=plan(left,terminal_role,path+"/outer")
            for source in inner:
                for target in outer:edges.append(_OccurrenceEdge(source,target,"question_type_compose_dependency"))
            return outer
        if current.op=="Compare":return plan(left,"comparison",path+"/left")+plan(right,"comparison",path+"/right")
        first=plan(right,"bridge_lookup",path+"/first");second=plan(left,"verification",path+"/verify")
        for source in first:
            for target in second:edges.append(_OccurrenceEdge(source,target,"question_type_intersect_verification"))
        return second
    if node.op=="leaf":plan(node,"primary_lookup","root")
    elif node.op=="Intersect":
        first_child=_INTERSECT_FIRST_CHILD.get(question_type)
        if first_child is None:return [],[]
        second_child=1-first_child;first=plan(node.children[first_child],"bridge_lookup","root/first");second=plan(node.children[second_child],"verification","root/verify")
        for source in first:
            for target in second:edges.append(_OccurrenceEdge(source,target,"question_type_intersect_verification"))
    elif node.op=="Compare":plan(node,"comparison","root")
    else:plan(node,"detail_lookup","root")
    return occurrences,edges
def _bind_occurrences(artifacts,occurrences):
    artifacts_by_modality=defaultdict(list);occurrences_by_modality=defaultdict(list)
    for index,artifact in enumerate(artifacts):artifacts_by_modality[str(artifact.get("modality") or "text")].append(index)
    for occurrence in occurrences:occurrences_by_modality[occurrence.modality].append(occurrence.occurrence_id)
    binding={item.occurrence_id:[] for item in occurrences};used=set()
    for modality,occurrence_ids in occurrences_by_modality.items():
        artifact_ids=artifacts_by_modality.get(modality,[])
        if not artifact_ids:continue
        if len(occurrence_ids)==1:binding[occurrence_ids[0]].extend(artifact_ids);used.update(artifact_ids);continue
        for occurrence_id,artifact_id in zip(occurrence_ids,artifact_ids):binding[occurrence_id].append(artifact_id);used.add(artifact_id)
        remaining=artifact_ids[len(occurrence_ids):];roles={occurrences[oid].role for oid in occurrence_ids}
        if len(roles)==1:
            for offset,artifact_id in enumerate(remaining):binding[occurrence_ids[offset%len(occurrence_ids)]].append(artifact_id);used.add(artifact_id)
    return binding,used
def _fallback_roles(artifacts):
    count=len(artifacts)
    if count==0:return [],[]
    if count==1:return ["primary_lookup"],["unsupported_question_type_single_artifact"]
    roles=[];bases=[]
    for index,artifact in enumerate(artifacts):
        modality=str(artifact.get("modality") or "text")
        if index==0:role,basis="identification","unsupported_question_type_seed_fallback"
        elif modality=="table":role,basis="comparison","unsupported_question_type_table_fallback"
        elif index==count-1:role,basis="detail_lookup","unsupported_question_type_terminal_fallback"
        else:role,basis="supporting_lookup","unsupported_question_type_support_fallback"
        roles.append(role);bases.append(basis)
    return roles,bases
def functionalize_bundle_payload(payload:Mapping[str,Any])->dict[str,Any]:
    result=dict(payload);artifacts=[dict(item) for item in payload.get("artifacts",()) if isinstance(item,Mapping)];raw_metadata=payload.get("metadata",{});metadata=dict(raw_metadata) if isinstance(raw_metadata,Mapping) else {};question_type=str(metadata.get("mmqa_question_type") or "").strip();occurrences,occurrence_edges=_plan_question_type(question_type);supported=bool(occurrences);occurrence_for_artifact={};binding={};used=set()
    if supported:
        binding,used=_bind_occurrences(artifacts,occurrences);roles=["supporting_lookup" for _ in artifacts];bases=["question_type_unbound_supporting_context" for _ in artifacts]
        for occurrence in occurrences:
            for artifact_index in binding.get(occurrence.occurrence_id,()):roles[artifact_index]=occurrence.role;bases[artifact_index]=occurrence.basis;occurrence_for_artifact[artifact_index]=occurrence
    else:roles,bases=_fallback_roles(artifacts)
    transitions=[];missing_edge_endpoint=0;seen_pairs=set()
    if supported:
        for edge in occurrence_edges:
            source_artifacts=binding.get(edge.source,());target_artifacts=binding.get(edge.target,())
            if not source_artifacts or not target_artifacts:missing_edge_endpoint+=1;continue
            source=source_artifacts[0];target=target_artifacts[0];pair=(source,target)
            if source==target or pair in seen_pairs:continue
            seen_pairs.add(pair);transitions.append({"from_artifact":str(artifacts[source].get("artifact_id")),"to_artifact":str(artifacts[target].get("artifact_id")),"bridge_values":[],"reason":edge.reason})
    outgoing=Counter();incoming=Counter();index_by_artifact_id={str(a.get("artifact_id")):i for i,a in enumerate(artifacts)}
    for edge in transitions:
        source=index_by_artifact_id.get(str(edge.get("from_artifact")));target=index_by_artifact_id.get(str(edge.get("to_artifact")))
        if source is not None:outgoing[source]+=1
        if target is not None:incoming[target]+=1
    updated=[]
    for index,artifact in enumerate(artifacts):
        artifact_metadata=dict(artifact.get("metadata",{})) if isinstance(artifact.get("metadata"),Mapping) else {};occurrence=occurrence_for_artifact.get(index);artifact_metadata.update({"functional_role_policy":QUESTION_TYPE_ROLE_POLICY,"functional_role_basis":bases[index],"functional_in_degree":int(incoming[index]),"functional_out_degree":int(outgoing[index]),"question_type_bound":occurrence is not None,"question_type_leaf":occurrence.leaf if occurrence else None,"question_type_path":occurrence.path if occurrence else None,"produces_intermediate_observation":roles[index] in {"identification","bridge_lookup"} and bool(outgoing[index])});artifact["role"]=roles[index];artifact["metadata"]=artifact_metadata;updated.append(artifact)
    metadata.update({"artifact_role_policy":QUESTION_TYPE_ROLE_POLICY,"transition_policy":QUESTION_TYPE_TRANSITION_POLICY,"position_only_role_assignment":False,"question_type_grammar_supported":supported,"question_type_occurrence_count":len(occurrences),"question_type_bound_artifact_count":len(used),"transition_count":len(transitions),"question_type_missing_edge_endpoint_count":missing_edge_endpoint,"intermediate_values_used_for_role_assignment":False});result["artifacts"]=updated;result["transitions"]=transitions;result["metadata"]=metadata;return result
def _rewrite_bundle_bank(path:Path)->dict[str,Any]:
    role_counts=Counter();role_sequences=Counter();question_type_counts=Counter();unsupported_types=Counter();transition_reasons=Counter();transition_count=bound_artifacts=total_artifacts=bundle_count=0;temp=path.with_suffix(path.suffix+".question_type.tmp")
    with path.open("r",encoding="utf-8") as reader,temp.open("w",encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():continue
            payload=json.loads(line)
            if not isinstance(payload,Mapping):continue
            updated=functionalize_bundle_payload(payload);writer.write(json.dumps(updated,ensure_ascii=False)+"\n");bundle_count+=1;metadata=updated.get("metadata",{});metadata=metadata if isinstance(metadata,Mapping) else {};question_type=str(metadata.get("mmqa_question_type") or "missing");question_type_counts[question_type]+=1
            if not metadata.get("question_type_grammar_supported"):unsupported_types[question_type]+=1
            bound_artifacts+=int(metadata.get("question_type_bound_artifact_count",0) or 0);roles=[str(item.get("role")) for item in updated.get("artifacts",()) if isinstance(item,Mapping)];total_artifacts+=len(roles);role_counts.update(roles)
            if roles:role_sequences[" -> ".join(roles)]+=1
            for edge in updated.get("transitions",()):
                if isinstance(edge,Mapping):transition_count+=1;transition_reasons[str(edge.get("reason") or "unknown")]+=1
    temp.replace(path);return {"bundles_rewritten":bundle_count,"artifact_role_counts":dict(role_counts.most_common()),"functional_role_sequence_count":len(role_sequences),"top_functional_role_sequences":[{"pattern":p,"count":c} for p,c in role_sequences.most_common(100)],"mmqa_question_type_counts":dict(question_type_counts.most_common()),"unsupported_question_type_counts":dict(unsupported_types.most_common()),"question_type_bound_artifact_count":bound_artifacts,"question_type_artifact_count":total_artifacts,"question_type_bound_artifact_rate":bound_artifacts/total_artifacts if total_artifacts else 0.0,"functional_discovery_transition_count":transition_count,"functional_transition_reason_counts":dict(transition_reasons.most_common())}
def build_mmqa_semantic_bundle_bank(paths:Sequence[str|Path],output_path:str|Path,*,summary_path:str|Path|None=None,max_records:int|None=None,progress:bool=False)->dict[str,Any]:
    with _tag_question_structure():summary=_build_enriched_bundle_bank(paths,output_path,summary_path=None,max_records=max_records,progress=progress)
    target=Path(output_path);grammar=_rewrite_bundle_bank(target);summary={**dict(summary),**grammar,"artifact_role_counts":grammar["artifact_role_counts"],"artifact_role_policy":QUESTION_TYPE_ROLE_POLICY,"transition_policy":QUESTION_TYPE_TRANSITION_POLICY,"position_only_role_assignment":False,"intermediate_values_used_for_role_assignment":False}
    if summary_path is not None:
        destination=Path(summary_path);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary
def extract_mmqa_semantic_bundle(record:Mapping[str,Any],*,record_index:int|None=None):
    with _tag_question_structure():bundle=_extract_enriched_bundle(record,record_index=record_index)
    if bundle is None:return None
    payload=functionalize_bundle_payload(bundle.to_dict());by_id={str(item.get("artifact_id")):item for item in payload.get("artifacts",()) if isinstance(item,Mapping)};artifacts=tuple(replace(artifact,role=str(by_id[artifact.artifact_id].get("role",artifact.role)),metadata=dict(by_id[artifact.artifact_id].get("metadata",artifact.metadata))) for artifact in bundle.artifacts);transitions=tuple(DiscoveryTransition(str(edge.get("from_artifact")),str(edge.get("to_artifact")),(),str(edge.get("reason") or "question_type_dependency")) for edge in payload.get("transitions",()) if isinstance(edge,Mapping));return replace(bundle,artifacts=artifacts,transitions=transitions,metadata=dict(payload.get("metadata",bundle.metadata)))
__all__=["QUESTION_TYPE_ROLE_POLICY","QUESTION_TYPE_TRANSITION_POLICY","build_mmqa_semantic_bundle_bank","extract_mmqa_semantic_bundle","functionalize_bundle_payload"]
