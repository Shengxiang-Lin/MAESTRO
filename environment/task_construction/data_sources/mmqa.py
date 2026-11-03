"""MultiModalQA adapter and compact scenario-prior analysis."""
from __future__ import annotations
import gzip,json
from collections import Counter
from pathlib import Path
from typing import Any,Dict,Iterator,Sequence
from .semantic_bundles import build_mmqa_semantic_bundle_bank,extract_mmqa_semantic_bundle

def iter_jsonl_gz(path:str|Path,*,progress:bool=False)->Iterator[Dict[str,Any]]:
    source=Path(path); opener=gzip.open if source.suffix.lower()==".gz" else open
    with opener(source,"rt",encoding="utf-8") as handle:
        iterator=handle
        if progress:
            from tqdm.auto import tqdm
            iterator=tqdm(handle,desc=f"MMQA {source.name}",unit="record")
        for line in iterator:
            if line.strip():
                item=json.loads(line)
                if isinstance(item,dict):yield item

def extract_source_motif(record:Dict[str,Any])->Dict[str,Any]:
    bundle=extract_mmqa_semantic_bundle(record)
    if bundle is None:return {"declared_type":None,"source_types":[],"supporting_source_count":0,"intermediate_steps":[]}
    metadata=record.get("metadata"); declared=metadata.get("type") if isinstance(metadata,dict) else None
    return {"declared_type":declared,"source_types":[item.modality for item in bundle.artifacts],"supporting_source_count":len(bundle.artifacts),"intermediate_steps":[{"step":i,"source_type":item.modality or "unknown","has_local_answer":bool(item.value)} for i,item in enumerate(bundle.intermediate_observations)]}

def analyze_source_motifs(paths:Sequence[str|Path],*,top_k:int=100,progress:bool=False)->Dict[str,Any]:
    total=cross_source=with_intermediate=intermediate_count=nested_intermediate_records=0; pattern_counts=Counter();scenario_pattern_counts=Counter();scenario_metadata={};declared_counts=Counter();source_count_hist=Counter()
    for path in paths:
        for record in iter_jsonl_gz(path,progress=progress):
            total+=1; bundle=extract_mmqa_semantic_bundle(record,record_index=total-1); motif=extract_source_motif(record); source_types=tuple(motif["source_types"]); cross_source+=int(len(set(source_types))>1); pattern_counts[" -> ".join(source_types) if source_types else "unknown"]+=1; declared_counts[str(motif.get("declared_type"))]+=1; source_count_hist[motif["supporting_source_count"]]+=1; steps=motif["intermediate_steps"]; with_intermediate+=int(bool(steps)); intermediate_count+=len(steps)
            raw=(record.get("metadata") or {}).get("intermediate_answers") if isinstance(record.get("metadata"),dict) else None
            if isinstance(raw,(list,tuple)) and any(isinstance(item,(list,tuple)) for item in raw):nested_intermediate_records+=1
            if bundle is None or not bundle.artifacts:continue
            scenario={"source_types":[i.modality for i in bundle.artifacts],"artifact_roles":[i.role for i in bundle.artifacts],"surface_styles":[i.surface_style for i in bundle.artifacts]}; key=json.dumps(scenario,sort_keys=True,ensure_ascii=False);scenario_pattern_counts[key]+=1;current=scenario_metadata.setdefault(key,{**scenario,"discovery_edge_count":0,"intermediate_observation_count":0});current["discovery_edge_count"]=max(int(current["discovery_edge_count"]),len(bundle.transitions));current["intermediate_observation_count"]=max(int(current["intermediate_observation_count"]),len(bundle.intermediate_observations))
    scenario_patterns=[]
    for key,count in scenario_pattern_counts.most_common(top_k*2):item=dict(scenario_metadata[key]);item["count"]=count;scenario_patterns.append(item)
    return {"records":total,"cross_source_records":cross_source,"cross_source_rate":cross_source/total if total else 0.0,"records_with_intermediate_steps":with_intermediate,"intermediate_step_count":intermediate_count,"nested_intermediate_records_seen":nested_intermediate_records,"declared_type_counts":dict(declared_counts.most_common()),"source_count_histogram":{str(k):v for k,v in sorted(source_count_hist.items())},"top_source_patterns":[{"pattern":p,"count":c} for p,c in pattern_counts.most_common(top_k)],"scenario_patterns":scenario_patterns,"scenario_schema":"scenario_semantic_bundle.v1_projection","perception_policy":"modality_conditioned_textual_observation_not_raw_media","note":"MMQA contributes artifact organization and progressive-discovery priors, not synthetic truth. Nested intermediate answers are recursively preserved by the semantic bundle adapter."}
