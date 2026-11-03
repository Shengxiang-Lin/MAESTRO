"""Symbolic modality-indispensability diagnostics for synthetic environments."""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
from typing import Any,Iterable
from ..executor import ExecutionError,SymbolicExecutor
from ..ir import ProgramIR
from ..validation import normalize_answer
from ..world import SyntheticWorld
from .pipeline import program_from_dict,world_from_dict
MEDIA_PROXY_TYPES=frozenset({"image_report","video_report","audio_transcript"});TEXTUAL_SOURCE_TYPES=frozenset({"text","table","database"})
def _source_fact_ids(world,source_types):
    wanted=set(map(str,source_types));return {fact_id for source in world.sources.values() if source.source_type in wanted for fact_id in source.fact_ids}
def _exclusive_fact_ids(world,removed_types):
    removed=set(map(str,removed_types));removed_facts=_source_fact_ids(world,removed);retained={fid for source in world.sources.values() if source.source_type not in removed for fid in source.fact_ids};return removed_facts-retained
def _answer_changed(program,world,fact_ids,*,baseline,executor):
    removed=set(fact_ids)
    if not removed:return False
    try:answer=executor.execute(program,world.without_facts(removed))
    except ExecutionError:return True
    return normalize_answer(answer)!=baseline
def modality_indispensability_profile(program,world,*,executor=None):
    executor=executor or SymbolicExecutor();baseline=normalize_answer(executor.execute(program,world));source_types=sorted({source.source_type for source in world.sources.values()});required={};exclusive={}
    for source_type in source_types:
        removed=_exclusive_fact_ids(world,{source_type});exclusive[source_type]=len(removed);required[source_type]=_answer_changed(program,world,removed,baseline=baseline,executor=executor)
    media_present=sorted(set(source_types)&MEDIA_PROXY_TYPES);media_removed=_exclusive_fact_ids(world,MEDIA_PROXY_TYPES);text_only=not _answer_changed(program,world,media_removed,baseline=baseline,executor=executor);required_modalities=sorted(k for k,v in required.items() if v);required_media=sorted(set(required_modalities)&MEDIA_PROXY_TYPES)
    return {"source_types":source_types,"media_proxy_types_present":media_present,"has_media_proxy":bool(media_present),"required_by_source_type":required,"exclusive_fact_count_by_source_type":exclusive,"required_modalities":required_modalities,"required_modality_count":len(required_modalities),"required_media_proxy_types":required_media,"media_proxy_required":bool(required_media),"cross_modal_required":len(required_modalities)>=2,"cross_media_proxy_required":len(required_media)>=2,"text_only_answerable":text_only}
def audit_modality_jsonl(input_path,*,max_records=None):
    path=Path(input_path);total=media_present=media_required=text_only=cross_modal=cross_media=0;source_presence=Counter();required_by_type=Counter();count_hist=Counter()
    with path.open("r",encoding="utf-8") as handle:
        for line in handle:
            if max_records is not None and total>=max_records:break
            if not line.strip():continue
            record=json.loads(line);profile=modality_indispensability_profile(program_from_dict(record["program"]),world_from_dict(record["world"]));total+=1;media_present+=int(profile["has_media_proxy"]);media_required+=int(profile["media_proxy_required"]);text_only+=int(profile["text_only_answerable"]);cross_modal+=int(profile["cross_modal_required"]);cross_media+=int(profile["cross_media_proxy_required"]);source_presence.update(profile["source_types"]);required_by_type.update(profile["required_modalities"]);count_hist[profile["required_modality_count"]]+=1
    rate=lambda value:value/total if total else 0.
    return {"task_count":total,"media_proxy_present_tasks":media_present,"media_proxy_present_rate":rate(media_present),"media_proxy_required_tasks":media_required,"media_proxy_required_rate":rate(media_required),"text_only_answerable_tasks":text_only,"text_only_answerable_rate":rate(text_only),"cross_modal_required_tasks":cross_modal,"cross_modal_required_rate":rate(cross_modal),"cross_media_proxy_required_tasks":cross_media,"cross_media_proxy_required_rate":rate(cross_media),"source_type_presence":dict(source_presence.most_common()),"required_by_source_type":dict(required_by_type.most_common()),"required_modality_count_histogram":{str(k):v for k,v in sorted(count_hist.items())},"note":"image_report/video_report/audio_transcript are modality-derived proxies, not raw media. A required proxy means its exclusive facts are symbolically indispensable."}
def main():
    p=argparse.ArgumentParser(description="Audit symbolic modality indispensability");p.add_argument("--input",required=True);p.add_argument("--max-records",type=int);a=p.parse_args();print(json.dumps(audit_modality_jsonl(a.input,max_records=a.max_records),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
