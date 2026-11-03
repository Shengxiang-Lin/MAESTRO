"""MultiModalQA context enrichment for scenario semantic bundles."""
from __future__ import annotations
import gzip,json
from contextlib import contextmanager
from pathlib import Path
from typing import Any,Iterator,Mapping,Sequence
from . import semantic_bundles as _sb
_CONTEXT_FILENAMES={"text":("texts.jsonl.gz","texts.jsonl","MMQA_texts.jsonl.gz","MMQA_texts.jsonl"),"table":("tables.jsonl.gz","tables.jsonl","MMQA_tables.jsonl.gz","MMQA_tables.jsonl"),"image":("images.jsonl.gz","images.jsonl","MMQA_images.jsonl.gz","MMQA_images.jsonl")}
def _iter_jsonl(path:str|Path)->Iterator[dict[str,Any]]:
    source=Path(path);opener=gzip.open if source.suffix.lower()==".gz" else open
    with opener(source,"rt",encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload=json.loads(line)
                if isinstance(payload,dict):yield payload
def _normalize_bank_modality(value:Any)->str:
    token=str(value or "text").strip().lower().replace("-","_").replace(" ","_")
    if token in {"image","img","photo","visual","image_report"}:return "image"
    if token in {"table","tabular"}:return "table"
    return "text"
def _bundle_modality(value:Any)->str:
    bank=_normalize_bank_modality(value);return "image_report" if bank=="image" else bank
def _source_type_with_doc_part(item:Mapping[str,Any])->str:
    for key in ("source_type","type","modality","media_type","doc_part","part"):
        if item.get(key) is not None:return _bundle_modality(item.get(key))
    return "text"
def _discover_context_banks(qa_paths:Sequence[str|Path])->dict[str,Path]:
    parents=[]
    for value in qa_paths:
        parent=Path(value).resolve().parent
        if parent not in parents:parents.append(parent)
    found={}
    for modality,names in _CONTEXT_FILENAMES.items():
        for parent in parents:
            match=next((parent/name for name in names if (parent/name).is_file()),None)
            if match is not None:found[modality]=match;break
    return found
def _supporting_context(record:Mapping[str,Any])->list[Mapping[str,Any]]:return _sb._supporting_context(record)
def _collect_needed_doc_ids(qa_paths:Sequence[str|Path],*,max_records:int|None)->tuple[dict[str,set[str]],int]:
    needed={"text":set(),"table":set(),"image":set()};scanned=0
    for path in qa_paths:
        for record in _iter_jsonl(path):
            if max_records is not None and scanned>=max_records:return needed,scanned
            scanned+=1
            for item in _supporting_context(record):
                doc_id=item.get("doc_id") or item.get("source_id") or item.get("id")
                if doc_id is None:continue
                modality=_normalize_bank_modality(item.get("doc_part") or item.get("part") or item.get("modality") or item.get("source_type") or item.get("type"));needed[modality].add(str(doc_id))
    return needed,scanned
def _cell_text(cell:Any)->str|None:
    value=(cell.get("text") or cell.get("value") or cell.get("name")) if isinstance(cell,Mapping) else cell
    if value is None:return None
    text=" ".join(str(value).split());return text or None
def _compact_table(record:Mapping[str,Any])->dict[str,Any]:
    table=record.get("table");table=table if isinstance(table,Mapping) else {};header_raw=table.get("header",());headers=[]
    if isinstance(header_raw,Sequence) and not isinstance(header_raw,(str,bytes)):
        for item in header_raw[:12]:
            value=(item.get("column_name") or item.get("text") or item.get("name")) if isinstance(item,Mapping) else item
            if value is not None and str(value).strip():headers.append(" ".join(str(value).split()))
    rows=[];raw_rows=table.get("table_rows") or table.get("rows") or ()
    if isinstance(raw_rows,Sequence) and not isinstance(raw_rows,(str,bytes)):
        for raw_row in raw_rows[:8]:
            if not isinstance(raw_row,Sequence) or isinstance(raw_row,(str,bytes)):continue
            row=[value for cell in raw_row[:10] if (value:=_cell_text(cell))]
            if row:rows.append(row)
    return {"id":record.get("id"),"title":record.get("title"),"table":{"table_name":table.get("table_name"),"headers":headers,"rows":rows}}
def _compact_context(record:Mapping[str,Any],modality:str)->dict[str,Any]:
    if modality=="table":return _compact_table(record)
    if modality=="image":return {"id":record.get("id"),"title":record.get("title"),"caption":record.get("caption") or record.get("image_caption"),"description":record.get("description") or record.get("alt_text")}
    return {"id":record.get("id"),"title":record.get("title"),"text":record.get("text") or record.get("passage") or record.get("content")}
def _load_context_index(banks:Mapping[str,Path],needed:Mapping[str,set[str]])->tuple[dict[tuple[str,str],dict[str,Any]],dict[str,Any]]:
    index={};stats={"banks":{},"needed":{k:len(v) for k,v in needed.items()}}
    for modality,path in banks.items():
        wanted=needed.get(modality,set());scanned=matched=0
        if wanted:
            for record in _iter_jsonl(path):
                scanned+=1;doc_id=record.get("id") or record.get("doc_id") or record.get("source_id")
                if doc_id is None or str(doc_id) not in wanted:continue
                index[(modality,str(doc_id))]=_compact_context(record,modality);matched+=1
                if matched>=len(wanted):break
        stats["banks"][modality]={"path":str(path),"scanned":scanned,"matched":matched,"requested":len(wanted)}
    stats["joined"]=len(index);return index,stats
@contextmanager
def _patched_bundle_parser(context_index:Mapping[tuple[str,str],Mapping[str,Any]]):
    original_source_type=_sb._source_type;original_supporting_context=_sb._supporting_context
    def enriched_supporting_context(record):
        result=[]
        for item in original_supporting_context(record):
            stub=dict(item);doc_id=stub.get("doc_id") or stub.get("source_id") or stub.get("id");modality=_normalize_bank_modality(stub.get("doc_part") or stub.get("part") or stub.get("modality") or stub.get("source_type") or stub.get("type"));merged={}
            if doc_id is not None:merged.update(context_index.get((modality,str(doc_id)),{}))
            merged.update(stub);merged["source_type"]=_bundle_modality(modality);result.append(merged)
        return result
    _sb._source_type=_source_type_with_doc_part;_sb._supporting_context=enriched_supporting_context
    try:yield
    finally:_sb._source_type=original_source_type;_sb._supporting_context=original_supporting_context
def extract_mmqa_semantic_bundle(record:Mapping[str,Any],*,record_index:int|None=None):
    with _patched_bundle_parser({}):return _sb.extract_mmqa_semantic_bundle(record,record_index=record_index)
def build_mmqa_semantic_bundle_bank(paths:Sequence[str|Path],output_path:str|Path,*,summary_path:str|Path|None=None,max_records:int|None=None,progress:bool=False)->dict[str,Any]:
    qa_paths=[Path(value) for value in paths];needed,qa_records_scanned=_collect_needed_doc_ids(qa_paths,max_records=max_records);banks=_discover_context_banks(qa_paths);index,enrichment=_load_context_index(banks,needed)
    with _patched_bundle_parser(index):summary=_sb.build_mmqa_semantic_bundle_bank(qa_paths,output_path,summary_path=summary_path,max_records=max_records,progress=progress)
    summary=dict(summary);summary["context_enrichment"]={**enrichment,"qa_records_scanned_for_ids":qa_records_scanned,"discovered_modalities":sorted(banks),"missing_context_banks":sorted(set(_CONTEXT_FILENAMES)-set(banks)),"policy":"join_supporting_context_doc_id_to_official_context_bank"};return summary
__all__=["build_mmqa_semantic_bundle_bank","extract_mmqa_semantic_bundle"]
