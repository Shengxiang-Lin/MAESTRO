"""Deterministic capability/tool hints for rendered evidence modalities."""
from __future__ import annotations
from typing import Mapping
_SOURCE_CAPABILITIES={
    "text":("document_reading","document_specialist",("read_text","search")),
    "table":("table_reasoning","table_specialist",("read_table","search")),
    "image":("visual_inspection","vision_specialist",("inspect_image",)),
    "image_report":("visual_inspection","vision_specialist",("inspect_image",)),
    "video":("video_inspection","video_specialist",("inspect_video",)),
    "video_report":("video_inspection","video_specialist",("inspect_video",)),
    "audio":("audio_inspection","audio_specialist",("inspect_audio",)),
    "audio_transcript":("audio_inspection","audio_specialist",("inspect_audio",)),
    "database":("structured_lookup","database_specialist",("query_database","search")),
}
def capability_hint(source_type:str):
    capability,specialist,tools=_SOURCE_CAPABILITIES.get(str(source_type),("document_reading","document_specialist",("read_text",)))
    return {"capability_id":capability,"specialist":specialist,"source_types":[str(source_type)],"tool_families":list(tools),"rationale":f"process {source_type} evidence"}
def available_tool_families(source_type:str):return tuple(capability_hint(source_type)["tool_families"])
def specialist_for_source(source_type:str):return str(capability_hint(source_type)["specialist"])
def capability_id_for_source(source_type:str):return str(capability_hint(source_type)["capability_id"])
__all__=["available_tool_families","capability_hint","capability_id_for_source","specialist_for_source"]
