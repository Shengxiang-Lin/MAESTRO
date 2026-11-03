"""Deterministic rendering for synthetic orchestration tasks."""
from .pipeline import render_jsonl,render_task_record,program_from_dict,world_from_dict
from .schema import RenderedSource,RenderedTask
__all__=["RenderedSource","RenderedTask","program_from_dict","render_jsonl","render_task_record","world_from_dict"]
