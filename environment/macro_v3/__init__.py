"""Experimental graph-grounded Macro-first Stage 1 (v3).

Kept separate from the frozen v2 scene/replay contract.
"""
from .composer import compose_scene, replay_scene, validate_scene
from .yago import KnowledgeGraph, load_yago

__all__ = ["compose_scene", "replay_scene", "validate_scene", "KnowledgeGraph", "load_yago"]
