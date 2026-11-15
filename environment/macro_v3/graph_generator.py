"""Compatibility exports for the typed Macro-first DAG sampler.

The canonical sampler is now part of the existing Stage-1 package.
"""
from environment.generate_question.typed_graph_generator import (
    VERSION, bind, waves, generate_graph,
)
