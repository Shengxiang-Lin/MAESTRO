"""Current-only Stage-1 Scene fixtures shared by downstream pipeline tests.

No retired approval/macro-first/typed-knowledge constructors are imported.
"""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache

@lru_cache(maxsize=128)
def _scene(i: int, difficulty: str):
    from environment.macro_v3.yago import Edge, KnowledgeGraph
    from environment.generate_question.planning.knowledge_index import YagoKnowledgeIndex
    from environment.generate_question.planning.task_plan import make_spec
    from environment.generate_question.authoring.question import reference_question
    from environment.generate_question.auditing.quality import compile_and_validate
    p = "http://schema.org/"
    graph = KnowledgeGraph(edges=[
        Edge("https://example.org/person", p+"birthPlace", "https://example.org/town"),
        Edge("https://example.org/town", p+"containedInPlace", "https://example.org/region"),
        Edge("https://example.org/region", p+"containedInPlace", "https://example.org/country"),
    ], input_file="current-stage1-fixture.ttl", scanned=3)
    spec = make_spec(
        graph, seed=97, ordinal=i, difficulty=difficulty,
        min_hops=2, max_hops=3, index=YagoKnowledgeIndex(graph),
    )
    return compile_and_validate(spec, reference_question(spec))

def current_scene(i: int = 0, difficulty: str = "hard") -> dict:
    if difficulty not in {"medium", "hard"}:
        difficulty = "medium"
    return deepcopy(_scene(i, difficulty))
