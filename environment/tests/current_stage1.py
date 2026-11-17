"""Current-only Stage-1 Scene fixtures shared by downstream pipeline tests.

No retired approval/macro-first/typed-knowledge constructors are imported.
"""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache

@lru_cache(maxsize=128)
def _scene(i: int, difficulty: str):
    from environment.generate_question.grounding.yago import Edge, KnowledgeGraph
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
    index = YagoKnowledgeIndex(graph)
    # A tiny three-edge fixture can have a coincidental reference-word/clue
    # collision for a handful of ordinals. Select the first fully guarded,
    # deterministic Task Plan rather than weakening the actual leakage guard.
    for attempt in range(32):
        ordinal = i + 1000 * attempt
        spec = make_spec(
            graph, seed=17, ordinal=ordinal, difficulty=difficulty,
            min_hops=2, max_hops=3, index=index,
        )
        try:
            return compile_and_validate(spec, reference_question(spec))
        except ValueError as exc:
            if "task_plan_answer_clues_leaked_into_question" not in str(exc):
                raise
    raise RuntimeError("unable_to_construct_valid_test_task_plan")

def current_scene(i: int = 0, difficulty: str = "hard") -> dict:
    if difficulty not in {"medium", "hard"}:
        difficulty = "medium"
    return deepcopy(_scene(i, difficulty))
