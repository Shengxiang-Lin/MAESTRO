"""YAGO path index used by the sole current output-first Stage-1 planner.

No historical typed-knowledge generator; only reusable graph operations.
"""
from __future__ import annotations
from collections import defaultdict
import re

class YagoKnowledgeIndex:
    """Build once for a large YAGO dump; share indexes across all scenes.

    Neither the index nor the full YAGO corpus enters a frozen ProblemSpec.
    Each scene records only its actual entity-to-entity factual witnesses.
    """

    def __init__(self, graph):
        self.graph = graph
        self.outgoing = graph.outgoing()
        self.by_predicate = defaultdict(list)
        for fact in graph.edges:
            self.by_predicate[fact.predicate].append(fact)
        self.joinable = [
            fact for fact in graph.edges
            if any(
                next_fact.object != fact.subject
                and next_fact.object != fact.object
                for next_fact in self.outgoing.get(fact.object, ())
            )
        ]

    def connected_path(self, rng, *, min_hops: int, max_hops: int):
        if not 1 <= min_hops <= max_hops <= 5:
            raise ValueError("invalid_knowledge_hop_limits")
        lengths = list(range(min_hops, max_hops + 1))
        rng.shuffle(lengths)
        lengths.sort(key=lambda n: n > 1, reverse=True)
        for target in lengths:
            choices = self.joinable if target > 1 else self.graph.edges
            if not choices:
                continue
            for _ in range(120):
                start = rng.choice(choices)
                path, visited = [start], {start.subject, start.object}
                while len(path) < target:
                    following = [
                        e for e in self.outgoing.get(path[-1].object, ())
                        if e.object not in visited and e.subject != e.object
                    ]
                    if not following:
                        break
                    fact = rng.choice(following)
                    path.append(fact)
                    visited.add(fact.object)
                if len(path) == target:
                    return path
        raise ValueError("no_witnessed_yago_path_for_requested_depth")

def predicate_label(iri: str) -> str:
    tail = iri.rstrip("/#").rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", tail).replace("_", " ").strip()
