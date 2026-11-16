"""Mine real YAGO relation motifs for hybrid-world task grounding.

The miner labels only RDF edges from the provided KnowledgeGraph as real.
Extra motif edges are optional *world context*, not automatically claimed
as necessary evidence or independent publishers. The generated task's
mandatory path is supplied by the causally checked knowledge v2 backbone.
"""
from __future__ import annotations

from collections import defaultdict
import random

from environment.schema import digest

VERSION = "maestro.real_world_motif.v1"


def _fact(edge, graph) -> dict:
    return {
        "subject": edge.subject,
        "predicate": edge.predicate,
        "object": edge.object,
        "subject_label": graph.label(edge.subject),
        "object_label": graph.label(edge.object),
        "origin": "yago",
        "fact_id": "yago:" + digest(
            edge.subject, edge.predicate, edge.object,
        )[:20],
    }


def mine_motif(graph, index, anchored_path: list[dict],
               *, seed: int, ordinal: int) -> dict:
    """Classify an observed chain and optionally a true fork/diamond.

    Diamond: another 2-hop route from A to the already observed B, with
    a different intermediate entity. Fork: an actual non-path outgoing
    relation from a node on the mandatory chain. Never create an RDF edge.
    """
    rng = random.Random(f"maestro-yago-motifs-v1:{seed}:{ordinal}")
    triples = {(e.subject, e.predicate, e.object): e for e in graph.edges}
    chain = []
    for row in anchored_path:
        key = row["subject"], row["predicate"], row["object"]
        if key not in triples:
            raise ValueError("knowledge_motif_missing_yago_anchor:" + str(key))
        chain.append(triples[key])
    if not chain:
        raise ValueError("knowledge_motif_empty_anchor")

    chain_set = {(e.subject, e.predicate, e.object) for e in chain}
    # A genuine alternate route to an existing anchor destination is
    # especially useful for future verification Macros.
    for i in range(len(chain)-1):
        start, end = chain[i].subject, chain[i+1].object
        direct_middle = chain[i].object
        proposals = list(index.outgoing.get(start, ()))
        rng.shuffle(proposals)
        for left in proposals[:120]:
            if left.object == direct_middle or left.object == start:
                continue
            for right in index.outgoing.get(left.object, ())[:250]:
                if right.object == end and right.subject == left.object:
                    return {
                        "version": VERSION, "kind": "diamond",
                        "anchor_facts": [_fact(e, graph) for e in chain],
                        "extra_facts": [_fact(left, graph), _fact(right, graph)],
                        "source_lineage": "YAGO-4.6",
                        "causal_use": "context_only_not_macro_evidence",
                    }

    possible = []
    chain_nodes = {e.subject for e in chain} | {chain[-1].object}
    for i, item in enumerate(chain):
        for edge in index.outgoing.get(item.subject, ())[:160]:
            key = edge.subject, edge.predicate, edge.object
            if key not in chain_set and edge.object not in chain_nodes:
                possible.append((i, edge))
    if possible:
        i, sibling = rng.choice(possible)
        return {
            "version": VERSION, "kind": "fork",
            "fork_at_chain_position": i,
            "anchor_facts": [_fact(e, graph) for e in chain],
            "extra_facts": [_fact(sibling, graph)],
            "source_lineage": "YAGO-4.6",
            "causal_use": "context_only_not_macro_evidence",
        }
    return {
        "version": VERSION, "kind": "chain",
        "anchor_facts": [_fact(e, graph) for e in chain],
        "extra_facts": [],
        "source_lineage": "YAGO-4.6",
        "causal_use": "context_only_not_macro_evidence",
    }


def validate_frozen_motif(motif: dict, anchored_path: list[dict]) -> list[str]:
    """Self-contained replay checks; never assert external RDF re-verification."""
    if motif.get("version") != VERSION or motif.get("kind") not in {
        "chain", "fork", "diamond",
    }:
        return ["hybrid_bad_real_motif_version"]
    anchor = motif.get("anchor_facts")
    if not isinstance(anchor, list) or len(anchor) != len(anchored_path):
        return ["hybrid_bad_real_motif_anchor_count"]
    for frozen, expected in zip(anchor, anchored_path):
        if any(frozen.get(k) != expected.get(k)
               for k in ("subject", "predicate", "object", "object_label")):
            return ["hybrid_real_motif_anchor_mismatch"]
        if frozen.get("origin") != "yago":
            return ["hybrid_real_motif_wrong_origin"]
    extra = motif.get("extra_facts", [])
    if len(extra) != {"chain": 0, "fork": 1, "diamond": 2}[motif["kind"]]:
        return ["hybrid_bad_motif_extra_fact_count"]
    for row in anchor + extra:
        if (row.get("origin") != "yago"
                or row.get("fact_id") != "yago:" + digest(
                    row["subject"], row["predicate"], row["object"],
                )[:20]):
            return ["hybrid_bad_motif_fact_id"]
    if motif.get("causal_use") != "context_only_not_macro_evidence":
        return ["hybrid_motif_unjustified_causal_claim"]
    return []
