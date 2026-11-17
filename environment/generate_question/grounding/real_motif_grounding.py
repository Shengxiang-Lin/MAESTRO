"""Causally grounded *real* YAGO forks for synthetic event validation.

One compatible extra real-world assertion per candidate, with a common
predicate and distinct RDF objects. Every edge is an actual observed
KnowledgeGraph Edge. The fork is a required Macro source in v4 (unlike the
context-only motif inventory from v3).
"""
from __future__ import annotations

from collections import defaultdict
import random

from environment.schema import digest

VERSION = "maestro.real_fork_grounding.v1"


def _serialize(graph, edge):
    return {
        "subject": edge.subject,
        "predicate": edge.predicate,
        "object": edge.object,
        "subject_label": graph.label(edge.subject),
        "object_label": graph.label(edge.object),
        "origin": "yago",
        "fact_id": "yago:" + digest(
            edge.subject, edge.predicate, edge.object
        )[:20],
    }


def mine_required_fork(graph, index, candidate_paths: dict,
                       *, seed: int, ordinal: int) -> dict:
    """Find a common secondary relation type for *every* candidate root.

    A missing fork for ANY candidate would reintroduce the v1
    shortcut (later lookup accidentally eliminates decoys), so fail
    closed instead of padding with invented links. Prefer a relation
    different from the primary chain predicate.
    """
    paths = list(candidate_paths.items())
    by_case = {}
    for case, chain in paths:
        root = chain[0]["subject"]
        excluded = {
            (fact["subject"], fact["predicate"], fact["object"])
            for fact in chain
        }
        by_pred = defaultdict(list)
        for edge in index.outgoing.get(root, ()):
            if ((edge.subject, edge.predicate, edge.object) not in excluded
                    and edge.subject != edge.object):
                by_pred[edge.predicate].append(edge)
        by_case[case] = by_pred
    if not paths:
        raise ValueError("real_fork_empty_candidate_paths")
    predicates = set.intersection(*[
        set(by_case[case]) for case, _ in paths
    ])
    if not predicates:
        raise ValueError("no_common_real_yago_fork_predicate")
    rng = random.Random(f"maestro-grounded-required-fork:{seed}:{ordinal}")
    order = sorted(predicates)
    rng.shuffle(order)
    order.sort(key=lambda p: (
        p == paths[0][1][0]["predicate"],
        max(len(by_case[case][p]) for case, _ in paths),
    ))
    for predicate in order:
        chosen, seen_objects = {}, set()
        # Hard cases first: reduce greedy collisions across RDF objects.
        randomized_cases = list(by_case)
        rng.shuffle(randomized_cases)
        randomized_cases.sort(key=lambda case: len(by_case[case][predicate]))
        failed = False
        for case in randomized_cases:
            options = list(by_case[case][predicate])
            rng.shuffle(options)
            item = next((edge for edge in options
                         if edge.object not in seen_objects), None)
            if item is None:
                failed = True
                break
            chosen[case] = _serialize(graph, item)
            seen_objects.add(item.object)
        if failed:
            continue
        return {
            "version": VERSION, "motif_kind": "real_fork",
            "predicate": predicate,
            "candidate_forks": {
                case: chosen[case] for case, _ in paths
            },
            "source_lineage": "YAGO-4.6",
            "causal_use": "required_real_fork_macro_evidence",
        }
    raise ValueError("no_distinct_real_yago_fork_objects_for_candidates")


def validate_frozen_required_fork(fork: dict, paths: dict) -> list[str]:
    """Replay checks the frozen RDF witnesses; creation checks actual KG."""
    try:
        if (fork["version"] != VERSION
                or fork["motif_kind"] != "real_fork"
                or fork["causal_use"] != "required_real_fork_macro_evidence"
                or fork["source_lineage"] != "YAGO-4.6"
                or set(fork["candidate_forks"]) != set(paths)):
            return ["required_fork_wrong_contract"]
        objects = set()
        for case, path in paths.items():
            item = fork["candidate_forks"][case]
            if (item["subject"] != path[0]["subject"]
                    or item["predicate"] != fork["predicate"]
                    or item["origin"] != "yago"
                    or (item["subject"], item["predicate"], item["object"])
                    in {(f["subject"], f["predicate"], f["object"])
                        for f in path}
                    or item["fact_id"] != "yago:" + digest(
                        item["subject"], item["predicate"], item["object"]
                    )[:20]):
                return ["required_fork_not_a_real_secondary_relation"]
            if item["object"] in objects:
                return ["required_fork_duplicate_real_objects"]
            objects.add(item["object"])
        return []
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return ["required_fork_invalid:" + str(exc)[:120]]
