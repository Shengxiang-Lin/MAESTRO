"""Bind an abstract typed Macro DAG to simulated sources and genuine YAGO facts.

Only this layer introduces names, modalities, predicates and source scope.
Synthetic dossier annotations and YAGO RDF facts have disjoint provenance.
"""
from __future__ import annotations

import random

from .composer import _source
from .v31 import BUNDLES, choose_real_path
from .yago import KnowledgeGraph, Edge

VERSION = "maestro.typed_world.v1"


def _dummy_rows(rng, target: str, distractors: list[str],
                *, source_index: int, clue_count: int, cue: str, alternate: str):
    records = []
    for case in [target] + distractors:
        if case == target:
            value = cue
        elif distractors.index(case) < clue_count:
            # Critical decoy i satisfies every clue except clue i.
            value = alternate if distractors.index(case) == source_index else cue
        else:
            value = alternate
        records.append({"case_id": case, "cue": value})
    rng.shuffle(records)
    return records


def _paths_for_investigations(graph: KnowledgeGraph, rng: random.Random,
                              anchor: list[Edge], count: int) -> list[list[Edge]]:
    """Use actual YAGO edges; allow short branches and path prefixes."""
    paths: list[list[Edge]] = [list(anchor)]
    # A same-subject alternative produces a genuinely different investigation.
    alternatives = [
        edge for edge in graph.outgoing().get(anchor[0].subject, [])
        if (edge.subject, edge.predicate, edge.object) !=
           (anchor[0].subject, anchor[0].predicate, anchor[0].object)
    ]
    rng.shuffle(alternatives)
    for edge in alternatives:
        if len(paths) >= count:
            break
        paths.append([edge])
    prefixes = [anchor[:i] for i in range(1, len(anchor))]
    rng.shuffle(prefixes)
    while len(paths) < count:
        if prefixes:
            paths.append(prefixes.pop())
        else:
            paths.append(list(anchor))
    return paths[:count]


def ground_graph(graph: KnowledgeGraph, abstract: dict, rng: random.Random,
                 *, min_hops: int = 1, max_hops: int = 3) -> tuple[dict, dict]:
    """Create one frozen World from the already-sampled abstract graph."""
    nodes = abstract["nodes"]
    discovery = [n for n in nodes if n["role"] == "discovery"]
    research = [n for n in nodes if n["role"] == "investigation"]
    verify = [n for n in nodes if n["role"] == "verification"]
    if not 2 <= len(discovery) <= len(BUNDLES):
        raise ValueError("typed_world_discovery_budget")

    anchor = choose_real_path(graph, rng, min_hops=min_hops, max_hops=max_hops)
    selected = f"Ref-{rng.randrange(100000, 999999)}"
    n_clues = 2 * len(discovery)
    other_ids = [f"Ref-{rng.randrange(100000, 999999)}-{i}"
                 for i in range(n_clues + rng.randint(2, 4))]
    cases = [selected] + other_ids
    rng.shuffle(cases)

    sources = {}
    criteria = {}
    public_hints = []
    for i, node in enumerate(discovery):
        choices = {}
        node["source_scope"] = []
        for j, (modality, title, attribute, values) in enumerate(BUNDLES[i]):
            idx = i*2+j
            desired = rng.choice(values)
            other = rng.choice([value for value in values if value != desired])
            sid = f"{node['macro_id']}_clue_{j+1}"
            rows = _dummy_rows(rng, selected, other_ids,
                               source_index=idx, clue_count=n_clues,
                               cue=desired, alternate=other)
            sources[sid] = _source(
                sid, modality, title + " for fictional dossiers",
                ["case_id", "cue"], rows,
            )
            node["source_scope"].append(sid)
            choices[sid] = desired
            public_hints.append({
                "source_id": sid, "title": sources[sid]["title"],
                "attribute": attribute, "cue": desired,
            })
        criteria[node["macro_id"]] = choices
        node["objective"] = (
            "Investigate two different archival source records, compare their "
            "candidate dossiers and return the matching cases with evidence"
        )

    index_id = "entity_crosswalk"
    known_subjects = list(dict.fromkeys(e.subject for e in graph.edges))
    # Avoid O(250K)-list shuffle per scene.
    rng.shuffle(known_subjects) if len(known_subjects) <= 1000 else None
    idx_rows = []
    for case in cases:
        if case == selected:
            uri = anchor[0].subject
        else:
            uri = "urn:maestro:synthetic:index:" + case
            for _ in range(60):
                candidate = rng.choice(known_subjects)
                if candidate != anchor[0].subject:
                    uri = candidate
                    break
        idx_rows.append({"case_id": case, "entity_uri": uri})
    rng.shuffle(idx_rows)
    sources[index_id] = _source(
        index_id, "table", "reference dossier entity crosswalk (synthetic)",
        ["case_id", "entity_uri"], idx_rows, mode="keyed_lookup",
    )
    identify = next(n for n in nodes if n["role"] == "identification")
    identify["source_scope"] = [index_id]
    identify["objective"] = (
        "Reconcile candidate dossiers across the provided source investigations "
        "and identify the unique YAGO-linked reference"
    )

    routes = _paths_for_investigations(graph, rng, anchor, len(research))
    relation_bindings = {}
    real_facts = []
    for node, path in zip(research, routes):
        node["source_scope"] = []
        relation_bindings[node["macro_id"]] = []
        for j, fact in enumerate(path, 1):
            sid = f"{node['macro_id']}_yago_{j}"
            rows = [{
                "subject_uri": fact.subject, "predicate": fact.predicate,
                "object_uri": fact.object, "object_label": graph.label(fact.object),
            }]
            seen = {(fact.subject, fact.predicate, fact.object)}
            for _ in range(50):
                if len(seen) >= 5:
                    break
                other = rng.choice(graph.edges)
                key = (other.subject, other.predicate, other.object)
                if other.subject == fact.subject or key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "subject_uri": other.subject, "predicate": other.predicate,
                    "object_uri": other.object,
                    "object_label": graph.label(other.object),
                })
            rng.shuffle(rows)
            sources[sid] = _source(
                sid, "text", "YAGO entity-relation evidence excerpt",
                ["subject_uri", "predicate", "object_uri", "object_label"],
                rows, mode="keyed_lookup", origin="yago",
            )
            node["source_scope"].append(sid)
            relation_bindings[node["macro_id"]].append({
                "source_id": sid, "subject": fact.subject,
                "predicate": fact.predicate, "object": fact.object,
            })
            real_facts.append({
                "subject": fact.subject, "predicate": fact.predicate,
                "object": fact.object, "source_id": sid, "origin": "yago",
            })
        node["objective"] = (
            "Investigate the documented relations " +
            " then ".join(graph.label(e.predicate) for e in path) +
            " from the identified entity using scoped evidence"
        )
        if "verified_context" in node["input_ports"]:
            node["objective"] += ", incorporating the preceding verified assessment"

    reviews = {}
    status_pool = ["confirmed", "provisional", "requires_followup"]
    for node in verify:
        src = node["input_ports"]["finding"]["from_macro"]
        path = relation_bindings[src]
        target_object = path[-1]["object"]
        ground_status = rng.choice(status_pool)
        conflict = rng.random() < 0.55
        other_status = rng.choice([x for x in status_pool if x != ground_status])
        ids = []
        for suffix, modality, name, status in (
            ("a", "text", "first independently written dossier assessment",
             ground_status),
            ("b", "table", "separate survey of dossier assessment",
             other_status if conflict else ground_status),
            ("review", "text", "case-specific synthetic adjudication record",
             ground_status),
        ):
            sid = f"{node['macro_id']}_{suffix}"
            sources[sid] = _source(
                sid, modality, name,
                ["object_uri", "assessment"],
                [{"object_uri": target_object, "assessment": status}],
                mode="keyed_lookup",
            )
            ids.append(sid)
        node["source_scope"] = ids
        reviews[node["macro_id"]] = ids
        node["objective"] = (
            "Verify the upstream YAGO-linked finding against two independent "
            "simulated dossier assessments and an adjudication record"
        )

    for node in nodes:
        if node["role"] in {"comparison", "synthesis"}:
            node["source_scope"] = []
            node["objective"] = (
                "Compare the two independently investigated relation outcomes"
                if node["role"] == "comparison"
                else "Combine the typed source-backed investigations, verified "
                     "claims and original witness into the requested conclusion"
            )
        node["input_contract"] = {
            name: dict(link) for name, link in node["input_ports"].items()
        }
    world = {
        "version": VERSION, "simulated_only": True,
        "sources": sources,
        "real_fact_path": real_facts,
        "private_witness": {
            "case_id": selected, "entity_uri": anchor[0].subject,
        },
    }
    binding = {
        "version": "maestro.world_binding.v1",
        "criteria": criteria,
        "relation_paths": relation_bindings,
        "review_sources": reviews,
        "question_hints": public_hints,
    }
    return world, binding
