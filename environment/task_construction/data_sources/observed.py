"""Offline graph-support augmentation for the compact YAGO predicate catalog.

The full facts file is streamed, but runtime keeps only bounded evidence: typed
edge statistics plus real sampled-center graph witnesses. Witness identities let
source binding use YAGO as a grounding source instead of interpreting missing
co-occurrence counts as semantic impossibility.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

from .yago import METADATA_PREDICATES, TYPE_PREDICATES, iter_turtle_triples

WITNESS_CAP_PER_TYPED_MOTIF = 32


def sampled_centers(catalog: Mapping[str, Any]) -> set[str]:
    return {str(edge["subject"]) for info in catalog["predicates"].values()
            if info.get("kind") != "metadata" for edge in info.get("samples", ())}


def augment_observed_support(catalog: dict[str, Any], *, facts_path=None, progress=False) -> dict:
    predicates = catalog["predicates"]
    centers = sampled_centers(catalog)
    endpoints = set(centers)
    for info in predicates.values():
        if info.get("kind") == "metadata":
            continue
        endpoints.update(str(e["object"]) for e in info.get("samples", ())
                         if not str(e["object"]).startswith(('"', "'")))
    types: dict[str, set[str]] = defaultdict(set)
    ports: dict[str, set[str]] = defaultdict(set)
    for triple in iter_turtle_triples(
        facts_path or catalog["source"], progress=progress,
        description="YAGO graph witness support",
    ):
        if triple.predicate in TYPE_PREDICATES:
            if triple.subject in endpoints:
                types[triple.subject].add(triple.object)
        elif triple.predicate not in METADATA_PREDICATES:
            if triple.subject in centers:
                ports[triple.subject].add("out|" + triple.predicate)
            if not triple.object_is_literal and triple.object in centers:
                ports[triple.object].add("in|" + triple.predicate)
    return compile_observed_support(catalog, types, ports, facts_path=facts_path)


def compile_observed_support(catalog, types, ports, *, facts_path=None):
    """Compile edge priors and shared-entity witness identities from one type pass."""
    predicates = catalog["predicates"]
    centers = sampled_centers(catalog)
    for info in predicates.values():
        if info.get("kind") == "metadata":
            continue
        counts = Counter()
        for edge in info.get("samples", ()):
            subject_types = tuple(sorted(types.get(edge["subject"], ())))
            literal = str(edge["object"]).startswith(('"', "'"))
            object_types = () if literal else tuple(sorted(types.get(edge["object"], ())))
            if subject_types and (literal or object_types):
                counts[(subject_types, object_types)] += 1
        info["observed_edge_types"] = [
            [list(subject_types), list(object_types), count]
            for (subject_types, object_types), count in sorted(counts.items())
        ]

    motif_counts = Counter()
    motif_witnesses: dict[tuple, list[str]] = defaultdict(list)
    for entity in sorted(centers):
        center_types = tuple(sorted(types.get(entity, ())))
        if not center_types:
            continue
        for pair in combinations(sorted(ports.get(entity, ())), 2):
            key = (pair, center_types)
            motif_counts[key] += 1
            bucket = motif_witnesses[key]
            if len(bucket) < WITNESS_CAP_PER_TYPED_MOTIF:
                bucket.append(entity)

    observed_rows = [
        {"ports": list(pair), "center_types": list(center_types), "count": count}
        for (pair, center_types), count in sorted(motif_counts.items())
    ]
    witness_rows = [
        {"ports": list(pair), "center_types": list(center_types), "count": count,
         "witnesses": list(motif_witnesses[(pair, center_types)])}
        for (pair, center_types), count in sorted(motif_counts.items())
    ]
    source = str(facts_path or catalog["source"])
    catalog["observed_support"] = {
        "version": 1, "source": source,
        "policy": "existing_reservoir_joint_edges_and_complete_incident_ports_on_sampled_centers",
        "center_count": len(centers), "typed_endpoint_count": len(types),
        "motifs": observed_rows,
        "motif_semantics": "two directed predicate ports co-observed on one real typed entity; endpoints checked separately",
    }
    catalog["graph_witness_support"] = {
        "version": 1,
        "schema_id": "graph_witness_support.v1",
        "source": source,
        "policy": "real_shared_entity_witnesses_on_sampled_centers",
        "witness_cap_per_typed_motif": WITNESS_CAP_PER_TYPED_MOTIF,
        "center_count": len(centers),
        "typed_endpoint_count": len(types),
        "motifs": witness_rows,
        "semantics": (
            "each witness id is one real YAGO entity on which all listed directed "
            "predicate ports were observed; multi-port grounding intersects witness ids"
        ),
    }
    if catalog.get("typed_literal_support", {}).get("version") != 1:
        from .typed_literals import compile_typed_literal_support
        compile_typed_literal_support(catalog, types)
    return catalog


def augment_catalog_file(input_path, output_path=None, *, facts_path=None, progress=False) -> Mapping[str, Any]:
    catalog = json.loads(Path(input_path).read_text(encoding="utf-8"))
    augment_observed_support(catalog, facts_path=facts_path, progress=progress)
    target = Path(output_path or input_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return {
        "predicates": len(catalog["predicates"]),
        "motifs": len(catalog["observed_support"]["motifs"]),
        "graph_witness_motifs": len(catalog["graph_witness_support"]["motifs"]),
        "typed_endpoint_count": catalog["observed_support"]["typed_endpoint_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--facts")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("refuse to overwrite an existing graph-witness catalog")
    result = augment_catalog_file(args.input, args.output, facts_path=args.facts, progress=args.progress)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
