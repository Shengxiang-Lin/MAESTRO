"""Constraint-based synthetic evidence World synthesis from a typed graph.

Generate a satisfying witness first, then three overlapping decoys and
source-scoped records. Entity labels and row order are randomized; source
facts remain stable under replay. No original MMQA media is accessed.
"""
from __future__ import annotations

import random

from environment.schema import digest


def generate_evidence_world(spec: dict) -> dict:
    from .graph_grammar import validate_evidence_spec
    from .world import _rows, WORLD_VERSION

    issues = validate_evidence_spec(spec)
    if issues:
        raise ValueError("invalid_evidence_graph:" + ",".join(issues))
    graph = spec["evidence_graph"]
    rng = random.Random(int(digest(
        spec["seed"], spec["task_id"], "evidence-world-v1"
    )[:15], 16))
    mode, hops = graph["archetype"], graph["lookup_hops"]
    features = graph["visual_features"]
    # Target is an internal witness, not a public label. Two decoys share
    # one feature each. Shuffle display identities independently of their
    # semantic target roles, so 'Item-01' is not an answer shortcut.
    if features is not None:
        from .graph_grammar import COLORS, MOTIFS
        color = features["emblem_color"]
        motif = features["emblem_shape"]
        other_color = rng.choice([x for x in COLORS if x != color])
        other_motif = rng.choice([x for x in MOTIFS if x != motif])
        descriptors = [
            (color, motif), (color, other_motif),
            (other_color, motif), (other_color, other_motif),
        ]
    else:
        descriptors = [(None, None)] * 4
    labels = [f"Item-{i:02d}" for i in range(1, 5)]
    rng.shuffle(labels)
    markers = rng.sample(range(100, 999), 4)
    years = rng.sample(range(1960, 2023), 4)
    # The four keys at each hop are unique, but the actual key values
    # are hidden until the preceding source has been consulted.
    keys_at_hop = [
        rng.sample(range(1000, 9999), 4)
        for _ in range(hops - 1)
    ]
    items = []
    for i, (item_color, item_motif) in enumerate(descriptors):
        entry = {
            "candidate_id": labels[i], "case_id": labels[i],
            "emblem_color": item_color, "emblem_shape": item_motif,
            "marker_code": f"M-{markers[i]:03d}",
            "accession_year": years[i],
        }
        for j, values in enumerate(keys_at_hop, 1):
            entry[f"key_{j}"] = f"AC{j}-{values[i]:04d}"
        items.append(entry)
    data = {}

    if mode == "chain":
        data["illustration"] = [{
            k: item[k] for k in (
                "candidate_id", "emblem_color", "emblem_shape", "marker_code"
            )
        } for item in items]
    elif mode == "parallel":
        data["illustration"] = [{
            k: item[k] for k in ("candidate_id", "emblem_color")
        } for item in items]
        data["caption"] = [{
            k: item[k] for k in (
                "candidate_id", "emblem_shape", "marker_code"
            )
        } for item in items]
    elif mode == "disambiguation":
        data["illustration"] = [{
            k: item[k] for k in (
                "candidate_id", "emblem_color", "marker_code"
            )
        } for item in items]
        data["notes"] = [{
            k: item[k] for k in ("candidate_id", "emblem_shape")
        } for item in items]
    elif mode == "conflict":
        data["report_a"] = [{
            "case_id": item["case_id"],
            "reported_year_a": item["accession_year"] - 2,
        } for item in items]
        data["report_b"] = [{
            "case_id": item["case_id"],
            # Only the target contains a discrepancy; the two reports for
            # every decoy agree, so neither report alone identifies it.
            "reported_year_b": (item["accession_year"] - 1 if i == 0
                                else item["accession_year"] - 2),
        } for i, item in enumerate(items)]
    else:
        raise ValueError("unsupported_evidence_world_archetype")

    prior_key = "case_id" if mode == "conflict" else "marker_code"
    for i in range(1, hops):
        next_key = f"key_{i}"
        data[f"index_{i}"] = [
            {
                prior_key: item[prior_key],
                next_key: item[next_key],
                **({"candidate_id": item["candidate_id"]}
                   if mode == "disambiguation" and i == 1 else {}),
            }
            for item in items
        ]
        prior_key = next_key
    data["archive"] = [
        {prior_key: item[prior_key],
         "accession_year": item["accession_year"]}
        for item in items
    ]

    sources = {}
    for role in spec["sources"]:
        sid = role["source_id"]
        rows = data[sid]
        rng.shuffle(rows)
        actual_schema = set(rows[0])
        if actual_schema != set(graph["source_schemas"][sid]):
            raise ValueError("synthesized_source_schema_mismatch:" + sid)
        policy = {"mode": role["access_mode"]}
        if policy["mode"] == "keyed_lookup":
            policy["key_field"] = role["key_field"]
        sources[sid] = {
            **_rows(sid, role["modality"], role["title"],
                    role["purpose"], rows),
            "access_policy": policy,
        }
    return {
        "version": WORLD_VERSION,
        "simulated_only": True,
        "sources": sources,
        "entities": [{
            "hidden_entity_id": f"fictional-evidence-item-{i:03d}",
            "identity_source": "synthetic_generation_only",
        } for i in range(4)],
        "generation_seed": spec["seed"],
        "value_sampling_contract": (
            "witness_first_typed_graph_decoys_with_randomized_identity"
        ),
    }


def mutate_evidence_witness(sources: dict, sid: str,
                            gold_workproducts: list[dict], spec: dict) -> dict:
    """One grounded counterfactual per source, not a source-name guess."""
    from copy import deepcopy

    revised = deepcopy(sources)
    refs = {
        ref for row in gold_workproducts
        for ref in row["source_fact_refs"]
        if ref.startswith(sid + ":")
    }
    selected = next(
        (row for row in revised[sid]["observations"]
         if refs.intersection(row["fact_refs"])),
        None,
    )
    if selected is None:
        raise ValueError("no_grounded_graph_witness:" + sid)
    values = selected["values"]
    mode = spec["evidence_graph"]["archetype"]
    if sid == "illustration":
        values["emblem_color"] = "__different_color__"
    elif sid in {"caption", "notes"}:
        values["emblem_shape"] = "__different_motif__"
    elif sid in {"report_a", "report_b"}:
        other = "report_b" if sid == "report_a" else "report_a"
        key = "reported_year_a" if sid == "report_a" else "reported_year_b"
        other_key = "reported_year_b" if sid == "report_a" else "reported_year_a"
        partner = next(
            row["values"] for row in revised[other]["observations"]
            if row["values"]["case_id"] == values["case_id"]
        )
        values[key] = partner[other_key]
    elif sid == "archive":
        values["accession_year"] += 1
    elif sid.startswith("index_"):
        output_key = f"key_{int(sid.split('_')[1])}"
        values[output_key] = "__unresolvable_downstream_key__"
    else:
        raise ValueError("unexpected_evidence_source:" + sid)
    return revised
