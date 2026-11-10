"""Clean, simulated clue-driven evidence acquisition benchmark motif.

A pictured motif must identify a unique marker. That marker unlocks one
index entry; the accession identifier then unlocks ONE archive record.
The archive/index are keyed stores, NOT downloadable tables. Two decoys
share one target feature each, so reading one feature is insufficient.

The source/world/Gold Program are deterministic and use simulated
structured image observations only; no real MMQA media is decoded.
"""
from __future__ import annotations

import random

from envir.schema import digest
from .world import _node, _rows

CLUE_PATTERN = "clue_chain"
CLUE_FAMILY = "accession_year"
VISUAL_PAIRS = (
    ("crimson", "twin-wing"),
    ("azure", "crescent"),
    ("amber", "spiral"),
    ("emerald", "trident"),
)


def clue_features(ordinal: int) -> dict:
    color, shape = VISUAL_PAIRS[ordinal % len(VISUAL_PAIRS)]
    return {"emblem_color": color, "emblem_shape": shape}


def compile_clue_program(spec: dict) -> dict:
    from .spec import spec_issues
    invalid = spec_issues(spec)
    if invalid:
        raise ValueError("invalid_clue_spec:" + ",".join(invalid))
    features = spec["investigation"]["visual_features"]
    nodes = [
        _node("illustration_observations", "scan", source_id="illustration"),
        _node("matching_color", "filter_eq", "illustration_observations",
              field="emblem_color", value=features["emblem_color"]),
        _node("matching_shape", "filter_eq", "matching_color",
              field="emblem_shape", value=features["emblem_shape"]),
        _node("observed_marker", "project", "matching_shape",
              fields=["marker_code"]),
        _node("index_lookup", "lookup_by_key", "observed_marker",
              source_id="index", input_key_field="marker_code",
              source_key_field="marker_code"),
        _node("accession_key", "project", "index_lookup",
              fields=["accession_id"]),
        _node("archive_lookup", "lookup_by_key", "accession_key",
              source_id="archive", input_key_field="accession_id",
              source_key_field="accession_id"),
        _node("answer_year", "project", "archive_lookup",
              fields=["accession_year"]),
        _node("unique_answer_year", "unique", "answer_year"),
    ]
    return {
        "version": "maestro.reasoning_program.v4",
        "semantic_operator_pattern": CLUE_PATTERN,
        "nodes": nodes,
        "output": "unique_answer_year",
        "authorized_source_ids": [role["source_id"] for role in spec["sources"]],
        "frozen_problem_spec_id": spec["task_id"],
    }


def generate_clue_world(spec: dict) -> dict:
    from .spec import spec_issues
    from .world import WORLD_VERSION
    invalid = spec_issues(spec)
    if invalid:
        raise ValueError("invalid_clue_spec:" + ",".join(invalid))
    rng = random.Random(int(digest(
        spec["seed"], spec["task_id"], "clue-world-v1"
    )[:15], 16))
    feature = spec["investigation"]["visual_features"]
    color = feature["emblem_color"]
    shape = feature["emblem_shape"]
    wrong_color = next(value for value, _ in VISUAL_PAIRS if value != color)
    wrong_shape = next(value for _, value in VISUAL_PAIRS if value != shape)
    # Every target feature occurs in >=2 pictured items; the INTERSECTION
    # is unique. Image annotations are simulated observations.
    objects = [
        {"emblem_color": color, "emblem_shape": shape},
        {"emblem_color": color, "emblem_shape": wrong_shape},
        {"emblem_color": wrong_color, "emblem_shape": shape},
        {"emblem_color": wrong_color, "emblem_shape": wrong_shape},
    ]
    marker_numbers = rng.sample(range(10, 90), 4)
    accession_numbers = rng.sample(range(100, 900), 4)
    accession_years = rng.sample(range(1960, 2020), 4)
    illustrations, index_rows, archive_rows = [], [], []
    for ix, descriptor in enumerate(objects):
        marker = f"M-{marker_numbers[ix]:02d}"
        accession = f"AC-{accession_numbers[ix]:03d}"
        illustrations.append({
            "display_label": f"Exhibit-{ix+1:02d}",
            **descriptor,
            "marker_code": marker,
        })
        index_rows.append({
            "marker_code": marker,
            "accession_id": accession,
        })
        archive_rows.append({
            "accession_id": accession,
            "accession_year": accession_years[ix],
        })
    for batch in (illustrations, index_rows, archive_rows):
        rng.shuffle(batch)
    data = {
        "illustration": illustrations, "index": index_rows, "archive": archive_rows,
    }
    policies = {
        "illustration": {"mode": "browse"},
        "index": {"mode": "keyed_lookup", "key_field": "marker_code"},
        "archive": {"mode": "keyed_lookup", "key_field": "accession_id"},
    }
    return {
        "version": WORLD_VERSION,
        "simulated_only": True,
        "sources": {
            role["source_id"]: {
                **_rows(
                    role["source_id"], role["modality"], role["title"],
                    role["purpose"], data[role["source_id"]],
                ),
                "access_policy": policies[role["source_id"]],
            }
            for role in spec["sources"]
        },
        "entities": [
            {"hidden_entity_id": f"fictional-exhibit-{ix:03d}",
             "identity_source": "synthetic_generation_only"}
            for ix in range(4)
        ],
        "generation_seed": spec["seed"],
        "value_sampling_contract": (
            "deterministic_keyed_clue_chain_world_with_visual_decoys"
        ),
    }
