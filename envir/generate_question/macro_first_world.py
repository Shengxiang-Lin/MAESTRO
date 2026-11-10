"""Witness-first synthetic World for frozen Macro-first task plans.

The global consistent witness is assigned BEFORE local evidence rows; each
source is then populated with a targeted fact and misleading but plausible
decoys. The generator never reads original MMQA questions or media.
"""
from __future__ import annotations

import copy
import random

from envir.schema import digest

from .macro_first import validate_macro_first_spec, YEARS


def generate_macro_first_world(spec: dict) -> dict:
    from .world import _rows, WORLD_VERSION

    errors = validate_macro_first_spec(spec)
    if errors:
        raise ValueError("invalid_macro_first_world:" + ",".join(errors))
    plan = spec["macro_plan"]
    rng = random.Random(int(digest(
        spec["seed"], spec["task_id"], "macro-first-world-v1"
    )[:15], 16))
    pattern = plan["pattern"]
    color = (plan["clues"] or {}).get("emblem_color")
    motif = (plan["clues"] or {}).get("emblem_shape")
    colors = ("crimson", "azure", "amber", "emerald", "violet")
    motifs = ("crescent", "spiral", "trident", "sunburst", "twin-wing")
    alternative_color = rng.choice([v for v in colors if v != color])
    alternative_motif = rng.choice([v for v in motifs if v != motif])
    features = [
        (color, motif), (color, alternative_motif),
        (alternative_color, motif), (alternative_color, alternative_motif),
    ]
    candidate_ids = [f"Reference-{i:02d}" for i in range(1, 5)]
    rng.shuffle(candidate_ids)
    marker_values = rng.sample(range(110, 990), 4)
    dates = rng.sample(range(1960, 2025), 4)
    record_keys = [
        rng.sample(range(1000, 9999), 4)
        for _ in range(plan["index_hops"])
    ]
    objects = []
    for j, (c, m) in enumerate(features):
        row = {
            "candidate_id": candidate_ids[j],
            "case_id": candidate_ids[j],
            "emblem_color": c, "emblem_shape": m,
            "marker_code": f"MK-{marker_values[j]:03d}",
            plan["goal"]["answer_field"]: dates[j],
        }
        for i, keys in enumerate(record_keys, 1):
            row[f"record_key_{i}"] = f"RK{i}-{keys[j]:04d}"
        objects.append(row)

    data = {}
    if pattern == "serial_discovery":
        data["visual"] = [{
            k: item[k] for k in
            ("candidate_id", "emblem_color", "emblem_shape", "marker_code")
        } for item in objects]
    elif pattern == "parallel_resolution":
        data["visual"] = [{
            k: item[k] for k in
            ("candidate_id", "emblem_color", "marker_code")
        } for item in objects]
        data["caption"] = [{
            k: item[k] for k in ("candidate_id", "emblem_shape")
        } for item in objects]
    elif pattern == "conflict_reconciliation":
        data["report_a"] = [{
            "case_id": item["case_id"],
            "reported_year_a": item[plan["goal"]["answer_field"]] - 3,
        } for item in objects]
        data["report_b"] = [{
            "case_id": item["case_id"],
            "reported_year_b": (
                item[plan["goal"]["answer_field"]] - 2
                if j == 0 else item[plan["goal"]["answer_field"]] - 3
            ),
        } for j, item in enumerate(objects)]
    else:
        raise ValueError("unknown_macro_first_pattern:" + pattern)

    current_key = ("case_id" if pattern == "conflict_reconciliation"
                   else "marker_code")
    for i in range(1, plan["index_hops"] + 1):
        following = f"record_key_{i}"
        sid = f"index_{i}"
        data[sid] = [{
            current_key: row[current_key],
            following: row[following],
        } for row in objects]
        current_key = following
    answer_field = plan["goal"]["answer_field"]
    data["archive"] = [{
        current_key: row[current_key],
        answer_field: row[answer_field],
    } for row in objects]

    world_sources = {}
    for role in spec["sources"]:
        sid = role["source_id"]
        rows = data[sid]
        rng.shuffle(rows)
        if set(rows[0]) != set(plan["source_schemas"][sid]):
            raise ValueError("macro_first_source_schema_drift:" + sid)
        access = {"mode": role["access_mode"]}
        if access["mode"] == "keyed_lookup":
            access["key_field"] = role["key_field"]
        world_sources[sid] = {
            **_rows(sid, role["modality"], role["title"],
                    role["purpose"], rows),
            "access_policy": access,
        }
    return {
        "version": WORLD_VERSION,
        "simulated_only": True,
        "sources": world_sources,
        "entities": [{
            "hidden_entity_id": f"fictional-reference-{i:03d}",
            "identity_source": "synthetic_generation_only",
        } for i in range(4)],
        "generation_seed": spec["seed"],
        "value_sampling_contract": (
            "macro_plan_witness_first_consistent_handoffs_with_decoys"
        ),
    }


def mutate_macro_first_witness(sources: dict, source_id: str,
                               gold_rows: list[dict], spec: dict) -> dict:
    """Intervene on the *cited* witness fact, never on the first source row."""
    revised = copy.deepcopy(sources)
    refs = {ref for row in gold_rows for ref in row["source_fact_refs"]
            if ref.startswith(source_id + ":")}
    selected = next(
        (row for row in revised[source_id]["observations"]
         if refs.intersection(row["fact_refs"])), None
    )
    if selected is None:
        raise ValueError("macro_first_source_has_no_gold_witness:" + source_id)
    values = selected["values"]
    if source_id == "visual":
        values["emblem_color"] = "__missing_target_color__"
    elif source_id == "caption":
        values["emblem_shape"] = "__missing_target_motif__"
    elif source_id in {"report_a", "report_b"}:
        partner_id = "report_b" if source_id == "report_a" else "report_a"
        key = "reported_year_a" if source_id == "report_a" else "reported_year_b"
        partner_key = (
            "reported_year_b" if source_id == "report_a" else "reported_year_a"
        )
        partner = next(
            row["values"] for row in revised[partner_id]["observations"]
            if row["values"]["case_id"] == values["case_id"]
        )
        values[key] = partner[partner_key]
    elif source_id.startswith("index_"):
        index = int(source_id.split("_")[1])
        values[f"record_key_{index}"] = "__unresolvable_record_key__"
    elif source_id == "archive":
        values[spec["macro_plan"]["goal"]["answer_field"]] += 1
    else:
        raise ValueError("unrecognized_macro_first_source:" + source_id)
    return revised
