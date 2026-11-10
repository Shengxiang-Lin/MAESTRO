"""Witness-first synthetic heterogeneous World for Macro-first v2.

Synthetic media observations are typed: still images have emblem features,
videos have action/time, audio has spoken cues/time, and text/table provide
independent records. They are NOT claims about real source media.
"""
from __future__ import annotations

import copy
import random

from envir.schema import digest
from .macro_first_v2 import validate_spec


def _goal_value(field: str, kind: str, i: int, rng: random.Random):
    if kind == "int":
        return 1975 + i * 3 + rng.randint(0, 2)
    if field == "match_result":
        return f"{i + 1}-{i % 3}"
    if field in {"event_venue", "screening_venue", "performance_venue"}:
        return f"{('Riverside', 'Central', 'Eastside', 'Harbor')[i % 4]} "
        f"{('Arena', 'Hall', 'Ground', 'Stage')[i % 4]} {i + 1}"
    if field.endswith("status"):
        return f"{('confirmed', 'under review', 'revised', 'closed')[i % 4]} "
        f"case {i + 1}"
    if field in {"survey_site"}:
        return f"Survey Sector {chr(65 + i % 26)}-{i + 1}"
    if field.endswith("title"):
        return f"Record {chr(65 + i % 26)}-{i + 1}"
    if field == "song_title":
        return f"Performance Track {chr(65 + i % 26)}"
    if field == "experiment_label":
        return f"Trial {chr(65 + i % 26)}-{i + 1}"
    return f"Reference-{i + 1}"


def generate_world(spec: dict) -> dict:
    from .world import _rows, WORLD_VERSION

    problems = validate_spec(spec)
    if problems:
        raise ValueError("invalid_macro_first_v2_world:" + ",".join(problems))
    plan, clues = spec["macro_plan"], spec["macro_plan"]["clues"]
    rng = random.Random(int(digest(
        spec["seed"], spec["task_id"], "macro-first-v2-world"
    )[:15], 16))
    source_targets = {}
    for task in plan["macro_tasks"]:
        source_targets.update(
            task["evidence_plan"]["source_observation_targets"]
        )
    if set(source_targets) != set(plan["source_schemas"]):
        raise ValueError("local_evidence_plan_source_coverage_mismatch")
    entity_count = max(source_targets.values())
    colors = ("amber", "blue", "green", "red", "violet")
    motifs = ("crescent", "spiral", "star", "wave", "triangle")
    actions = ("entrance", "handoff", "turn", "signal", "departure")
    other_color = rng.choice([c for c in colors if c != clues["emblem_color"]])
    other_motif = rng.choice([c for c in motifs if c != clues["emblem_shape"]])
    other_action = rng.choice([c for c in actions if c != clues["scene_action"]])
    other_tag = "__independent_other_log_tag__"
    labels = [f"Case-{i + 1:03d}" for i in range(entity_count)]
    rng.shuffle(labels)
    marker_codes = rng.sample(range(100, 9999), entity_count)
    index_codes = [
        rng.sample(range(1000, 99999), entity_count)
        for _ in range(plan["index_hops"])
    ]
    base_years = rng.sample(range(1980, 2025), entity_count)
    objects = []
    field, kind = plan["goal"]["answer_field"], plan["goal"]["answer_type"]
    for i in range(entity_count):
        # Controlled ambiguity: target i=0 is the sole complete match.
        # i=1 matches visual cues but misses the independent cue or timing.
        # i=2 matches the second cue but misses the visual cue.
        color = clues["emblem_color"] if i in (0, 1) else other_color
        motif = clues["emblem_shape"] if i in (0, 2) else other_motif
        action = clues["scene_action"] if i in (0, 1, 3) else other_action
        timestamp = clues["time_threshold_ms"] + 10000 if i in (0, 1, 3) else 20000
        # For multimodal_event, candidate 1 has both cues but wrong ordering.
        if plan["pattern"] == "multimodal_event":
            motif = clues["emblem_shape"] if i in (0, 1, 2) else other_motif
        case_tag = clues["case_tag"] if i in (0, 1, 2) else other_tag
        item = {
            "candidate_id": labels[i], "case_id": labels[i],
            "marker_code": f"MK-{marker_codes[i]:04d}",
            "emblem_color": color, "emblem_shape": motif,
            "scene_action": action,
            "frame_time_ms": timestamp,
            "footage_time_ms": timestamp,
            "commentary_time_ms": (
                timestamp + 5000 if i != 1 else timestamp - 5000
            ),
            "spoken_cue": motif, "caption_motif": motif,
            "utterance_time_ms": timestamp + 5000,
            "case_tag": case_tag, "reported_year_a": base_years[i],
            "reported_year_b": base_years[i] + 1 if i == 0 else base_years[i],
            field: _goal_value(field, kind, i, rng),
        }
        for j, keys in enumerate(index_codes, 1):
            item[f"record_key_{j}"] = f"KEY{j}-{keys[i]:05d}"
        objects.append(item)

    kind_map = {
        "visual": ["candidate_id", "marker_code", "emblem_color"] + (
            ["emblem_shape"] if clues["visual_modality"] == "image"
            else ["scene_action", "frame_time_ms"]
        ),
        "caption": (
            ["candidate_id", "spoken_cue", "utterance_time_ms"]
            if clues["caption_modality"] == "audio"
            else ["candidate_id", "caption_motif"]
        ),
        "footage": ["candidate_id", "marker_code", "scene_action",
                    "footage_time_ms"],
        "commentary": ["candidate_id", "spoken_cue", "commentary_time_ms"],
        "context": ["candidate_id", "case_tag"],
        "report_a": ["case_id", "reported_year_a"],
        "report_b": ["case_id", "reported_year_b"],
    }
    key_field = ("case_id" if plan["pattern"] == "disputed_reports"
                 else "marker_code")
    for i in range(1, plan["index_hops"] + 1):
        next_key = f"record_key_{i}"
        kind_map[f"index_{i}"] = [key_field, next_key]
        key_field = next_key
    kind_map["archive"] = [key_field, field]
    world_sources = {}
    for role in spec["sources"]:
        sid = role["source_id"]
        target_count = source_targets[sid]
        required = list(plan["source_schemas"][sid])
        if set(required) != set(kind_map[sid]):
            raise ValueError("macro_first_v2_schema_sampler_mismatch:" + sid)
        rows = [{key: item[key] for key in required}
                for item in objects[:target_count]]
        rng.shuffle(rows)
        policy = {"mode": role["access_mode"]}
        if policy["mode"] == "keyed_lookup":
            policy["key_field"] = role["key_field"]
        world_sources[sid] = {
            **_rows(sid, role["modality"], role["title"],
                    role["purpose"], rows),
            "access_policy": policy,
        }
    return {
        "version": WORLD_VERSION, "simulated_only": True,
        "sources": world_sources,
        "entities": [
            {"hidden_entity_id": f"synthetic-{i:03d}",
             "identity_source": "generated_witness_only"}
            for i in range(entity_count)
        ],
        "generation_seed": spec["seed"],
        "value_sampling_contract": "v2_global_witness_variable_local_evidence",
    }


def mutate_witness(sources: dict, sid: str, gold_rows: list[dict],
                   spec: dict) -> dict:
    revised = copy.deepcopy(sources)
    refs = {ref for row in gold_rows for ref in row["source_fact_refs"]
            if ref.startswith(sid + ":")}
    selected = next(
        (row for row in revised[sid]["observations"]
         if refs.intersection(row["fact_refs"])), None
    )
    if selected is None:
        raise ValueError("macro_first_v2_missing_cited_witness:" + sid)
    values = selected["values"]
    if sid in {"visual", "footage"}:
        if "scene_action" in values:
            values["scene_action"] = "__other_action__"
        else:
            values["emblem_color"] = "__other_color__"
    elif sid in {"caption", "commentary"}:
        key = "spoken_cue" if "spoken_cue" in values else "caption_motif"
        values[key] = "__other_cue__"
    elif sid == "context":
        values["case_tag"] = "__other_case_tag__"
    elif sid in {"report_a", "report_b"}:
        peer_id = "report_b" if sid == "report_a" else "report_a"
        peer_key = ("reported_year_b" if sid == "report_a"
                    else "reported_year_a")
        local_key = ("reported_year_a" if sid == "report_a"
                     else "reported_year_b")
        peer = next(row["values"] for row in revised[peer_id]["observations"]
                    if row["values"]["case_id"] == values["case_id"])
        values[local_key] = peer[peer_key]
    elif sid.startswith("index_"):
        i = int(sid.split("_")[-1])
        values[f"record_key_{i}"] = "__broken_key__"
    elif sid == "archive":
        field = spec["macro_plan"]["goal"]["answer_field"]
        if spec["macro_plan"]["goal"]["answer_type"] == "int":
            values[field] += 1
        else:
            values[field] += " (revised)"
    else:
        raise ValueError("unknown_macro_first_v2_source:" + sid)
    return revised
