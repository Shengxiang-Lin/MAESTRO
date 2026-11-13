"""Witness-first synthetic heterogeneous World for Macro-first v2.

Synthetic media observations are typed: still images have emblem features,
videos have action/time, audio has spoken cues/time, and text/table provide
independent records. They are NOT claims about real source media.
"""
from __future__ import annotations

import copy
import random

from environment.schema import digest
from .macro_planning import validate_spec


def _goal_value(field: str, kind: str, i: int, rng: random.Random):
    """Generate plausible within-workspace values, not real-world claims."""
    if kind == "int":
        return rng.randint(1973, 2025)
    if field == "match_result":
        return f"{rng.randint(0, 4)}-{rng.randint(0, 4)}"
    places = ("Riverside", "Central", "North", "Harbor", "Union",
              "Eastside", "Westport", "Garden", "Civic")
    venue_types = ("Hall", "Arena", "Ground", "Auditorium", "Stage")
    if field in {
        "event_venue", "screening_venue", "performance_venue",
        "arrival_site", "campus_site", "civic_site",
    }:
        return f"{rng.choice(places)} {rng.choice(venue_types)}"
    if field in {
        "review_status", "resolution_status", "permit_status",
        "delivery_status",
    }:
        return rng.choice((
            "confirmed", "under review", "resolved", "withdrawn", "revised"
        ))
    if field == "survey_site":
        return f"{rng.choice(places)} survey sector {rng.randint(1, 12)}"
    adjectives = ("Northern", "Hidden", "Second", "Distant", "Last",
                  "Silver", "Quiet", "Crossing", "Open", "Early")
    nouns = ("Signal", "Window", "Crossing", "Departure", "Echo",
             "Record", "Return", "Journey", "Interval", "Horizon")
    title = f"{rng.choice(adjectives)} {rng.choice(nouns)}"
    if field in {"song_title", "exhibition_title", "course_title"}:
        return title
    if field == "episode_title":
        return "The " + title
    if field == "edition_title":
        return f"{title}, {rng.choice(('revised', 'annotated', 'expanded'))} edition"
    if field == "artifact_title":
        return f"{title} exhibit"
    if field == "case_title":
        return f"{title} review file"
    if field == "species_label":
        return rng.choice(("river willow", "red-backed finch",
                           "marsh orchid", "silver birch", "field lark"))
    if field == "experiment_label":
        return f"{rng.choice(('Phase', 'Trial', 'Protocol'))} {rng.choice(adjectives)}-{i+1:02d}"
    return title

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
    variable_profile = (
        plan.get("candidate_profile")
        if spec.get("scene_realism_version")
        and plan["pattern"] in {
            "multimodal_event", "corroborated_dispute",
        } else None
    )
    if variable_profile and any(
        target < variable_profile["union_candidate_count"]
        for target in source_targets.values()
    ):
        raise ValueError("branch_candidates_not_materialized_in_all_sources")
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
        action = clues["scene_action"] if i in (0, 1) else other_action
        timestamp = clues["time_threshold_ms"] + 10000 if i in (0, 1) else 20000
        if (plan["pattern"] == "serial_identity"
                and clues["visual_modality"] == "video" and i == 1):
            timestamp = clues["time_threshold_ms"] - 10000
        if plan["pattern"] == "multimodal_event" and i == 3:
            action = clues["scene_action"]
            timestamp = clues["time_threshold_ms"] + 10000
        # For multimodal_event, candidate 1 has both cues but wrong ordering.
        if plan["pattern"] == "multimodal_event":
            motif = clues["emblem_shape"] if i in (0, 1, 2) else other_motif
        information_necessary = bool(
            spec.get("macro_information_necessity_version")
        )
        if information_necessary and plan["pattern"] == "multimodal_event":
            # Both branches must meaningfully reduce *different* candidate
            # sets. Video+audio gives {target, candidate-1}; the event log
            # gives {target, candidate-2}. Neither is unique alone.
            case_tag = clues["case_tag"] if i in (0, 2) else other_tag
        else:
            case_tag = clues["case_tag"] if i in (0, 1, 2) else other_tag
        if variable_profile and plan["pattern"] == "corroborated_dispute":
            left_count = variable_profile["left_candidate_count"]
            union_count = variable_profile["union_candidate_count"]
            disputed = i < left_count
            log_match = i == 0 or left_count <= i < union_count
            case_tag = clues["case_tag"] if log_match else other_tag
        if variable_profile and plan["pattern"] == "multimodal_event":
            left_count = variable_profile["left_candidate_count"]
            union_count = variable_profile["union_candidate_count"]
            left_member = i < left_count
            right_member = i == 0 or left_count <= i < union_count
            # The video/audio branch keeps the target plus left decoys;
            # the independent event log keeps the target plus right
            # decoys. Both are non-singleton; only the join is unique.
            action = clues["scene_action"] if left_member else other_action
            motif = clues["emblem_shape"] if left_member else other_motif
            timestamp = (
                clues["time_threshold_ms"] + 10000
                if left_member else 20000
            )
            case_tag = clues["case_tag"] if right_member else other_tag
        item = {
            "candidate_id": labels[i], "case_id": labels[i],
            "marker_code": f"MK-{marker_codes[i]:04d}",
            "emblem_color": color, "emblem_shape": motif,
            "scene_action": action,
            "frame_time_ms": timestamp,
            "footage_time_ms": timestamp,
            "commentary_time_ms": (
                timestamp + 5000 if variable_profile else
                timestamp + 5000 if (i != 1 or (
                    information_necessary and
                    plan["pattern"] == "multimodal_event"
                )) else timestamp - 5000
            ),
            "spoken_cue": motif, "caption_motif": motif,
            "utterance_time_ms": timestamp + 5000,
            "case_tag": case_tag, "reported_year_a": base_years[i],
            "reported_year_b": base_years[i] + 1 if (
                (variable_profile
                 and plan["pattern"] == "corroborated_dispute"
                 and i < variable_profile["left_candidate_count"])
                or i == 0
            ) else base_years[i],
            field: _goal_value(field, kind, i, rng),
            # Each branch is a separate synthetic observation. The target
            # and alternate identities imply different final outcomes.
            "metric_a": 90 if i == 0 else 20 + i,
            "metric_b": 25 if i == 0 else 90 + i,
            "completed_count": 36 if i == 0 else 10 + i,
            "total_count": 48 if i == 0 else 60 + i,
        }
        for j, keys in enumerate(index_codes, 1):
            item[f"record_key_{j}"] = f"KEY{j}-{keys[i]:05d}"
        objects.append(item)

    if spec.get("numeric_goal_world_version"):
        # Separate PRNG stream: only the two numeric source payloads change.
        # Older frozen worlds without this marker preserve 90/25, 36/48.
        from .numeric_goal_world import generate_numeric_goal_rows
        sampled = generate_numeric_goal_rows(
            spec, entity_count=entity_count,
        )
        for row, numbers in zip(objects, sampled):
            row.update(numbers)

    if variable_profile:
        # Every candidate that survives either branch must lead to a
        # different final answer from the target; alternative candidates
        # may share a value (e.g. one of five possible statuses).
        target_value = objects[0][field]
        for index, row in enumerate(
            objects[1:variable_profile["union_candidate_count"]], 1
        ):
            for attempt in range(60):
                if row[field] != target_value:
                    break
                row[field] = _goal_value(
                    field, kind, entity_count + index + attempt, rng
                )
            if row[field] == target_value:
                raise ValueError("alternative_has_same_goal_as_target")
    elif (spec.get("macro_information_necessity_version")
            and plan["pattern"] == "multimodal_event"):
        # Information necessity should matter to the ANSWER, not merely
        # to a candidate label. The target and competing candidates must
        # resolve to different final values (statuses, dates, titles, etc.).
        values_seen = set()
        for index, row in enumerate(objects[:3]):
            for attempt in range(40):
                if row[field] not in values_seen:
                    break
                row[field] = _goal_value(
                    field, kind, entity_count + index + attempt, rng
                )
            if row[field] in values_seen:
                raise ValueError("nonunique_competing_goal_values")
            values_seen.add(row[field])

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
        "context": (
            ["case_id", "case_tag"]
            if plan["pattern"] == "corroborated_dispute"
            else ["candidate_id", "case_tag"]
        ),
        "report_a": ["case_id", "reported_year_a"],
        "report_b": ["case_id", "reported_year_b"],
    }
    key_field = (
        "case_id" if plan["pattern"] in {
            "disputed_reports", "corroborated_dispute",
        } else "marker_code"
    )
    for i in range(1, plan["index_hops"] + 1):
        next_key = f"record_key_{i}"
        kind_map[f"index_{i}"] = [key_field, next_key]
        key_field = next_key
    goal_kind = plan.get("goal_resolution", {}).get("kind", "direct_lookup")
    if goal_kind == "direct_lookup":
        kind_map["archive"] = [key_field, field]
    else:
        left_name, right_name = (
            ("metric_a", "metric_b") if goal_kind == "comparison"
            else ("completed_count", "total_count")
        )
        kind_map["stat_a"] = [key_field, left_name]
        kind_map["stat_b"] = [key_field, right_name]
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
            # This is authored World provenance, not inferred independence
            # from two Sub-Agent reports mentioning the same original record.
            **({"evidence_origin_id": "independent_stream:" + sid}
               if spec.get("evidence_relation_version")
               and sid in {"report_a", "report_b"} else {}),
            # A Direct-Extract URL is known before delegation; URLs discovered
            # through Search -> Extract remain private until the simulated
            # search observation returns them.
            **({"resource_url": role["simulated_page_url"]}
               if role.get("acquisition_route") == "direct_extract" else {}),
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
    elif sid in {"stat_a", "stat_b"}:
        plan = spec["macro_plan"]
        kind = plan["goal_resolution"]["kind"]
        # The correlation key is the unique shared field of the two
        # independent statistic source schemas. Never infer it from dict
        # insertion order: with an index hop, metric_b sorts before
        # record_key_1/2 in the frozen JSON Output Schema.
        left_schema = plan["source_schemas"]["stat_a"]
        right_schema = plan["source_schemas"]["stat_b"]
        common_fields = set(left_schema) & set(right_schema)
        if (len(common_fields) != 1 or any(
            left_schema[field] != right_schema[field]
            for field in common_fields
        )):
            raise ValueError("statistic_source_key_contract_ambiguous")
        key_field = next(iter(common_fields))
        peer_id = "stat_b" if sid == "stat_a" else "stat_a"
        peer = next(
            row["values"] for row in revised[peer_id]["observations"]
            if row["values"][key_field] == values[key_field]
        )
        if kind == "comparison":
            left = values if sid == "stat_a" else peer
            right = values if sid == "stat_b" else peer
            if sid == "stat_a":
                values["metric_a"] = (
                    right["metric_b"] - 1000
                    if values["metric_a"] > right["metric_b"]
                    else right["metric_b"] + 1000
                )
            else:
                values["metric_b"] = (
                    left["metric_a"] - 1000
                    if values["metric_b"] > left["metric_a"]
                    else left["metric_a"] + 1000
                )
        elif kind == "calculation":
            if sid == "stat_a":
                values["completed_count"] = 0
            else:
                values["total_count"] = peer["completed_count"]
        else:
            raise ValueError("unknown_mutation_goal_kind")
    else:
        raise ValueError("unknown_macro_first_v2_source:" + sid)
    return revised
