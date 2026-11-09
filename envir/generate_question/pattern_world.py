"""Executable semantic *alternatives* to the audit/approval relation grammar.

These patterns require different cross-source predicates:
  time_window         — per-entry time lies inside source-specific interval
  status_disagreement — conflicting binary reports for the same resolved group
  dynamic_threshold   — per-entry measure meets another source's group threshold

All media information is a synthetic structured observation. The modality-
specific fields participate in the Gold Program; no original media is read.
"""
from __future__ import annotations

import random

from envir.schema import digest


def modality_fields(modality: str, pattern: str):
    """Name functional observation fields after their simulated media role."""
    if pattern == "time_window":
        return {
            "video": ("clip_step_start", "clip_step_end"),
            "audio": ("utterance_step_start", "utterance_step_end"),
            "image": ("diagram_step_start", "diagram_step_end"),
        }.get(modality, ("interval_step_start", "interval_step_end"))
    if pattern == "status_disagreement":
        return {
            "video": "clip_reported_status",
            "audio": "utterance_reported_status",
            "image": "visual_marker_status",
        }.get(modality, "observed_status")
    if pattern == "dynamic_threshold":
        return {
            "video": "clip_threshold",
            "audio": "spoken_threshold",
            "image": "visual_threshold",
        }.get(modality, "minimum_required")
    raise ValueError("unknown_pattern:" + pattern)


def _vary_pattern_core(nodes: list[dict], pattern: str,
                       variant: int) -> tuple[list[dict], str | None]:
    """Re-plan *equivalent* relational tasks with real dependency changes.

    This only applies to newly generated specs with macro_plan_variant.
    Historical v4 specs lacking that field reproduce their original graph.
    The plans move eligible filters across inner joins or vary which
    two sources reconcile first; all conditions and Gold values remain
    unchanged, but task parallelism/dependency shape can differ.
    """
    if variant == 0:
        return nodes, None
    by_id = {node["id"]: node for node in nodes}
    final_override = None

    def set_inputs(name, *inputs, **params):
        node = by_id[name]
        node["inputs"] = list(inputs)
        node.update(params)

    if pattern == "time_window":
        if variant == 1:
            set_inputs("mapped_events", "register_rows", "codebook_rows")
            set_inputs("active_events", "mapped_events")
            set_inputs("events_with_windows", "active_events", "window_rows")
        elif variant == 2:
            set_inputs("mapped_events", "register_rows", "codebook_rows")
            set_inputs("events_with_windows", "mapped_events", "window_rows")
            set_inputs("active_events", "events_with_windows")
            set_inputs("after_window_start", "active_events")
        else:
            set_inputs("mapped_events", "codebook_rows", "window_rows",
                       left_on="group_id", right_on="group_id")
            set_inputs("events_with_windows", "active_events", "mapped_events",
                       left_on="route_code", right_on="route_code")
    elif pattern == "status_disagreement":
        if variant in (1, 2):
            set_inputs("observation_codes", "register_rows", "codebook_rows",
                       left_on="route_code", right_on="route_code")
            if variant == 1:
                set_inputs("linked_reports", "observation_codes", "observations",
                           left_on="group_id", right_on="group_id")
                set_inputs("active_reports", "linked_reports")
                set_inputs("conflicting_reports", "active_reports")
            else:
                set_inputs("active_reports", "observation_codes")
                set_inputs("linked_reports", "active_reports", "observations",
                           left_on="group_id", right_on="group_id")
        else:
            set_inputs("linked_reports", "register_rows", "observation_codes")
            set_inputs("active_reports", "linked_reports")
            set_inputs("conflicting_reports", "active_reports")
    elif pattern == "dynamic_threshold":
        if variant in (1, 2):
            if variant == 1:
                set_inputs("mapped_entries", "codebook_rows", "policy_rows",
                           left_on="group_id", right_on="group_id")
                set_inputs("active_entries", "register_rows")
                set_inputs("with_thresholds", "active_entries", "mapped_entries",
                           left_on="route_code", right_on="route_code")
                set_inputs("qualified_by_threshold", "with_thresholds")
            else:
                set_inputs("active_entries", "register_rows")
                set_inputs("mapped_entries", "active_entries", "codebook_rows")
                set_inputs("qualified_by_threshold", "with_thresholds")
        else:
            set_inputs("qualified_by_threshold", "with_thresholds")
            set_inputs("active_entries", "qualified_by_threshold")
            final_override = "active_entries"
    else:
        raise ValueError("unknown_pattern_for_plan:" + pattern)

    # Canonical topological sort; changing scan/linear placement without
    # dependencies alone would NOT create a new graph.
    remaining = list(nodes)
    ordered, seen = [], set()
    while remaining:
        ready = next((node for node in remaining
                      if all(parent in seen for parent in node["inputs"])), None)
        if ready is None:
            raise ValueError("non_topological_macro_plan_variant")
        ordered.append(ready)
        seen.add(ready["id"])
        remaining.remove(ready)
    return ordered, final_override


def compile_pattern_program(spec: dict) -> dict:
    from .world import _node
    pattern = spec["reasoning_pattern"]
    difficulty = spec["difficulty_target"]
    modality = next(x["modality"] for x in spec["sources"]
                    if x["source_id"] == "annotations")
    if pattern == "time_window":
        nodes = [
            _node("register_rows", "scan", source_id="register"),
            _node("active_events", "filter_eq", "register_rows",
                  field="active", value=True),
            _node("codebook_rows", "scan", source_id="glossary"),
            _node("mapped_events", "join", "active_events", "codebook_rows",
                  left_on="route_code", right_on="route_code"),
            _node("window_rows", "scan", source_id="annotations"),
            _node("events_with_windows", "join",
                  "mapped_events", "window_rows",
                  left_on="group_id", right_on="group_id"),
        ]
        start, end = modality_fields(modality, pattern)
        nodes += [
            _node("after_window_start", "compare_fields",
                  "events_with_windows", left_field="event_step",
                  right_field=start, comparison="gte"),
            _node("inside_window", "compare_fields", "after_window_start",
                  left_field="event_step", right_field=end,
                  comparison="lte"),
        ]
        final = "inside_window"
    elif pattern == "status_disagreement":
        observed = modality_fields(modality, pattern)
        nodes = [
            _node("observations", "scan", source_id="annotations"),
            _node("codebook_rows", "scan", source_id="glossary"),
            _node("observation_codes", "join",
                  "codebook_rows", "observations",
                  left_on="group_id", right_on="group_id"),
            _node("register_rows", "scan", source_id="register"),
            _node("active_reports", "filter_eq", "register_rows",
                  field="active", value=True),
            _node("linked_reports", "join", "active_reports",
                  "observation_codes",
                  left_on="route_code", right_on="route_code"),
            _node("conflicting_reports", "compare_fields",
                  "linked_reports", left_field="reported_status",
                  right_field=observed, comparison="ne"),
        ]
        final = "conflicting_reports"
    elif pattern == "dynamic_threshold":
        threshold = modality_fields(modality, pattern)
        nodes = [
            _node("register_rows", "scan", source_id="register"),
            _node("codebook_rows", "scan", source_id="glossary"),
            _node("mapped_entries", "join", "register_rows", "codebook_rows",
                  left_on="route_code", right_on="route_code"),
            _node("policy_rows", "scan", source_id="annotations"),
            _node("with_thresholds", "join", "mapped_entries", "policy_rows",
                  left_on="group_id", right_on="group_id"),
            _node("active_entries", "filter_eq", "with_thresholds",
                  field="active", value=True),
            _node("qualified_by_threshold", "compare_fields",
                  "active_entries", left_field="measured_value",
                  right_field=threshold, comparison="gte"),
        ]
        final = "qualified_by_threshold"
    else:
        raise ValueError("unsupported_semantic_pattern:" + pattern)

    plan_variant = spec.get("macro_plan_variant", 0)
    if plan_variant:
        nodes, final_override = _vary_pattern_core(nodes, pattern, plan_variant)
        if final_override:
            final = final_override

    if difficulty == "hard":
        nodes += [
            _node("clearance_rows", "scan", source_id="clearance"),
            _node("individually_cleared", "filter_eq", "clearance_rows",
                  field="cleared", value=True),
            _node("cleared_candidates", "join",
                  final, "individually_cleared",
                  left_on="entry_label", right_on="entry_label"),
        ]
        final = "cleared_candidates"

    nodes += [
        _node("answer_entries", "project", final, fields=["entry_label"]),
        _node("answer_unique", "unique", "answer_entries"),
    ]
    return {
        "version": "maestro.reasoning_program.v4",
        "nodes": nodes,
        "output": "answer_unique",
        "authorized_source_ids": [
            x["source_id"] for x in spec["sources"]
        ],
        "frozen_problem_spec_id": spec["task_id"],
        "semantic_operator_pattern": pattern,
        **({"macro_plan_variant": plan_variant}
           if "macro_plan_variant" in spec else {}),
    }


def generate_pattern_world(spec: dict) -> dict:
    """Bounded exact relational facts, generated after a frozen Program."""
    from .world import _rows
    pattern = spec["reasoning_pattern"]
    rng = random.Random(int(digest(spec["seed"], spec["task_id"],
                                   "semantic-pattern-world-v4")[:15], 16))
    group_count = rng.randint(3, 5)
    entry_count = rng.randint(8, 12)
    groups = [f"Collection group {chr(65+i)}"
              for i in range(group_count)]
    codes = [f"R-{i+1:02d}" for i in range(group_count)]
    assigned = list(range(group_count)) + [
        rng.randrange(group_count) for _ in range(entry_count - group_count)
    ]
    rng.shuffle(assigned)
    anchor = rng.randrange(entry_count)
    active = [rng.random() < 0.8 for _ in range(entry_count)]
    active[anchor] = True
    modality = next(x["modality"] for x in spec["sources"]
                    if x["source_id"] == "annotations")

    glossary = [{
        "route_code": codes[i], "group_id": groups[i],
    } for i in range(group_count)]
    media_fields = modality_fields(modality, pattern)
    observations = []
    register = []
    for group_id in range(group_count):
        values = {"group_id": groups[group_id]}
        if pattern == "time_window":
            start, end = media_fields
            lower = rng.randint(10, 65)
            values.update({start: lower, end: lower + rng.randint(12, 30)})
        elif pattern == "status_disagreement":
            values[media_fields] = rng.random() < 0.5
        else:
            values[media_fields] = rng.randint(35, 75)
        observations.append(values)
    for index, group in enumerate(assigned):
        values = {
            "entry_label": f"Entry-{index+1:02d}",
            "route_code": codes[group],
            "active": active[index],
        }
        evidence = observations[group]
        if pattern == "time_window":
            start, end = media_fields
            values["event_step"] = rng.randint(0, 105)
            if index == anchor:
                values["event_step"] = rng.randint(
                    evidence[start], evidence[end])
        elif pattern == "status_disagreement":
            values["reported_status"] = rng.random() < 0.5
            if index == anchor:
                values["reported_status"] = not evidence[media_fields]
        elif pattern == "dynamic_threshold":
            values["measured_value"] = rng.randint(10, 100)
            if index == anchor:
                values["measured_value"] = evidence[media_fields] + 2
        register.append(values)

    # Force an active hard negative and ensure its local evidence disagrees
    # with the requested operation; it fools register-only shortcuts.
    decoys = [i for i in range(entry_count) if i != anchor]
    decoy = rng.choice(decoys)
    register[decoy]["active"] = True
    evidence = observations[assigned[decoy]]
    if pattern == "time_window":
        register[decoy]["event_step"] = 1000
    elif pattern == "status_disagreement":
        register[decoy]["reported_status"] = evidence[media_fields]
    else:
        register[decoy]["measured_value"] = 0

    clearance = [{
        "entry_label": f"Entry-{i+1:02d}",
        "cleared": (i == anchor or rng.random() < 0.35),
    } for i in range(entry_count)]
    records = {
        "register": register,
        "glossary": glossary,
        "annotations": observations,
        "clearance": clearance,
    }
    for items in records.values():
        rng.shuffle(items)
    sources = {
        role["source_id"]: _rows(
            role["source_id"], role["modality"], role["title"],
            role["purpose"], records[role["source_id"]]
        ) for role in spec["sources"]
    }
    entities = [{
        "hidden_entity_id": f"fictional-{i+1:03d}",
        "entry_label": f"Entry-{i+1:02d}",
        "group_id": groups[assigned[i]],
        "identity_source": "fictional_sampling_only",
    } for i in range(entry_count)]
    return {
        "version": "maestro.source_observation_world.v4",
        "simulated_only": True,
        "sources": sources,
        "entities": entities,
        "generation_seed": spec["seed"],
        "value_sampling_contract": "deterministic_pattern_specific_source_facts",
        "media_observation_fields_used": [
            media_fields] if isinstance(media_fields, str) else list(media_fields),
    }
