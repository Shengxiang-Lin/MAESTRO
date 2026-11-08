"""Replay simulated media and evidence dependencies, without real pixels."""
from __future__ import annotations

import re
from collections import defaultdict


def extract_reference(media: dict) -> str | None:
    modality = media["modality"]
    values = []
    for obs in media["observations"]:
        if modality == "audio":
            if obs.get("kind") == "speech" and obs.get("speaker") == "announcer":
                values.extend(re.findall(r"MX-\d{5}", obs.get("value", "")))
        elif obs.get("kind") == "visible_identifier" and obs.get("subject") == "target":
            values.append(obs.get("value"))
    # An audio announcer may repeat the same reference in later segments.
    unique = {v for v in values if v}
    return next(iter(unique)) if len(unique) == 1 else None


def extract_assignment(source: dict, reference: str) -> str | None:
    matches = []
    for obs in source["observations"]:
        if obs["kind"] == "table_row" and obs.get("reference") == reference:
            matches.append(obs["assignment"])
        elif obs["kind"] == "written_record":
            match = re.fullmatch(
                r"Reference (MX-\d{5}) is assigned to (.+)\.",
                obs.get("value", ""),
            )
            if match and match.group(1) == reference:
                matches.append(match.group(2))
    return matches[0] if len(matches) == 1 else None


def extract_registry(source: dict, reference: str) -> dict | None:
    matches = [
        {"status": obs["registration_status"],
         "location": obs["registered_location"]}
        for obs in source["observations"]
        if obs["kind"] == "registry_record" and obs.get("reference") == reference
    ]
    return matches[0] if len(matches) == 1 else None


def derive_answer(scene: dict, *, hidden_sources: set[str] | None = None):
    """Compute output from public observations, not stored Gold answer."""
    hidden_sources = hidden_sources or set()
    by_id = {s["source_id"]: s for s in scene["sources"]
             if s["source_id"] not in hidden_sources}
    sid = scene["scene_id"]
    media = by_id.get(f"{sid}:media")
    registry = by_id.get(f"{sid}:registry")
    if not media or not registry:
        return None
    code = extract_reference(media)
    if not code:
        return None
    reg = extract_registry(registry, code)
    if reg is None:
        return None
    if scene["graph_pattern"] == "chain":
        return reg
    schedule = by_id.get(f"{sid}:schedule")
    if schedule is None:
        return None
    assignment = extract_assignment(schedule, code)
    if assignment is None:
        return None
    if scene["graph_pattern"] == "cross_check":
        return {"agree": assignment == reg["location"],
                "schedule": assignment, "registry": reg["location"]}
    return {"assignment": assignment, "registration": reg["status"]}


def validate_scene(scene: dict) -> list[str]:
    issues = []
    by_id = {}
    for source in scene.get("sources", []):
        if source.get("source_id") in by_id:
            issues.append("duplicate_source_id")
        by_id[source.get("source_id")] = source
        if source.get("synthetic") is not True or source.get("asset_path") is not None:
            issues.append("source_not_declared_synthetic")
        ids = [o["observation_id"] for o in source["observations"]]
        if len(ids) != len(set(ids)):
            issues.append("duplicate_observation_id")
    required = scene.get("required_source_ids", [])
    if not required or not set(required) <= set(by_id):
        issues.append("missing_required_source")
    if set(required) & set(scene.get("optional_source_ids", [])):
        issues.append("distractor_is_required")
    answer = derive_answer(scene)
    if answer is None or answer != scene.get("gold_answer"):
        issues.append("answer_source_replay_mismatch")
    if any(derive_answer(scene, hidden_sources={src}) is not None for src in required):
        issues.append("required_source_not_indispensable")
    world = scene.get("world", {})
    media = by_id.get(f"{scene['scene_id']}:media")
    if media and extract_reference(media) != world.get("target_reference"):
        issues.append("world_media_mismatch")
    schedule = by_id.get(f"{scene['scene_id']}:schedule")
    if schedule and extract_assignment(
        schedule, world.get("target_reference", "")
    ) != world.get("schedule_assignment"):
        issues.append("world_schedule_mismatch")
    registry = by_id.get(f"{scene['scene_id']}:registry")
    if registry:
        rec = extract_registry(registry, world.get("target_reference", ""))
        if rec != {"status": world.get("registry_status"),
                   "location": world.get("registry_assignment")}:
            issues.append("world_registry_mismatch")
    graph = scene.get("task_graph", {})
    nodes = graph.get("nodes", [])
    ids = [n["subtask_id"] for n in nodes]
    if len(ids) != len(set(ids)) or graph.get("final_node") not in ids:
        issues.append("invalid_graph_nodes")
    edges = graph.get("edges", [])
    if any(e.get("from") not in ids or e.get("to") not in ids
           for e in edges):
        issues.append("invalid_graph_edges")
    incoming = defaultdict(list)
    for edge in edges:
        incoming[edge["to"]].append(edge["from"])
    seen, active = set(), set()

    def visit(node):
        if node in active:
            return False
        if node in seen:
            return True
        active.add(node)
        if any(not visit(p) for p in incoming[node]):
            return False
        active.remove(node)
        seen.add(node)
        return True

    if any(not visit(n) for n in ids):
        issues.append("cyclic_task_graph")
    required_set = set(graph.get("required_subtasks", []))
    if required_set != set(ids):
        issues.append("unused_required_subtask")
    # Compare every teacher node to a re-materialization from source content.
    # The replay function must not trust stored answer or summary_trace fields.
    try:
        from .generator import materialize_graph
        from .taxonomy import BY_DOMAIN
        rebuilt = materialize_graph(scene, BY_DOMAIN[scene["domain"]])
        if rebuilt["task_graph"] != graph:
            issues.append("subtask_feedback_or_binding_drift")
    except (KeyError, StopIteration, ValueError, TypeError):
        issues.append("subtask_replay_error")
    if not scene.get("question") or not scene["question_brief"].get(
        "private_answers_excluded"
    ):
        issues.append("question_brief_missing")
    if world.get("target_reference") in scene.get("question", ""):
        issues.append("private_identifier_leaked_to_question")
    if scene.get("verification", {}).get("real_media_perception") is not False:
        issues.append("unjustified_real_media_claim")
    variants = scene.get("feedback_variants") or {}
    if variants.get("clean", {}).get("reported_reference") != world.get("target_reference"):
        issues.append("clean_feedback_not_supported")
    if variants.get("misleading", {}).get("reported_reference") == world.get("target_reference"):
        issues.append("misleading_feedback_not_corrupted")
    return sorted(set(issues))
