"""MMQA-motif-guided scenario generation for simulated orchestration.

No image/video/audio bytes or actual perception is required. The virtual
sources have rich, structured observations; tools reveal them on demand.
All answers and summary traces are derived from the same teacher-side world.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict

from envir.schema import digest
from .taxonomy import BY_DOMAIN, PACKS, ScenePack


def _other_code(code: str) -> str:
    return code[:-1] + str((int(code[-1]) + 3) % 10)


def _media_observations(modality: str, pack: ScenePack, code: str,
                        other_code: str, scene_type: str, setting: str) -> list[dict]:
    base = [
        {"observation_id": "o1", "kind": "entity", "subject": "target",
         "value": pack.asset, "region": "center"},
        {"observation_id": "o2", "kind": "visible_identifier",
         "subject": "target", "value": code, "region": "upper-right label",
         "certainty": "clear"},
        {"observation_id": "o3", "kind": "distractor_identifier",
         "subject": "background", "value": other_code, "region": "left background"},
        {"observation_id": "o4", "kind": "spatial_relation",
         "subject": "target", "value": f"near the {pack.location} sign",
         "region": "foreground"},
        {"observation_id": "o5", "kind": "scene_context",
         "value": f"{pack.setting} / {scene_type}", "region": "wide view"},
    ]
    if modality == "image":
        return base + [
            {"observation_id": "o6", "kind": "visual_ambiguity",
             "value": "The background object's reference label is not the target label",
             "region": "left background"},
        ]
    if modality == "video":
        return [
            {"observation_id": "o1", "timestamp": "00:02", "kind": "scene_context",
             "value": f"Wide view of {setting}"},
            {"observation_id": "o2", "timestamp": "00:05", "kind": "entity",
             "subject": "target", "value": pack.asset},
            {"observation_id": "o3", "timestamp": "00:08", "kind": "visible_identifier",
             "subject": "target", "value": code, "certainty": "clear"},
            {"observation_id": "o4", "timestamp": "00:11",
             "kind": "distractor_identifier", "value": other_code,
             "subject": "background"},
            {"observation_id": "o5", "timestamp": "00:16",
             "kind": "spatial_relation", "subject": "target",
             "value": f"moves past the {pack.location} sign"},
        ]
    if modality == "audio":
        return [
            {"observation_id": "o1", "timestamp": "00:00", "kind": "acoustic_context",
             "value": f"Public address recording at {setting}", "channel": "ambient"},
            {"observation_id": "o2", "timestamp": "00:06", "kind": "speech",
             "speaker": "announcer", "value": f"Attention: {pack.asset} {code}",
             "channel": "foreground"},
            {"observation_id": "o3", "timestamp": "00:10", "kind": "speech",
             "speaker": "background radio",
             "value": f"Background reference {other_code}",
             "channel": "background"},
            {"observation_id": "o4", "timestamp": "00:15", "kind": "speech",
             "speaker": "announcer", "value": f"Proceed to the {pack.location} desk",
             "channel": "foreground"},
        ]
    raise ValueError(f"unsupported media modality: {modality}")


def _source(source_id: str, modality: str, title: str, description: str,
            observations: list[dict], *, searchable: bool = False) -> dict:
    return {
        "source_id": source_id, "modality": modality, "title": title,
        "description": description, "observations": observations,
        "searchable": searchable, "synthetic": True,
        "asset_path": None, "simulator": "structured_observation_v1",
    }


def _task(task_id: str, kind: str, goal: str, answer, evidence_ids: list[str],
          events: list[dict], trace: list[str], *, binds: dict | None = None) -> dict:
    return {
        "subtask_id": task_id, "kind": kind, "goal": goal,
        "inputs": dict(binds or {}),
        "answer": {"value": answer},
        "evidence_ids": evidence_ids, "tool_events": events,
        "summary": trace[-1], "summary_trace": trace,
        "verification": "simulation_replay_verified",
    }


def build_scene(motif: dict, ordinal: int, *, seed: int, domain: str | None = None,
                modality: str | None = None, pattern: str | None = None) -> dict:
    if ordinal < 0:
        raise ValueError("ordinal must not be negative")
    rng = random.Random(seed + ordinal * 100003)
    inferred_domain = motif.get("domain_hint", "general")
    pack = BY_DOMAIN[domain or inferred_domain] if (domain or inferred_domain) in BY_DOMAIN else BY_DOMAIN["general"]
    modality = modality or "image"
    if modality not in {"image", "video", "audio"}:
        raise ValueError("modality must be image/video/audio")
    pattern = pattern or "fork_join"
    if pattern not in {"chain", "fork_join", "cross_check"}:
        raise ValueError("pattern must be chain/fork_join/cross_check")
    scene_type = rng.choice(pack.scene_types)
    assigned = rng.choice(pack.units)
    other_assigned = rng.choice([v for v in pack.units if v != assigned])
    registry_assigned = assigned if ordinal % 2 == 0 else other_assigned
    states = ("Registered", "Pending review", "Suspended", "Registered")
    state = states[ordinal % len(states)]
    code = f"MX-{ordinal+1:05d}"
    distractor = _other_code(code)
    sid = "sim-" + digest(seed, ordinal, motif["motif_id"], pack.domain, scene_type)
    media_id, schedule_id, registry_id = (f"{sid}:{role}"
        for role in ("media", "schedule", "registry"))
    decoy_id = f"{sid}:decoy"
    descriptions = {
        "image": f"A simulated still from {pack.setting}: a {pack.asset} is shown "
                 f"at a {scene_type.replace('_',' ')}. Another similar item is in the background.",
        "video": f"A simulated 18-second recording at {pack.setting}, featuring a "
                 f"{pack.asset}, an unrelated background item and several timed observations.",
        "audio": f"A simulated 18-second announcement recording at {pack.setting}, "
                 f"with a foreground reference and a separate background radio announcement.",
    }
    media = _source(media_id, modality, f"{scene_type.replace('_',' ').title()}",
                    descriptions[modality],
                    _media_observations(modality, pack, code, distractor, scene_type, pack.setting))
    second_modality = "table" if "table" in motif.get("source_modalities", []) else (
        "text" if "text" in motif.get("source_modalities", []) and ordinal % 3 == 0 else "table")
    other_location = rng.choice([v for v in pack.units if v != assigned])
    schedule_rows = [
        {"reference": distractor, "assignment": other_location},
        {"reference": code, "assignment": assigned},
        {"reference": f"MX-{ordinal+101:05d}", "assignment": rng.choice(pack.units)},
    ]
    schedule_obs = (
        [{"observation_id": f"row-{i+1}", "kind": "table_row", **v}
         for i, v in enumerate(schedule_rows)]
        if second_modality == "table"
        else [{"observation_id": f"paragraph-{i+1}", "kind": "written_record",
               "value": f"Reference {v['reference']} is assigned to {v['assignment']}."}
              for i, v in enumerate(schedule_rows)]
    )
    schedule = _source(
        schedule_id, second_modality, pack.schedule.title(),
        f"Simulated {pack.schedule} for the {pack.setting}, with several references.",
        schedule_obs,
    )
    registry_claim = {
        "reference": code, "registration_status": state,
        "registered_location": registry_assigned,
    }
    registry = _source(
        registry_id, "search", pack.registry.title(),
        f"Simulated {pack.registry} entry for reference {code}.",
        [{"observation_id": "registry-record", "kind": "registry_record",
          **registry_claim}],
        searchable=True,
    )
    decoy = _source(
        decoy_id, "search", f"Other {pack.registry}",
        f"Unrelated {pack.registry} entry for {distractor}.",
        [{"observation_id": "registry-decoy", "kind": "registry_record",
          "reference": distractor, "registration_status": "Registered",
          "registered_location": other_assigned}],
        searchable=True,
    )

    scene = {
        "schema_version": "synthetic_mmqa_scene_v1",
        "scene_id": sid,
        "setting": pack.setting, "domain": pack.domain,
        "scene_type": scene_type, "primary_modality": modality,
        "secondary_modality": second_modality, "graph_pattern": pattern,
        "source_motif": {
            "motif_id": motif["motif_id"], "record_id": motif.get("record_id"),
            "question_hint": motif["question_hint"],
            "question_type": motif.get("question_type", ""),
            "logical_pattern": motif.get("logical_pattern", ""),
            "source_modalities": motif.get("source_modalities", []),
            "alignment_policy": (
                "topic_and_structure" if pack.domain == inferred_domain
                else "structure_only_synthetic_domain_extension"
            ),
            "media_extension": modality in {"video", "audio"},
        },
        "world": {
            "target_reference": code,
            "distractor_reference": distractor,
            "target_kind": pack.asset,
            "schedule_assignment": assigned,
            "registry_status": state,
            "registry_assignment": registry_assigned,
        },
        "sources": [media, schedule, registry, decoy],
        "required_source_ids": [media_id, schedule_id, registry_id]
            if pattern != "chain" else [media_id, registry_id],
        "optional_source_ids": [decoy_id],
    }
    scene["source_motif"]["logical_pattern_alignment"] = motif.get(
        "logical_pattern") in {
            "chain": ("composition", "lookup"),
            "fork_join": ("lookup", "aggregation", "intersection"),
            "cross_check": ("comparison", "intersection"),
        }[pattern]
    sim_graph = materialize_graph(scene, pack)
    scene.update(sim_graph)
    return scene


def materialize_graph(scene: dict, pack: ScenePack) -> dict:
    """Deterministically derive teacher feedback from source observations."""
    sid = scene["scene_id"]
    source_by_id = {s["source_id"]: s for s in scene["sources"]}
    media_id, sched_id, reg_id = scene["required_source_ids"][0], f"{sid}:schedule", f"{sid}:registry"
    code = next(
        o["value"] for o in source_by_id[media_id]["observations"]
        if o["kind"] == "visible_identifier"
    ) if scene["primary_modality"] != "audio" else next(
        scene["world"]["target_reference"] for o in source_by_id[media_id]["observations"]
        if o["kind"] == "speech" and o.get("speaker") == "announcer"
        and scene["world"]["target_reference"] in o["value"]
    )
    schedule_source = source_by_id[sched_id]
    if scene["secondary_modality"] == "table":
        assignment = next(
            o["assignment"] for o in schedule_source["observations"]
            if o.get("reference") == code
        )
    else:
        phrase = next(o["value"] for o in schedule_source["observations"]
                      if f"Reference {code} is assigned to " in o.get("value", ""))
        assignment = phrase.split(" is assigned to ", 1)[1].rstrip(".")
    registry = next(
        o for o in source_by_id[reg_id]["observations"]
        if o["kind"] == "registry_record" and o["reference"] == code
    )
    state, registry_location = registry["registration_status"], registry["registered_location"]
    media_node_id, table_node_id, search_node_id = (f"{sid}:{role}"
        for role in ("S1", "S2", "S3"))
    media_node = _task(
        media_node_id, f"{scene['primary_modality']}_identify",
        f"Identify the {pack.characteristic} of the target {pack.asset}.",
        code, [media_id],
        [{"action": "InspectSimulatedMedia", "source_id": media_id,
          "observation_ids": [o["observation_id"]
                              for o in source_by_id[media_id]["observations"]]}],
        [f"Inspected simulated {scene['primary_modality']} source {media_id}.",
         f"The target {pack.characteristic} is {code}."],
        binds={"source_id": media_id},
    )
    table_node = _task(
        table_node_id, "lookup_assignment",
        f"Find the {pack.location} assignment for reference {code}.",
        assignment, [sched_id],
        [{"action": "ReadSimulatedDocument", "source_id": sched_id,
          "reference": code}],
        [f"Looked up reference {code} in {pack.schedule}.",
         f"The {pack.location} assignment is {assignment}."],
        binds={"reference": "$" + media_node_id + ".answer"},
    )
    search_node = _task(
        search_node_id, "search_registry",
        f"Find the current {pack.registry} record for reference {code}.",
        {"status": state, "location": registry_location},
        [reg_id],
        [{"action": "SimulatedSearch", "query": f"{code} {pack.registry}",
          "returned_ids": [reg_id, f"{sid}:decoy"]},
         {"action": "OpenSimulatedPage", "source_id": reg_id}],
        [f"Searched the simulated {pack.registry} for {code}.",
         f"Registry status is {state}, with location {registry_location}."],
        binds={"reference": "$" + media_node_id + ".answer"},
    )
    base = [media_node, search_node] if scene["graph_pattern"] == "chain" else [
        media_node, table_node, search_node
    ]
    dependencies = [{
        "from": media_node_id, "to": search_node_id, "binding": "reference"
    }]
    if scene["graph_pattern"] != "chain":
        dependencies.append({
            "from": media_node_id, "to": table_node_id, "binding": "reference"
        })

    reference_phrase = (
        "mentioned in the supplied audio" if scene["primary_modality"] == "audio"
        else f"shown in the supplied {scene['primary_modality']}"
    )
    if scene["graph_pattern"] == "chain":
        question = (f"For the {pack.asset} identified in the supplied "
                    f"{scene['primary_modality']} from the {pack.setting}, "
                    f"what is its current {pack.registry} status and registered {pack.location}?")
        gold = {"status": state, "location": registry_location}
        final = search_node_id
    else:
        if scene["graph_pattern"] == "cross_check":
            gold = {"agree": assignment == registry_location,
                    "schedule": assignment, "registry": registry_location}
            question = (
                f"Do the {pack.schedule} and {pack.registry} give the same "
                f"{pack.location} for the {pack.asset} {reference_phrase}? "
                f"Report both recorded locations."
            )
        else:
            gold = {"assignment": assignment, "registration": state}
            question = (
                f"For the {pack.asset} {reference_phrase} "
                f"from the {pack.setting}, "
                f"what {pack.location} is it assigned to, and what is "
                f"its current {pack.registry} status?"
            )
        final = f"{sid}:S4"
        combine = _task(
            final, "join_results", "Combine the independently obtained records.",
            gold, [sched_id, reg_id],
            [{"action": "Combine", "inputs": [table_node_id, search_node_id]}],
            [f"Compared {pack.schedule} with {pack.registry}.",
             "Returned both source-specific results without dropping either constraint."],
            binds={"schedule": "$" + table_node_id + ".answer",
                   "registry": "$" + search_node_id + ".answer"},
        )
        base.append(combine)
        dependencies.extend([
            {"from": table_node_id, "to": final, "binding": "answer"},
            {"from": search_node_id, "to": final, "binding": "answer"},
        ])
    return {
        "task_graph": {
            "pattern": scene["graph_pattern"],
            "nodes": base, "edges": dependencies, "final_node": final,
            "required_subtasks": [t["subtask_id"] for t in base],
            "optional_subtask_ids": [],
        },
        "question_brief": {
            "final_goal": question,
            "public_anchors": [media_id],
            "required_sources": list(scene["required_source_ids"]),
            "answer_type": "record" if scene["graph_pattern"] != "cross_check" else "comparison",
            "private_answers_excluded": True,
        },
        "question": question,
        "gold_answer": gold,
        "feedback_variants": {
            "clean": {
                "affected_subtask": media_node_id,
                "status": "supported",
                "reported_reference": code,
                "source_observation_id": "o2" if scene["primary_modality"] == "image"
                                         else "o3" if scene["primary_modality"] == "video"
                                         else "o2",
            },
            "partial": {
                "affected_subtask": media_node_id,
                "status": "ambiguous",
                "reported_candidates": [code, scene["world"]["distractor_reference"]],
                "reason": "target identifier partly unreadable in the simulated report",
            },
            "misleading": {
                "affected_subtask": media_node_id,
                "status": "unsupported_claim",
                "reported_reference": scene["world"]["distractor_reference"],
                "reason": "an intentionally incorrect sub-agent feedback variant",
            },
        },
        "verification": {
            "level": "simulation_replay_verified",
            "world_consistency": True,
            "real_media_perception": False,
            "required_source_ablation": "all_required_for_complete_output",
        },
    }


def generate_scenes(motifs: list[dict], *, target: int = 100,
                    seed: int = 17, extensions: bool = True) -> tuple[list[dict], dict]:
    if not motifs or target < 1:
        raise ValueError("nonempty MMQA motifs and positive target are required")
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for motif in motifs:
        buckets.setdefault(motif.get("domain_hint", "general"), []).append(motif)
    for group in buckets.values():
        rng.shuffle(group)
    usage: Counter = Counter()
    scenes = []
    for ordinal in range(target):
        # Balanced domains regardless of skew in the first MMQA data prefix.
        pack = PACKS[ordinal % len(PACKS)]
        source_pool = buckets.get(pack.domain) or motifs
        desired_pattern = ("fork_join", "chain", "cross_check")[ordinal % 3]
        compatible = {
            "chain": {"composition", "lookup"},
            "fork_join": {"lookup", "aggregation", "intersection"},
            "cross_check": {"comparison", "intersection"},
        }[desired_pattern]
        aligned_pool = [m for m in source_pool
                        if m.get("logical_pattern") in compatible]
        candidates = aligned_pool or source_pool
        motif = candidates[(ordinal // len(PACKS)) % len(candidates)]
        # Image remains the dominant visual mode. Video/audio are explicitly
        # synthetic extensions, not represented as official MMQA modalities.
        media = ("audio" if ordinal % 11 == 7 else
                 "video" if extensions and ordinal % 7 == 4 else "image")
        if not extensions:
            media = "image"
        pattern = desired_pattern
        scene = build_scene(
            motif, ordinal, seed=seed, domain=pack.domain,
            modality=media, pattern=pattern,
        )
        from .validation import validate_scene
        errors = validate_scene(scene)
        if errors:
            raise ValueError(f"{scene['scene_id']}: {errors}")
        scenes.append(scene)
        usage.update([
            "aligned:" + str(scene["source_motif"]["logical_pattern_alignment"]),
            "domain:" + scene["domain"], "type:" + scene["scene_type"],
            "media:" + media, "graph:" + pattern,
            "motif_pattern:" + motif.get("logical_pattern", ""),
        ])
    return scenes, {
        "scenes": len(scenes),
        "domains": {key[7:]: value for key, value in usage.items() if key.startswith("domain:")},
        "scene_types": {key[5:]: value for key, value in usage.items() if key.startswith("type:")},
        "media": {key[6:]: value for key, value in usage.items() if key.startswith("media:")},
        "graph_patterns": {key[6:]: value for key, value in usage.items() if key.startswith("graph:")},
        "mmqa_motif_patterns": {key[14:]: value for key, value in usage.items()
                                if key.startswith("motif_pattern:")},
        "mmqa_pattern_alignment": {
            key[8:]: value for key, value in usage.items() if key.startswith("aligned:")
        },
        "policy": "MMQA-pattern-guided_fully_synthetic_media_and_facts",
    }
