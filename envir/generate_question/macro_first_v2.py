"""Macro-first v2: goal-conditioned Macro DAGs and per-Macro evidence plans.

This opt-in schema coexists with v1: frozen v1 scenes keep their exact replay.
V2 has natural user tasks, typed non-year goals, heterogeneous simulated
modalities and independently sampled evidence budgets per Macro.
"""
from __future__ import annotations

import random

from envir.schema import digest
from .macro_first import _source

VERSION = "maestro.macro_plan.v2"
PATTERNS = ("serial_identity", "parallel_cues",
            "disputed_reports", "multimodal_event")
# High-precision paraphrases; a changed target (e.g. match year instead
# of match result) is never a valid semantic variant.
GOAL_PHRASES = {
    "match_result": ("match result", "final score", "match score"),
    "event_venue": ("event venue", "venue for the event", "event location"),
    "match_year": ("match year", "year of the match"),
    "episode_title": ("episode title", "title of the episode", "episode's title"),
    "screening_venue": ("screening venue", "screening location"),
    "production_year": ("production year", "year of production"),
    "performance_venue": ("performance venue", "performance location"),
    "song_title": ("song title", "title of the song"),
    "performance_year": ("performance year", "year of the performance"),
    "edition_title": ("edition title", "title of the edition"),
    "publication_year": ("publication year", "year of publication"),
    "artifact_title": ("artifact title", "title of the artifact"),
    "record_year": ("record year", "year of the record"),
    "survey_site": ("survey site", "site of the survey"),
    "survey_year": ("survey year", "year of the survey"),
    "experiment_label": ("experiment label", "experiment identifier"),
    "review_status": ("review status", "status of the review"),
    "resolution_status": ("resolution status", "status of the resolution"),
    "filing_year": ("filing year", "year of filing"),
    "case_title": ("case title", "title of the case"),
    "review_year": ("review year", "year of the review"),
}

DOMAIN_GOALS = {
    "sports": (("match_result", "str"), ("event_venue", "str"),
               ("match_year", "int")),
    "film": (("episode_title", "str"), ("screening_venue", "str"),
             ("production_year", "int")),
    "music": (("performance_venue", "str"), ("song_title", "str"),
              ("performance_year", "int")),
    "book": (("edition_title", "str"), ("publication_year", "int")),
    "history": (("artifact_title", "str"), ("record_year", "int")),
    "geography": (("survey_site", "str"), ("survey_year", "int")),
    "science": (("experiment_label", "str"), ("review_status", "str")),
    "business": (("resolution_status", "str"), ("filing_year", "int")),
    "general": (("case_title", "str"), ("review_status", "str"),
                ("review_year", "int")),
}

def domain_hint(topic: str, hint: str) -> str:
    blob = (topic + " " + hint).lower()
    aliases = {
        "sports": ("football", "soccer", "league", "nba", "tennis",
                   "basketball", "world cup", "racing", "olympic", "cricket"),
        "film": ("film", "movie", "cinema", "episode", "television",
                 "tv series", "netflix", "the office"),
        "music": ("song", "album", "music", "singer", "concert",
                  "band", "pearl jam", "dion"),
        "book": ("book", "novel", "author", "publisher", "literature"),
        "geography": ("geography", "river", "city", "mountain", "geographic"),
        "science": ("biology", "physics", "science", "experiment",
                    "chemistry", "laboratory"),
        "history": ("history", "museum", "artifact", "archive", "historical"),
        "business": ("company", "business", "finance", "trade", "corporation"),
    }
    for category, words in aliases.items():
        if any(word in blob for word in words):
            return category
    return "general"


def sample_macro_plan(*, seed: int, ordinal: int, record_id: str, topic: str,
                      difficulty: str, domain: str) -> tuple[dict, list[dict]]:
    if difficulty not in {"medium", "hard"}:
        raise ValueError("macro_first_v2_requires_medium_or_hard")
    if domain not in DOMAIN_GOALS:
        raise ValueError("unknown_macro_first_v2_domain")
    rng = random.Random(int(digest(
        seed, ordinal, record_id, "macro-first-v2-plan"
    )[:15], 16))
    pattern = rng.choice(PATTERNS)
    goal_field, goal_type = rng.choice(DOMAIN_GOALS[domain])
    color = rng.choice(("amber", "blue", "green", "red", "violet"))
    motif = rng.choice(("crescent", "spiral", "star", "wave", "triangle"))
    action = rng.choice(("entrance", "handoff", "turn", "signal", "departure"))
    tag = rng.choice(("reviewed", "priority", "incident", "follow-up", "flagged"))
    threshold = rng.choice((30000, 45000, 60000))
    sources, schemas, tasks = [], {}, []

    def source(sid, modality, schema, purpose, key=None):
        item, fields = _source(sid, topic, modality, schema,
                               key=key, purpose=purpose)
        kind = {
            "visual": "image observations" if modality == "image"
                      else "video segment annotations",
            "caption": "written captions" if modality == "text"
                       else "commentary utterances",
            "footage": "event footage timeline",
            "commentary": "audio commentary transcript",
            "context": "event log excerpts",
            "report_a": "first incident report",
            "report_b": "independent incident register",
            "archive": "verified outcome records",
        }.get(sid, "cross-reference index")
        item["title"] = f"{topic} — {kind}"
        sources.append(item)
        schemas[sid] = fields

    def macro(role, parents, output, scoped=()):
        mid = "M" + str(len(tasks) + 1)
        grade = rng.choices(("simple", "moderate", "complex"),
                            weights=(3, 5, 2), k=1)[0]
        lo, hi = {
            "simple": (4, 6), "moderate": (7, 11), "complex": (12, 17)
        }[grade]
        targets = {sid: rng.randint(lo, hi) for sid in scoped}
        tasks.append({
            "macro_id": mid,
            "role": role,
            "depends_on": list(parents),
            "input_contract": {
                pid: next(t["output_schema"] for t in tasks
                          if t["macro_id"] == pid) for pid in parents
            },
            "output_schema": dict(output),
            "source_scope": list(scoped),
            "solver_template": role,
            "evidence_plan": {
                "evidence_volume_grade": grade,
                "source_observation_targets": targets,
                "candidate_observations_to_examine": sum(targets.values()),
                "description": "source-row examination volume; not calibrated reasoning difficulty",
            },
        })
        return mid

    visual_modality = rng.choice(("image", "video"))
    caption_modality = rng.choice(("audio", "text"))
    if pattern in {"serial_identity", "parallel_cues"}:
        visual_fields = {
            "candidate_id": "str", "marker_code": "str",
            "emblem_color": "str"
        }
        if visual_modality == "image":
            visual_fields["emblem_shape"] = "str"
        else:
            visual_fields.update({"scene_action": "str", "frame_time_ms": "int"})
        source("visual", visual_modality, visual_fields,
               "identify the subject from a media observation")
        if pattern == "serial_identity":
            prior = macro("visual_identity", [], {"marker_code": "str"}, ["visual"])
        else:
            source("caption", caption_modality, (
                {"candidate_id": "str", "spoken_cue": "str",
                 "utterance_time_ms": "int"} if caption_modality == "audio"
                else {"candidate_id": "str", "caption_motif": "str"}
            ), "independent spoken or written clue")
            first = macro("visual_candidates", [],
                          {"candidate_id": "str", "marker_code": "str"}, ["visual"])
            second = macro("caption_candidates", [],
                           {"candidate_id": "str"}, ["caption"])
            prior = macro("candidate_fusion", [first, second],
                          {"marker_code": "str"})
        key_field = "marker_code"
    elif pattern == "multimodal_event":
        source("footage", "video", {
            "candidate_id": "str", "marker_code": "str", "scene_action": "str",
            "footage_time_ms": "int",
        }, "identify and time a visible event")
        source("commentary", "audio", {
            "candidate_id": "str", "spoken_cue": "str",
            "commentary_time_ms": "int",
        }, "locate a spoken report after the event")
        source("context", "text", {
            "candidate_id": "str", "case_tag": "str",
        }, "independently check an event log label")
        first = macro("video_audio_alignment", [],
                      {"candidate_id": "str", "marker_code": "str"},
                      ["footage", "commentary"])
        second = macro("context_candidates", [], {"candidate_id": "str"},
                       ["context"])
        prior = macro("event_fusion", [first, second], {"marker_code": "str"})
        key_field = "marker_code"
    else:
        source("report_a", "text",
               {"case_id": "str", "reported_year_a": "int"},
               "first record of a reported event year")
        source("report_b", "table",
               {"case_id": "str", "reported_year_b": "int"},
               "independent record of the same event")
        first = macro("first_report", [],
                      {"case_id": "str", "reported_year_a": "int"}, ["report_a"])
        second = macro("second_report", [],
                       {"case_id": "str", "reported_year_b": "int"}, ["report_b"])
        prior = macro("dispute_resolution", [first, second], {"case_id": "str"})
        key_field = "case_id"

    # Macro count is sampled independently of each Macro's evidence budget.
    # Hard changes the mix probabilistically, not via a fixed +1 lookup.
    hops = rng.choices((0, 1, 2), weights=(
        (2, 4, 3) if difficulty == "medium" else (1, 3, 5)
    ), k=1)[0]
    for i in range(1, hops + 1):
        new_key = f"record_key_{i}"
        sid = f"index_{i}"
        source(sid, "text", {key_field: "str", new_key: "str"},
               "retrieve a linked reference by an upstream identifier",
               key=key_field)
        prior = macro(f"key_resolution_{i}", [prior],
                      {new_key: "str"}, [sid])
        key_field = new_key
    source("archive", "table", {key_field: "str", goal_field: goal_type},
           "confirm the requested detail for a discovered case",
           key=key_field)
    terminal = macro("authoritative_result", [prior],
                     {goal_field: goal_type}, ["archive"])
    plan = {
        "version": VERSION, "pattern": pattern, "difficulty": difficulty,
        "domain": domain,
        "goal": {"answer_field": goal_field, "answer_type": goal_type,
                 "public_label": goal_field.replace("_", " "),
                 "cardinality": "exactly_one",
                 "meaning": f"confirmed {goal_field.replace('_', ' ')} for the matching item"},
        "clues": {
            "emblem_color": color, "emblem_shape": motif,
            "scene_action": action, "spoken_cue": motif,
            "case_tag": tag, "time_threshold_ms": threshold,
            "visual_modality": visual_modality,
            "caption_modality": caption_modality,
        },
        "macro_tasks": tasks, "terminal_macro_id": terminal,
        "source_schemas": schemas, "index_hops": hops,
        "media_interpretation": "simulated_typed_observations_not_media_decoding",
    }
    return plan, sources


def make_spec(*, seed_record: dict, ordinal: int, run_seed: int,
              difficulty: str, topic: str, record_id: str, version: str) -> dict:
    domain = domain_hint(topic, str(seed_record.get("domain_hint", "")))
    plan, roles = sample_macro_plan(
        seed=run_seed, ordinal=ordinal, record_id=record_id, topic=topic,
        difficulty=difficulty, domain=domain,
    )
    from .semantic_evidence import CONTRACT_VERSION
    return {
        "version": version,
        "semantic_evidence_contract_version": CONTRACT_VERSION,
        "task_id": f"question-{ordinal:06d}-{digest(record_id, run_seed)[:8]}",
        "scene_id": record_id,
        "source_anchor": {
            "dataset": "MultiModalQA_topic_only",
            "source_record_id": record_id,
            "original_question_reused": False,
            "original_answer_reused": False,
            "original_media_used": False,
            "domain_hint": str(seed_record.get("domain_hint", "")),
        },
        "difficulty_target": difficulty,
        "family": "macro_first_v2", "reasoning_pattern": "macro_first_v2",
        "scenario": {
            "topic": topic, "setting": f"an information investigation about {topic}",
            "user_role": "investigator",
            "purpose": "resolve the requested fact by combining independent material",
            "motivation": "check the result against relevant sources",
            "entity_kind": "cases and events",
        },
        "sources": roles, "macro_plan": plan,
        "requirements": {
            "must_resolve_aliases": False,
            "must_match_independent_reviews": False,
            "must_check_independent_clearance": False,
            "answer_kind": "macro_value",
            "must_preserve_source_provenance": True,
            "simulation_only": True,
        },
        "seed": run_seed, "ordinal": ordinal,
    }


def validate_spec(spec: dict) -> list[str]:
    plan = spec.get("macro_plan")
    if not isinstance(plan, dict) or plan.get("version") != VERSION:
        return ["invalid_macro_first_v2_plan"]
    from .semantic_evidence import CONTRACT_VERSION
    if spec.get("semantic_evidence_contract_version") not in {
        None, CONTRACT_VERSION
    }:
        return ["invalid_semantic_evidence_contract_version"]
    try:
        expected, roles = sample_macro_plan(
            seed=spec["seed"], ordinal=spec["ordinal"],
            record_id=spec["source_anchor"]["source_record_id"],
            topic=spec["scenario"]["topic"],
            difficulty=spec["difficulty_target"],
            domain=domain_hint(
                spec["scenario"]["topic"],
                spec["source_anchor"].get("domain_hint", "")
            ),
        )
        errors = []
        if plan != expected:
            errors.append("macro_first_v2_plan_not_reproducible")
        def contract(role):
            return (role.get("source_id"), role.get("modality"),
                    role.get("access_mode"), role.get("key_field"))
        if ([contract(role) for role in spec["sources"]]
                != [contract(role) for role in roles]):
            errors.append("macro_first_v2_source_scope_changed")
        if (not plan.get("macro_tasks")
                or plan["terminal_macro_id"] != plan["macro_tasks"][-1]["macro_id"]):
            errors.append("macro_first_v2_terminal_changed")
        return errors
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return ["invalid_macro_first_v2_contract:" + str(exc)[:120]]


def compile_macro_solvers(spec: dict) -> dict:
    errors = validate_spec(spec)
    if errors:
        raise ValueError("invalid_macro_first_v2_spec:" + ",".join(errors))
    plan = spec["macro_plan"]
    clues = plan["clues"]
    nodes, groups, outputs = [], [], {}
    prior_key = ("case_id" if plan["pattern"] == "disputed_reports"
                 else "marker_code")
    for task in plan["macro_tasks"]:
        mid, role = task["macro_id"], task["role"]
        local = []
        def add(op, *inputs, **fields):
            nid = f"{mid.lower()}_op{len(local)+1:02d}"
            local.append({"id": nid, "op": op, "inputs": list(inputs), **fields})
            return nid

        parents = [outputs[pid] for pid in task["depends_on"]]
        if role in {"visual_identity", "visual_candidates"}:
            src = add("scan", source_id="visual")
            if clues["visual_modality"] == "image":
                selected = add("filter_eq", src, field="emblem_color",
                               value=clues["emblem_color"])
                if role == "visual_identity":
                    selected = add("filter_eq", selected, field="emblem_shape",
                                   value=clues["emblem_shape"])
            else:
                selected = add("filter_eq", src, field="scene_action",
                               value=clues["scene_action"])
                selected = add("filter_gte", selected, field="frame_time_ms",
                               value=clues["time_threshold_ms"])
            last = add("project", selected, fields=(
                ["marker_code"] if role == "visual_identity"
                else ["candidate_id", "marker_code"]
            ))
        elif role == "caption_candidates":
            src = add("scan", source_id="caption")
            field = ("spoken_cue" if clues["caption_modality"] == "audio"
                     else "caption_motif")
            matched = add("filter_eq", src, field=field,
                          value=clues["emblem_shape"])
            last = add("project", matched, fields=["candidate_id"])
        elif role in {"first_report", "second_report"}:
            sid = "report_a" if role == "first_report" else "report_b"
            src = add("scan", source_id=sid)
            last = add("project", src, fields=[
                "case_id", ("reported_year_a" if sid == "report_a"
                            else "reported_year_b"),
            ])
        elif role in {"candidate_fusion", "event_fusion"}:
            matched = add("join", *parents, left_on="candidate_id",
                          right_on="candidate_id")
            last = add("project", matched, fields=["marker_code"])
        elif role == "dispute_resolution":
            matched = add("join", *parents, left_on="case_id",
                          right_on="case_id")
            selected = add("compare_fields", matched,
                           left_field="reported_year_a",
                           right_field="reported_year_b", comparison="ne")
            last = add("project", selected, fields=["case_id"])
        elif role == "video_audio_alignment":
            video = add("scan", source_id="footage")
            event = add("filter_eq", video, field="scene_action",
                        value=clues["scene_action"])
            audio = add("scan", source_id="commentary")
            spoken = add("filter_eq", audio, field="spoken_cue",
                         value=clues["spoken_cue"])
            matched = add("join", event, spoken, left_on="candidate_id",
                          right_on="candidate_id")
            ordered = add("compare_fields", matched,
                          left_field="footage_time_ms",
                          right_field="commentary_time_ms", comparison="lte")
            last = add("project", ordered, fields=["candidate_id", "marker_code"])
        elif role == "context_candidates":
            src = add("scan", source_id="context")
            selected = add("filter_eq", src, field="case_tag",
                           value=clues["case_tag"])
            last = add("project", selected, fields=["candidate_id"])
        elif role.startswith("key_resolution_"):
            i = int(role.split("_")[-1])
            following = f"record_key_{i}"
            key = add("lookup_by_key", *parents, source_id=f"index_{i}",
                      input_key_field=prior_key, source_key_field=prior_key)
            last = add("project", key, fields=[following])
            prior_key = following
        elif role == "authoritative_result":
            key = add("lookup_by_key", *parents, source_id="archive",
                      input_key_field=prior_key, source_key_field=prior_key)
            projected = add("project", key,
                            fields=[plan["goal"]["answer_field"]])
            last = add("unique", projected)
        else:
            raise ValueError("unknown_macro_first_v2_solver:" + role)
        nodes.extend(local)
        groups.append({
            "macro_id": mid, "node_ids": [n["id"] for n in local],
            "output_node_id": last, "external_input_nodes": parents,
        })
        outputs[mid] = last
    return {
        "version": "maestro.reasoning_program.v4",
        "semantic_operator_pattern": "macro_first_v2",
        "nodes": nodes, "output": outputs[plan["terminal_macro_id"]],
        "authorized_source_ids": [source["source_id"]
                                   for source in spec["sources"]],
        "frozen_problem_spec_id": spec["task_id"],
        "macro_solver_groups": groups, "macro_plan_version": VERSION,
    }
