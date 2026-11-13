"""Macro-first v2: goal-conditioned Macro DAGs and per-Macro evidence plans.

This opt-in schema coexists with v1: frozen v1 scenes keep their exact replay.
V2 has natural user tasks, typed non-year goals, heterogeneous simulated
modalities and independently sampled evidence budgets per Macro.
"""
from __future__ import annotations

import random

from environment.schema import digest
from .macro_first import _source
from .evidence_tools import (
    EVIDENCE_TOOL_CONTRACT_VERSION, LEGACY_EVIDENCE_TOOL_CONTRACT_VERSION,
    source_acquisition_policy, acquisition_for_operator,
    synthetic_page_url, web_acquisition_route,
)

VERSION = "maestro.macro_plan.v2"
INFORMATION_NECESSITY_VERSION = "maestro.cross_branch_candidates.v1"
GRAMMAR_VERSION = "maestro.macro_grammar.v3"
PATTERNS = ("serial_identity", "parallel_cues",
            "disputed_reports", "multimodal_event")
EXTENDED_PATTERNS = PATTERNS + ("corroborated_dispute",)
# High-precision paraphrases; a changed target (e.g. match year instead
# of match result) is never a valid semantic variant.
GOAL_PHRASES = {
    "match_result": (
        "match result", "final score", "match score", "match outcome",
        "game result", "game score", "final scoreline",
    ),
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
    "higher_count_source": ("higher count", "larger count", "which source"),
    "completion_rate_percent": ("completion rate", "percentage completed",
                                "completion percentage"),
    "exhibition_title": ("exhibition title", "title of the exhibition"),
    "exhibition_year": ("exhibition year", "year of the exhibition"),
    "course_title": ("course title", "title of the course"),
    "campus_site": ("campus site", "campus location"),
    "academic_year": ("academic year", "year of the academic record"),
    "delivery_status": ("delivery status", "status of the delivery"),
    "arrival_site": ("arrival site", "arrival location"),
    "shipment_year": ("shipment year", "year of the shipment"),
    "civic_site": ("civic site", "civic location"),
    "permit_status": ("permit status", "status of the permit"),
    "inspection_year": ("inspection year", "year of the inspection"),
    "species_label": ("species label", "species name"),
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
    "culture": (("exhibition_title", "str"), ("performance_venue", "str"),
                ("exhibition_year", "int")),
    "education": (("course_title", "str"), ("campus_site", "str"),
                  ("academic_year", "int")),
    "logistics": (("delivery_status", "str"), ("arrival_site", "str"),
                  ("shipment_year", "int")),
    "civic": (("civic_site", "str"), ("permit_status", "str"),
              ("inspection_year", "int")),
    "environment": (("survey_site", "str"), ("species_label", "str"),
                    ("survey_year", "int")),
}

def classify_domain_from_keywords(topic: str, hint: str) -> str:
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


def classify_domain_from_metadata(topic: str, hint: str) -> str:
    """Use reliable MMQA category metadata before sparse name keywords.

    This only applies to newly versioned semantic-contract scenes, so
    previously frozen V2 plans continue using classify_domain_from_keywords unchanged.
    Avoid speculative culture/civic remaps without a recognizable subject.
    """
    value = str(hint or "").strip().lower()
    if value == "sports":
        return "sports"
    if value == "commerce":
        return "business"
    return classify_domain_from_keywords(topic, hint)


def classify_domain_from_taxonomy(topic: str, hint: str) -> str:
    """More faithful domain taxonomy for NEW information-contract scenes.

    Keep classify_domain_from_metadata unchanged: the older semantic V2 scenes must replay
    against their original topic/Goal mapping.
    """
    value = str(hint or "").strip().lower()
    by_metadata = {
        "commerce": "business", "culture": "culture",
        "education": "education", "logistics": "logistics",
        "civic": "civic", "environment": "environment",
        "science": "science", "sports": "sports",
    }
    named = classify_domain_from_keywords(topic, "")
    return named if named != "general" else by_metadata.get(
        value, classify_domain_from_metadata(topic, hint)
    )


def classify_domain_from_explicit_cue(topic: str, hint: str, cue: str | None) -> str:
    """Prefer unambiguous topic-category cues over unrelated broad metadata."""
    return cue if cue in DOMAIN_GOALS else classify_domain_from_taxonomy(topic, hint)


def sample_macro_plan(*, seed: int, ordinal: int, record_id: str, topic: str,
                      difficulty: str, domain: str,
                      information_need: bool = False,
                      realism_v3: bool = False,
                      extended_grammar: bool = False,
                      goal_resolution_enabled: bool = False,
                      evidence_tools_enabled: bool = False,
                      evidence_tools_version: str | None = None,
                      operator_grammar_enabled: bool = False,
                      ) -> tuple[dict, list[dict]]:
    if difficulty not in {"medium", "hard"}:
        raise ValueError("macro_first_v2_requires_medium_or_hard")
    if domain not in DOMAIN_GOALS:
        raise ValueError("unknown_macro_first_v2_domain")
    rng = random.Random(int(digest(
        seed, ordinal, record_id, "macro-first-v2-plan"
    )[:15], 16))
    pattern = rng.choice(
        EXTENDED_PATTERNS if extended_grammar else PATTERNS
    )
    candidate_profile = None
    if realism_v3 and pattern in {
        "multimodal_event", "corroborated_dispute",
    }:
        # Independent branches have varied ambiguity, with Hard sampling
        # larger disjoint false-positive sets than Medium on average.
        low, high = ((3, 4) if difficulty == "hard" else (2, 3))
        left_count = rng.randint(low, high)
        right_count = rng.randint(low, high)
        candidate_profile = {
            "left_candidate_count": left_count,
            "right_candidate_count": right_count,
            "union_candidate_count": left_count + right_count - 1,
            "intersection_candidate_count": 1,
        }
    goal_kind = (
        rng.choices(("direct_lookup", "comparison", "calculation"),
                    weights=(4, 3, 3), k=1)[0]
        if goal_resolution_enabled else "direct_lookup"
    )
    from .goal_resolution import LABELS
    if goal_kind == "direct_lookup":
        goal_field, goal_type = rng.choice(DOMAIN_GOALS[domain])
    else:
        goal_field = LABELS[goal_kind]["answer_field"]
        goal_type = LABELS[goal_kind]["answer_type"]
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
        if evidence_tools_enabled:
            if evidence_tools_version == EVIDENCE_TOOL_CONTRACT_VERSION:
                if modality in {"text", "table"}:
                    route = web_acquisition_route(
                        seed=seed, ordinal=ordinal, record_id=record_id,
                        source_id=sid,
                    )
                    item["acquisition_route"] = route
                    if route != "search_only":
                        item["simulated_page_url"] = synthetic_page_url(
                            seed=seed, ordinal=ordinal, record_id=record_id,
                            source_id=sid,
                        )
                else:
                    item["acquisition_route"] = "media_observation"
            item["evidence_access"] = source_acquisition_policy(item)
        sources.append(item)
        schemas[sid] = fields

    def macro(role, parents, output, scoped=()):
        mid = "M" + str(len(tasks) + 1)
        grade = rng.choices(("simple", "moderate", "complex"),
                            weights=(3, 5, 2), k=1)[0]
        lo, hi = {
            "simple": (4, 6), "moderate": (7, 11), "complex": (12, 17)
        }[grade]
        required_rows = (
            candidate_profile["union_candidate_count"]
            if candidate_profile is not None else 0
        )
        if required_rows > hi and grade == "simple":
            grade = "moderate"
            lo, hi = (7, 11)
        if required_rows > hi:
            grade = "complex"
            lo, hi = (12, 17)
        targets = {sid: rng.randint(max(lo, required_rows), hi)
                   for sid in scoped}
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
        if pattern == "corroborated_dispute":
            source("context", "text", {
                "case_id": "str", "case_tag": "str",
            }, "independent log corroborating a disputed case")
            log = macro("dispute_context_candidates", [],
                        {"case_id": "str"}, ["context"])
            prior = macro("dispute_context_fusion", [prior, log],
                          {"case_id": "str"})
        key_field = "case_id"

    # Macro count is sampled independently of each Macro's evidence budget.
    # Hard changes the mix probabilistically, not via a fixed +1 lookup.
    hops = rng.choices((0, 1, 2), weights=(
        (2, 4, 3) if difficulty == "medium" else (1, 3, 5)
    ), k=1)[0]
    if goal_kind != "direct_lookup" and pattern == "corroborated_dispute":
        # Five discovery Macros + one keyed hop + three resolution Macros
        # stays within the existing nine-Macro orchestration budget.
        hops = min(hops, 1)
    for i in range(1, hops + 1):
        new_key = f"record_key_{i}"
        sid = f"index_{i}"
        source(sid, "text", {key_field: "str", new_key: "str"},
               "retrieve a linked reference by an upstream identifier",
               key=key_field)
        prior = macro(f"key_resolution_{i}", [prior],
                      {new_key: "str"}, [sid])
        key_field = new_key
    if goal_kind == "direct_lookup":
        source("archive", "table", {key_field: "str", goal_field: goal_type},
               "confirm the requested detail for a discovered case",
               key=key_field)
        terminal = macro("authoritative_result", [prior],
                         {goal_field: goal_type}, ["archive"])
    else:
        a_field, b_field = (
            ("metric_a", "metric_b") if goal_kind == "comparison"
            else ("completed_count", "total_count")
        )
        source("stat_a", "table", {key_field: "str", a_field: "int"},
               "first independent observation count for identified event",
               key=key_field)
        source("stat_b", "table", {key_field: "str", b_field: "int"},
               "second independent observation count for identified event",
               key=key_field)
        first_count = macro("metric_a_lookup", [prior],
                            {key_field: "str", a_field: "int"}, ["stat_a"])
        second_count = macro("metric_b_lookup", [prior],
                             {key_field: "str", b_field: "int"}, ["stat_b"])
        terminal = macro(
            "goal_resolution", [first_count, second_count],
            {goal_field: goal_type},
        )
    plan = {
        "version": VERSION, "pattern": pattern, "difficulty": difficulty,
        "domain": domain,
        "goal": {
            "answer_field": goal_field,
            "answer_type": goal_type,
            "public_label": (
                goal_field.replace("_", " ") if goal_kind == "direct_lookup"
                else LABELS[goal_kind]["public_label"]
            ),
            "cardinality": "exactly_one",
            "meaning": (
                f"confirmed {goal_field.replace('_', ' ')} for the matching item"
                if goal_kind == "direct_lookup"
                else LABELS[goal_kind]["meaning"]
            ),
        },
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
    if goal_resolution_enabled:
        from .goal_resolution import GOAL_RESOLUTION_VERSION
        plan["goal_resolution"] = {
            "version": GOAL_RESOLUTION_VERSION, "kind": goal_kind,
            "input_source_ids": (
                ["archive"] if goal_kind == "direct_lookup"
                else ["stat_a", "stat_b"]
            ),
        }
    if information_need and pattern in {
        "multimodal_event", "corroborated_dispute",
    }:
        plan["information_necessity_policy"] = (
            "reported_year_disagreement_plus_context_v1"
            if pattern == "corroborated_dispute" else
            "variable_nonunique_candidate_sets_single_intersection_v2"
            if realism_v3 else
            "two_nonunique_candidate_sets_with_single_intersection_v1"
        )
        if candidate_profile is not None:
            plan["candidate_profile"] = candidate_profile
    if operator_grammar_enabled:
        from .operator_grammar import choose_solver_variant
        for task in tasks:
            task["solver_variant"] = choose_solver_variant(
                seed=seed, ordinal=ordinal, record_id=record_id,
                macro_id=task["macro_id"], role=task["role"], clues=plan["clues"],
            )
    return plan, sources


def make_spec(*, seed_record: dict, ordinal: int, run_seed: int,
              difficulty: str, topic: str, record_id: str, version: str,
              operator_grammar_enabled: bool = False) -> dict:
    from .topic_domain import infer_topic_domain, TOPIC_DOMAIN_VERSION
    category_cue = infer_topic_domain(seed_record)
    domain = classify_domain_from_explicit_cue(
        topic, str(seed_record.get("domain_hint", "")), category_cue
    )
    plan, roles = sample_macro_plan(
        seed=run_seed, ordinal=ordinal, record_id=record_id, topic=topic,
        difficulty=difficulty, domain=domain,
        information_need=True,
        realism_v3=True,
        extended_grammar=True,
        goal_resolution_enabled=True,
        evidence_tools_enabled=True,
        evidence_tools_version=EVIDENCE_TOOL_CONTRACT_VERSION,
        operator_grammar_enabled=operator_grammar_enabled,
    )
    from .goal_resolution import GOAL_RESOLUTION_VERSION
    from .semantic_evidence import CONTRACT_VERSION
    from .question_fidelity import QUESTION_FIDELITY_VERSION
    from .source_title_contract import (
        SCENE_REALISM_VERSION, expected_source_titles, DOMAIN_FRAMES,
    )
    _, domain_materials = DOMAIN_FRAMES[domain]
    spec = {
        "version": version,
        "semantic_evidence_contract_version": CONTRACT_VERSION,
        "macro_information_necessity_version": INFORMATION_NECESSITY_VERSION,
        "scene_realism_version": SCENE_REALISM_VERSION,
        "question_fidelity_version": QUESTION_FIDELITY_VERSION,
        "topic_domain_version": TOPIC_DOMAIN_VERSION,
        "macro_grammar_version": GRAMMAR_VERSION,
        "goal_resolution_version": GOAL_RESOLUTION_VERSION,
        "evidence_tool_contract_version": EVIDENCE_TOOL_CONTRACT_VERSION,
        "task_id": f"question-{ordinal:06d}-{digest(record_id, run_seed)[:8]}",
        "scene_id": record_id,
        "source_anchor": {
            "dataset": "MultiModalQA_topic_only",
            "source_record_id": record_id,
            "original_question_reused": False,
            "original_answer_reused": False,
            "original_media_used": False,
            "domain_hint": str(seed_record.get("domain_hint", "")),
            "topic_domain_cue": category_cue,
        },
        "difficulty_target": difficulty,
        "family": "macro_first_v2", "reasoning_pattern": "macro_first_v2",
        "scenario": {
            "topic": topic,
            "setting": f"A source-based investigation of {topic} {domain_materials}",
            "user_role": "research analyst",
            "purpose": f"determine the confirmed {plan['goal']['public_label']}",
            "motivation": "resolve the identity using independent working evidence",
            "entity_kind": "events and records",
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
    if operator_grammar_enabled:
        from .operator_grammar import VERSION as OPERATOR_GRAMMAR_VERSION
        spec["operator_grammar_version"] = OPERATOR_GRAMMAR_VERSION
    title_by_source = expected_source_titles(spec)
    for source in spec["sources"]:
        source["title"] = title_by_source[source["source_id"]]
    return spec


def validate_spec(spec: dict) -> list[str]:
    plan = spec.get("macro_plan")
    if not isinstance(plan, dict) or plan.get("version") != VERSION:
        return ["invalid_macro_first_v2_plan"]
    from .semantic_evidence import CONTRACT_VERSION
    from .operator_grammar import VERSION as OPERATOR_GRAMMAR_VERSION
    if spec.get("operator_grammar_version") not in {
        None, OPERATOR_GRAMMAR_VERSION,
    }:
        return ["invalid_operator_grammar_version"]
    if spec.get("semantic_evidence_contract_version") not in {
        None, CONTRACT_VERSION
    }:
        return ["invalid_semantic_evidence_contract_version"]
    if spec.get("macro_information_necessity_version") not in {
        None, INFORMATION_NECESSITY_VERSION
    }:
        return ["invalid_macro_information_necessity_version"]
    from .source_title_contract import SCENE_REALISM_VERSION
    if spec.get("scene_realism_version") not in {None, SCENE_REALISM_VERSION}:
        return ["invalid_scene_realism_version"]
    from .question_fidelity import QUESTION_FIDELITY_VERSION
    if spec.get("question_fidelity_version") not in {
        None, QUESTION_FIDELITY_VERSION,
    }:
        return ["invalid_question_fidelity_version"]
    from .topic_domain import TOPIC_DOMAIN_VERSION
    if spec.get("topic_domain_version") not in {
        None, TOPIC_DOMAIN_VERSION,
    }:
        return ["invalid_topic_domain_version"]
    if spec.get("macro_grammar_version") not in {None, GRAMMAR_VERSION}:
        return ["invalid_macro_grammar_version"]
    from .goal_resolution import GOAL_RESOLUTION_VERSION
    if spec.get("goal_resolution_version") not in {
        None, GOAL_RESOLUTION_VERSION
    }:
        return ["invalid_goal_resolution_version"]
    if spec.get("evidence_tool_contract_version") not in {
        None, LEGACY_EVIDENCE_TOOL_CONTRACT_VERSION,
        EVIDENCE_TOOL_CONTRACT_VERSION,
    }:
        return ["invalid_evidence_tool_contract_version"]
    try:
        expected, roles = sample_macro_plan(
            seed=spec["seed"], ordinal=spec["ordinal"],
            record_id=spec["source_anchor"]["source_record_id"],
            topic=spec["scenario"]["topic"],
            difficulty=spec["difficulty_target"],
            domain=(
                classify_domain_from_explicit_cue(
                    spec["scenario"]["topic"],
                    spec["source_anchor"].get("domain_hint", ""),
                    spec["source_anchor"].get("topic_domain_cue"),
                ) if (spec.get("topic_domain_version")
                      and spec.get("scene_realism_version")) else
                (classify_domain_from_taxonomy if spec.get("macro_information_necessity_version")
                 else classify_domain_from_metadata if spec.get("semantic_evidence_contract_version")
                 else classify_domain_from_keywords)(
                    spec["scenario"]["topic"],
                    spec["source_anchor"].get("domain_hint", ""),
                )
            ),
            information_need=bool(spec.get("macro_information_necessity_version")),
            realism_v3=bool(spec.get("scene_realism_version")),
            extended_grammar=bool(
                spec.get("macro_grammar_version")
                and spec.get("scene_realism_version")
            ),
            goal_resolution_enabled=bool(spec.get("goal_resolution_version")),
            evidence_tools_enabled=bool(spec.get("evidence_tool_contract_version")),
            evidence_tools_version=spec.get("evidence_tool_contract_version"),
            operator_grammar_enabled=bool(spec.get("operator_grammar_version")),
        )
        errors = []
        if plan != expected:
            errors.append("macro_first_v2_plan_not_reproducible")
        def contract(role):
            # Historical tests intentionally remove version markers from
            # newer specs. Unmarked schemas must ignore new-only metadata.
            return (role.get("source_id"), role.get("modality"),
                    role.get("access_mode"), role.get("key_field"),
                    role.get("evidence_access")
                    if spec.get("evidence_tool_contract_version") else None,
                    role.get("acquisition_route")
                    if spec.get("evidence_tool_contract_version")
                    == EVIDENCE_TOOL_CONTRACT_VERSION else None,
                    role.get("simulated_page_url")
                    if spec.get("evidence_tool_contract_version")
                    == EVIDENCE_TOOL_CONTRACT_VERSION else None)
        if ([contract(role) for role in spec["sources"]]
                != [contract(role) for role in roles]):
            errors.append("macro_first_v2_source_scope_changed")
        if (not plan.get("macro_tasks")
                or plan["terminal_macro_id"] != plan["macro_tasks"][-1]["macro_id"]):
            errors.append("macro_first_v2_terminal_changed")
        if spec.get("scene_realism_version"):
            from .source_title_contract import expected_source_titles
            titles = expected_source_titles(spec)
            if any(source["title"] != titles[source["source_id"]]
                   for source in spec["sources"]):
                errors.append("scene_source_title_not_schema_frozen")
        return errors
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return ["invalid_macro_first_v2_contract:" + str(exc)[:120]]


def compile_macro_solvers(spec: dict) -> dict:
    errors = validate_spec(spec)
    if errors:
        raise ValueError("invalid_macro_first_v2_spec:" + ",".join(errors))
    plan = spec["macro_plan"]
    clues = plan["clues"]
    source_roles = {row["source_id"]: row for row in spec["sources"]}
    has_evidence_contract = bool(spec.get("evidence_tool_contract_version"))
    nodes, groups, outputs = [], [], {}
    prior_key = (
        "case_id" if plan["pattern"] in {
            "disputed_reports", "corroborated_dispute",
        } else "marker_code"
    )
    for task in plan["macro_tasks"]:
        mid, role = task["macro_id"], task["role"]
        variant = task.get("solver_variant", "canonical")
        from .operator_grammar import validate_solver_variant
        validate_solver_variant(role, variant, clues)
        local = []
        def add(op, *inputs, **fields):
            nid = f"{mid.lower()}_op{len(local)+1:02d}"
            node = {"id": nid, "op": op, "inputs": list(inputs), **fields}
            if has_evidence_contract and op in {"scan", "lookup_by_key"}:
                node["evidence_acquisition"] = acquisition_for_operator(
                    node, source_roles[node["source_id"]]
                )
            local.append(node)
            return nid

        parents = [outputs[pid] for pid in task["depends_on"]]
        if role in {"visual_identity", "visual_candidates"}:
            src = add("scan", source_id="visual")
            predicates = (
                [("filter_eq", "emblem_color", clues["emblem_color"])]
                + ([("filter_eq", "emblem_shape", clues["emblem_shape"])]
                   if role == "visual_identity" else [])
                if clues["visual_modality"] == "image" else
                [("filter_eq", "scene_action", clues["scene_action"]),
                 ("filter_gte", "frame_time_ms", clues["time_threshold_ms"])]
            )
            if variant == "reverse_filters":
                predicates.reverse()
            selected = src
            for operator, field, value in predicates:
                selected = add(operator, selected, field=field, value=value)
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
            ordered_parents = (
                list(reversed(parents)) if variant == "right_first"
                else parents
            )
            matched = add("join", *ordered_parents, left_on="candidate_id",
                          right_on="candidate_id")
            last = add("project", matched, fields=["marker_code"])
        elif role == "dispute_resolution":
            ordered_parents = (
                list(reversed(parents)) if variant == "right_first"
                else parents
            )
            matched = add("join", *ordered_parents, left_on="case_id",
                          right_on="case_id")
            selected = add("compare_fields", matched,
                           left_field="reported_year_a",
                           right_field="reported_year_b", comparison="ne")
            last = add("project", selected, fields=["case_id"])
        elif role == "dispute_context_candidates":
            rows = add("scan", source_id="context")
            matching = add("filter_eq", rows, field="case_tag",
                           value=clues["case_tag"])
            last = add("project", matching, fields=["case_id"])
        elif role == "dispute_context_fusion":
            ordered_parents = (
                list(reversed(parents)) if variant == "right_first"
                else parents
            )
            aligned = add("join", *ordered_parents, left_on="case_id",
                          right_on="case_id")
            last = add("project", aligned, fields=["case_id"])
        elif role == "video_audio_alignment":
            # Equivalent relational plans: filter before an inner Join,
            # after the Join, or one branch on either side. These are
            # actual operator-dependency rewrites, not filler operations.
            video = add("scan", source_id="footage")
            video_early = variant in {"canonical", "pushdown_video_only"}
            audio_early = variant in {"canonical", "pushdown_audio_only"}
            event = (
                add("filter_eq", video, field="scene_action",
                    value=clues["scene_action"]) if video_early else video
            )
            audio = add("scan", source_id="commentary")
            spoken = (
                add("filter_eq", audio, field="spoken_cue",
                    value=clues["spoken_cue"]) if audio_early else audio
            )
            matched = add("join", event, spoken, left_on="candidate_id",
                          right_on="candidate_id")
            if not video_early:
                matched = add("filter_eq", matched, field="scene_action",
                              value=clues["scene_action"])
            if not audio_early:
                matched = add("filter_eq", matched, field="spoken_cue",
                              value=clues["spoken_cue"])
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
        elif role in {"metric_a_lookup", "metric_b_lookup"}:
            sid = "stat_a" if role == "metric_a_lookup" else "stat_b"
            kind = plan["goal_resolution"]["kind"]
            metric_field = (
                "metric_a" if sid == "stat_a" else "metric_b"
            ) if kind == "comparison" else (
                "completed_count" if sid == "stat_a" else "total_count"
            )
            acquired = add(
                "lookup_by_key", *parents, source_id=sid,
                input_key_field=prior_key, source_key_field=prior_key,
            )
            last = add("project", acquired,
                       fields=[prior_key, metric_field])
        elif role == "goal_resolution":
            kind = plan["goal_resolution"]["kind"]
            ordered_parents = (
                list(reversed(parents)) if variant == "right_first"
                else parents
            )
            paired = add("join", *ordered_parents, left_on=prior_key,
                         right_on=prior_key)
            if kind == "comparison":
                resolved = add(
                    "derive_comparison", paired, left_field="metric_a",
                    right_field="metric_b",
                    output_field=plan["goal"]["answer_field"],
                )
            elif kind == "calculation":
                resolved = add(
                    "derive_percentage", paired,
                    left_field="completed_count",
                    right_field="total_count",
                    output_field=plan["goal"]["answer_field"],
                )
            else:
                raise ValueError("unsupported_goal_resolution_kind:" + kind)
            last = add("unique", resolved)
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
