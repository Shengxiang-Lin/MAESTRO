"""Frozen Question Specification for the MAESTRO Stage-1 generator.

The original MMQA question, answer and media are NOT input to this spec.
Its scene/topic/source hints are used only to create new fictional tasks.
"""
from __future__ import annotations

import random
import re
from envir.schema import digest

VERSION = "maestro.problem_spec.v4"
RELATION_PATTERNS = (
    "approval", "time_window", "status_disagreement", "dynamic_threshold", "clue_chain", "evidence_graph", "macro_first",
)

FAMILIES = (
    "reconcile",
    "coverage_count",
    "eligibility",
    "priority",
    "chronology",
    "evidence_reconcile",
    "category_distribution",
    "window_alignment",
    "status_disagreement",
    "threshold_eligibility",
    "accession_year",
    "evidence_answer",
    "macro_first",
)
DIFFICULTIES = ("easy", "medium", "hard")

SCENE_PROFILES = {
    "sports": ("roster analyst", "prepare a staffing cross-check", "roster entries"),
    "film": ("production archivist", "reconcile production planning notes", "production entries"),
    "movie": ("production archivist", "reconcile production planning notes", "production entries"),
    "music": ("collection curator", "cross-check catalog annotations", "catalog entries"),
    "book": ("library curator", "reconcile a book collection", "catalog entries"),
    "history": ("archive researcher", "prepare a historical collection inventory", "archive entries"),
    "geography": ("regional researcher", "reconcile a local reference collection", "location entries"),
    "business": ("records coordinator", "reconcile a fictional operations audit", "case entries"),
    "science": ("research coordinator", "reconcile a fictional project register", "research entries"),
    "general": ("records coordinator", "prepare a fictional reference audit", "reference entries"),
}
PURPOSES = (
    "resolve mismatched terminology between independently prepared records",
    "avoid reporting entries that lack the required cross-source confirmation",
    "prepare a reproducible report with traceable source references",
    "identify which tracked entries satisfy the documented review conditions",
    "reconcile multiple partial inventories before finalizing a summary",
    "avoid confusing a source label with an entity identity",
)
MEDIA = ("image", "video", "audio", "text", "table")


def normalize_topic(topic: str) -> str:
    value = " ".join(str(topic or "").split())[:100]
    if not value or "?" in value or "\n" in value:
        raise ValueError("invalid_or_missing_topic")
    return value


def source_profile(domain_hint: str) -> tuple[str, str, str]:
    normalized = (domain_hint or "").lower()
    for key, profile in SCENE_PROFILES.items():
        if key != "general" and key in normalized:
            return profile
    return SCENE_PROFILES["general"]


def _choose_media(seed_record: dict, ordinal: int, rng: random.Random) -> str:
    given = [
        x for x in seed_record.get("source_modalities", [])
        if x in MEDIA
    ]
    if not given:
        given = [
            x.get("modality") for x in seed_record.get("supporting_sources", [])
            if isinstance(x, dict) and x.get("modality") in MEDIA
        ]
    # MMQA only provides image/table/text roles. Permit simulated video/audio
    # as NEW fictional source roles, clearly distinct from MMQA provenance.
    desired = ("image", "video", "audio", "table", "text")[ordinal % 5]
    return desired if ordinal % 3 == 0 else (rng.choice(given) if given else desired)


def make_spec(*, seed_record: dict, ordinal: int, run_seed: int,
              difficulty: str, clue_chain_share: int = 0,
              evidence_graph_share: int = 0,
              macro_first_share: int = 0) -> dict:
    if difficulty not in DIFFICULTIES:
        raise ValueError("difficulty must be easy, medium or hard")
    if (type(clue_chain_share) is not int
            or not 0 <= clue_chain_share <= 100):
        raise ValueError("clue_chain_share must be between 0 and 100")
    if type(evidence_graph_share) is not int or not 0 <= evidence_graph_share <= 100:
        raise ValueError("evidence_graph_share must be between 0 and 100")
    if type(macro_first_share) is not int or not 0 <= macro_first_share <= 100:
        raise ValueError("macro_first_share must be between 0 and 100")
    topic = normalize_topic(seed_record.get("topic", ""))
    record_id = str(seed_record.get("record_id") or seed_record.get("scene_id")
                    or f"topic-{ordinal}")
    rng = random.Random(int(digest(run_seed, ordinal, record_id, "v2")[:15], 16))
    # Select the Macro-level DAG FIRST, before any primitive solver or
    # evidence World is sampled. This opt-in overrides the older
    # primitive-first evidence graph only for selected medium/hard scenes.
    if (difficulty in {"medium", "hard"} and
            int(digest(run_seed, ordinal, "macro-first-mix-v1")[:8], 16) % 100
            < macro_first_share):
        from .macro_first import make_macro_first_spec
        return make_macro_first_spec(
            seed_record=seed_record, ordinal=ordinal, run_seed=run_seed,
            difficulty=difficulty, topic=topic, record_id=record_id,
            version=VERSION,
        )
    # Opt-in seeded graph grammar. Legacy specs and clue-chain remain
    # byte-identical when this flag is zero.
    if (difficulty in {"medium", "hard"} and
            int(digest(run_seed, ordinal, "evidence-graph-mix-v1")[:8], 16) % 100
            < evidence_graph_share):
        from .graph_grammar import make_evidence_spec
        return make_evidence_spec(
            seed_record=seed_record, ordinal=ordinal, run_seed=run_seed,
            difficulty=difficulty, topic=topic, record_id=record_id,
            version=VERSION,
        )
    # Semantic families change the actual relations/predicates, not only
    # the execution order. Easy remains a one-source independent task.
    # Opt-in: a bounded share of MEDIUM tasks uses real keyed evidence
    # acquisition, not static scan/join computation. No old mode changes.
    keyed = (
        difficulty == "medium"
        and int(digest(run_seed, ordinal, "clue-mix-v1")[:8], 16) % 100
        < clue_chain_share
    )
    pattern = (
        "clue_chain" if keyed else
        "approval" if difficulty == "easy" else
        ("approval", "time_window", "status_disagreement", "dynamic_threshold")[
            (ordinal + run_seed) % 4
        ]
    )
    legacy_families = FAMILIES[:7]
    family = (
        "accession_year" if keyed else
        legacy_families[(ordinal + run_seed) % len(legacy_families)]
        if pattern == "approval" else
        {
            "time_window": "window_alignment",
            "status_disagreement": "status_disagreement",
            "dynamic_threshold": "threshold_eligibility",
            "clue_chain": "accession_year",
            "evidence_graph": "evidence_answer",
            "macro_first": "macro_first",
        }[pattern]
    )
    if keyed:
        from .clue_chain import clue_features
        feature = clue_features(ordinal)
        return {
            "version": VERSION,
            "task_id": f"question-{ordinal:06d}-{digest(record_id, run_seed)[:8]}",
            "scene_id": record_id,
            "source_anchor": {
                "dataset": "MultiModalQA_topic_only" if seed_record.get("record_id")
                           else "scene_bank_topic_only",
                "source_record_id": record_id,
                "original_question_reused": False,
                "original_answer_reused": False,
                "original_media_used": False,
            },
            "difficulty_target": "medium",
            "family": "accession_year",
            "reasoning_pattern": "clue_chain",
            "macro_plan_variant": 0,
            "scenario": {
                "topic": topic,
                "setting": (
                    f"a fictional exhibit catalog dedicated to {topic}"
                ),
                "user_role": "collection researcher",
                "purpose": (
                    "identify an exhibit from a pictured visual clue "
                    "and retrieve its accession year"
                ),
                "motivation": (
                    "the illustrated emblem is the only starting clue "
                    "to locate the archival accession record"
                ),
                "entity_kind": "exhibits",
            },
            "sources": [
                {
                    "source_id": "illustration", "modality": "image",
                    "title": f"{topic} exhibit illustration",
                    "purpose": (
                        "simulated visual features and pictured marker codes"
                    ),
                    "access_mode": "browse",
                },
                {
                    "source_id": "index", "modality": "text",
                    "title": f"{topic} exhibit marker index",
                    "purpose": (
                        "lookup an accession identifier using the observed "
                        "marker code; records are keyed, not browseable"
                    ),
                    "access_mode": "keyed_lookup",
                },
                {
                    "source_id": "archive", "modality": "table",
                    "title": f"{topic} accession archive",
                    "purpose": (
                        "lookup an accession year by the identifier returned "
                        "from the marker index; records are keyed"
                    ),
                    "access_mode": "keyed_lookup",
                },
            ],
            "investigation": {
                "visual_features": feature,
                "acquisition_chain": [
                    {"source_id": "illustration", "mode": "browse",
                     "produces": "marker_code"},
                    {"source_id": "index", "mode": "keyed_lookup",
                     "input_key": "marker_code", "produces": "accession_id"},
                    {"source_id": "archive", "mode": "keyed_lookup",
                     "input_key": "accession_id", "produces": "accession_year"},
                ],
            },
            "requirements": {
                "must_resolve_aliases": False,
                "must_match_independent_reviews": False,
                "must_check_independent_clearance": False,
                "answer_kind": "accession_year",
                "must_preserve_source_provenance": True,
                "simulation_only": True,
            },
            "seed": run_seed,
            "ordinal": ordinal,
        }
    if difficulty == "easy" and family == "category_distribution":
        family = "coverage_count"
    role, purpose, entity_kind = source_profile(seed_record.get("domain_hint", ""))
    media = _choose_media(seed_record, ordinal, rng)
    scenario = {
        "topic": topic,
        "setting": f"a fictional reference collection organized around {topic}",
        "user_role": role,
        "purpose": purpose,
        "motivation": PURPOSES[(ordinal + run_seed * 3) % len(PURPOSES)],
        "entity_kind": entity_kind,
    }
    source_roles = [{
        "source_id": "register",
        "modality": "table",
        "title": f"{topic} working register",
        "purpose": "the tracked entries, their abbreviated labels, and recorded status",
    }]
    if difficulty != "easy":
        source_roles += [
            {"source_id": "glossary", "modality": "text",
             "title": f"{topic} terminology crosswalk",
             "purpose": "translate source-specific codes to shared labels"},
            {"source_id": "annotations", "modality": media,
             "title": f"{topic} independent annotation record",
             "purpose": "review judgments, sequence and category descriptions"},
        ]
    if pattern != "approval":
        semantics = {
            "time_window": (
                "tracked events with recorded relative occurrence steps",
                "map event codes to simulated sessions",
                "simulated event windows and their time boundaries",
            ),
            "status_disagreement": (
                "tracked entries and their reported binary status",
                "resolve status codes to their record groups",
                "independently reported binary status by group",
            ),
            "dynamic_threshold": (
                "tracked entries with numerical measured quantities",
                "translate policy codes to policy groups",
                "independent per-group minimum acceptable values",
            ),
        }[pattern]
        for entry, meaning in zip(source_roles, semantics):
            entry["purpose"] = meaning
    if difficulty == "hard":
        source_roles.append({
            "source_id": "clearance", "modality": "text",
            "title": f"{topic} clearance notes",
            "purpose": "independent entry-specific clearance conditions",
        })
    return {
        "version": VERSION,
        "task_id": f"question-{ordinal:06d}-{digest(record_id, run_seed)[:8]}",
        "scene_id": record_id,
        "source_anchor": {
            "dataset": "MultiModalQA_topic_only" if seed_record.get("record_id")
                       else "scene_bank_topic_only",
            "source_record_id": record_id,
            "original_question_reused": False,
            "original_answer_reused": False,
            "original_media_used": False,
        },
        "difficulty_target": difficulty,
        "family": family,
        "reasoning_pattern": pattern,
        # An executable alternative join/filter dependency plan; this
        # affects the MacroGraph, NOT just names or prompt wording.
        "macro_plan_variant": int(digest(
            run_seed, ordinal, record_id, "macro-graph-plan-v1"
        )[:8], 16) % 4,
        "scenario": scenario,
        "sources": source_roles,
        "requirements": {
            "must_resolve_aliases": difficulty != "easy",
            "must_match_independent_reviews": difficulty != "easy",
            "must_check_independent_clearance": difficulty == "hard",
            "answer_kind": (
                "count" if family == "coverage_count" else
                "category_counts" if family == "category_distribution" else
                "single_entry" if family in {"priority", "chronology"} else
                "entry_evidence" if family == "evidence_reconcile" else
                "entry_list"
            ),
            "must_preserve_source_provenance": True,
            "simulation_only": True,
        },
        "seed": run_seed,
        "ordinal": ordinal,
    }


def spec_issues(spec: dict) -> list[str]:
    errors = []
    if spec.get("version") != VERSION:
        errors.append("invalid_problem_version")
    if spec.get("difficulty_target") not in DIFFICULTIES:
        errors.append("invalid_difficulty")
    if spec.get("family") not in FAMILIES:
        errors.append("invalid_family")
    if spec.get("version") == VERSION:
        pattern = spec.get("reasoning_pattern")
        mapped = {
            "time_window": "window_alignment",
            "status_disagreement": "status_disagreement",
            "dynamic_threshold": "threshold_eligibility",
            "clue_chain": "accession_year",
            "evidence_graph": "evidence_answer",
        }
        if pattern not in RELATION_PATTERNS:
            errors.append("invalid_reasoning_pattern")
        elif pattern == "approval":
            if spec.get("family") not in FAMILIES[:7]:
                errors.append("approval_family_mismatch")
        elif spec.get("family") != mapped[pattern]:
            errors.append("reasoning_pattern_family_mismatch")
    plan_variant = spec.get("macro_plan_variant")
    if plan_variant is not None and (
        type(plan_variant) is not int or plan_variant not in range(4)
    ):
        errors.append("invalid_macro_plan_variant")
    scenario = spec.get("scenario", {})
    for key in ("topic", "setting", "user_role", "purpose", "motivation", "entity_kind"):
        if not isinstance(scenario.get(key), str) or not scenario[key].strip():
            errors.append("missing_scenario_" + key)
    sources = spec.get("sources", [])
    keyed = spec.get("reasoning_pattern") == "clue_chain"
    evidence = spec.get("reasoning_pattern") == "evidence_graph"
    macro_first = spec.get("reasoning_pattern") == "macro_first"
    n = (len(sources) if (evidence or macro_first) else 3 if keyed else
         {"easy": 1, "medium": 3, "hard": 4}.get(
             spec.get("difficulty_target"), -1))
    if len(sources) != n:
        errors.append("wrong_source_count")
    ids = [s.get("source_id") for s in sources]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_source_id")
    expected_ids = ([s.get("source_id") for s in sources]
                    if (evidence or macro_first) else
                    ["illustration", "index", "archive"] if keyed else ["register"])
    if not keyed and not (evidence or macro_first) and spec.get("difficulty_target") in {"medium", "hard"}:
        expected_ids.extend(["glossary", "annotations"])
    if not keyed and not (evidence or macro_first) and spec.get("difficulty_target") == "hard":
        expected_ids.append("clearance")
    if ids != expected_ids:
        errors.append("source_role_contract_mismatch")
    needs = spec.get("requirements", {})
    expected_answer = (
        "macro_value" if macro_first else
        "accession_year" if (keyed or evidence) else
        "count" if spec.get("family") == "coverage_count"
        else "category_counts" if spec.get("family") == "category_distribution"
        else "single_entry" if spec.get("family") in {"priority", "chronology"}
        else "entry_evidence" if (
            spec.get("version") == VERSION
            and spec.get("family") == "evidence_reconcile"
        ) else "entry_list"
    )
    if needs.get("answer_kind") != expected_answer:
        errors.append("answer_kind_contract_mismatch")
    required_aliases = False if (keyed or evidence or macro_first) else spec.get("difficulty_target") != "easy"
    if needs.get("must_resolve_aliases") is not required_aliases:
        errors.append("alias_requirement_contract_mismatch")
    if needs.get("must_match_independent_reviews") is not required_aliases:
        errors.append("review_requirement_contract_mismatch")
    if needs.get("must_check_independent_clearance") is not (
        spec.get("difficulty_target") == "hard" and not (keyed or evidence or macro_first)
    ):
        errors.append("clearance_requirement_contract_mismatch")
    if needs.get("must_preserve_source_provenance") is not True:
        errors.append("provenance_requirement_missing")
    for source in sources:
        if source.get("modality") not in MEDIA:
            errors.append("unsupported_modality")
        for key in ("source_id", "title", "purpose"):
            if not isinstance(source.get(key), str) or not source[key].strip():
                errors.append("source_missing_" + key)
    if keyed:
        if spec.get("difficulty_target") != "medium":
            errors.append("clue_chain_requires_medium")
        if [(role.get("access_mode")) for role in sources] != [
            "browse", "keyed_lookup", "keyed_lookup"
        ]:
            errors.append("clue_chain_access_mode_contract")
        investigation = spec.get("investigation", {})
        from .clue_chain import clue_features
        if investigation.get("visual_features") != clue_features(
            spec.get("ordinal", 0)
        ):
            errors.append("clue_chain_visual_feature_mismatch")
        path = investigation.get("acquisition_chain", [])
        if [(x.get("source_id"), x.get("mode"), x.get("input_key"),
             x.get("produces")) for x in path if isinstance(x, dict)] != [
            ("illustration", "browse", None, "marker_code"),
            ("index", "keyed_lookup", "marker_code", "accession_id"),
            ("archive", "keyed_lookup", "accession_id", "accession_year"),
        ]:
            errors.append("clue_chain_acquisition_path_mismatch")
    if macro_first:
        from .macro_first import validate_macro_first_spec
        errors.extend(validate_macro_first_spec(spec))
    if evidence:
        from .graph_grammar import validate_evidence_spec
        errors.extend(validate_evidence_spec(spec))
    if spec.get("requirements", {}).get("simulation_only") is not True:
        errors.append("non_simulated_spec")
    return sorted(set(errors))


def difficulty_sequence(*, target: int, mix: tuple[int, int, int],
                        run_seed: int) -> list[str]:
    """Largest-remainder allocation with no silent target-count drift."""
    if target <= 0 or len(mix) != 3 or any(x < 0 for x in mix) or sum(mix) != 100:
        raise ValueError("mix must be three nonnegative percentages summing to 100")
    floats = [target * x / 100 for x in mix]
    counts = [int(n) for n in floats]
    missing = target - sum(counts)
    order = sorted(range(3), key=lambda i: (-(floats[i] - counts[i]), i))
    for index in order[:missing]:
        counts[index] += 1
    sequence = [d for d, count in zip(DIFFICULTIES, counts)
                for _ in range(count)]
    random.Random(run_seed + 881).shuffle(sequence)
    return sequence
