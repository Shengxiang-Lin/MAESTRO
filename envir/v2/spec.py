"""Frozen ProblemSpec for the independently generated MAESTRO V2 pipeline.

The original MMQA question, answer and media are NOT input to this spec.
Its scene/topic/source hints are used only to create new fictional tasks.
"""
from __future__ import annotations

import random
import re
from envir.schema import digest

VERSION = "maestro.problem_spec.v2"

FAMILIES = (
    "reconcile",
    "coverage_count",
    "eligibility",
    "priority",
    "chronology",
    "evidence_reconcile",
    "category_distribution",
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
              difficulty: str) -> dict:
    if difficulty not in DIFFICULTIES:
        raise ValueError("difficulty must be easy, medium or hard")
    topic = normalize_topic(seed_record.get("topic", ""))
    record_id = str(seed_record.get("record_id") or seed_record.get("scene_id")
                    or f"topic-{ordinal}")
    rng = random.Random(int(digest(run_seed, ordinal, record_id, "v2")[:15], 16))
    family = FAMILIES[(ordinal + run_seed) % len(FAMILIES)]
    if difficulty == "easy" and family == "category_distribution":
        family = "coverage_count"  # no cross-source category key in Easy
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
    if difficulty == "hard":
        source_roles.append({
            "source_id": "clearance", "modality": "text",
            "title": f"{topic} clearance notes",
            "purpose": "independent entry-specific clearance conditions",
        })
    return {
        "version": VERSION,
        "task_id": f"v2-{ordinal:06d}-{digest(record_id, run_seed)[:8]}",
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
    scenario = spec.get("scenario", {})
    for key in ("topic", "setting", "user_role", "purpose", "motivation", "entity_kind"):
        if not isinstance(scenario.get(key), str) or not scenario[key].strip():
            errors.append("missing_scenario_" + key)
    sources = spec.get("sources", [])
    n = {"easy": 1, "medium": 3, "hard": 4}.get(
        spec.get("difficulty_target"), -1
    )
    if len(sources) != n:
        errors.append("wrong_source_count")
    ids = [s.get("source_id") for s in sources]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_source_id")
    for source in sources:
        if source.get("modality") not in MEDIA:
            errors.append("unsupported_modality")
        for key in ("source_id", "title", "purpose"):
            if not isinstance(source.get(key), str) or not source[key].strip():
                errors.append("source_missing_" + key)
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
