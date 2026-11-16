"""Frozen Question Specification for the MAESTRO Stage-1 generator.

The original MMQA question, answer and media are NOT input to this spec.
Its scene/topic/source hints are used only to create new fictional tasks.
"""
from __future__ import annotations

import random
from environment.schema import digest

VERSION = "maestro.problem_spec.v4"
RELATION_PATTERNS = ("approval", "macro_first_v2", "typed_knowledge_v1", "typed_knowledge_v2", "typed_knowledge_v3")

FAMILIES = (
    "reconcile", "coverage_count", "eligibility", "priority",
    "chronology", "evidence_reconcile", "category_distribution",
    "macro_first_v2", "typed_knowledge_v1", "typed_knowledge_v2", "typed_knowledge_v3",
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


def make_spec(*, seed_record: dict, ordinal: int, run_seed: int,
              difficulty: str, macro_first_v2_share: int = 100,
              operator_grammar_share: int = 100,
              numeric_goal_diversity_share: int = 100,
              question_semantic_lock_share: int = 100) -> dict:
    """Canonical Stage 1: Easy approval and Medium/Hard Macro-first V2."""
    if difficulty not in DIFFICULTIES:
        raise ValueError("difficulty must be easy, medium or hard")
    if type(macro_first_v2_share) is not int or macro_first_v2_share != 100:
        raise ValueError("macro_planning_share_must_be_100")
    for name, share in (
        ("operator_grammar_share", operator_grammar_share),
        ("numeric_goal_diversity_share", numeric_goal_diversity_share),
        ("question_semantic_lock_share", question_semantic_lock_share),
    ):
        if type(share) is not int or not 0 <= share <= 100:
            raise ValueError(name + " must be between 0 and 100")
    topic = normalize_topic(seed_record.get("topic", ""))
    record_id = str(seed_record.get("record_id") or seed_record.get("scene_id")
                    or f"topic-{ordinal}")
    if difficulty != "easy":
        from .macro_planning import make_spec as make_macro_v2_spec
        return make_macro_v2_spec(
            seed_record=seed_record, ordinal=ordinal, run_seed=run_seed,
            difficulty=difficulty, topic=topic, record_id=record_id,
            version=VERSION,
            operator_grammar_enabled=(
                int(digest(run_seed, ordinal, record_id,
                           "typed-operator-grammar-mix-v1")[:8], 16) % 100
                < operator_grammar_share
            ),
            numeric_goal_diversity_enabled=(
                int(digest(run_seed, ordinal, record_id,
                           "numeric-goal-diversity-mix-v1")[:8], 16) % 100
                < numeric_goal_diversity_share
            ),
            question_semantic_lock_enabled=(
                int(digest(run_seed, ordinal, record_id,
                           "question-semantic-lock-mix-v1")[:8], 16) % 100
                < question_semantic_lock_share
            ),
        )
    # Easy: preserve the existing one-source semantic and answer contract.
    family = FAMILIES[:7][(ordinal + run_seed) % 7]
    if family == "category_distribution":
        family = "coverage_count"
    role, purpose, entity_kind = source_profile(seed_record.get("domain_hint", ""))
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
        "difficulty_target": "easy",
        "family": family, "reasoning_pattern": "approval",
        "macro_plan_variant": int(digest(
            run_seed, ordinal, record_id, "macro-graph-plan-v1"
        )[:8], 16) % 4,
        "scenario": {
            "topic": topic,
            "setting": f"a fictional reference collection organized around {topic}",
            "user_role": role, "purpose": purpose,
            "motivation": PURPOSES[(ordinal + run_seed * 3) % len(PURPOSES)],
            "entity_kind": entity_kind,
        },
        "sources": [{
            "source_id": "register", "modality": "table",
            "title": f"{topic} working register",
            "purpose": "the tracked entries, their abbreviated labels, and recorded status",
        }],
        "requirements": {
            "must_resolve_aliases": False,
            "must_match_independent_reviews": False,
            "must_check_independent_clearance": False,
            "answer_kind": (
                "count" if family == "coverage_count" else
                "single_entry" if family in {"priority", "chronology"} else
                "entry_evidence" if family == "evidence_reconcile" else "entry_list"
            ),
            "must_preserve_source_provenance": True,
            "simulation_only": True,
        },
        "seed": run_seed, "ordinal": ordinal,
    }


def spec_issues(spec: dict) -> list[str]:
    """Reject old graph generators; preserve current source and output contracts."""
    errors = []
    if spec.get("version") != VERSION:
        errors.append("invalid_problem_version")
    difficulty = spec.get("difficulty_target")
    if difficulty not in DIFFICULTIES:
        errors.append("invalid_difficulty")
    family = spec.get("family")
    if family not in FAMILIES:
        errors.append("invalid_family")
    pattern = spec.get("reasoning_pattern")
    if pattern not in RELATION_PATTERNS:
        errors.append("invalid_reasoning_pattern")
    if pattern == "approval" and (difficulty != "easy" or family not in FAMILIES[:7]):
        errors.append("approval_requires_easy")
    if pattern == "macro_first_v2" and (
        difficulty not in {"medium", "hard"} or family != "macro_first_v2"
    ):
        errors.append("macro_first_v2_requires_medium_or_hard")
    if pattern in {"typed_knowledge_v1", "typed_knowledge_v2", "typed_knowledge_v3"} and (
        difficulty not in {"medium", "hard"} or family != pattern
    ):
        errors.append("typed_knowledge_requires_medium_or_hard")
    variant = spec.get("macro_plan_variant")
    if variant is not None and (type(variant) is not int or variant not in range(4)):
        errors.append("invalid_macro_plan_variant")
    scenario = spec.get("scenario", {})
    for key in ("topic", "setting", "user_role", "purpose", "motivation", "entity_kind"):
        if not isinstance(scenario.get(key), str) or not scenario[key].strip():
            errors.append("missing_scenario_" + key)
    sources = spec.get("sources", [])
    ids = [source.get("source_id") for source in sources]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_source_id")
    if pattern == "approval" and ids != ["register"]:
        errors.append("source_role_contract_mismatch")
    if pattern == "macro_first_v2" and not sources:
        errors.append("wrong_source_count")
    needs = spec.get("requirements", {})
    expected_answer = (
        "macro_value" if pattern in {"macro_first_v2", "typed_knowledge_v1", "typed_knowledge_v2", "typed_knowledge_v3"} else
        "count" if family == "coverage_count" else
        "category_counts" if family == "category_distribution" else
        "single_entry" if family in {"priority", "chronology"} else
        "entry_evidence" if family == "evidence_reconcile" else "entry_list"
    )
    if needs.get("answer_kind") != expected_answer:
        errors.append("answer_kind_contract_mismatch")
    for field in ("must_resolve_aliases", "must_match_independent_reviews",
                  "must_check_independent_clearance"):
        if needs.get(field) is not False:
            errors.append(field + "_contract_mismatch")
    if needs.get("must_preserve_source_provenance") is not True:
        errors.append("provenance_requirement_missing")
    for source in sources:
        if source.get("modality") not in MEDIA:
            errors.append("unsupported_modality")
        for key in ("source_id", "title", "purpose"):
            if not isinstance(source.get(key), str) or not source[key].strip():
                errors.append("source_missing_" + key)
    if pattern == "macro_first_v2":
        from .macro_planning import validate_spec
        errors.extend(validate_spec(spec))
    if pattern == "typed_knowledge_v3":
        from .hybrid_world import validate_spec
        errors.extend(validate_spec(spec))
    if pattern == "typed_knowledge_v2":
        from .knowledge_macro_v2 import validate_spec
        errors.extend(validate_spec(spec))
    if pattern == "typed_knowledge_v1":
        from .knowledge_macro import validate_spec
        errors.extend(validate_spec(spec))
    if needs.get("simulation_only") is not True:
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
