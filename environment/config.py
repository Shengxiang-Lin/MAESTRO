"""YAML settings for question generation and deterministic Replay."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "questions.yaml"

SECTIONS = {
    "input": {"mmqa", "max_records"},
    "generation": {"output_dir", "target", "seed", "difficulty_mix",
                   "overwrite", "no_progress", "flush_every",
                   "macro_planning_share",
                   "operator_grammar_share",
                   "numeric_goal_diversity_share",
                   "question_semantic_lock_share",
                   "knowledge_yago", "knowledge_max_triples",
                   "knowledge_max_edges", "knowledge_min_hops",
                   "knowledge_max_hops", "knowledge_variant"},
    "scenario": {"api_base", "model", "critic_api_base", "critic_model",
                 "api_key_env", "critic_api_key_env", "workers",
                 "max_revisions", "timeout"},
    "question": {"api_base", "model", "critic_api_base", "critic_model",
                 "api_key_env", "workers", "max_revisions", "timeout"},
}


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    location = Path(path).expanduser().resolve()
    if not location.is_file():
        raise FileNotFoundError(f"Generator config not found: {location}")
    try:
        payload = yaml.safe_load(location.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {location}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Configuration must be a mapping with version: 1")
    if set(payload) != {"version", *SECTIONS}:
        raise ValueError("Configuration must contain exactly: version, input, "
                         "generation, scenario, question")
    for section, allowed in SECTIONS.items():
        values = payload[section]
        if not isinstance(values, dict):
            raise ValueError(f"config.{section} must be a mapping")
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown config.{section} key(s): {sorted(unknown)}")
    return payload


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def args_from_config(config: dict, *, output_dir=None, target=None,
                     no_progress=False) -> argparse.Namespace:
    data = config["input"]
    generation = config["generation"]
    scenario = config["scenario"]
    question = config["question"]
    if not data.get("mmqa"):
        raise ValueError("input.mmqa is required")
    knowledge = generation.get("knowledge_yago")
    for section, value in (("scenario", scenario), ("question", question)):
        # Knowledge-mode dry runs can disable LLM naturalization, while
        # existing production configurations retain their strict settings.
        if not knowledge and (not value.get("api_base") or not value.get("model")):
            raise ValueError(f"{section}.api_base and {section}.model are required")
        if knowledge and bool(value.get("api_base")) != bool(value.get("model")):
            raise ValueError(f"{section}.api_base_and_model_must_be_paired")
        for field in ("workers", "timeout"):
            if not isinstance(value.get(field), int) or value[field] < 1:
                raise ValueError(f"{section}.{field} must be a positive integer")
        revisions = value.get("max_revisions")
        if not isinstance(revisions, int) or not 0 <= revisions <= 8:
            raise ValueError(f"{section}.max_revisions must be between 0 and 8")
    mix = generation.get("difficulty_mix")
    if (not isinstance(mix, list) or len(mix) != 3
            or any(type(n) is not int or n < 0 for n in mix)
            or sum(mix) != 100):
        raise ValueError("generation.difficulty_mix must be three integer "
                         "percentages summing to 100")
    chosen_target = target if target is not None else generation.get("target")
    max_records = data.get("max_records")
    if (type(chosen_target) is not int or chosen_target < 1
            or type(max_records) is not int or max_records < chosen_target):
        raise ValueError("generation.target must be positive and <= input.max_records")
    seed = generation.get("seed")
    if type(seed) is not int:
        raise ValueError("generation.seed must be an integer")
    flush_every = generation.get("flush_every", 1)
    if type(flush_every) is not int or flush_every < 1:
        raise ValueError("generation.flush_every must be a positive integer")
    macro_first_v2_share = generation.get("macro_planning_share", 100)
    if type(macro_first_v2_share) is not int or macro_first_v2_share != 100:
        raise ValueError("generation.macro_planning_share must be 100 for the supported Stage-1 pipeline")
    operator_grammar_share = generation.get("operator_grammar_share", 0)
    if (type(operator_grammar_share) is not int
            or not 0 <= operator_grammar_share <= 100):
        raise ValueError("generation.operator_grammar_share must be 0..100")
    if operator_grammar_share and not macro_first_v2_share:
        raise ValueError("operator_grammar_requires_macro_planning")
    numeric_goal_diversity_share = generation.get(
        "numeric_goal_diversity_share", 0
    )
    question_semantic_lock_share = generation.get(
        "question_semantic_lock_share", 0
    )
    for name, share in (
        ("numeric_goal_diversity_share", numeric_goal_diversity_share),
        ("question_semantic_lock_share", question_semantic_lock_share),
    ):
        if type(share) is not int or not 0 <= share <= 100:
            raise ValueError("generation." + name + " must be 0..100")
        if share and not macro_first_v2_share:
            raise ValueError(name + "_requires_macro_planning")
    if knowledge:
        if generation.get("knowledge_variant", "v1") not in {"v1", "v2", "v3", "v4", "graph_first"}:
            raise ValueError("unsupported_knowledge_variant")
        if (type(generation.get("knowledge_max_triples", 1000000)) is not int
            or generation.get("knowledge_max_triples", 1000000) < 1
            or type(generation.get("knowledge_max_edges", 250000)) is not int
            or generation.get("knowledge_max_edges", 250000) < 1):
            raise ValueError("invalid_knowledge_yago_resource_limits")
        lo = generation.get("knowledge_min_hops", 2)
        hi = generation.get("knowledge_max_hops", 3)
        if not (type(lo) is int and type(hi) is int and 1 <= lo <= hi <= 5):
            raise ValueError("invalid_knowledge_relation_hops")
        if mix[0] != 0:
            raise ValueError("knowledge_v1_currently_requires_no_easy_tasks")
    directory = output_dir or generation.get("output_dir")
    if not directory:
        raise ValueError("generation.output_dir is required")
    return argparse.Namespace(
        mmqa=_path(data["mmqa"]), scenes=None,
        output_dir=_path(directory), target=chosen_target,
        max_records=max_records, seed=seed,
        difficulty_mix=",".join(map(str, mix)),
        overwrite=bool(generation.get("overwrite", False)),
        flush_every=flush_every,
        macro_first_v2_share=macro_first_v2_share,
        operator_grammar_share=operator_grammar_share,
        numeric_goal_diversity_share=numeric_goal_diversity_share,
        question_semantic_lock_share=question_semantic_lock_share,
        knowledge_yago=_path(knowledge) if knowledge else None,
        knowledge_variant=generation.get("knowledge_variant", "v1"),
        knowledge_max_triples=generation.get("knowledge_max_triples", 1000000),
        knowledge_max_edges=generation.get("knowledge_max_edges", 250000),
        knowledge_min_hops=generation.get("knowledge_min_hops", 2),
        knowledge_max_hops=generation.get("knowledge_max_hops", 3),
        no_progress=no_progress or bool(generation.get("no_progress", False)),
        scenario_api_base=scenario["api_base"],
        scenario_model=scenario["model"],
        scenario_critic_api_base=scenario.get("critic_api_base"),
        scenario_critic_model=scenario.get("critic_model"),
        scenario_api_key_env=scenario.get("api_key_env"),
        scenario_critic_api_key_env=scenario.get("critic_api_key_env"),
        scenario_workers=scenario["workers"],
        scenario_max_revisions=scenario["max_revisions"],
        scenario_timeout=scenario["timeout"],
        question_api_base=question["api_base"],
        question_model=question["model"],
        question_critic_api_base=question.get("critic_api_base"),
        question_critic_model=question.get("critic_model"),
        question_api_key_env=question.get("api_key_env"),
        question_workers=question["workers"],
        max_revisions=question["max_revisions"],
        question_timeout=question["timeout"],
    )
