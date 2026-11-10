"""Configuration for the only supported MAESTRO Stage-1 question generator."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")

SECTIONS = {
    "input": {"mmqa", "max_records"},
    "generation": {"output_dir", "target", "seed", "difficulty_mix",
                   "overwrite", "no_progress", "flush_every", "clue_chain_share",
                   "evidence_graph_share", "macro_first_share",
                   "macro_first_v2_share"},
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
    for section, value in (("scenario", scenario), ("question", question)):
        if not value.get("api_base") or not value.get("model"):
            raise ValueError(f"{section}.api_base and {section}.model are required")
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
    clue_chain_share = generation.get("clue_chain_share", 0)
    if (type(clue_chain_share) is not int
            or not 0 <= clue_chain_share <= 100):
        raise ValueError("generation.clue_chain_share must be 0..100")
    evidence_graph_share = generation.get("evidence_graph_share", 0)
    if (type(evidence_graph_share) is not int
            or not 0 <= evidence_graph_share <= 100):
        raise ValueError("generation.evidence_graph_share must be 0..100")
    macro_first_share = generation.get("macro_first_share", 0)
    if type(macro_first_share) is not int or not 0 <= macro_first_share <= 100:
        raise ValueError("generation.macro_first_share must be 0..100")
    quota_refill = generation.get("quota_refill", False)
    if type(quota_refill) is not bool:
        raise ValueError("generation.quota_refill must be a boolean")
    max_candidate_multiplier = generation.get("max_candidate_multiplier", 3)
    if (type(max_candidate_multiplier) is not int
            or not 1 <= max_candidate_multiplier <= 10):
        raise ValueError("generation.max_candidate_multiplier must be 1..10")
    macro_first_v2_share = generation.get("macro_first_v2_share", 0)
    if type(macro_first_v2_share) is not int or not 0 <= macro_first_v2_share <= 100:
        raise ValueError("generation.macro_first_v2_share must be 0..100")
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
        clue_chain_share=clue_chain_share,
        evidence_graph_share=evidence_graph_share,
        macro_first_share=macro_first_share,
        macro_first_v2_share=macro_first_v2_share,
        quota_refill=quota_refill,
        max_candidate_multiplier=max_candidate_multiplier,
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
