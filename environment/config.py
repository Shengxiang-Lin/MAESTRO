"""Validated configuration for the canonical Stage 1 generator."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "questions.yaml"

SECTIONS = {
    "input": {"max_records"},
    "generation": {
        "output_dir", "target", "seed", "difficulty_mix", "overwrite",
        "no_progress", "flush_every", "knowledge_yago",
        "knowledge_max_triples", "knowledge_max_edges",
        "knowledge_min_hops", "knowledge_max_hops",
    },
    "scenario": {
        "api_base", "model", "critic_api_base", "critic_model",
        "api_key_env", "critic_api_key_env", "workers",
        "max_revisions", "timeout",
    },
    "question": {
        "registry", "api_base", "model", "critic_api_base", "critic_model",
        "api_key_env", "critic_api_key_env", "workers",
        "max_revisions", "timeout",
    },
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
        raise ValueError("Stage 1 configuration requires version: 1")
    if set(payload) != {"version", *SECTIONS}:
        raise ValueError("Config must contain version, input, generation, "
                         "scenario and question")
    for section, allowed in SECTIONS.items():
        values = payload[section]
        if not isinstance(values, dict):
            raise ValueError("config." + section + " must be a mapping")
        unknown = set(values) - allowed
        if unknown:
            raise ValueError("Unsupported config." + section +
                             " field(s): " + repr(sorted(unknown)))
    return payload


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _resolve_question_registry(data: dict) -> tuple[dict, dict, dict]:
    """Resolve only the selected remote API models from the shared registry.

    Stage-1 config owns model selection and secret *environment variable names*.
    The central registry owns base URLs and sampling parameters. Existing
    explicit endpoints remain supported when registry is omitted.
    """
    values = dict(data)
    registry = values.get("registry")
    if not registry:
        return values, {}, {}
    path = _path(registry)
    if not path.is_file():
        raise FileNotFoundError("question_model_registry_missing:" + str(path))
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        raise ValueError("question_model_registry_invalid")
    models = payload["models"]

    def resolve(model_field: str, endpoint_field: str) -> dict:
        name = values.get(model_field)
        if not name:
            return {}
        entry = models.get(name)
        if not isinstance(entry, dict):
            raise ValueError("question_model_not_in_registry:" + str(name))
        if entry.get("backend") != "api" or entry.get("api_type") != "openai":
            raise ValueError("question_registry_model_not_openai_api:" + str(name))
        base = entry.get("base_url")
        if not isinstance(base, str) or not base.startswith(("https://", "http://")):
            raise ValueError("question_model_registry_missing_base_url:" + str(name))
        if not values.get(endpoint_field):
            values[endpoint_field] = base
        return entry

    author = resolve("model", "api_base")
    critic = resolve("critic_model", "critic_api_base")
    if not values.get("critic_model") and author:
        critic = author
    return values, author, critic


def _client_settings(name: str, data: dict, *, required: bool):
    if bool(data.get("api_base")) != bool(data.get("model")):
        raise ValueError(name + "_api_base_and_model_must_be_paired")
    if required and not data.get("model"):
        raise ValueError(name + "_model_required")
    for field in ("workers", "timeout"):
        if type(data.get(field)) is not int or data[field] < 1:
            raise ValueError(name + "." + field + "_must_be_positive")
    revisions = data.get("max_revisions")
    if type(revisions) is not int or not 0 <= revisions <= 8:
        raise ValueError(name + ".max_revisions_out_of_range")


def args_from_config(config: dict, *, output_dir=None, target=None,
                     no_progress=False) -> argparse.Namespace:
    generation = config["generation"]
    scenario = config["scenario"]
    question, author_model_config, critic_model_config = (
        _resolve_question_registry(config["question"])
    )
    _client_settings("scenario", scenario, required=False)
    _client_settings("question", question, required=False)

    max_records = config["input"].get("max_records")
    chosen_target = target if target is not None else generation.get("target")
    if (type(chosen_target) is not int or chosen_target < 1 or
            type(max_records) is not int or max_records < chosen_target):
        raise ValueError("target_must_be_positive_and_at_most_max_records")
    difficulty = generation.get("difficulty_mix")
    if (not isinstance(difficulty, list) or len(difficulty) != 3
            or any(type(x) is not int or x < 0 for x in difficulty)
            or sum(difficulty) != 100 or difficulty[0] != 0):
        raise ValueError("difficulty_mix_must_be_[0,medium,hard]_percentages")
    if type(generation.get("seed")) is not int:
        raise ValueError("seed_must_be_an_integer")
    flush = generation.get("flush_every", 1)
    if type(flush) is not int or flush < 1:
        raise ValueError("flush_every_must_be_positive")
    knowledge = generation.get("knowledge_yago")
    if not isinstance(knowledge, str) or not knowledge:
        raise ValueError("knowledge_yago_required")
    for field in ("knowledge_max_triples", "knowledge_max_edges"):
        value = generation.get(field, 1000000)
        if type(value) is not int or value < 1:
            raise ValueError(field + "_must_be_positive")
    lo = generation.get("knowledge_min_hops", 2)
    hi = generation.get("knowledge_max_hops", 3)
    if type(lo) is not int or type(hi) is not int or not 2 <= lo <= hi <= 4:
        raise ValueError("invalid_relation_hops")
    directory = output_dir or generation.get("output_dir")
    if not directory:
        raise ValueError("output_dir_required")

    return argparse.Namespace(
        scenes=None, output_dir=_path(directory), target=chosen_target,
        max_records=max_records, seed=generation["seed"],
        difficulty_mix=",".join(map(str, difficulty)),
        overwrite=bool(generation.get("overwrite", False)),
        flush_every=flush,
        knowledge_yago=_path(knowledge),
        knowledge_max_triples=generation.get("knowledge_max_triples", 1000000),
        knowledge_max_edges=generation.get("knowledge_max_edges", 250000),
        knowledge_min_hops=lo, knowledge_max_hops=hi,
        no_progress=no_progress or bool(generation.get("no_progress", False)),
        scenario_api_base=scenario.get("api_base"),
        scenario_model=scenario.get("model"),
        scenario_critic_api_base=scenario.get("critic_api_base"),
        scenario_critic_model=scenario.get("critic_model"),
        scenario_api_key_env=scenario.get("api_key_env"),
        scenario_critic_api_key_env=scenario.get("critic_api_key_env"),
        scenario_workers=scenario["workers"],
        scenario_max_revisions=scenario["max_revisions"],
        scenario_timeout=scenario["timeout"],
        question_api_base=question.get("api_base"),
        question_model=question.get("model"),
        question_critic_api_base=question.get("critic_api_base"),
        question_critic_model=question.get("critic_model"),
        question_api_key_env=question.get("api_key_env"),
        question_critic_api_key_env=question.get("critic_api_key_env"),
        question_author_temperature=author_model_config.get("temperature"),
        question_author_top_p=author_model_config.get("top_p"),
        question_critic_temperature=critic_model_config.get("temperature"),
        question_critic_top_p=critic_model_config.get("top_p"),
        question_workers=question["workers"],
        max_revisions=question["max_revisions"],
        question_timeout=question["timeout"],
    )
