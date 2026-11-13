"""Configuration-driven trusted Main-Agent trajectory generation.

This module owns only configuration and model wiring. The trusted trajectory
builder remains independent so a future untrusted orchestration workflow can
reuse the same Scene/report contracts without importing this CLI.
"""
from __future__ import annotations

from pathlib import Path
import os

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "environment" / "configs" / "base_orchestration.yaml"

_FIELDS = {
    "input": {"scenes", "reports"},
    "generation": {
        "output_dir", "max_scenes", "workers", "mode", "max_revisions",
        "max_attempts", "sub_model", "model_alias", "overwrite",
        "no_progress", "verbose_progress", "freeze_dispatch",
    },
    "model": {
        "author_api_base", "author_model", "critic_api_base",
        "critic_model", "api_key_env", "temperature",
        "author_max_tokens", "critic_max_tokens",
    },
}


def _path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("trajectory_config_path_must_be_nonempty")
    result = Path(value).expanduser()
    return result if result.is_absolute() else ROOT / result


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    file = Path(path).expanduser().resolve()
    if not file.is_file():
        raise FileNotFoundError("trajectory_config_not_found:" + str(file))
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError("trajectory_invalid_yaml:" + str(file)) from exc
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("trajectory_config_requires_version_1")
    if set(data) != {"version", *list(_FIELDS)}:
        raise ValueError("trajectory_config_sections_invalid")
    for section, allowed in _FIELDS.items():
        entry = data[section]
        if not isinstance(entry, dict):
            raise ValueError("trajectory_config_section_not_mapping:" + section)
        unknown = set(entry) - allowed
        if unknown:
            raise ValueError("trajectory_config_unknown_" + section +
                             ":" + ",".join(sorted(unknown)))

    incoming, options, models = (
        data["input"], data["generation"], data["model"]
    )
    config = {
        "scenes_path": _path(incoming.get("scenes")),
        "reports_path": _path(incoming.get("reports")),
        "output_dir": _path(options.get("output_dir")),
        "max_scenes": options.get("max_scenes"),
        "workers": options.get("workers", 4),
        "mode": options.get("mode", "llm"),
        "max_revisions": options.get("max_revisions", 2),
        "max_attempts": options.get("max_attempts", 16),
        "sub_model": options.get("sub_model", "qwen3-8b"),
        "model_alias": options.get("model_alias", "model_1"),
        "overwrite": options.get("overwrite", False),
        "no_progress": options.get("no_progress", False),
        "verbose_progress": options.get("verbose_progress", False),
        "freeze_dispatch": options.get("freeze_dispatch", False),
        "author_api_base": models.get("author_api_base"),
        "author_model": models.get("author_model"),
        "critic_api_base": models.get("critic_api_base"),
        "critic_model": models.get("critic_model"),
        "api_key_env": models.get("api_key_env", "OPENAI_API_KEY"),
        "temperature": models.get("temperature", 0.4),
        "author_max_tokens": models.get("author_max_tokens", 2800),
        "critic_max_tokens": models.get("critic_max_tokens", 1200),
    }
    if config["mode"] not in {"llm", "template"}:
        raise ValueError("trajectory_mode_must_be_llm_or_template")
    for key in ("workers", "max_attempts"):
        if type(config[key]) is not int or config[key] < (
                2 if key == "max_attempts" else 1):
            raise ValueError("trajectory_invalid_" + key)
    if config["workers"] > 64:
        raise ValueError("trajectory_workers_above_64")
    if (type(config["max_revisions"]) is not int
            or not 0 <= config["max_revisions"] <= 8):
        raise ValueError("trajectory_invalid_max_revisions")
    if (config["max_scenes"] is not None and (
            type(config["max_scenes"]) is not int
            or config["max_scenes"] < 1)):
        raise ValueError("trajectory_invalid_max_scenes")
    for key in ("overwrite", "no_progress", "verbose_progress",
                "freeze_dispatch"):
        if type(config[key]) is not bool:
            raise ValueError("trajectory_invalid_boolean_" + key)
    for key in ("sub_model", "model_alias"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError("trajectory_missing_" + key)
    temperature = config["temperature"]
    if (type(temperature) not in (int, float)
            or not 0 <= temperature <= 2):
        raise ValueError("trajectory_invalid_temperature")
    for key in ("author_max_tokens", "critic_max_tokens"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("trajectory_invalid_" + key)
    if (config["api_key_env"] is not None
            and (not isinstance(config["api_key_env"], str)
                 or not config["api_key_env"].strip())):
        raise ValueError("trajectory_invalid_api_key_env")
    if config["mode"] == "llm":
        for key in ("author_api_base", "author_model"):
            if (not isinstance(config[key], str)
                    or not config[key].strip()):
                raise ValueError("trajectory_missing_" + key)
    return config


def generate_from_config(path: str | Path = DEFAULT_CONFIG,
                         **overrides) -> dict:
    """Load a complete run from YAML; supplied CLI overrides are optional."""
    options = load_config(path)
    for key, value in overrides.items():
        if key not in options:
            raise ValueError("trajectory_unknown_override:" + key)
        if value is not None:
            options[key] = value
    # Resolve overrides with the same root-relative semantics as YAML paths.
    for key in ("scenes_path", "reports_path", "output_dir"):
        options[key] = _path(str(options[key]))
    from environment.generate_question.backend import OpenAICompatibleBackend
    from .generate_turns import generate_turns

    author = critic = None
    if options["mode"] == "llm":
        api_key = (os.getenv(options["api_key_env"])
                   if options["api_key_env"] else None)
        author = OpenAICompatibleBackend(
            base_url=options["author_api_base"],
            model=options["author_model"], api_key=api_key,
            temperature=options["temperature"],
            max_tokens=options["author_max_tokens"],
        )
        critic = OpenAICompatibleBackend(
            base_url=options["critic_api_base"] or options["author_api_base"],
            model=options["critic_model"] or options["author_model"],
            api_key=api_key, temperature=0.0,
            max_tokens=options["critic_max_tokens"],
        )
    return generate_turns(
        scenes_path=options["scenes_path"],
        reports_path=options["reports_path"],
        output_dir=options["output_dir"],
        max_scenes=options["max_scenes"],
        workers=options["workers"],
        author=author, critic=critic,
        max_revisions=options["max_revisions"],
        max_attempts=options["max_attempts"],
        sub_model=options["sub_model"],
        model_alias=options["model_alias"],
        author_model=options["author_model"] if author else None,
        critic_model=(options["critic_model"] or options["author_model"])
                     if critic else None,
        overwrite=options["overwrite"],
        no_progress=options["no_progress"],
        verbose_progress=options["verbose_progress"],
        freeze_dispatch=options["freeze_dispatch"],
    )
