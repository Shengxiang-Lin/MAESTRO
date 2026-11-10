"""Strict, root-relative YAML configuration for untrusted round branches."""
from __future__ import annotations

from pathlib import Path
import os
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "environment" / "configs" / "untrusted_orchestration.yaml"

_REQUIRED = {
    "input": {"scenes", "reports", "base_episodes"},
    "generation": {
        "output_dir", "max_scenes", "branches_per_scene", "seed",
        "fault_modes", "persistence", "multi_target_probability",
        "max_rounds", "max_cost", "max_retries", "max_parallel_tasks",
        "workers", "mode", "max_revisions", "sub_model", "model_alias",
        "overwrite", "no_progress",
    },
    "model": {
        "author_api_base", "author_model", "critic_api_base",
        "critic_model", "temperature", "author_max_tokens",
        "critic_max_tokens", "api_key_env",
    },
}


def _path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("untrusted_path_must_be_nonempty")
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    config_file = Path(path).expanduser().resolve()
    if not config_file.is_file():
        raise FileNotFoundError("untrusted_config_not_found:" + str(config_file))
    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("untrusted_config_requires_version_1")
    if set(data) != {"version", *set(_REQUIRED)}:
        raise ValueError("untrusted_config_invalid_sections")
    for section, allowed in _REQUIRED.items():
        actual = data[section]
        if not isinstance(actual, dict):
            raise ValueError("untrusted_config_section_not_mapping:" + section)
        if set(actual) - allowed:
            raise ValueError("untrusted_unknown_config_key:" + section)
    source, run, model = data["input"], data["generation"], data["model"]
    config = {
        "scenes_path": _path(source.get("scenes")),
        "reports_path": _path(source.get("reports")),
        "base_episodes_path": _path(source.get("base_episodes")),
        "output_dir": _path(run.get("output_dir")),
        "max_scenes": run.get("max_scenes"),
        "branches_per_scene": run.get("branches_per_scene", 1),
        "seed": run.get("seed", 17),
        "fault_modes": run.get("fault_modes", ["missing", "partial", "wrong"]),
        "persistence": run.get("persistence", "once"),
        "multi_target_probability": run.get("multi_target_probability", 0.0),
        "max_rounds": run.get("max_rounds", 16),
        "max_cost": run.get("max_cost", 80),
        "max_retries": run.get("max_retries", 1),
        "max_parallel_tasks": run.get("max_parallel_tasks", 4),
        "workers": run.get("workers", 4),
        "mode": run.get("mode", "template"),
        "max_revisions": run.get("max_revisions", 2),
        "sub_model": run.get("sub_model", "qwen3-8b"),
        "model_alias": run.get("model_alias", "model_1"),
        "overwrite": run.get("overwrite", False),
        "no_progress": run.get("no_progress", False),
        "author_api_base": model.get("author_api_base"),
        "author_model": model.get("author_model"),
        "critic_api_base": model.get("critic_api_base"),
        "critic_model": model.get("critic_model"),
        "temperature": model.get("temperature", 0.3),
        "author_max_tokens": model.get("author_max_tokens", 2800),
        "critic_max_tokens": model.get("critic_max_tokens", 1200),
        "api_key_env": model.get("api_key_env", "OPENAI_API_KEY"),
    }
    for key in ("branches_per_scene", "max_rounds", "max_cost",
                "max_parallel_tasks", "workers", "author_max_tokens",
                "critic_max_tokens"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("untrusted_invalid_" + key)
    for key in ("seed", "max_retries", "max_revisions"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError("untrusted_invalid_" + key)
    if config["max_scenes"] is not None and (
        type(config["max_scenes"]) is not int or config["max_scenes"] < 1
    ):
        raise ValueError("untrusted_invalid_max_scenes")
    for key in ("overwrite", "no_progress"):
        if type(config[key]) is not bool:
            raise ValueError("untrusted_invalid_" + key)
    if config["mode"] not in {"llm", "template"}:
        raise ValueError("untrusted_mode_requires_template_or_llm")
    if config["persistence"] not in {"once", "same_path"}:
        raise ValueError("untrusted_persistence_invalid")
    modes = config["fault_modes"]
    if (not isinstance(modes, list) or not modes
            or len(set(modes)) != len(modes)
            or not set(modes) <= {"missing", "partial", "wrong"}):
        raise ValueError("untrusted_invalid_fault_modes")
    p = config["multi_target_probability"]
    if type(p) not in (int, float) or not 0 <= p <= 1:
        raise ValueError("untrusted_invalid_multi_target_probability")
    if config["workers"] > 64 or config["max_rounds"] > 64:
        raise ValueError("untrusted_worker_or_round_limit")
    for key in ("sub_model", "model_alias"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError("untrusted_missing_" + key)
    if config["mode"] == "llm":
        for key in ("author_api_base", "author_model"):
            if not isinstance(config[key], str) or not config[key].strip():
                raise ValueError("untrusted_missing_" + key)
    return config


def generate_from_config(
    path: str | Path = DEFAULT_CONFIG, **overrides,
) -> dict:
    cfg = load_config(path)
    for key, value in overrides.items():
        if key not in cfg:
            raise ValueError("unknown_untrusted_override:" + key)
        if value is not None:
            cfg[key] = value
    for key in ("scenes_path", "reports_path",
                "base_episodes_path", "output_dir"):
        cfg[key] = _path(cfg[key])
    from environment.generate_question.backend import OpenAICompatibleBackend
    from .generate_turns import generate_turns

    author = critic = None
    if cfg["mode"] == "llm":
        api_key = (os.getenv(cfg["api_key_env"])
                   if cfg["api_key_env"] else None)
        author = OpenAICompatibleBackend(
            base_url=cfg["author_api_base"],
            model=cfg["author_model"], api_key=api_key,
            temperature=cfg["temperature"],
            max_tokens=cfg["author_max_tokens"],
        )
        critic = OpenAICompatibleBackend(
            base_url=cfg["critic_api_base"] or cfg["author_api_base"],
            model=cfg["critic_model"] or cfg["author_model"],
            api_key=api_key, temperature=0.0,
            max_tokens=cfg["critic_max_tokens"],
        )
    for key in ("mode", "author_api_base", "critic_api_base",
                "temperature", "author_max_tokens",
                "critic_max_tokens", "api_key_env"):
        cfg.pop(key)
    return generate_turns(
        **cfg, author=author, critic=critic,
        report_author=author, report_critic=critic,
    )
