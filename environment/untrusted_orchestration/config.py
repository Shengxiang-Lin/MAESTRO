"""Strict, root-relative YAML configuration for untrusted round branches."""
from __future__ import annotations

from pathlib import Path
import os
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "environment" / "configs" / "untrusted_orchestration.yaml"

_REQUIRED = {
    "input": {"scenes", "reports", "clean_episodes"},
    "generation": {
        "output_dir", "max_scenes", "branches_per_scene",
        "round_sampling_strategy", "max_post_fork_rounds", "seed",
        "max_rounds", "max_cost", "max_retries", "max_parallel_tasks",
        "workers", "mode", "report_max_revisions", "main_max_revisions",
        "sub_model", "model_alias", "overwrite", "no_progress",
        "atomic_fault_kinds", "max_candidates_per_round",
        "max_distractors_per_fault", "max_selected_per_round",
        "selection_strategy", "max_diagnostics_per_scene",
        "max_language_attempts_per_round",
    },
    "model": {
        "author_api_base", "author_model", "critic_api_base",
        "critic_model", "temperature", "author_max_tokens",
        "critic_max_tokens", "api_key_env",
    },
    "subagent_model": {
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
    report_model = data["subagent_model"]
    config = {
        "scenes_path": _path(source.get("scenes")),
        "reports_path": _path(source.get("reports")),
        "clean_episodes_path": (
            _path(source["clean_episodes"])
            if source.get("clean_episodes") else None
        ),
        "output_dir": _path(run.get("output_dir")),
        "max_scenes": run.get("max_scenes"),
        "branches_per_scene": run.get("branches_per_scene"),
        "round_sampling_strategy": run.get(
            "round_sampling_strategy", "chronological"
        ),
        "max_post_fork_rounds": run.get("max_post_fork_rounds"),
        "seed": run.get("seed", 17),
        "fault_modes": ["missing", "partial", "wrong"],
        "scenario_mode": "event_driven",
        "candidate_strategy": "enumerate",
        "max_candidates_per_round": run.get(
            "max_candidates_per_round", 24
        ),
        "max_distractors_per_fault": run.get(
            "max_distractors_per_fault", 2
        ),
        "max_selected_per_round": run.get(
            "max_selected_per_round", 1
        ),
        "selection_strategy": run.get(
            "selection_strategy", "causal_rank"
        ),
        "max_diagnostics_per_scene": run.get(
            "max_diagnostics_per_scene", 0
        ),
        "max_language_attempts_per_round": run.get(
            "max_language_attempts_per_round", 1
        ),
        "atomic_fault_kinds": run.get("atomic_fault_kinds", [
            "no_result", "incomplete_retrieval", "wrong_retrieval",
            "wrong_entity_binding", "report_missing",
            "result_summary_conflict", "fabricated_claim",
        ]),
        "max_fault_atoms": 1,
        "delayed_evidence_probability": 0.0,
        "persistence": "once",
        "multi_target_probability": 0.0,
        "max_rounds": run.get("max_rounds", 16),
        "max_cost": run.get("max_cost", 80),
        "max_retries": run.get("max_retries", 1),
        "max_parallel_tasks": run.get("max_parallel_tasks", 4),
        "workers": run.get("workers", 4),
        "mode": run.get("mode", "template"),
        "max_revisions": 2,
        "report_max_revisions": run.get("report_max_revisions", 2),
        "main_max_revisions": run.get("main_max_revisions", 2),
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
        "report_author_api_base": report_model.get("author_api_base"),
        "report_author_model": report_model.get("author_model"),
        "report_critic_api_base": report_model.get("critic_api_base"),
        "report_critic_model": report_model.get("critic_model"),
        "report_temperature": report_model.get("temperature", 0.3),
        "report_author_max_tokens": report_model.get("author_max_tokens", 2800),
        "report_critic_max_tokens": report_model.get("critic_max_tokens", 1200),
        "report_api_key_env": report_model.get("api_key_env", "OPENAI_API_KEY"),
    }
    # None means every accepted Main feedback round in every Question.
    # The optional integer cap is diagnostic/backwards-compatible only.
    if config["branches_per_scene"] is not None and (
        type(config["branches_per_scene"]) is not int
        or config["branches_per_scene"] < 1
    ):
        raise ValueError("untrusted_invalid_round_debug_cap")
    if config["round_sampling_strategy"] not in {
        "chronological", "stratified",
    }:
        raise ValueError("untrusted_invalid_round_sampling_strategy")
    if config["max_post_fork_rounds"] is not None and (
        type(config["max_post_fork_rounds"]) is not int
        or config["max_post_fork_rounds"] < 1
    ):
        raise ValueError("untrusted_invalid_max_post_fork_rounds")
    for key in ("max_rounds", "max_cost",
                "max_parallel_tasks", "workers", "author_max_tokens",
                "critic_max_tokens", "report_author_max_tokens",
                "report_critic_max_tokens"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("untrusted_invalid_" + key)
    for key in (
        "seed", "max_retries", "max_revisions",
        "report_max_revisions", "main_max_revisions",
    ):
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
    from .atomic_faults import ATOM_TO_MODE
    strategy = config["candidate_strategy"]
    if strategy not in {"sample_one", "enumerate"}:
        raise ValueError("untrusted_invalid_candidate_strategy")
    if strategy == "enumerate" and config["scenario_mode"] != "event_driven":
        raise ValueError("untrusted_enumeration_requires_event_driven")
    if config["selection_strategy"] not in {
        "causal_rank", "stratified",
    }:
        raise ValueError("untrusted_invalid_selection_strategy")
    if (type(config["max_diagnostics_per_scene"]) is not int
            or not 0 <= config["max_diagnostics_per_scene"] <= 8):
        raise ValueError("untrusted_invalid_max_diagnostics_per_scene")
    for field, minimum, maximum in (
        ("max_candidates_per_round", 1, 128),
        ("max_distractors_per_fault", 1, 10),
        ("max_selected_per_round", 1, 8),
        ("max_language_attempts_per_round", 1, 8),
    ):
        value = config[field]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError("untrusted_invalid_" + field)
    if config["candidate_strategy"] == "enumerate" and (
        config["max_language_attempts_per_round"] <
        config["max_selected_per_round"]
    ):
        raise ValueError("untrusted_language_budget_smaller_than_selection")
    if config["scenario_mode"] not in {"round_fork", "event_driven"}:
        raise ValueError("untrusted_invalid_scenario_mode")
    kinds = config["atomic_fault_kinds"]
    if (not isinstance(kinds, list) or not kinds
            or any(not isinstance(k, str) for k in kinds)
            or len(set(kinds)) != len(kinds)
            or not set(kinds) <= set(ATOM_TO_MODE)):
        raise ValueError("untrusted_invalid_atomic_fault_kinds")
    if (type(config["max_fault_atoms"]) is not int
            or not 1 <= config["max_fault_atoms"] <= 2):
        raise ValueError("untrusted_invalid_max_fault_atoms")
    delay_probability = config["delayed_evidence_probability"]
    if (type(delay_probability) not in (int, float)
            or not 0 <= delay_probability <= 1):
        raise ValueError("untrusted_invalid_delayed_evidence_probability")
    p = config["multi_target_probability"]
    if type(p) not in (int, float) or not 0 <= p <= 1:
        raise ValueError("untrusted_invalid_multi_target_probability")
    if config["workers"] > 64 or config["max_rounds"] > 64:
        raise ValueError("untrusted_worker_or_round_limit")
    for key in ("sub_model", "model_alias"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError("untrusted_missing_" + key)
    main_mode = report_mode = config["mode"]
    for mode, keys in (
        (main_mode, ("author_api_base", "author_model")),
        (report_mode, ("report_author_api_base", "report_author_model")),
    ):
        if mode == "llm":
            for key in keys:
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
    for key in ("scenes_path", "reports_path", "output_dir"):
        cfg[key] = _path(cfg[key])

    from environment.generate_question.authoring.backend import OpenAICompatibleBackend
    from .generate_turns import generate_turns

    main_mode = report_mode = cfg["mode"]
    author = critic = report_author = report_critic = None

    if main_mode == "llm":
        api_key = os.getenv(cfg["api_key_env"]) if cfg["api_key_env"] else None
        author = OpenAICompatibleBackend(
            base_url=cfg["author_api_base"],
            model=cfg["author_model"],
            api_key=api_key, temperature=cfg["temperature"],
            max_tokens=cfg["author_max_tokens"],
        )
        critic = OpenAICompatibleBackend(
            base_url=cfg["critic_api_base"] or cfg["author_api_base"],
            model=cfg["critic_model"] or cfg["author_model"],
            api_key=api_key, temperature=0.0,
            max_tokens=cfg["critic_max_tokens"],
        )

    if report_mode == "llm":
        api_key = (os.getenv(cfg["report_api_key_env"])
                   if cfg["report_api_key_env"] else None)
        report_author = OpenAICompatibleBackend(
            base_url=cfg["report_author_api_base"],
            model=cfg["report_author_model"],
            api_key=api_key,
            temperature=cfg["report_temperature"],
            max_tokens=cfg["report_author_max_tokens"],
        )
        report_critic = OpenAICompatibleBackend(
            base_url=(
                cfg["report_critic_api_base"]
                or cfg["report_author_api_base"]
            ),
            model=(
                cfg["report_critic_model"]
                or cfg["report_author_model"]
            ),
            api_key=api_key, temperature=0.0,
            max_tokens=cfg["report_critic_max_tokens"],
        )

    # Pass only executable generation parameters to the runner. Model
    # endpoint secrets and local Author/Critic clients never enter public
    # main_turns.jsonl or private fault-alignment metadata.
    for key in (
        "mode",
        "author_api_base", "critic_api_base", "temperature",
        "author_max_tokens", "critic_max_tokens", "api_key_env",
        "report_author_api_base", "report_critic_api_base",
        "report_temperature", "report_author_max_tokens",
        "report_critic_max_tokens", "report_api_key_env",
    ):
        cfg.pop(key)
    return generate_turns(
        **cfg, author=author, critic=critic,
        report_author=report_author, report_critic=report_critic,
        main_mode=main_mode, report_mode=report_mode,
    )
