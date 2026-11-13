"""YAML settings for trusted Sub-Agent evidence and report generation."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "subagent_reports.yaml"
_ALLOWED = {
    "scenes", "output_dir", "max_records", "evidence_limit",
    "strict_replay", "no_progress", "overwrite",
}


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Stage-2 config not found: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid Stage-2 YAML: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Stage-2 config requires version: 1")
    if not {"version", "implementation"} <= set(data) or (
        set(data) - {"version", "implementation", "reporting"}
    ):
        raise ValueError(
            "Stage-2 config supports version, implementation, and reporting"
        )
    values = data["implementation"]
    if not isinstance(values, dict):
        raise ValueError("implementation must be a mapping")
    unknown = set(values) - _ALLOWED
    if unknown:
        raise ValueError(f"Unknown Stage-2 config keys: {sorted(unknown)}")
    if not values.get("scenes") or not values.get("output_dir"):
        raise ValueError("implementation.scenes and output_dir required")
    limit = values.get("max_records")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("max_records must be null or positive int")
    count = values.get("evidence_limit", 12)
    if type(count) is not int or count < 0:
        raise ValueError("evidence_limit must be a nonnegative integer")
    for name in ("strict_replay", "no_progress", "overwrite"):
        if type(values.get(name, False)) is not bool:
            raise ValueError(f"{name} must be a boolean")
    return {
        "scenes_path": _resolve(values["scenes"]),
        "output_dir": _resolve(values["output_dir"]),
        "max_records": limit,
        "evidence_limit": count,
        "strict_replay": values.get("strict_replay", True),
        "no_progress": values.get("no_progress", False),
        "overwrite": values.get("overwrite", False),
    }


_REPORTING_FIELDS = {
    "source_subtasks", "source_scene_index", "output_dir",
    "api_base", "model", "critic_api_base", "critic_model",
    "api_key_env", "critic_api_key_env", "workers",
    "max_revisions", "timeout", "max_scenes", "no_progress", "overwrite", "flush_every", "reuse_from",
    "result_policy",
}


def load_reporting_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Stage-2 config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Stage-2 config requires version: 1")
    values = data.get("reporting")
    if not isinstance(values, dict):
        raise ValueError("Stage-2 reporting section required")
    unknown = set(values) - _REPORTING_FIELDS
    if unknown:
        raise ValueError(f"Unknown reporting keys: {sorted(unknown)}")
    for name in ("source_subtasks", "source_scene_index", "output_dir",
                 "api_base", "model"):
        if not values.get(name):
            raise ValueError(f"reporting.{name} required")
    for name in ("workers", "timeout"):
        value = values.get(name)
        if type(value) is not int or value < 1:
            raise ValueError(f"reporting.{name} must be positive")
    revisions = values.get("max_revisions")
    if type(revisions) is not int or revisions not in range(9):
        raise ValueError("reporting.max_revisions must be between 0 and 8")
    maximum = values.get("max_scenes")
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise ValueError("reporting.max_scenes must be null or positive integer")
    for field in ("no_progress", "overwrite"):
        if type(values.get(field, False)) is not bool:
            raise ValueError(f"reporting.{field} must be boolean")
    policy = values.get("result_policy", "llm_authored")
    if policy not in {"llm_authored", "code_owned"}:
        raise ValueError("reporting.result_policy must be llm_authored or code_owned")
    flush_every = values.get("flush_every", 1)
    if type(flush_every) is not int or flush_every < 1:
        raise ValueError("reporting.flush_every must be positive")
    return {
        "flush_every": flush_every,
        "result_policy": policy,
        "reuse_from": (
            _resolve(values["reuse_from"])
            if values.get("reuse_from") else None
        ),
        "source_subtasks": _resolve(values["source_subtasks"]),
        "source_scene_index": _resolve(values["source_scene_index"]),
        "output_dir": _resolve(values["output_dir"]),
        "api_base": values["api_base"],
        "model": values["model"],
        "critic_api_base": values.get("critic_api_base"),
        "critic_model": values.get("critic_model"),
        "api_key_env": values.get("api_key_env"),
        "critic_api_key_env": values.get("critic_api_key_env"),
        "workers": values["workers"],
        "max_revisions": revisions,
        "timeout": values["timeout"],
        "max_scenes": maximum,
        "no_progress": values.get("no_progress", False),
        "overwrite": values.get("overwrite", False),
    }
