"""Independent YAML configuration for MAESTRO Stage 2 Subtask execution oracle."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
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
    if set(data) != {"version", "implementation"}:
        raise ValueError("Stage-2 config must contain version and implementation")
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
