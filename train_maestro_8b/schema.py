"""Data contracts for learnable MAESTRO Main-Agent training."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable

from orchestra.modes import MODE_TO_INDEX, ORCHESTRATION_MODES

ORCH_TOKEN = "<|orch|>"


def _require_text(record: Dict[str, Any], key: str) -> str:
    value = str(record.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def validate_stage1_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one SFT + macro-supervision sample.

    Public Main input must be an Orchestra-o1-equivalent prompt. Privileged
    simulator labels may exist elsewhere in the source trajectory but must not
    be copied into ``input``.
    """
    item = dict(record)
    item["input"] = _require_text(item, "input")
    item["target_json"] = _require_text(item, "target_json")
    macro = str(item.get("macro_label") or "").strip().lower()
    if macro not in MODE_TO_INDEX:
        raise ValueError(
            f"macro_label must be one of {ORCHESTRATION_MODES}, got {macro!r}"
        )
    item["macro_label"] = macro
    try:
        parsed = json.loads(item["target_json"])
    except json.JSONDecodeError as exc:
        raise ValueError("target_json must contain valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("target_json must decode to a JSON object")
    if parsed.get("action") not in {"delegate_task", "complete"}:
        raise ValueError("target_json.action must be delegate_task or complete")
    return item


def validate_critic_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one partial/full action-value target.

    ``macro_values`` may contain only the actions actually rolled out. Missing
    modes are masked rather than treated as zero-value counterfactuals.
    """
    item = dict(record)
    item["input"] = _require_text(item, "input")
    raw_values = item.get("macro_values")
    if not isinstance(raw_values, dict) or not raw_values:
        raise ValueError("macro_values must be a non-empty mapping")
    values: Dict[str, float] = {}
    for mode, value in raw_values.items():
        mode = str(mode).strip().lower()
        if mode not in MODE_TO_INDEX:
            raise ValueError(f"unknown macro mode: {mode}")
        values[mode] = float(value)
    item["macro_values"] = values
    return item


def macro_vector(
    macro_values: Dict[str, float],
    *,
    fill: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Convert a sparse macro-value mapping into target and mask vectors."""
    targets = [float(fill)] * len(ORCHESTRATION_MODES)
    mask = [0.0] * len(ORCHESTRATION_MODES)
    for mode, value in macro_values.items():
        index = MODE_TO_INDEX[mode]
        targets[index] = float(value)
        mask[index] = 1.0
    return targets, mask


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield item


__all__ = [
    "ORCH_TOKEN",
    "validate_stage1_record",
    "validate_critic_record",
    "macro_vector",
    "iter_jsonl",
]
