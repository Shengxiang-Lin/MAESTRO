"""Stage-2 reward selection.

DEFAULT is deterministic, private-state evidence orchestration quality.
Legacy Teacher Action text-matching remains explicitly opt-in only for
comparison/debugging; neither function performs future environment rollouts.
"""
from __future__ import annotations
from difflib import SequenceMatcher
import importlib
import json


def parse_action(text):
    text = text.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        parts = text.splitlines()
        if len(parts) >= 3 and parts[-1].startswith(fence):
            text = "\n".join(parts[1:-1])
    try:
        result = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(result, dict) or result.get("action") not in ("delegate_task", "complete"):
        return None
    if not isinstance(result.get("params"), dict):
        return None
    return result


def teacher_proxy_reward(candidate, record):
    """A limited reference imitation score; NEVER interpret as recovery success."""
    action = parse_action(candidate)
    if action is None:
        return 0.0
    ref = record.get("reference_action") or {}
    score = 0.1
    if action["action"] != ref.get("action"):
        return score
    score += 0.2
    if action["action"] == "complete":
        predicted = str(action["params"].get("answer", "")).strip().casefold()
        target = str(ref.get("params", {}).get("answer", "")).strip().casefold()
        return score + 0.7 * float(bool(predicted) and predicted == target)
    tasks = action["params"].get("tasks")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 8:
        return score
    if not all(isinstance(t, dict) and
               isinstance(t.get("task_instruction"), str) and
               len(t["task_instruction"].strip()) >= 8 and
               isinstance(t.get("tools"), list) for t in tasks):
        return score
    score += 0.2
    expected = ref.get("params", {}).get("tasks", [])
    if len(tasks) == len(expected):
        score += 0.2
    predicted_text = " ".join(t["task_instruction"] for t in tasks)
    expected_text = " ".join(str(t.get("task_instruction", "")) for t in expected)
    score += 0.3 * SequenceMatcher(
        None, predicted_text.casefold(), expected_text.casefold()
    ).ratio()
    return min(1.0, score)


def resolve_reward(spec=None):
    if spec is None:
        from environment.untrusted_orchestration.orchestration_reward import (
            orchestration_reward,
        )
        return orchestration_reward
    module, separator, function_name = spec.partition(":")
    if not separator or not module or not function_name:
        raise ValueError("expected reward callable as module.path:function")
    fn = getattr(importlib.import_module(module), function_name)
    if not callable(fn):
        raise ValueError("reward plugin must be callable")
    return fn
