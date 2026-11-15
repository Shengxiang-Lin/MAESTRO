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



def _format_progress(candidate):
    """Conservative, label-free JSON-shape progress for free-form generation.

    Crucially this does NOT clean, repair, trim, constrain or parse a
    generated Action on behalf of the strict executor. It supplies a
    small, bounded learning signal when the *entire* text is invalid JSON.
    The earliest apparent object is used; repeating JSON keywords far
    into natural language is deliberately worth very little.
    """
    if not isinstance(candidate, str):
        return 0.0
    text = candidate.strip()
    if not text:
        return 0.0
    brace = text.find("{")
    if brace < 0:
        return 0.0
    # Object beginning after a long natural-language preamble earns
    # much less credit than starting the reply with the required {.
    proximity = 1.0 / (1.0 + brace / 64.0)
    fragment = text[brace:brace + 4096]
    score = 0.03 + 0.12 * proximity
    if '"action"' in fragment:
        score += 0.055 * proximity
    if '"params"' in fragment:
        score += 0.045 * proximity
    if '"delegate_task"' in fragment or '"complete"' in fragment:
        score += 0.04 * proximity
    # A well-formed JSON object surrounded by chatter is still invalid,
    # but it is closer to the desired standalone JSON than bare prose.
    try:
        value, _ = json.JSONDecoder().raw_decode(text[brace:])
        if isinstance(value, dict):
            score += 0.035 * proximity
    except (ValueError, TypeError):
        pass
    return round(min(0.335, score), 6)


def score_training_candidate(candidate, state, *, mode="curriculum"):
    """Train with graded format feedback; ALWAYS evaluate strict legality.

    The original deterministic oracle is left untouched and remains the
    definitive evaluation metric. There is no Teacher/Gold text matching,
    no future-state access, and no automatic JSON repair.

    Strictly legal actions earn 0.65-1.0. Strictly invalid actions never
    exceed 0.55. A structurally plausible but invalid JSON object earns
    more than a malformed one, while premature completion gets a penalty
    to avoid incentivizing a trivial empty/short 'complete' shortcut.
    """
    from environment.untrusted_orchestration.orchestration_reward import (
        score_decision,
    )
    if mode not in ("strict", "curriculum"):
        raise ValueError("training reward mode must be strict or curriculum")
    strict = score_decision(candidate, state)
    progress = _format_progress(candidate)
    result = dict(strict)
    result["strict_reward"] = strict["reward"]
    result["format_progress"] = progress
    result["reward_mode"] = mode

    if mode == "strict":
        result["training_reward"] = strict["reward"]
        result["json_valid"] = strict["reason"] != "invalid_json"
        return result

    if strict["valid"]:
        shaped = 0.65 + 0.35 * max(0.0, min(1.0, strict["reward"]))
        result["json_valid"] = True
    elif strict["reason"] == "invalid_json":
        shaped = progress
        result["json_valid"] = False
    elif strict["reason"] in (
        "premature_completion", "unnecessary_dispatch_after_completion",
    ):
        shaped = 0.35
        result["json_valid"] = True
    else:
        # Well-formed JSON, still invalid for the action contract/binding.
        # It never outranks an actually legal Main-Agent decision.
        shaped = 0.4 + 0.15 * min(1.0, progress / 0.335)
        result["json_valid"] = True
    result["training_reward"] = round(float(shaped), 8)
    return result


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
