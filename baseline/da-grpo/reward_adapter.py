"""Bridge TRL's per-completion reward contract to the original O1 Judge.

Ground truth and expert decisions stay in dataset reward-only columns; they are
never concatenated into the policy prompt. One callable preserves the original
weighted 4D reward (format 0.1, action 0.1, tool 0.2, decision 0.6).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from typing import Any


def completion_text(completion: Any) -> str:
    """TRL conversational completions are assistant-message lists."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        chunks = []
        for item in completion:
            if not isinstance(item, dict) or item.get("role") != "assistant":
                raise ValueError("Expected assistant message(s) from TRL")
            value = item.get("content")
            if not isinstance(value, str):
                raise ValueError("Nontext GRPO completion unsupported")
            chunks.append(value)
        if chunks:
            return "\n".join(chunks)
    raise ValueError(f"Unsupported completion type: {type(completion).__name__}")


def make_reward_function(workers: int):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive int")

    def orchestra_o1_judge(
        completions: list,
        data_source: list[str],
        ground_truth: list[str],
        extra_info_json: list[str],
        **_: Any,
    ) -> list[float]:
        """Returns one O1 reward per generated completion (TRL repeats columns)."""
        n = len(completions)
        if not (len(data_source) == len(ground_truth) == len(extra_info_json) == n):
            raise ValueError("TRL completion/metadata length mismatch")

        metadata = [json.loads(item) for item in extra_info_json]
        if any(not isinstance(item, dict) or not item.get("has_expert") for item in metadata):
            raise ValueError("Missing reward-only expert metadata")
        from reward_fn import compute_score

        def score(index: int) -> float:
            scored = compute_score(
                data_source[index],
                completion_text(completions[index]),
                ground_truth[index],
                metadata[index],
            )
            value = float(scored["score"] if isinstance(scored, dict) else scored)
            if not 0 <= value <= 1:
                raise ValueError(f"O1 Judge reward out of range: {value}")
            return value

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(score, range(n)))

    return orchestra_o1_judge
