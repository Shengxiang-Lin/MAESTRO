"""Bridge TRL's per-completion reward contract to the original O1 Judge.

Ground truth and expert decisions stay in dataset reward-only columns; they are
never concatenated into the policy prompt. One callable preserves the original
weighted 4D reward (format 0.1, action 0.1, tool 0.2, decision 0.6).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


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


def make_reward_function(
    workers: int, *, diagnostics_path: Path | None = None, num_generations: int = 1,
):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive int")
    if diagnostics_path is not None:
        if type(num_generations) is not int or num_generations < 1:
            raise ValueError("num_generations must be positive int")
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    score_options = {"include_decision_hash": True} if diagnostics_path is not None else {}

    def orchestra_o1_judge(
        completions: list,
        data_source: list[str],
        ground_truth: list[str],
        extra_info_json: list[str],
        trainer_state: Any = None,
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

        def score(index: int) -> tuple[float, dict]:
            scored = compute_score(
                data_source[index],
                completion_text(completions[index]),
                ground_truth[index],
                metadata[index],
                **score_options,
            )
            value = float(scored["score"] if isinstance(scored, dict) else scored)
            if not 0 <= value <= 1:
                raise ValueError(f"O1 Judge reward out of range: {value}")
            return value, scored if isinstance(scored, dict) else {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(score, range(n)))

        if diagnostics_path is not None:
            batch_id = uuid4().hex
            step = getattr(trainer_state, "global_step", None)
            records = []
            for index, (value, details) in enumerate(results):
                group, candidate = divmod(index, num_generations)
                scores = {
                    name: round(details[name] * scale) if name in details else None
                    for name, scale in (("format_correct", 1), ("action_valid", 1),
                                        ("tool_reasonable", 3), ("decision_quality", 3))
                }
                records.append(json.dumps({
                    "batch_id": batch_id, "trainer_step": step,
                    "group_index": group, "candidate_index": candidate,
                    "sample_id": metadata[index].get("sample_id"),
                    "decision_sha256": details.get("decision_sha256"),
                    "scores": scores, "reward": value,
                }, ensure_ascii=False) + "\n")
            with diagnostics_path.open("a", encoding="utf-8") as file:
                file.write("".join(records))
        return [value for value, _ in results]

    return orchestra_o1_judge
