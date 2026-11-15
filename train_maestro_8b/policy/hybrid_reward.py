"""Hard environment legality gate plus optional semantic quality reward."""
from __future__ import annotations

from environment.untrusted_orchestration.orchestration_reward import score_decision


class HybridReward:
    """Group-scoring interface consumed by RiskSteeredGRPOTrainer.

    Only *legal* candidates are submitted to the external judge. Invalid
    Actions remain exactly -1 and are never rescued by semantic opinions.
    """

    def __init__(self, judge, *, mode: str = "hybrid", judge_weight: float = 0.4):
        if mode not in {"hybrid", "judge_only"}:
            raise ValueError("HybridReward mode must be hybrid or judge_only")
        if not 0.0 <= judge_weight <= 1.0:
            raise ValueError("judge_weight must lie in [0,1]")
        self.judge = judge
        self.mode = mode
        self.judge_weight = judge_weight

    def score_group(self, candidates: list[str], record: dict) -> list[dict]:
        state = record["decision_state"]
        results = [score_decision(candidate, state) for candidate in candidates]
        valid_indices = [i for i, result in enumerate(results) if result["valid"]]
        semantic = self.judge.score_many(
            record["prompt"], [candidates[i] for i in valid_indices],
        ) if valid_indices else []
        if len(semantic) != len(valid_indices):
            raise RuntimeError("Semantic Judge returned misaligned candidate rewards")
        for result in results:
            result["env_reward"] = result["reward"]
            result["judge_reward"] = None
            result["judge_dimensions"] = None
            result["judge_reason"] = None
            result["judge_cached"] = None
            result["reward_mode"] = self.mode
        for index, judge_result in zip(valid_indices, semantic):
            result = results[index]
            score = float(judge_result["reward"])
            if not 0.0 <= score <= 1.0:
                raise ValueError("Semantic Judge reward outside [0,1]")
            result["judge_reward"] = score
            result["judge_dimensions"] = judge_result["dimensions"]
            result["judge_reason"] = judge_result.get("reason", "")
            result["judge_cached"] = bool(judge_result.get("cached", False))
            if self.mode == "judge_only":
                result["reward"] = score
            else:
                result["reward"] = round(
                    (1 - self.judge_weight) * result["env_reward"] +
                    self.judge_weight * score, 8,
                )
        return results
