"""Compile trusted O1 history directly from Stage-1 scenes and Stage-2 reports.

This is a deterministic internal bootstrap for Stage 4, not a third
LLM generation or review stage. It has no separate input artifact and
never publishes template-only Main turns as positive SFT data.
"""
from __future__ import annotations

from collections import defaultdict

from environment.base_orchestration.trajectory_builder import build_episode


def compile_trusted_prefixes(
    scenes: list[dict], reports: list[dict], *,
    max_rounds: int, sub_model: str, model_alias: str,
) -> tuple[dict[str, dict], list[dict]]:
    grouped = defaultdict(list)
    for row in reports:
        grouped[row["scene_task_id"]].append(row)
    accepted, reviews = {}, []
    for scene in scenes:
        task_id = scene["task_id"]
        try:
            episode, _turns, _audit, status = build_episode(
                scene, grouped[task_id],
                author=None, critic=None, max_attempts=max_rounds,
                sub_model=sub_model, model_alias=model_alias,
            )
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            episode = None
            status = {"status": "rejected", "cause": str(exc)[:250]}
        if episode is not None and status.get("status") == "accepted":
            accepted[task_id] = episode
        reviews.append({
            "task_id": task_id,
            "status": "accepted" if task_id in accepted else "rejected",
            "cause": status.get("cause"),
            "trusted_decision_rounds": (
                len(episode["trajectory"]) if episode else 0
            ),
            "source": "stage1_scene_and_stage2_reports",
            "mode": "deterministic_template_not_published",
        })
    return accepted, reviews
