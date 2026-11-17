"""Compile trusted O1 history directly from Stage-1 scenes and Stage-2 reports.

This is a deterministic internal bootstrap for Stage 4, not a third
LLM generation or review stage. It has no separate input artifact and
never publishes template-only Main turns as positive SFT data.
"""
from __future__ import annotations

from collections import defaultdict

from environment.base_orchestration.trajectory_builder import build_episode


def validate_clean_episode(expected: dict, supplied: dict) -> None:
    """Reject Stage-3 artifacts that changed trusted Stage-2 observations.

    Main reasoning/instructions may differ; delegated batches and actually
    returned Finish Results must match the frozen, executable baseline.
    """
    if (supplied.get("task_id") != expected["task_id"]
            or supplied.get("clean_reference_id") !=
            expected["clean_reference_id"]
            or supplied.get("trusted_subagent_feedback") is not True):
        raise ValueError("stage4_external_clean_reference_identity_mismatch")
    reference_steps = expected["trajectory"]
    candidate_steps = supplied.get("trajectory")
    if (not isinstance(candidate_steps, list)
            or len(candidate_steps) != len(reference_steps)):
        raise ValueError("stage4_external_clean_schedule_mismatch")
    for reference, candidate in zip(reference_steps, candidate_steps):
        if (candidate.get("attempt") != reference["attempt"]
                or candidate.get("action") != reference["action"]):
            raise ValueError("stage4_external_clean_action_mismatch")
        if reference["action"] == "complete":
            if candidate.get("params", {}).get("answer") != (
                reference["params"]["answer"]
            ):
                raise ValueError("stage4_external_clean_answer_mismatch")
            continue
        actual = (candidate.get("result") or {}).get("subtask_results")
        trusted = reference["result"]["subtask_results"]
        assigned = (candidate.get("params") or {}).get("tasks")
        if (not isinstance(actual, list) or not isinstance(assigned, list)
                or len(actual) != len(trusted)
                or len(assigned) != len(trusted)):
            raise ValueError("stage4_external_clean_batch_mismatch")
        for delivered, known in zip(actual, trusted):
            if (delivered.get("finish_result") != known["finish_result"]
                    or delivered.get("model") != known["model"]):
                raise ValueError("stage4_external_clean_report_mismatch")


def compile_trusted_prefixes(
    scenes: list[dict], reports: list[dict], *,
    max_rounds: int, sub_model: str, model_alias: str,
    clean_episodes: dict[str, dict] | None = None,
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
            if clean_episodes is not None:
                supplied = clean_episodes.get(task_id)
                try:
                    if supplied is None:
                        raise ValueError("stage4_external_clean_reference_missing")
                    validate_clean_episode(episode, supplied)
                except (ValueError, TypeError, AttributeError) as exc:
                    status = {"status": "rejected", "cause": str(exc)[:250]}
                else:
                    episode = supplied
            if status.get("status") == "accepted":
                accepted[task_id] = episode
        reviews.append({
            "task_id": task_id,
            "status": "accepted" if task_id in accepted else "rejected",
            "cause": status.get("cause"),
            "trusted_decision_rounds": (
                len(episode["trajectory"]) if episode else 0
            ),
            "source": (
                "validated_stage3_clean_reference"
                if clean_episodes is not None else
                "stage1_scene_and_stage2_reports"
            ),
            "mode": (
                "reviewed_stage3_clean_reference"
                if clean_episodes is not None else
                "deterministic_template_not_published"
            ),
        })
    return accepted, reviews
