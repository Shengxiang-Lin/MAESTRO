"""Exact Orchestra-o1 Main prompt/history adapter for turn-level training.

This imports the *actual* baseline prompt renderer and history formatter:
never substitute a loose paraphrase for the model-visible input.
"""
from __future__ import annotations

from types import SimpleNamespace

from orchestra.base_main_agent import BaseMainAgent
from orchestra.prompts.omnigaia import OmniGAIAMainAgentPrompt
from environment.generate_question.evidence_tools import o1_tool_definitions

def source_catalog(scene: dict) -> dict[str, dict]:
    entries = scene["public"]["available_source_descriptions"]
    sources = {item["source_id"]: item for item in entries}
    if len(sources) != len(entries):
        raise ValueError("stage3_duplicate_public_source")
    return sources


def o1_history(task_entries: list[dict]) -> str:
    """Call the baseline's real formatter on a minimal history-only view."""
    state = SimpleNamespace(
        task_entries=task_entries, mask_model_names=False, model_to_alias={},
    )
    return BaseMainAgent._format_subtask_history(state)


def o1_prompt_from_history(
    scene: dict, subtask_history: str, *, attempt: int,
    max_attempts: int, sub_model: str, model_alias: str = "model_1",
) -> str:
    """Exactly reconstruct the O1 prompt from shared Scene + turn state.

    The on-disk per-turn SFT input stores only the cumulative subtask
    history. The fixed prompt template, Question and source catalog are
    resolved from the Stage-1 Scene and these run-level rendering settings.
    """
    if attempt < 1 or attempt > max_attempts:
        raise ValueError("stage3_attempt_exceeds_budget")
    if not isinstance(subtask_history, str):
        raise TypeError("stage3_subtask_history_must_be_string")
    sources = source_catalog(scene)
    meta = {
        "task_type": "Multi-source investigation",
        "category": scene.get("problem_spec", {}).get("macro_plan", {}).get(
            "domain", "general",
        ),
        "modal_inputs": [
            {
                "type": source["modality_hint"],
                "id": sid,
                # Known URL is visible from the outset only for Direct
                # Extract. Search-discovered URLs are absent until the
                # Sub-Agent's own simulated Search observation.
                "path": source.get("resource_url") or (
                    "simulated-web-index://" + sid
                    if (scene.get("problem_spec", {}).get(
                            "evidence_tool_contract_version", ""
                        ).endswith(".v2")
                        and source["modality_hint"] in {"text", "table"})
                    else "workspace://" + sid
                ),
            } for sid, source in sources.items()
        ],
    }
    # Exact descriptions and parameter schemas from baseline Action classes.
    tool_objects = [SimpleNamespace(**definition)
                    for definition in o1_tool_definitions()]
    return OmniGAIAMainAgentPrompt.build_prompt(
        instruction=scene["question"],
        meta=meta,
        prior_context="",  # The O1 builder currently does not render it.
        attempt_index=attempt,
        max_attempts=max_attempts,
        sub_models=[model_alias],
        subtask_history=subtask_history,
        model_to_alias={sub_model: model_alias},
        tools=tool_objects,
    )


def o1_prompt(
    scene: dict, task_entries: list[dict], *, attempt: int,
    max_attempts: int, sub_model: str, model_alias: str = "model_1",
) -> str:
    """Render from live O1 structured history (Author/Critic inference)."""
    return o1_prompt_from_history(
        scene, o1_history(task_entries), attempt=attempt,
        max_attempts=max_attempts, sub_model=sub_model,
        model_alias=model_alias,
    )


def o1_prompt_from_turn(
    scene: dict, turn: dict, *, max_attempts: int,
    sub_model: str, model_alias: str = "model_1",
) -> str:
    """Rebuild a full baseline O1 prompt for an exported compact SFT turn."""
    if turn.get("task_id") != scene.get("task_id"):
        raise ValueError("stage3_turn_scene_mismatch")
    if type(turn.get("round")) is not int:
        raise ValueError("stage3_turn_round_invalid")
    inputs = turn.get("input")
    if not isinstance(inputs, dict) or set(inputs) != {"subtask_history"}:
        raise ValueError("stage3_turn_input_not_compact_history")
    return o1_prompt_from_history(
        scene, inputs["subtask_history"], attempt=turn["round"],
        max_attempts=max_attempts, sub_model=sub_model,
        model_alias=model_alias,
    )
