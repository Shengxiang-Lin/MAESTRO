"""Exact Orchestra-o1 Main prompt/history adapter for turn-level training.

This imports the *actual* baseline prompt renderer and history formatter:
never substitute a loose paraphrase for the model-visible input.
"""
from __future__ import annotations

from types import SimpleNamespace

from orchestra.base_main_agent import BaseMainAgent
from orchestra.prompts.omnigaia import OmniGAIAMainAgentPrompt
from environment.generate_question.execution.evidence_tools import o1_tool_definitions

def source_catalog(scene: dict) -> dict[str, dict]:
    entries = scene["public"]["available_source_descriptions"]
    sources = {item["source_id"]: item for item in entries}
    if len(sources) != len(entries):
        raise ValueError("stage3_duplicate_public_source")
    return sources


def o1_initial_instruction(scene: dict) -> str:
    """Render user question, already-observed inputs and unqueried sources.

    The current Task Plan uses textual simulations of media observations.
    Simulated source handles are NOT attached image/audio files.
    """
    question = scene["question"]
    card = scene["public"].get("task_input")
    if card is None:
        return question
    if card.get("version") != "macro_goal_reference_card_v2":
        raise ValueError("stage3_unknown_task_reference_card")
    observations = card["observations"]
    if not observations:
        raise ValueError("stage3_empty_initial_task_input")
    counts = {}
    lines = [
        "==== INITIAL TASK INPUTS (PROVIDED SIMULATED OBSERVATIONS) ====",
        "These observations were already provided by the user. They are "
        "text descriptions, not attached media or Sub-Agent findings.",
    ]
    for obs in observations:
        modality = obs["modality"]
        counts[modality] = counts.get(modality, 0) + 1
        marker = modality + "_" + str(counts[modality])
        lines.append(
            "[" + marker + " / observed " + modality + "] " + obs["content"]
        )
    lines.extend([
        "",
        "==== AVAILABLE EVIDENCE SOURCES (NOT INITIAL ATTACHMENTS) ====",
        "The following sources have not been queried. Their identifiers "
        "denote simulated investigation workspaces, not physical media.",
    ])
    for entry in scene["public"]["available_source_descriptions"]:
        lines.append(
            "- " + entry["source_id"] + " [" + entry["modality_hint"] + ", " +
            entry.get("access_mode", "browse") + "]: " + entry["title"]
        )
    lines.append(
        "Investigate candidate records and verify as needed. Do not "
        "re-extract reference observations already provided above."
    )
    return question + "\n\n" + "\n".join(lines)



def stage3_author_context(
    scene: dict, subtask_history: str, *,
    attempt: int, max_attempts: int,
) -> str:
    """Compact generation-only view; NOT the model's training prompt.

    Reuse the exact initial user-visible instruction and O1 history without
    truncation, while omitting the baseline's static orchestration tutorial,
    pricing table, tool JSON schemas and output-format examples. Current
    dispatch/source/output contracts are supplied separately by Stage 3.
    """
    if attempt < 1 or attempt > max_attempts:
        raise ValueError("stage3_attempt_exceeds_budget")
    if not isinstance(subtask_history, str):
        raise TypeError("stage3_subtask_history_must_be_string")
    instruction = o1_initial_instruction(scene)
    if "task_input" not in scene["public"]:
        # Legacy scenes enumerate sources in O1 modal_inputs rather than the
        # initial instruction. Keep every publicly visible source and URL.
        sources = source_catalog(scene)
        source_lines = [
            "==== AVAILABLE EVIDENCE SOURCES ====",
        ]
        for sid, source in sources.items():
            path = source.get("resource_url") or (
                "simulated-web-index://" + sid
                if (scene.get("problem_spec", {}).get(
                    "evidence_tool_contract_version", ""
                ).endswith(".v2")
                    and source["modality_hint"] in {"text", "table"})
                else "workspace://" + sid
            )
            source_lines.append(
                "- " + sid + " [" + source["modality_hint"] + ", " +
                source.get("access_mode", "browse") + "]: " +
                source["title"] + "; path: " + path
            )
        instruction += "\n\n" + "\n".join(source_lines)
    category = scene.get("problem_spec", {}).get("macro_plan", {}).get(
        "domain", "general",
    )
    return (
        "Task Type: Multi-source investigation\n" +
        "Category: " + str(category) +
        "\n\n==== QUESTION ====\n" +
        instruction +
        "\n\n==== PROGRESS ====\n" +
        f"Attempt {attempt}/{max_attempts}; "
        f"remaining {max_attempts - attempt} attempts." +
        "\n\n==== SUBTASK HISTORY ====\n" +
        subtask_history
    )


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

    The on-disk per-turn reviewed-turn input stores only the cumulative subtask
    history. The fixed prompt template, Question and source catalog are
    resolved from the Stage-1 Scene and these run-level rendering settings.
    """
    if attempt < 1 or attempt > max_attempts:
        raise ValueError("stage3_attempt_exceeds_budget")
    if not isinstance(subtask_history, str):
        raise TypeError("stage3_subtask_history_must_be_string")
    sources = source_catalog(scene)
    has_reference_inputs = "task_input" in scene["public"]
    meta = {
        "task_type": "Multi-source investigation",
        "category": scene.get("problem_spec", {}).get("macro_plan", {}).get(
            "domain", "general",
        ),
        "modal_inputs": [] if has_reference_inputs else [
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
        instruction=o1_initial_instruction(scene),
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
    """Rebuild a full baseline O1 prompt for an exported compact reviewed-turn turn."""
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
