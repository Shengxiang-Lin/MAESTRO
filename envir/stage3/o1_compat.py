"""Exact Orchestra-o1 Main prompt/history adapter for turn-level training.

This imports the *actual* baseline prompt renderer and history formatter:
never substitute a loose paraphrase for the model-visible input.
"""
from __future__ import annotations

from types import SimpleNamespace

from orchestra.base_main_agent import BaseMainAgent
from orchestra.prompts.omnigaia import OmniGAIAMainAgentPrompt
from envir.generate_question.evidence_tools import o1_tool_definitions

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


def o1_prompt(
    scene: dict, task_entries: list[dict], *, attempt: int,
    max_attempts: int, sub_model: str, model_alias: str = "model_1",
) -> str:
    if attempt < 1 or attempt > max_attempts:
        raise ValueError("stage3_attempt_exceeds_budget")
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
                "path": source.get("resource_url", "workspace://" + sid),
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
        subtask_history=o1_history(task_entries),
        model_to_alias={sub_model: model_alias},
        tools=tool_objects,
    )
