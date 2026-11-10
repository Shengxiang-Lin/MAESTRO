"""Exact Orchestra-o1 Main prompt/history adapter for turn-level training.

This imports the *actual* baseline prompt renderer and history formatter:
never substitute a loose paraphrase for the model-visible input.
"""
from __future__ import annotations

from types import SimpleNamespace

from orchestra.base_main_agent import BaseMainAgent
from orchestra.prompts.omnigaia import OmniGAIAMainAgentPrompt

TOOL_DEFINITIONS = {
    "ImageAnalysisAction": (
        "Inspect the task-provided image observation in the controlled "
        "workspace; report identifiers and visual evidence.",
        ["image"],
    ),
    "VideoAnalysisAction": (
        "Inspect the task-provided video observations, including timestamped "
        "events, from the controlled workspace.",
        ["video"],
    ),
    "ParseAudioAction": (
        "Inspect the supplied commentary or audio observations, including "
        "spoken cues and timing, from the controlled workspace.",
        ["audio"],
    ),
    "WorkspaceSearchAction": (
        "Search the named text records and linked indices within the "
        "controlled task workspace, not the public internet.",
        ["text"],
    ),
    "TableLookupAction": (
        "Look up an identified record in the supplied workspace table using "
        "a key learned from earlier findings; do not scan a keyed table blindly.",
        ["table"],
    ),
    "ExecuteCodeAction": (
        "Cross-reference previously retrieved evidence or perform a "
        "calculation using findings already returned by SubAgents.",
        [],
    ),
}

MODALITY_TOOL = {
    "image": "ImageAnalysisAction",
    "video": "VideoAnalysisAction",
    "audio": "ParseAudioAction",
    "text": "WorkspaceSearchAction",
    "table": "TableLookupAction",
}


def tools_for_sources(source_ids: list[str], catalog: dict[str, dict]) -> list[str]:
    if not source_ids:
        return ["ExecuteCodeAction"]
    names = []
    for sid in source_ids:
        source = catalog[sid]
        tool = MODALITY_TOOL.get(source["modality_hint"])
        if tool is None:
            raise ValueError("stage3_unknown_source_modality:" + sid)
        if tool not in names:
            names.append(tool)
    return names


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
                "path": "workspace://" + sid,
            } for sid, source in sources.items()
        ],
    }
    tool_objects = []
    for name, (description, _) in TOOL_DEFINITIONS.items():
        tool_objects.append(SimpleNamespace(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "query": {"type": "string"},
                },
            },
        ))
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
