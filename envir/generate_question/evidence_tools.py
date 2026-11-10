"""Frozen evidence-acquisition -> Orchestra-o1 tool contracts.

Source access, not Macro-task identity, determines tool requirements. The
baseline's six registered OmniGAIA Action classes are the authoritative tool
registry; their metadata is read from their source files without importing
heavy video/audio or network dependencies during offline dataset generation.

Synthetic workspace observations DO NOT represent executed tool calls. A
contract records the real O1 tool interface that a *corresponding* live
acquisition would use, while provenance and execution stay simulated.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

EVIDENCE_TOOL_CONTRACT_VERSION = "maestro.evidence_acquisition.o1.v1"

# These are the exact Action classes registered by
# benchmark.omnigaia.tools.build_omnigaia_tools, not invented workspace tools.
_O1_ACTION_FILES = (
    ("GoogleSearchAction", "google_search.py"),
    ("ExecuteCodeAction", "execute_code.py"),
    ("ExtractUrlContentAction", "extract_url_jina.py"),
    ("ImageAnalysisAction", "multimodal_toolkit.py"),
    ("ParseAudioAction", "audio_analysis.py"),
    ("VideoAnalysisAction", "video_analysis.py"),
)
_ACQUISITION_OPS = frozenset({"scan", "lookup_by_key"})
_MEDIA_ACTIONS = {
    "image": "ImageAnalysisAction",
    "video": "VideoAnalysisAction",
    "audio": "ParseAudioAction",
}


@lru_cache(maxsize=1)
def o1_tool_definitions() -> tuple[dict, ...]:
    """Read exact name/description/parameters from baseline tool classes.

    AST inspection avoids importing cv2, ffmpeg and optional network SDKs
    merely to build a deterministic offline Stage-1 contract.
    """
    root = Path(__file__).resolve().parents[2]
    tools_root = root / "benchmark" / "omnigaia" / "tools"
    result = []
    for name, file in _O1_ACTION_FILES:
        tree = ast.parse((tools_root / file).read_text(encoding="utf-8"))
        classes = [
            item for item in tree.body
            if isinstance(item, ast.ClassDef) and item.name == name
        ]
        if len(classes) != 1:
            raise ValueError("o1_tool_registry_class_mismatch:" + name)
        members = {}
        for item in classes[0].body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                if item.target.id in {"name", "description", "parameters"}:
                    members[item.target.id] = ast.literal_eval(item.value)
        if members.get("name") != name:
            raise ValueError("o1_tool_registry_name_mismatch:" + name)
        if (not isinstance(members.get("description"), str)
                or not isinstance(members.get("parameters"), dict)):
            raise ValueError("o1_tool_registry_schema_missing:" + name)
        result.append({
            "name": name, "description": members["description"],
            "parameters": members["parameters"],
        })
    return tuple(result)


def registered_o1_tool_names() -> tuple[str, ...]:
    return tuple(tool["name"] for tool in o1_tool_definitions())


def source_acquisition_policy(source: dict) -> dict:
    """Choose an access strategy for a *source*, not for a whole Macro.

    Web documents without a real URL use O1 search-by-query semantics. The
    URL extraction Action is only appropriate when a usable real URL is
    actually supplied, which current synthetic worlds do not provide.
    """
    modality = source["modality"]
    access_mode = source["access_mode"]
    if access_mode not in {"browse", "keyed_lookup"}:
        raise ValueError("evidence_unknown_access_mode:" + str(access_mode))
    if modality in _MEDIA_ACTIONS:
        if access_mode != "browse":
            raise ValueError("evidence_media_keyed_lookup_unsupported")
        tool = _MEDIA_ACTIONS[modality]
        strategy = "media_observation"
        binding = "source_media"
    elif modality in {"text", "table"}:
        url = source.get("resource_url")
        if url is not None:
            if (not isinstance(url, str)
                    or not url.startswith(("https://", "http://"))):
                raise ValueError("evidence_invalid_real_url")
            tool = "ExtractUrlContentAction"
            strategy = "url_content_extraction"
            binding = "upstream_url" if access_mode == "keyed_lookup" else "source_url"
        else:
            tool = "GoogleSearchAction"
            strategy = (
                "keyed_search" if access_mode == "keyed_lookup"
                else "search_discovery"
            )
            binding = (
                "upstream_key_query" if access_mode == "keyed_lookup"
                else "question_cues_query"
            )
    else:
        raise ValueError("evidence_unsupported_modality:" + str(modality))
    if tool not in registered_o1_tool_names():
        raise ValueError("evidence_unregistered_o1_tool:" + tool)
    return {
        "version": EVIDENCE_TOOL_CONTRACT_VERSION,
        "modality": modality,
        "access_mode": access_mode,
        "strategy": strategy,
        "tool": tool,
        "input_binding": binding,
        "execution_mode": "simulated_no_tool_execution",
    }


def acquisition_for_operator(node: dict, source: dict) -> dict:
    """Freeze the source's O1 capability at each actual acquisition node."""
    op = node.get("op")
    if op not in _ACQUISITION_OPS:
        raise ValueError("evidence_not_an_acquisition_operator")
    sid = node["source_id"]
    if sid != source["source_id"]:
        raise ValueError("evidence_operator_source_mismatch")
    mode = source.get("access_mode")
    if (op == "scan" and mode != "browse") or (
        op == "lookup_by_key" and mode != "keyed_lookup"
    ):
        raise ValueError("evidence_access_mode_operator_mismatch")
    policy = source_acquisition_policy(source)
    if source.get("evidence_access") != policy:
        raise ValueError("evidence_frozen_source_policy_mismatch")
    contract = {
        **deepcopy(policy),
        "source_id": sid,
        "operator": op,
    }
    if op == "lookup_by_key":
        if node.get("source_key_field") != source.get("key_field"):
            raise ValueError("evidence_lookup_key_contract_mismatch")
        contract["upstream_key_field"] = node["input_key_field"]
        contract["source_key_field"] = node["source_key_field"]
    return contract


def tools_from_primitive_steps(steps: list[dict]) -> list[str]:
    """Ordered de-duplicated union of tools on lower-level acquisition steps."""
    tools = []
    for step in steps:
        acquisition = step.get("evidence_acquisition")
        if acquisition is None:
            continue
        if step.get("op") not in _ACQUISITION_OPS:
            raise ValueError("evidence_contract_on_non_acquisition")
        name = acquisition.get("tool")
        if name not in registered_o1_tool_names():
            raise ValueError("evidence_unknown_step_tool")
        if name not in tools:
            tools.append(name)
    return tools


def validate_scene_evidence_tools(scene: dict) -> list[str]:
    """Fail closed on source, primitive and Macro aggregation drift."""
    spec = scene["problem_spec"]
    if not spec.get("evidence_tool_contract_version"):
        return []  # Legacy source snapshots intentionally have no tool metadata.
    issues = []
    if spec["evidence_tool_contract_version"] != EVIDENCE_TOOL_CONTRACT_VERSION:
        return ["evidence_tool_contract_version_mismatch"]
    sources = {role["source_id"]: role for role in spec["sources"]}
    for sid, source in sources.items():
        try:
            if source.get("evidence_access") != source_acquisition_policy(source):
                issues.append("evidence_source_policy_mismatch:" + sid)
        except (KeyError, TypeError, ValueError) as exc:
            issues.append("evidence_source_invalid:" + sid + ":" + str(exc))
    seen = set()
    for task in scene["macro_graph"]["tasks"]:
        for step in task["operator_chain"]:
            expected = None
            try:
                if step["op"] in _ACQUISITION_OPS:
                    expected = acquisition_for_operator(
                        step, sources[step["source_id"]],
                    )
                if step.get("evidence_acquisition") != expected:
                    issues.append("evidence_step_contract_mismatch:" + step["id"])
            except (KeyError, TypeError, ValueError) as exc:
                issues.append("evidence_step_invalid:" + step["id"] + ":" + str(exc))
            if step["id"] in seen:
                issues.append("evidence_duplicate_step:" + step["id"])
            seen.add(step["id"])
        try:
            if task.get("required_tools") != tools_from_primitive_steps(
                task["operator_chain"]
            ):
                issues.append("evidence_macro_tool_aggregation_mismatch:" +
                              task["task_id"])
        except (KeyError, TypeError, ValueError) as exc:
            issues.append("evidence_macro_invalid:" + str(exc))
    return issues
