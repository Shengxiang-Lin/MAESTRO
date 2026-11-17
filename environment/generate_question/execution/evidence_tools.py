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
from urllib.parse import urlparse
from pathlib import Path

EVIDENCE_TOOL_CONTRACT_VERSION = "maestro.evidence_acquisition.o1.v2"
LEGACY_EVIDENCE_TOOL_CONTRACT_VERSION = "maestro.evidence_acquisition.o1.v1"

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


def synthetic_page_url(*, seed: int, ordinal: int,
                       record_id: str, source_id: str) -> str:
    """A stable *non-routable* HTTP-shaped fixture; never a live website."""
    from environment.schema import digest
    token = digest(seed, ordinal, record_id, source_id, "virtual-web-page")[:16]
    return f"https://evidence-{token}.example.invalid/{source_id}/record"


def web_acquisition_route(*, seed: int, ordinal: int,
                          record_id: str, source_id: str) -> str:
    """Select three real O1 access patterns without perturbing the World RNG."""
    from environment.schema import digest
    choices = ("search_only", "direct_extract", "search_then_extract")
    val = int(digest(seed, ordinal, record_id, source_id, "web-route")[:8], 16)
    return choices[val % len(choices)]


def _legacy_source_acquisition_policy(source: dict) -> dict:
    """Preserve old v1 Scene regeneration and deterministic Replay."""
    modality = source["modality"]
    access_mode = source["access_mode"]
    if access_mode not in {"browse", "keyed_lookup"}:
        raise ValueError("evidence_unknown_access_mode:" + str(access_mode))
    if modality in _MEDIA_ACTIONS:
        if access_mode != "browse":
            raise ValueError("evidence_media_keyed_lookup_unsupported")
        tool = _MEDIA_ACTIONS[modality]
        strategy, binding = "media_observation", "source_media"
    elif modality in {"text", "table"}:
        url = source.get("resource_url")
        if url is not None:
            if not isinstance(url, str) or not url.startswith(("https://", "http://")):
                raise ValueError("evidence_invalid_real_url")
            tool = "ExtractUrlContentAction"
            strategy = "url_content_extraction"
            binding = "upstream_url" if access_mode == "keyed_lookup" else "source_url"
        else:
            tool = "GoogleSearchAction"
            strategy = "keyed_search" if access_mode == "keyed_lookup" else "search_discovery"
            binding = "upstream_key_query" if access_mode == "keyed_lookup" else "question_cues_query"
    else:
        raise ValueError("evidence_unsupported_modality:" + str(modality))
    if tool not in registered_o1_tool_names():
        raise ValueError("evidence_unregistered_o1_tool:" + tool)
    return {
        "version": LEGACY_EVIDENCE_TOOL_CONTRACT_VERSION,
        "modality": modality, "access_mode": access_mode,
        "strategy": strategy, "tool": tool, "input_binding": binding,
        "execution_mode": "simulated_no_tool_execution",
    }


def source_acquisition_policy(source: dict) -> dict:
    """Freeze real O1 tools at *individual evidence acquisition*, not Macro.

    For v2 synthetic web sources, a query can be sufficient, a known URL
    can be extracted directly, or a query can discover a URL that must be
    extracted in a subsequent local Sub-Agent step. A search result is not
    an Extract call, and a discovered URL is not available before search.
    """
    route = source.get("acquisition_route")
    if route is None:
        return _legacy_source_acquisition_policy(source)
    modality, access_mode = source["modality"], source["access_mode"]
    if access_mode not in {"browse", "keyed_lookup"}:
        raise ValueError("evidence_unknown_access_mode:" + str(access_mode))
    if modality in _MEDIA_ACTIONS:
        if route != "media_observation" or access_mode != "browse":
            raise ValueError("evidence_media_route_invalid")
        if source.get("simulated_page_url") is not None:
            raise ValueError("evidence_media_must_not_have_url")
        calls = [{
            "step_id": "analyze", "tool": _MEDIA_ACTIONS[modality],
            "depends_on": [], "input_binding": "source_media",
            "output": "source_observations",
        }]
        binding, primary = "source_media", _MEDIA_ACTIONS[modality]
    elif modality in {"text", "table"}:
        if route not in {"search_only", "direct_extract", "search_then_extract"}:
            raise ValueError("evidence_unknown_web_route:" + str(route))
        query_binding = ("upstream_key_query" if access_mode == "keyed_lookup"
                         else "question_cues_query")
        if route == "search_only":
            if source.get("simulated_page_url") is not None:
                raise ValueError("evidence_search_only_has_hidden_page")
            calls = [{
                "step_id": "search", "tool": "GoogleSearchAction",
                "depends_on": [], "input_binding": query_binding,
                "output": "search_snippet_with_fact_refs",
            }]
            primary, binding = "GoogleSearchAction", query_binding
        else:
            url = source.get("simulated_page_url")
            if not isinstance(url, str) or not url.startswith("https://"):
                raise ValueError("evidence_extraction_page_url_missing")
            parsed = urlparse(url)
            if (not parsed.hostname or not parsed.hostname.endswith(".example.invalid")
                    or parsed.username or parsed.password or parsed.port
                    or parsed.fragment or parsed.query or not parsed.path):
                raise ValueError("evidence_not_a_reserved_synthetic_url")
            calls = []
            if route == "search_then_extract":
                calls.append({
                    "step_id": "discover", "tool": "GoogleSearchAction",
                    "depends_on": [], "input_binding": query_binding,
                    "output": "search_result_with_selected_url",
                })
                url_binding = "discover.selected_result.url"
            else:
                url_binding = "known_source_url"
            calls.append({
                "step_id": "extract", "tool": "ExtractUrlContentAction",
                "depends_on": ["discover"] if route == "search_then_extract" else [],
                "input_binding": url_binding,
                "output": "page_facts_with_refs",
            })
            primary, binding = "ExtractUrlContentAction", url_binding
    else:
        raise ValueError("evidence_unsupported_modality:" + str(modality))
    if any(c["tool"] not in registered_o1_tool_names() for c in calls):
        raise ValueError("evidence_unregistered_tool")
    return {
        "version": EVIDENCE_TOOL_CONTRACT_VERSION,
        "modality": modality, "access_mode": access_mode,
        "strategy": route, "tool": primary, "input_binding": binding,
        "tool_steps": calls,
        "execution_mode": "simulated_no_tool_execution",
    }


def simulate_acquisition_trace(node: dict, source: dict,
                               supporting_fact_refs: list[str],
                               key_values: list[str] | None = None) -> list[dict]:
    """Produce an *offline fixture*, never an observed network tool trace.

    Search->Extract passes exactly the URL returned by the immediately
    preceding synthetic search result. An acquired FactRef first appears in
    Search-only or Extract's observation, not the discovery URL result.
    """
    contract = node["evidence_acquisition"]
    if contract["version"] != EVIDENCE_TOOL_CONTRACT_VERSION:
        return []  # v1 preserved without inventing a historical call history.
    steps = contract["tool_steps"]
    source_id = node["source_id"]
    if source_id != source["source_id"]:
        raise ValueError("simulated_trace_source_mismatch")
    query = (f"Find {source['purpose']} in {source['title']}."
             if node["op"] == "scan" else
             f"Find {source['purpose']} in {source['title']} using "
             f"{node['input_key_field']}={','.join(key_values or [])}.")
    url = source.get("simulated_page_url")
    calls = []
    for step in steps:
        call_id = node["id"] + "." + step["step_id"]
        params = {}
        observation = {"synthetic": True, "source_id": source_id}
        if step["tool"] == "GoogleSearchAction":
            params = {"query": query}
            if step["output"] == "search_result_with_selected_url":
                observation["selected_result"] = {
                    "url": url, "title": source["title"],
                    "source_id": source_id,
                }
            else:
                observation["content"] = (
                    "Simulated search snippet for " + source["title"]
                )
                observation["source_fact_refs"] = list(supporting_fact_refs)
        elif step["tool"] == "ExtractUrlContentAction":
            if step["input_binding"] == "discover.selected_result.url":
                if not calls or calls[-1]["tool"] != "GoogleSearchAction":
                    raise ValueError("simulated_extract_missing_discovery")
                selected_url = calls[-1]["observation"]["selected_result"]["url"]
                if selected_url != url:
                    raise ValueError("simulated_extraction_url_mismatch")
                params = {"url": selected_url, "browse_query": query}
            else:
                params = {"url": url, "browse_query": query}
            observation["url"] = params["url"]
            observation["source_fact_refs"] = list(supporting_fact_refs)
            observation["content"] = (
                "Simulated extracted page evidence for " + source["title"]
            )
        else:
            # Synthetic media annotations are not real decoding.
            path_field = {
                "ImageAnalysisAction": "image_path",
                "VideoAnalysisAction": "video_path",
                "ParseAudioAction": "audio_path",
            }.get(step["tool"])
            if path_field is None:
                raise ValueError("simulated_unsupported_tool:" + step["tool"])
            params = {"query": query, path_field: "workspace://" + source_id}
            observation["source_fact_refs"] = list(supporting_fact_refs)
        tool_schema = next(
            x["parameters"] for x in o1_tool_definitions()
            if x["name"] == step["tool"]
        )
        if not set(tool_schema.get("required", [])).issubset(params):
            raise ValueError("simulated_tool_missing_required_params")
        if set(params) - set(tool_schema.get("properties", {})):
            raise ValueError("simulated_tool_has_invalid_params")
        calls.append({
            "call_id": call_id, "tool": step["tool"],
            "params": params,
            "depends_on": [node["id"] + "." + x
                           for x in step["depends_on"]],
            "observation": observation,
            "execution_mode": "simulated_no_tool_execution",
            "tool_executed": False,
        })
    return calls


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
        names = [item["tool"] for item in acquisition.get("tool_steps", [])]
        if not names:
            names = [acquisition.get("tool")]
        for name in names:
            if name not in registered_o1_tool_names():
                raise ValueError("evidence_unknown_step_tool")
            if name not in tools:
                tools.append(name)
    return tools


def legacy_tools_from_task(task: dict, public_catalog: dict) -> list[str]:
    """Read-only compatibility for pre-contract Stage-1 Scene snapshots.

    This is NOT a new frozen Scene contract. Old Scenes have no acquisition
    metadata; regenerate with Stage 1 for a fully audited contract.
    """
    result = []
    for step in task["operator_chain"]:
        if step["op"] not in _ACQUISITION_OPS:
            continue
        sid = step["source_id"]
        modality = public_catalog[sid]["modality_hint"]
        if modality in _MEDIA_ACTIONS:
            tool = _MEDIA_ACTIONS[modality]
        elif modality in {"text", "table"}:
            # Synthetic records expose no real URL, so an O1 search-by-query
            # affordance is closer than pretending URL extraction is possible.
            tool = "GoogleSearchAction"
        else:
            raise ValueError("evidence_legacy_unknown_modality:" + str(modality))
        if tool not in result:
            result.append(tool)
    return result


def validate_scene_evidence_tools(scene: dict) -> list[str]:
    """Fail closed on source, primitive and Macro aggregation drift."""
    spec = scene["problem_spec"]
    if not spec.get("evidence_tool_contract_version"):
        return []  # Legacy source snapshots intentionally have no tool metadata.
    issues = []
    if spec["evidence_tool_contract_version"] not in {
        LEGACY_EVIDENCE_TOOL_CONTRACT_VERSION, EVIDENCE_TOOL_CONTRACT_VERSION,
    }:
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
