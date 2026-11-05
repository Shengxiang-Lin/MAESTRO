"""Verified, *simulated* tool provenance for clean Stage-3 Sub-Agent reports.

Actual O1 traces pair native tool actions with observations. Stage 3 has
frozen public source witnesses rather than a live browser/media tool, so this
adapter constructs only those source-acquisition/inspection actions that the
witness and native tool profile support. These are NOT live tool call logs.

No query strings, video timestamps, URLs, failures, timings, or observations
are guessed. Tool-origin provenance remains private bank metadata for audit.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .tools import tool_profile


def reconstruct_tool_observations(
    fragment: Mapping[str, Any],
) -> dict[str, Any]:
    profile = tool_profile(fragment.get("scene", "omnigaia"))
    fresh = {item["resource_id"]: item for item in fragment.get("observations", [])}
    known = {item["resource_id"]: item for item in fragment.get("known_observations", [])}
    if fresh.keys() & known.keys():
        raise ValueError("source_is_both_fresh_and_previously_known")

    steps = []
    emitted = set()

    def append_fresh(item: Mapping[str, Any]) -> None:
        source_id = item["resource_id"]
        if source_id in emitted:
            return
        emitted.add(source_id)
        actions = profile.for_resource(item)
        allowed = set(fragment.get("suggested_tools", []))
        if not set(actions) <= allowed:
            raise ValueError(f"tool_not_assigned_for_public_source:{source_id}")
        for tool in actions:
            steps.append({
                "kind": "source_discovery" if tool in profile.search
                        else "source_inspection",
                "tool": tool,
                "resource_id": source_id,
                "source_title": item["title"],
                "source_type": item["source_type"],
                # Reference exact public claim IDs rather than generating
                # speculative search snippets or multimodal transcripts.
                "claim_keys": [
                    claim.get("observation_key")
                    for claim in item.get("claims", [])
                    if claim.get("observation_key") is not None
                ],
            })

    # Causally valid order from the original resource-acquisition DAG.
    for wave in fragment.get("acquisition_waves", []):
        for source_id in wave:
            if source_id not in fresh:
                raise ValueError(f"missing_public_acquisition_source:{source_id}")
            append_fresh(fresh[source_id])
    for item in fragment.get("observations", []):
        # Compatibility with old witnesses with no acquisition wave metadata.
        append_fresh(item)
    for item in fragment.get("known_observations", []):
        steps.append({
            "kind": "previously_available_evidence",
            "tool": None, "resource_id": item["resource_id"],
            "source_title": item["title"],
            "source_type": item["source_type"],
            "claim_keys": [
                c.get("observation_key") for c in item.get("claims", [])
                if c.get("observation_key") is not None
            ],
        })
    return {
        "origin": "reconstructed_from_frozen_public_witnesses",
        "live_tool_calls": False,
        "steps": steps,
    }


def permitted_tool_names(plan: Mapping[str, Any]) -> set[str]:
    return {step["tool"] for step in plan.get("steps", [])
            if isinstance(step.get("tool"), str)}


def source_evidence_guide(
    fragment: Mapping[str, Any], plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Join the permitted source-access steps with exactly those source's facts.

    The model should not have to infer which source supports a relationship by
    separately joining long lists of tool calls and ungrouped claims. This is
    derived solely from the same public witnesses used by replay.
    """
    sources = {
        item["resource_id"]: item
        for item in [
            *fragment.get("observations", []),
            *fragment.get("known_observations", []),
        ]
    }
    groups: dict[str, dict[str, Any]] = {}
    if not plan.get("steps"):
        # Direct writer/reviewer tests and legacy consumers can supply
        # source observations without a tool-plan packet. Never infer a tool
        # call from source availability alone in this fallback path.
        return [{
            "source_id": sid,
            "title": source["title"],
            "source_type": source.get("source_type", "unspecified"),
            "source_access": "source_witness_only",
            "permitted_tools": [],
            "claims": source.get("claims", []),
        } for sid, source in sources.items()]
    for event in plan.get("steps", []):
        sid = event["resource_id"]
        if sid not in sources:
            raise ValueError("tool_plan_refers_to_unobserved_source:" + sid)
        source = sources[sid]
        if sid not in groups:
            groups[sid] = {
                "source_id": sid,
                "title": source["title"],
                "source_type": source["source_type"],
                "source_access": "reused_prior_evidence"
                    if event["kind"] == "previously_available_evidence"
                    else "newly_available_public_evidence",
                "permitted_tools": [],
                "claims": source.get("claims", []),
            }
        tool = event.get("tool")
        if tool and tool not in groups[sid]["permitted_tools"]:
            groups[sid]["permitted_tools"].append(tool)
    if set(groups) != set(sources):
        raise ValueError("tool_plan_does_not_cover_all_scoped_sources")
    return list(groups.values())


def unsupported_tool_actions(text: str, plan: Mapping[str, Any]) -> set[str]:
    """Detect explicit/implied unsupported *tool actions*, not plain prose.

    Human O1 trace summaries naturally say 'examined the clip' instead of
    'VideoAnalysisAction was executed'. Do not require literal class names,
    but still reject action claims that have no corresponding witness-backed
    tool family. Factual attribution remains separately language-reviewed.
    """
    import re

    allowed = permitted_tool_names(plan)
    known = set(tool_profile("omnigaia").available)
    missing = {
        name for name in known
        if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text)
        and name not in allowed
    }
    implicit = {
        "GoogleSearchAction": (
            r"\b(?:web search|online search|search results|search engine|"
            r"(?:initial|refined|targeted|follow-up) search|"
            r"searched (?:the web|online)|performed (?:a )?search)\b"
        ),
        "VideoAnalysisAction": (
            r"\b(?:analy[sz]ed|examined|inspected|reviewed)\s+"
            r"(?:the |a |an )?(?:supplied |attached |provided )?"
            r"(?:video|footage|clip|video frames?)\b|"
            r"\b(?:video|clip|footage)\s+(?:was |were )?"
            r"(?:analy[sz]ed|examined|inspected|reviewed)\b"
        ),
        "ImageAnalysisAction": (
            r"\b(?:analy[sz]ed|examined|inspected)\s+"
            r"(?:the |a |an )?(?:supplied |attached |provided )?"
            r"(?:image|photo|photograph|picture)\b"
        ),
        "ParseAudioAction": (
            r"\b(?:transcribed|listened to|analy[sz]ed)\s+"
            r"(?:the |a |an )?(?:supplied |attached |provided )?"
            r"(?:audio|recording|speech)\b|"
            r"\b(?:(?:audio|speech) )?recording\s+(?:was |were )?"
            r"(?:transcribed|analy[sz]ed)\b"
        ),
        "ExtractUrlContentAction": (
            r"\b(?:extracted|opened|fetched)\s+(?:the |a |an |supplied )?"
            r"(?:webpage|web page|website|URL)\b"
        ),
        "ExecuteCodeAction": (
            r"\b(?:ran|executed)\s+(?:a |the )?"
            r"(?:Python (?:script|code)|script|code)\b"
        ),
    }
    for name, pattern in implicit.items():
        if name not in allowed and re.search(pattern, text, re.I):
            missing.add(name)
    # No generic click/browser commands are reconstructed by this adapter.
    if re.search(r"\b(?:I clicked|clicked (?:on |through )|browser automation)\b",
                 text, re.I):
        missing.add("unobserved_browser_action")
    return missing


__all__ = ["reconstruct_tool_observations", "permitted_tool_names",
           "source_evidence_guide", "unsupported_tool_actions"]
