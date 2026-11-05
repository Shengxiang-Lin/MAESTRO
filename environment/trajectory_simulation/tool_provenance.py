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


__all__ = ["reconstruct_tool_observations", "permitted_tool_names"]
