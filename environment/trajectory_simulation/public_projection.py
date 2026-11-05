"""Conservative projection of deliberately concealed Stage-1 identities.

The deterministic question renderer may conceal a singleton FIND because an
independent branch of the same INTERSECT already identifies the entity
uniquely. That direct FIND is NOT a publicly executable O1 subtask.
Only in this exact documented case may we remove the concealed FIND and its
redundant INTERSECT, keeping the genuinely observable evidence branch.
All other latent identities must pass public replay or be quarantined.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

from .compiler import _same, ready_fragments


def _consumer_ids(fragments: list[dict[str, Any]], fragment_id: str) -> set[str]:
    result = set()
    for fragment in fragments:
        if fragment_id in fragment["depends_on"] or any(
            binding.get("kind") == "completed" and binding.get("fragment_id") == fragment_id
            for op in fragment["operations"]
            for binding in op.get("input_bindings", ())
        ):
            result.add(fragment["fragment_id"])
    return result


def project_concealed_intersections(
    original: Mapping[str, Any], rendered: Any,
) -> dict[str, Any]:
    """Eliminate *only* provably redundant private singleton checks.

    Preconditions are checked against the explicitly frozen concealment
    metadata, not inferred from a matching answer alone.
    """
    plan = copy.deepcopy(dict(original))
    query_plan = (getattr(rendered, "hidden_metadata", {}) or {}).get(
        "public_query_plan", {}
    )
    entries = [
        *query_plan.get("latent_entities", []),
        *plan.get("projection_candidates", []),
    ]
    plan["public_projections"] = []
    plan["public_projection_diagnostics"] = []
    for rule in entries:
        if rule.get("rule") not in {
            "unique_intersection_clue", "witness_verified_intersection",
        }:
            continue
        for find_id in rule.get("find_node_ids", []):
            fragments = plan["fragments"]
            by_output = {f["output_node_id"]: f for f in fragments}
            hidden = by_output.get(find_id)
            if hidden is None:
                continue
            # A concealed root may only be removed if it is a standalone
            # ungrounded singleton, not a genuine tool/evidence subtask.
            if (
                len(hidden["operations"]) != 1
                or hidden["operations"][0]["op"] != "FIND"
                or hidden["depends_on"] or hidden["observations"]
                or hidden["resource_ids"]
            ):
                continue
            # The metadata-independent proof is usable only for a genuinely
            # concealed singleton, never an explicitly named entity.
            if rule["rule"] == "witness_verified_intersection":
                public_input = hidden["operations"][0].get("inputs", [""])[0]
                if not str(public_input).startswith(
                    "the entity matching the question clues"
                ):
                    continue
            if not isinstance(hidden["expected_answer"], list) or len(hidden["expected_answer"]) != 1:
                continue
            for visible_id in rule.get("identifying_node_ids", []):
                visible = by_output.get(visible_id)
                if visible is None or visible is hidden:
                    continue
                # The identifying branch must have actually observable evidence.
                # Evidence can be read in this fragment OR inherited from
                # completed predecessors. Do not demand a redundant new read.
                if not visible["support_resource_ids"]:
                    continue
                if not (visible["observations"] or visible.get("known_observations")
                        or visible["depends_on"]):
                    continue
                if not _same(hidden["expected_answer"], visible["expected_answer"]):
                    continue
                for join in list(fragments):
                    if join in (hidden, visible):
                        continue
                    if (rule.get("join_node_id") is not None
                        and join["output_node_id"] != rule["join_node_id"]):
                        continue
                    if len(join["operations"]) != 1:
                        continue
                    op = join["operations"][0]
                    if op["op"] != "INTERSECT":
                        continue
                    if set(op["dependencies"]) != {find_id, visible_id} or len(op["dependencies"]) != 2:
                        continue
                    if set(join["depends_on"]) != {
                        hidden["fragment_id"], visible["fragment_id"]
                    }:
                        continue
                    if join["observations"] or join["resource_ids"]:
                        continue
                    if not _same(join["expected_answer"], visible["expected_answer"]):
                        continue
                    if not set(join["support_resource_ids"]).issubset(
                        set(visible["support_resource_ids"])
                    ):
                        continue
                    if _consumer_ids(fragments, hidden["fragment_id"]) != {
                        join["fragment_id"]
                    }:
                        continue
                    if _consumer_ids(fragments, visible["fragment_id"]) - {
                        join["fragment_id"]
                    }:
                        # Reusing the visible branch elsewhere is safe, but
                        # avoid modifying more complex shared shapes for now.
                        continue

                    removed = {hidden["fragment_id"], join["fragment_id"]}
                    replacement = visible["fragment_id"]
                    for fragment in fragments:
                        if fragment["fragment_id"] in removed:
                            continue
                        fragment["depends_on"] = list(dict.fromkeys(
                            replacement if item == join["fragment_id"] else item
                            for item in fragment["depends_on"]
                        ))
                        for operation in fragment["operations"]:
                            for binding in operation.get("input_bindings", []):
                                if (binding.get("kind") == "completed"
                                    and binding.get("fragment_id") == join["fragment_id"]):
                                    binding["fragment_id"] = replacement
                    plan["fragments"] = [
                        fragment for fragment in fragments
                        if fragment["fragment_id"] not in removed
                    ]
                    if plan["final_fragment_id"] in removed:
                        plan["final_fragment_id"] = replacement
                    plan["terminal"]["depends_on"] = list(dict.fromkeys(
                        replacement if item == join["fragment_id"] else item
                        for item in plan["terminal"]["depends_on"]
                        if item != hidden["fragment_id"]
                    ))
                    plan["public_projections"].append({
                        "rule": rule["rule"],
                        "concealed_find_node_id": find_id,
                        "identifying_node_id": visible_id,
                        "removed_fragment_ids": sorted(removed),
                        "retained_fragment_id": replacement,
                    })
                    break
                else:
                    continue
                break

    # All candidates and outcomes stay in private diagnostics; never show
    # hidden node identities or the proof to student Main/Sub-Agent prompts.
    projected = {entry["concealed_find_node_id"] for entry in plan["public_projections"]}
    plan["public_projection_diagnostics"] = [{
        "find_node_ids": list(rule.get("find_node_ids", ())),
        "identifying_node_ids": list(rule.get("identifying_node_ids", ())),
        "rule": rule["rule"],
        "status": "projected" if any(
            int(fid) in projected for fid in rule.get("find_node_ids", ())
        ) else "not_proven_safe",
    } for rule in entries if rule.get("rule") in {
        "unique_intersection_clue", "witness_verified_intersection",
    }]
    if not plan["public_projections"]:
        return plan
    # The projected graph retains the private symbolic trace/oracle and
    # only changes the delegation view. Recompute causal parallel waves.
    completed: set[str] = set()
    waves: list[list[str]] = []
    while len(completed) < len(plan["fragments"]):
        ready = ready_fragments(plan, completed)
        if not ready:
            raise ValueError("invalid public projection: unresolved fragment dependencies")
        wave = [fragment["fragment_id"] for fragment in ready]
        waves.append(wave)
        completed.update(wave)
    plan["waves"] = waves
    return plan


__all__ = ["project_concealed_intersections"]
