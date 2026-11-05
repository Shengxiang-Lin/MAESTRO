"""Only resource prerequisite checks needed for fragment evidence witnesses.

AND across required anchors, OR across providers of the same anchor.
This is an offline evidence constraint, NOT a Main free-exploration policy.
"""
from __future__ import annotations
from typing import Any, Mapping

INITIAL = {"attached", "provided", "initial", "visible", "provided_initial"}


def resolve_dependencies(resources: list[Mapping[str, Any]], registry: Mapping[str, Any]):
    by_id = {str(r["resource_id"]): r for r in resources}
    if len(by_id) != len(resources):
        raise ValueError("duplicate resource ID")
    question_anchors = set(registry.get("question_anchor_ids", ()))
    has_anchors = bool(registry) or any("requires_anchors" in r for r in resources)
    providers: dict[str, set[str]] = {}
    for sid, resource in by_id.items():
        for anchor in resource.get("produces_anchors", ()):
            providers.setdefault(str(anchor), set()).add(sid)
    dependencies: dict[str, list[set[str]]] = {}
    for sid, resource in by_id.items():
        if resource.get("visibility") in INITIAL:
            dependencies[sid] = []
        elif resource.get("metadata", {}).get("search_entry") == "public_question":
            # A searchable public index has a query entry point rather than
            # a fabricated initial document. The query ranking and witness
            # reachability are audited separately by the search simulator.
            dependencies[sid] = []
        elif has_anchors:
            anchors = set(map(str, resource.get("requires_anchors", ())))
            if anchors:
                dependencies[sid] = [
                    providers.get(anchor, set()) - {sid}
                    for anchor in sorted(anchors - question_anchors)
                ]
            else:
                # Some frozen worlds store explicit provider source IDs
                # rather than a named anchor registry.
                parents = set(map(str, resource.get("anchor_dependency_resource_ids", ())))
                if not parents:
                    raise ValueError(f"discoverable resource {sid} has no frozen acquisition prerequisite")
                if not parents.issubset(by_id) or sid in parents:
                    raise ValueError(f"invalid acquisition parents for {sid}")
                dependencies[sid] = [{parent} for parent in sorted(parents)]
        else:
            parents = set(map(str, resource.get("anchor_dependency_resource_ids", ())))
            if not parents.issubset(by_id) or sid in parents:
                raise ValueError(f"invalid resource dependencies for {sid}")
            dependencies[sid] = [{parent} for parent in sorted(parents)]
    pending, done, levels = set(by_id), set(), {}
    while pending:
        ready = {sid for sid in pending
                 if all(group & done for group in dependencies[sid])}
        if not ready:
            raise ValueError("unreachable or cyclic resource discovery anchors")
        next_level = len(levels) + 1
        for sid in ready:
            levels[sid] = next_level
        done.update(ready)
        pending.difference_update(ready)
    return dependencies, levels


def reachable_subset(source_ids, dependencies) -> bool:
    pending, done = set(source_ids), set()
    if not pending.issubset(dependencies):
        return False
    while pending:
        ready = {sid for sid in pending
                 if all(group & done for group in dependencies[sid])}
        if not ready:
            return False
        done.update(ready)
        pending.difference_update(ready)
    return True


__all__ = ["resolve_dependencies", "reachable_subset"]
