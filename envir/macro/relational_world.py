"""Deterministic, source-cited relational structure for synthetic World.

Identity, categorical co-membership, explicit cross-source references and
relative temporal order are FOUR DIFFERENT relations. Neither shared group
nor cross-reference implies that two source records denote the same entity.
All relationships are fictional, never inferred from original media.
"""
from __future__ import annotations

from itertools import combinations

from envir.schema import digest

RELATION_FIELDS = (
    "synthetic_entity_key", "synthetic_reference_target_source_ids",
    "synthetic_precedes_source_ids",
)


def _number(*parts: object) -> int:
    return int(digest(*(str(p) for p in parts))[:12], 16)


def relational_layout(*, spec: dict, case_ref: str,
                      source_ids: list[str], goal_id: str) -> dict[str, dict]:
    """Return stable per-source facts including explicitly distinct relations.

    Conditional overlap provides positive AND empty counterexamples. A
    source's own entity key is not its observation/candidate key. For the
    composition Goal, explicit cross-source reference links can exist even
    without identity; for intersection, shared category does not establish
    shared entity. Ordinals are unique, and precedence links are ordered.
    """
    ids = sorted(set(source_ids))
    if len(ids) != len(source_ids) or not ids:
        raise ValueError("duplicate_or_empty_source_inventory")
    base = digest(spec["proposal_fingerprint"], case_ref, "relational_world_v1")
    order = sorted(ids, key=lambda sid: digest(base, sid, "temporal_order"))
    ordinals = {sid: i + 1 for i, sid in enumerate(order)}
    positive_overlap = _number(base, "category_overlap") % 4 != 0
    repeated_identity = _number(base, "entity_overlap") % 3 == 0
    rows = {}
    for i, sid in enumerate(order):
        # Category and identity are intentionally independent. In a positive
        # intersection, two distinct source handles share a CATEGORY only.
        if goal_id == "intersection":
            category = (
                "group-shared" if positive_overlap and i < 2
                else f"group-{i + 1}"
            )
        else:
            category = (
                f"group-{i // 2}" if _number(base, "cohort_distribution") % 3
                else f"group-{i}"
            )
        entity = (
            "entity-" + digest(base, order[0], "same_entity")[:10]
            if repeated_identity and i == 1 else
            "entity-" + digest(base, sid, "same_entity")[:10]
        )
        reference_targets = (
            [order[i - 1]]
            if i > 0 and (goal_id == "composition" or
                          _number(base, sid, "reference") % 4 == 0)
            else []
        )
        precedes = [order[i + 1]] if i + 1 < len(order) else []
        rows[sid] = {
            "candidate_key": "candidate-" + digest(base, sid, "observation")[:10],
            "entity_key": entity,
            "score": _number(base, sid, "score") % 101,
            "positive_flag": _number(base, sid, "binary_mark") % 2 == 0,
            "relative_time_index": ordinals[sid],
            "group_key": category,
            "reference_target_source_ids": reference_targets,
            "precedes_source_ids": precedes,
        }
    # For explicit entity overlap, use the same entity key for first two
    # sources, while preserving their *distinct* candidate observation IDs.
    if repeated_identity and len(order) > 1:
        rows[order[1]]["entity_key"] = rows[order[0]]["entity_key"]
    return rows


def world_relations(sources: dict) -> list[dict]:
    """Derive fully source-traceable typed edges from World source facts."""
    facts = {
        sid: {f["field"]: f for f in src["facts"]}
        for sid, src in sources.items()
    }
    if not facts or any(not set(RELATION_FIELDS) <= set(row)
                        for row in facts.values()):
        return []
    edges = []
    ids = sorted(facts)
    for left, right in combinations(ids, 2):
        a, b = facts[left], facts[right]
        if a["synthetic_entity_key"]["value"] == b["synthetic_entity_key"]["value"]:
            edges.append({
                "kind": "same_entity",
                "source_ids": [left, right],
                "entity_key": a["synthetic_entity_key"]["value"],
                "fact_refs": [
                    f"{left}:{a['synthetic_entity_key']['fact_id']}",
                    f"{right}:{b['synthetic_entity_key']['fact_id']}",
                ],
            })
        if a["synthetic_group_key"]["value"] == b["synthetic_group_key"]["value"]:
            edges.append({
                "kind": "shared_category",
                "source_ids": [left, right],
                "group_key": a["synthetic_group_key"]["value"],
                "fact_refs": [
                    f"{left}:{a['synthetic_group_key']['fact_id']}",
                    f"{right}:{b['synthetic_group_key']['fact_id']}",
                ],
            })
    for sid in ids:
        row = facts[sid]
        for target in row["synthetic_reference_target_source_ids"]["value"]:
            if target not in sources or target == sid:
                raise ValueError("invalid_reference_target:" + str(target))
            edges.append({
                "kind": "references",
                "source_ids": [sid, target],
                "fact_refs": [
                    f"{sid}:{row['synthetic_reference_target_source_ids']['fact_id']}"
                ],
            })
        for target in row["synthetic_precedes_source_ids"]["value"]:
            if target not in sources or target == sid:
                raise ValueError("invalid_precedence_target:" + str(target))
            if row["synthetic_time_index"]["value"] >= facts[target][
                    "synthetic_time_index"]["value"]:
                raise ValueError("nonforward_precedence:" + sid)
            edges.append({
                "kind": "precedes",
                "source_ids": [sid, target],
                "fact_refs": [
                    f"{sid}:{row['synthetic_precedes_source_ids']['fact_id']}",
                    f"{sid}:{row['synthetic_time_index']['fact_id']}",
                    f"{target}:{facts[target]['synthetic_time_index']['fact_id']}",
                ],
            })
    return sorted(edges, key=lambda x: (
        x["kind"], x["source_ids"], x.get("entity_key", ""),
        x.get("group_key", ""),
    ))
