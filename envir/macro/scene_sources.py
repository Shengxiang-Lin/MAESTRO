"""Source inventory for the Task-First Macro planner.

Scene Bank entries supply source handles and modality hints, not original
question logic. Source packages may be grouped by modality, but all original
supporting handles must remain reachable from the terminal work product.
"""
from __future__ import annotations

import random


MODALITY_ROLES = {
    "table": "records",
    "text": "context",
    "image": "material",
}
MODALITY_CAPABILITIES = {
    "table": "TableAnalysis",
    "text": "DocumentAnalysis",
    "image": "ImageAnalysis",
}


class IneligibleScene(ValueError):
    pass


def source_key(item: dict) -> tuple[str, str | None]:
    return item["modality"], item.get("doc_id")


def unique_sources(seed: dict) -> list[dict]:
    seen, result = set(), []
    for item in seed.get("supporting_sources", []):
        modality = item.get("modality")
        if modality not in MODALITY_ROLES:
            continue
        key = (modality, item.get("doc_id"))
        if key in seen:
            continue
        result.append(dict(item))
        seen.add(key)
    for modality in seed.get("source_modalities", []):
        if modality in MODALITY_ROLES and not any(
            item["modality"] == modality for item in result
        ):
            item = {"modality": modality, "doc_id": None,
                    "provenance": "mmqa_modality_role_only"}
            result.append(item)
    return result


def partition_sources(seed: dict, rng: random.Random,
                      max_roots: int = 4) -> tuple[list[list[dict]], dict]:
    sources = unique_sources(seed)
    if len(sources) < 2:
        raise IneligibleScene("at least two distinct supporting sources required")
    modality_groups = {
        mod: [item for item in sources if item["modality"] == mod]
        for mod in MODALITY_ROLES if any(s["modality"] == mod for s in sources)
    }
    buckets = [items for items in modality_groups.values()]
    if len(buckets) > max_roots:
        raise IneligibleScene("too many distinct modality groups")
    desired = rng.randint(max(2, len(buckets)), min(max_roots, len(sources)))
    while len(buckets) < desired:
        candidates = [i for i, bucket in enumerate(buckets) if len(bucket) > 1]
        if not candidates:
            break
        i = max(candidates, key=lambda j: (len(buckets[j]), rng.random()))
        split = buckets[i].pop()
        buckets.append([split])
    rng.shuffle(buckets)
    expected = {source_key(item) for item in sources}
    observed = {source_key(item) for bucket in buckets for item in bucket}
    if observed != expected:
        raise IneligibleScene("source partition coverage mismatch")
    return buckets, {
        "expected": len(expected), "assigned": len(observed),
        "source_keys": [
            {"modality": mod, "doc_id": doc_id}
            for mod, doc_id in sorted(expected, key=lambda x: (x[0], str(x[1])))
        ],
    }


def requirement(source: dict, *, topic: str) -> dict:
    mod, doc_id = source_key(source)
    return {
        "role": MODALITY_ROLES[mod],
        "description": (
            f"MultiModalQA {mod} reference {doc_id}" if doc_id is not None
            else f"simulated {mod} role for {topic}"
        ),
        "access": "synthetic_observation_only",
        "candidate_modalities": [mod],
        "source_origin": source["provenance"],
        "original_doc_id": doc_id,
        "source_leaf": "supporting_context",
        "synthetic": True,
    }
