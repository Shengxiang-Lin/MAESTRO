"""Validate grounded simulated observations and preserve semantic boundaries.

Report whether information used in the frozen Gold is actually rendered as
a modality-specific observation. Annotation text is NOT a new truth source.
"""
from __future__ import annotations

from collections import Counter


MEDIA = frozenset({"image", "audio", "video"})
PREDICATES = {
    "image": {"visual_measurement_reading", "visual_feature_recognition"},
    "audio": {"spoken_measurement_extraction", "spoken_identifier_extraction"},
    "video": {"video_event_recognition"},
}


def audit_scene_grounding(scene: dict) -> dict:
    spec = scene["problem_spec"]
    expected = {s["source_id"]: s for s in spec["sources"]}
    source_ops = {
        sid for task in spec["macro_plan"]["macro_tasks"]
        for sid in task["source_scope"]
    }
    counts = Counter()
    for sid, source in scene["world"]["sources"].items():
        role = expected[sid]
        for item in source["observations"]:
            counts["observations"] += 1
            if "document_context" in item:
                counts["observations_with_context"] += 1
            semantics = item.get("observation_semantics")
            if semantics is None:
                counts["missing_semantic_annotations"] += 1
                continue
            fields = semantics["derived_from_value_fields"]
            text = item.get("observed_content", "")
            if (semantics.get("modality") != role["modality"]
                    or any(f not in item["values"] for f in fields)
                    or any(str(item["values"][f]) not in text for f in fields)):
                counts["value_mismatches"] += 1
            if role["modality"] in MEDIA:
                counts["media_observations"] += 1
                if (sid in source_ops and semantics["predicate"] in
                        PREDICATES[role["modality"]]):
                    counts["operational_media_observations"] += 1
                else:
                    counts["unlicensed_media_predicates"] += 1
                if sid in {"metric_a", "metric_b"}:
                    counts["media_goal_measurement_observations"] += 1
            # A required target-discriminating affiliation is restricted
            # to its intended source; other document contexts must NOT
            # reveal the shared collection ID/label via metadata.
            required = spec["macro_plan"]["world_binding"].get(
                "world_model", {}).get("required_context_relation")
            if required and "document_context" in item:
                keys = set(item["document_context"])
                if keys.intersection({
                    "collection_id", "collection_title", "anchor_label"
                }):
                    counts["context_leakage_warnings"] += 1
    return {name: counts[name] for name in (
        "observations", "observations_with_context",
        "missing_semantic_annotations", "value_mismatches",
        "media_observations", "operational_media_observations",
        "unlicensed_media_predicates",
        "media_goal_measurement_observations",
        "context_leakage_warnings",
    )}


def summarize_grounded_evidence(scenes: list[dict]) -> dict:
    results = [audit_scene_grounding(scene) for scene in scenes]
    sums = Counter()
    for row in results:
        sums.update(row)
    return {
        "scenes": len(results),
        **{name: sums[name] for name in (
            "observations", "observations_with_context",
            "missing_semantic_annotations", "value_mismatches",
            "media_observations", "operational_media_observations",
            "unlicensed_media_predicates",
            "media_goal_measurement_observations",
            "context_leakage_warnings",
        )},
        "scenes_with_operational_media": sum(
            r["operational_media_observations"] > 0 for r in results
        ),
        "scenes_with_media_derived_goal_facts": sum(
            r["media_goal_measurement_observations"] > 0 for r in results
        ),
        "all_grounding_checks_pass": all(
            r["missing_semantic_annotations"] == 0
            and r["value_mismatches"] == 0
            and r["context_leakage_warnings"] == 0
            and r["unlicensed_media_predicates"] == 0
            for r in results
        ),
        "scope": (
            "semantic fidelity of simulated source annotations and "
            "Goal evidence, NOT binary media understanding"
        ),
    }
