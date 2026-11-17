"""Frozen Stage-1 evidence quality and dependency-structure metrics.

The metrics characterize executable source consistency and partial-order
freedom; they do not label alternative schedules as optimal policies or
claim binary image/audio/video perception.
"""
from __future__ import annotations

# --- dependency and scheduling-opportunity audit ---
from collections import Counter
from functools import lru_cache
from statistics import median


def _partial_order_stats(tasks: list[dict]) -> dict:
    """Quantify scheduling freedom under *actual* information dependencies.

    Counts topological orders exactly for the bounded (<=11 Macro) DAG.
    A legal order is only a scheduling alternative, NOT a distinct learned
    decision policy or an optional feedback-dependent action.
    """
    n = len(tasks)
    if n > 20:
        raise ValueError("unbounded_macro_partial_order_audit")
    lookup = {task["macro_id"]: i for i, task in enumerate(tasks)}
    if len(lookup) != n:
        raise ValueError("duplicate_macro_partial_order_audit")
    parents = []
    ancestors = []
    for task in tasks:
        mask = 0
        closure = 0
        for parent in task["depends_on"]:
            if parent not in lookup or lookup[parent] >= len(parents):
                raise ValueError("non_topological_partial_order_audit")
            j = lookup[parent]
            mask |= 1 << j
            closure |= 1 << j
            closure |= ancestors[j]
        parents.append(mask)
        ancestors.append(closure)

    @lru_cache(maxsize=None)
    def extensions(done: int) -> int:
        if done == (1 << n) - 1:
            return 1
        return sum(
            extensions(done | (1 << j))
            for j in range(n)
            if not done & (1 << j) and parents[j] & done == parents[j]
        )

    incomparable = sum(
        not (ancestors[i] & (1 << j) or ancestors[j] & (1 << i))
        for i in range(n) for j in range(i + 1, n)
    )
    possible_pairs = n * (n - 1) // 2
    return {
        "legal_topological_orders": extensions(0),
        "incomparable_task_pairs": incomparable,
        "incomparable_pair_share": round(
            incomparable / possible_pairs, 4
        ) if possible_pairs else 0.0,
        "source_independent_roots": sum(not mask for mask in parents),
    }


def audit_decision_opportunities(spec: dict) -> dict:
    plan = spec["macro_plan"]
    tasks = plan["macro_tasks"]
    by_id = {t["macro_id"]: t for t in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("duplicate_macro_in_decision_audit")
    remaining = set(by_id)
    completed = set()
    waves = []
    while remaining:
        ready = sorted(
            mid for mid in remaining
            if set(by_id[mid]["depends_on"]) <= completed
        )
        if not ready:
            raise ValueError("cyclic_macro_decision_audit")
        waves.append(ready)
        completed.update(ready)
        remaining.difference_update(ready)
    uses = Counter(parent for task in tasks
                   for parent in task["depends_on"])
    sources = {s["source_id"]: s for s in spec["sources"]}
    media_fact_ids = sorted({
        sid for task in tasks
        if task["operation"] == "fetch_typed_evidence"
        for sid in task["source_scope"]
        if sources[sid]["modality"] in {"image", "audio", "video"}
    })
    return {
        **_partial_order_stats(tasks),
        "max_parallel_width": max((len(x) for x in waves), default=0),
        "parallel_waves": sum(len(x) > 1 for x in waves),
        "keyed_handoffs": sum(
            bool(task["depends_on"]) and bool(task["source_scope"]) and
            all(sources[sid]["access_mode"] == "keyed_lookup"
                for sid in task["source_scope"])
            for task in tasks
        ),
        "shared_work_products": sum(used > 1 for used in uses.values()),
        "independent_roots": len(waves[0]) if waves else 0,
        "evidence_corroboration_macros": sum(
            task["operation"] == "verify_real_fork" for task in tasks
        ),
        "media_fact_source_ids": media_fact_ids,
        # Plans are frozen DAGs; they do not encode optional future actions.
        "independent_media_acquisitions": sum(
            task["operation"] == "fetch_typed_evidence"
            and not task["depends_on"]
            and any(sources[sid]["modality"] in {"image", "audio", "video"}
                    for sid in task["source_scope"])
            for task in tasks
        ),
        "conditional_feedback_dispatch": False,
    }


def summarize_decision_opportunities(scenes: list[dict]) -> dict:
    rows = [audit_decision_opportunities(scene["problem_spec"])
            for scene in scenes]
    return {
        "scenes": len(rows),
        "with_parallel_waves": sum(r["parallel_waves"] > 0 for r in rows),
        "with_keyed_handoffs": sum(r["keyed_handoffs"] > 0 for r in rows),
        "with_reusable_work": sum(r["shared_work_products"] > 0 for r in rows),
        "with_evidence_corroboration": sum(
            r["evidence_corroboration_macros"] > 0 for r in rows
        ),
        "with_media_derived_facts": sum(
            bool(r["media_fact_source_ids"]) for r in rows
        ),
        "mean_parallel_width": round(
            sum(r["max_parallel_width"] for r in rows)/len(rows), 3
        ) if rows else 0.0,
        "with_independent_media_acquisition": sum(
            r["independent_media_acquisitions"] > 0 for r in rows
        ),
        "conditional_feedback_dispatch_scenes": 0,
        "median_legal_topological_orders": median(
            r["legal_topological_orders"] for r in rows
        ) if rows else 0,
        "mean_incomparable_task_pair_share": round(
            sum(r["incomparable_pair_share"] for r in rows) / len(rows), 4
        ) if rows else 0.0,
        "mean_independent_roots": round(
            sum(r["source_independent_roots"] for r in rows) / len(rows), 3
        ) if rows else 0.0,
        "scope": (
            "Structural freedom under frozen Macro DAG dependencies; "
            "topological orders are NOT distinct feedback-dependent policies."
        ),
    }

# --- modality and source-evidence integrity audit ---
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
