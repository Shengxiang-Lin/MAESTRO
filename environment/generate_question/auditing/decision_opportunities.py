"""Structural orchestration opportunities in a frozen trusted Stage-1 plan.

These are not learned-agent scores, conditional policies or Stage-4 fault
labels. They describe decisions that a Main-Agent may make with visible
evidence and explicitly distinguish numerical comparison from provenance
verification.
"""
from __future__ import annotations

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
