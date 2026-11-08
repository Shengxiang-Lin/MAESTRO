"""Macro TaskGraph -> initial Question -> synthetic task evidence -> clean feedback.

For the dataset-driven path, the MMQA question and composition semantics are
the anchors; the Macro TaskGraph is compiled first, then the deterministic
question is derived from its terminal work goal. No fabricated media decoding
or Gold factual answer is claimed.
"""
from __future__ import annotations

import random
from collections import Counter

from .data_taskfirst import UnsuitableSeed, compose_seed_taskfirst
from .v4_taskfirst import compose_taskfirst
from .v4_world import build_clean_world
from .v4_evidence import run_clean_tasks


def build_one(*, ordinal: int, seed: int, kqa: list[dict],
              mmqa: list[dict] | None = None,
              seed_record: dict | None = None) -> dict:
    if seed_record is not None:
        plan = compose_seed_taskfirst(
            seed_record=seed_record, ordinal=ordinal, run_seed=seed, kqa=kqa)
    else:
        # Retained only for existing direct unit-test fixtures. The public
        # CLI uses MMQA-derived scenario seeds by default.
        plan = compose_taskfirst(ordinal=ordinal, seed=seed, kqa=kqa, mmqa=mmqa or [])
    world = build_clean_world(plan, seed=seed)
    execution = run_clean_tasks(plan, world)
    public_sources = [
        {"source_id": sid, "title": s["title"],
         "modality_hint": s["modality_hint"],
         "representation": s["representation"]}
        for sid, s in sorted(world["sources"].items())
    ]
    scene = {
        **plan,
        "construction_status": "clean_task_evidence_replayed",
        "world": world,
        **execution,
        "public": {
            "task_id": plan["task_id"],
            "question": plan["question"],
            "available_source_descriptions": public_sources,
            "max_concurrent_subagents": 10,
            "status": "synthetic_clean_world",
        },
        "clean_assumptions": {
            "all_required_evidence_obtained": True,
            "all_subagents_reliable": True,
            "no_evidence_conflicts": True,
            "no_untrusted_injection_or_recovery": True,
            "real_multimodal_perception_performed": False,
        },
    }
    from .v4_validation import validate_clean_scene
    errors = validate_clean_scene(scene)
    if errors:
        raise ValueError("invalid V4.5 scene: " + "; ".join(errors))
    return scene


def build_batch(*, kqa: list[dict], mmqa: list[dict] | None = None,
                scenario_seeds: list[dict] | None = None,
                target: int = 100, seed: int = 17) -> tuple[list[dict], dict]:
    if target < 1 or not kqa:
        raise ValueError("positive target and nonempty KQA program bank required")
    rejected = Counter()
    if scenario_seeds is None:
        if not mmqa:
            raise ValueError("MMQA priors required")
        scenes = [build_one(ordinal=i, seed=seed, kqa=kqa, mmqa=mmqa)
                  for i in range(target)]
    else:
        # Shuffle original data IDs reproducibly, never repeat a question just
        # to satisfy the requested count. Unsuitable 1-source/simple items
        # are reported and skipped, rather than padded with dummy tasks.
        order = list(scenario_seeds)
        random.Random(seed * 271828).shuffle(order)
        scenes = []
        for index, item in enumerate(order):
            if len(scenes) >= target:
                break
            try:
                scene = build_one(
                    ordinal=index, seed=seed, kqa=kqa, seed_record=item)
            except UnsuitableSeed as exc:
                rejected[str(exc)] += 1
                continue
            scenes.append(scene)
        if len(scenes) < target:
            raise ValueError(
                f"Only {len(scenes)} of {len(order)} MMQA seeds yielded valid "
                f"3-9-task compositions; target={target}; rejections={dict(rejected)}"
            )
    n_tasks = sum(len(s["macro_graph"]["tasks"]) for s in scenes)
    n_evidence = sum(
        len(g["nodes"]) for s in scenes for g in s["task_evidence_graphs"].values()
    )
    kinds = Counter(t["work_type"] for s in scenes
                    for t in s["macro_graph"]["tasks"])
    intents = Counter(s["question_brief"]["intent"] for s in scenes)
    graph_sig = lambda s: (
        tuple(t["work_type"] for t in s["macro_graph"]["tasks"]),
        tuple(sorted((e["from"], e["to"], e["input_slot"])
                     for e in s["macro_graph"]["edges"])),
    )
    return scenes, {
        "generated": len(scenes),
        "unique_question_texts": len({s["question"] for s in scenes}),
        "macro_task_count": n_tasks,
        "evidence_node_count": n_evidence,
        "mean_evidence_nodes_per_task": round(n_evidence / n_tasks, 2),
        "unique_macro_work_types": len(kinds),
        "macro_work_types": dict(kinds),
        "macro_graph_signatures": len({graph_sig(s) for s in scenes}),
        "intents": dict(intents),
        "macro_tasks_per_question": dict(Counter(
            len(s["macro_graph"]["tasks"]) for s in scenes)),
        "evidence_nodes_per_task": dict(Counter(
            len(g["nodes"]) for s in scenes
            for g in s["task_evidence_graphs"].values())),
        "clean_evidence_coverage": sum(
            s["goal_evidence_coverage"]["can_finish_cleanly"] for s in scenes),
        "fixed_max_concurrency": 10,
        "final_gold_answer_generated": False,
        "untrusted_scenarios_generated": False,
        "media_policy": "synthetic_text_records_only",
        "scenario_seed_rejections": dict(rejected),
        "question_provenance": (
            "MMQA original question encoded in terminal task goal"
            if scenario_seeds is not None else "deterministic task goal template"),
    }
