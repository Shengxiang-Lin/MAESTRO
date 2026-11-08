"""Dataset-driven Macro TaskGraph -> Question -> simulated clean evidence."""
from __future__ import annotations

import random
from collections import Counter

from .recipe_planner import UnsuitableSeed, compose_seed_taskfirst
from .evidence import run_clean_tasks
from .world import build_clean_world
from .graph import canonical_topology


def build_one(*, ordinal: int, seed: int, kqa: list[dict],
              seed_record: dict, contexts: dict | None = None) -> dict:
    plan = compose_seed_taskfirst(
        seed_record=seed_record, ordinal=ordinal, run_seed=seed, kqa=kqa)
    world = build_clean_world(plan, seed=seed, contexts=contexts)
    execution = run_clean_tasks(plan, world)
    public_sources = [
        {"source_id": sid, "title": source["title"],
         "modality_hint": source["modality_hint"],
         "representation": source["representation"]}
        for sid, source in sorted(world["sources"].items())
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
    from .validation import validate_clean_scene
    issues = validate_clean_scene(scene)
    if issues:
        raise ValueError("invalid data-derived clean scene: " + "; ".join(issues))
    return scene


def build_batch(*, kqa: list[dict], scenario_seeds: list[dict],
                target: int = 100, seed: int = 17,
                contexts: dict | None = None) -> tuple[list[dict], dict]:
    if target < 1 or not kqa or not scenario_seeds:
        raise ValueError("positive target and nonempty KQA/MMQA seed pools required")
    order = list(scenario_seeds)
    random.Random(seed * 271828).shuffle(order)
    scenes = []
    seen_question_texts = set()
    rejected = Counter()
    for ordinal, seed_record in enumerate(order):
        if len(scenes) >= target:
            break
        try:
            scene = build_one(
                ordinal=ordinal, seed=seed, kqa=kqa,
                seed_record=seed_record, contexts=contexts)
        except UnsuitableSeed as exc:
            rejected[str(exc)] += 1
            continue
        from envir.schema import normalize_label
        qnorm = normalize_label(scene["question"])
        if qnorm in seen_question_texts:
            rejected["duplicate_new_question_from_scene_goal_combination"] += 1
            continue
        seen_question_texts.add(qnorm)
        scenes.append(scene)
    if len(scenes) < target:
        raise ValueError(
            f"Only {len(scenes)} of {len(order)} MMQA seeds yielded a valid "
            f"3-9-task graph; target={target}; rejections={dict(rejected)}")
    n_tasks = sum(len(s["macro_graph"]["tasks"]) for s in scenes)
    depths = [s["macro_graph"]["complexity"]["macro_dependency_depth"]
              for s in scenes]
    topologies = Counter(s["macro_graph"]["motif"] for s in scenes)
    context_count = sum(
        "original_context_excerpt" in {
            fact["field"] for fact in source["facts"]
        }
        for scene in scenes for source in scene["world"]["sources"].values()
    )
    missing = 0
    for scene in scenes:
        expected = {
            (item["modality"], item.get("doc_id"))
            for item in scene["source_priors"]["MultiModalQA"]["supporting_sources"]
            if item.get("provenance") == "mmqa_supporting_context"
        }
        present = {
            (src["modality_hint"], src.get("original_doc_id"))
            for src in scene["world"]["sources"].values()
            if src.get("source_origin") == "mmqa_supporting_context"
        }
        missing += len(expected - present)
    n_evidence = sum(
        len(g["nodes"]) for s in scenes
        for g in s["task_evidence_graphs"].values())
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
        "mean_macro_tasks_per_question": round(n_tasks / len(scenes), 3),
        "mean_dependency_depth": round(sum(depths) / len(depths), 3),
        "dependency_depth_distribution": dict(Counter(depths)),
        "topology_distribution": dict(topologies),
        "missing_original_supporting_sources": missing,
        "original_mmqa_context_excerpts_used": context_count,
        "evidence_node_count": n_evidence,
        "mean_evidence_nodes_per_task": round(n_evidence / n_tasks, 2),
        "unique_macro_work_types": len(kinds),
        "macro_work_types": dict(kinds),
        "macro_graph_signatures": len({graph_sig(s) for s in scenes}),
        "unlabeled_dag_topologies": len({
            canonical_topology(
                s["macro_graph"]["tasks"],
                s["macro_graph"]["edges"],
                s["macro_graph"]["reference_waves"],
            ) for s in scenes
        }),
        "unique_recipe_sequences": len({
            tuple(t["recipe_id"] for t in s["macro_graph"]["tasks"])
            for s in scenes
        }),
        "unique_recipe_ids": len({
            t["recipe_id"] for s in scenes for t in s["macro_graph"]["tasks"]
        }),
        "recipe_selection": "backward_recursive_typed_workproduct_grammar",
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
        "question_provenance": "novel_question_derived_from_terminal_macro_task_blueprint",
        "original_mmqa_question_copies": sum(
            scene["question"].casefold().strip() ==
            scene["source_priors"]["MultiModalQA"]["original_question"].casefold().strip()
            for scene in scenes
        ),
        "unique_scene_record_ids": len({
            scene["source_priors"]["MultiModalQA"]["record_id"] for scene in scenes
        }),
        "scene_goal_combination_count": len({
            (scene["source_priors"]["MultiModalQA"]["record_id"],
             scene["goal"]["intent"])
            for scene in scenes
        }),
        "goal_pattern_selection": "reproducible_random_independent_of_mmqa_original_question",
    }
