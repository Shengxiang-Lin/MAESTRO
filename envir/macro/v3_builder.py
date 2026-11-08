"""Full controlled-world macro task construction and evidence certification.

The question's concrete information requirements are chosen BEFORE building
the macro graph. A faithful standalone simulated-world execution supplies
clean Answer, Summary, Summary Trace and a final Gold result. This is not a
real image/video/audio perception benchmark.
"""
from __future__ import annotations

import copy
from collections import Counter

from envir.schema import digest
from .v3_semantics import (
    choose_question_seed, make_world, public_question, seed_statistics,
)
from .v3_graph import build_graph
from .v3_runtime import evaluate, validate_grounded


def counterfactual_sensitivity(scene: dict) -> dict:
    """Intervene on a concrete SOURCE observation and replay the full graph.

    Unlike a flag-based ablation, this measures whether a change in actual
    case evidence changes the final answer. It does not prove that every
    source or edge is necessary in every possible world.
    """
    altered = copy.deepcopy(scene["world"])
    sources = altered["sources"]
    ids = altered["source_ids"]
    intent = scene["information_need"]["intent"]
    if intent in {"compliance", "impact", "feasibility"}:
        chosen = ids["records"]
        records = sources[chosen]["observations"]["records"]
        for record in records:
            record["status"] = "valid" if intent != "feasibility" else "expired"
    elif intent == "comparison":
        chosen = ids["records"]
        records = sources[chosen]["observations"]["records"]
        records[0]["units"] = max(r["units"] for r in records) + 100
    elif intent == "prioritization":
        chosen = ids["records"]
        sources[chosen]["observations"]["records"][0]["urgency"] = 100
    elif intent == "mitigation":
        chosen = ids["capacity"]
        for item in sources[chosen]["observations"]["capacities"]:
            item["available_units"] = 100
    elif intent == "timeline":
        chosen = ids["independent"]
        sources[chosen]["observations"]["observations"][0]["reported_minute"] += 300
    elif intent == "reconciliation":
        chosen = ids["independent"]
        events = sources[ids["event"]]["observations"]["events"]
        minute_map = {row["case_id"]: row["event_minute"] for row in events}
        for item in sources[chosen]["observations"]["observations"]:
            if item["case_id"] in minute_map:
                item["reported_minute"] = minute_map[item["case_id"]]
    else:
        raise ValueError("unsupported counterfactual intent: " + intent)
    changed = evaluate(scene, altered)["gold_answer"]
    original = scene["gold_answer"]
    return {
        "source_id": chosen,
        "kind": "fact_value_intervention_and_full_replay",
        "baseline": original,
        "counterfactual": changed,
        "answer_changed": original != changed,
    }


def build_one(*, ordinal: int, seed: int, kqa: list[dict],
              mmqa: list[dict], omni_styles=None) -> dict:
    need = choose_question_seed(
        ordinal=ordinal, seed=seed, kqa=kqa, mmqa=mmqa,
        omni_styles=omni_styles)
    world = make_world(need, seed=seed)
    graph = build_graph(need, world)
    question = public_question(need)
    public_sources = [
        {"source_id": item["source_id"],
         "title": item["title"], "modality": item["modality"],
         "representation": item["representation"]}
        for item in world["sources"].values()
    ]
    scene = {
        "schema": "maestro.grounded_macro.v3",
        "task_id": "macro-v3-" + digest(seed, need["seed_id"]),
        "construction_status": "source_replay_verified_synthetic_world",
        "question": question,
        "information_need": need,
        "macro_graph": graph,
        "world": world,
        "public": {
            "task_id": "macro-v3-" + digest(seed, need["seed_id"]),
            "question": question,
            "provided_sources": public_sources,
            "max_concurrent_subagents": 10,
            "media_policy": "simulated_textual_observations_no_actual_media",
        },
    }
    result = evaluate(scene, world)
    scene["gold_answer"] = result["gold_answer"]
    scene["feedback_bank"] = result["feedback_bank"]
    scene["verification"] = {
        "policy": result["verification_policy"],
        "all_sources_are_private_synthetic_observations": True,
        "real_media_perception": False,
        "original_dataset_answer_copied": False,
        "all_subtasks_replayed_from_authorized_inputs": True,
        "source_ablation_scope": "execution_dependency_not_proof_of_unique_semantic_necessity",
    }
    errors = validate_grounded(scene)
    if errors:
        raise ValueError("invalid controlled world: " + ", ".join(errors))
    witness = counterfactual_sensitivity(scene)
    if not witness["answer_changed"]:
        raise ValueError("candidate has no decisive source-value intervention")
    scene["verification"]["counterfactual_sensitivity"] = witness
    return scene


def build_many(*, kqa: list[dict], mmqa: list[dict],
               target: int = 100, seed: int = 17, omni_styles=None):
    if target <= 0 or not kqa or not mmqa:
        raise ValueError("target must be positive and KQA/MMQA priors nonempty")
    scenes = [build_one(ordinal=i, seed=seed, kqa=kqa, mmqa=mmqa,
                        omni_styles=omni_styles)
              for i in range(target)]
    intents = Counter(s["information_need"]["intent"] for s in scenes)
    domain = Counter(s["information_need"]["domain"] for s in scenes)
    roots = Counter(s["macro_graph"]["complexity"]["root_tasks"] for s in scenes)
    sizes = Counter(len(s["macro_graph"]["tasks"]) for s in scenes)
    depths = Counter(s["macro_graph"]["complexity"]["macro_dependency_depth"]
                     for s in scenes)
    topology_signature = lambda s: (
        tuple(t["work_type"] + "/" + t["scope_key"]
              for t in s["macro_graph"]["tasks"]),
        tuple(sorted((e["from"], e["to"], e["input_slot"])
                     for e in s["macro_graph"]["edges"])),
    )
    return scenes, {
        "generated": len(scenes),
        "domains": dict(domain),
        "intents": dict(intents),
        "unique_question_texts": len({s["question"] for s in scenes}),
        "unique_macro_work_graphs": len({topology_signature(s) for s in scenes}),
        "macro_node_distribution": dict(sizes),
        "macro_depth_distribution": dict(depths),
        "root_work_count_distribution": dict(roots),
        "coarsening_actual_merges": sum(
            s["macro_graph"]["complexity"]["actual_work_boundary_merges"]
            for s in scenes),
        "total_reliable_feedback_packets": sum(len(s["feedback_bank"]) for s in scenes),
        "value_counterfactuals_changed_answer": sum(
            s["verification"]["counterfactual_sensitivity"]["answer_changed"]
            for s in scenes),
        "source_ablation_checked": True,
        "max_concurrent_subagents": 10,
        "runtime_variants_generated": False,
        "source_priors": seed_statistics(
            [s["information_need"] for s in scenes]),
        "verification_scope": (
            "controlled_simulated_world_replay_and_one_decisive_value_intervention "
            "per_case_not_real_media_or_external_world_truth"
        ),
    }
