"""Observable structural difficulty features, NOT a learned-agent score.

These features support eventual independent calibration of Medium/Hard labels;
they deliberately avoid claiming that a larger Gold DAG or more candidate
observations automatically entails a harder task for a live agent.
"""
from __future__ import annotations


def scene_difficulty_profile(scene: dict, *, information_report=None) -> dict:
    spec = scene["problem_spec"]
    plan = spec["macro_plan"]
    world = scene["world"]
    graph = scene["macro_graph"]
    macro_ops = [
        len(task["operator_chain"]) for task in graph["tasks"]
    ]
    observations = [
        len(item["observations"]) for item in world["sources"].values()
    ]
    result = {
        "difficulty_target": spec["difficulty_target"],
        "macro_task_count": graph["macro_task_count"],
        "dependency_depth": graph["critical_path_depth"],
        "source_count": len(observations),
        "source_observation_total": sum(observations),
        "source_observation_max": max(observations, default=0),
        "keyed_index_hops": plan["index_hops"],
        "primitive_ops_total": sum(macro_ops),
        "max_primitive_ops_per_macro": max(macro_ops, default=0),
        "joint_video_audio_macro": plan["pattern"] == "multimodal_event",
        "scope": "Observed structure only; no claim of empirical agent hardness.",
    }
    if information_report is not None:
        result["left_candidate_count"] = information_report[
            "left_candidate_count"
        ]
        result["right_candidate_count"] = information_report[
            "right_candidate_count"
        ]
        result["alternative_candidates"] = (
            information_report["left_unique_candidates"]
            + information_report["right_unique_candidates"]
        )
        result["candidate_intersection_count"] = information_report[
            "intersection_count"
        ]
    return result
