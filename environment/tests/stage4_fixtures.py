"""Frozen Stage-1/2/3 fixtures for current single-turn Stage-4 tests."""
from __future__ import annotations

from copy import deepcopy
import json

from environment.tests.current_stage1 import current_scene
from environment.implement_subtask.realize import realize_scene
from environment.base_orchestration.trajectory_builder import build_episode


def sample(i=0):
    scene = current_scene(i)
    reports = [{
        **deepcopy(row),
        "finish_result": {
            "status": "done",
            "result": "Observed: " + json.dumps(row["answer"]),
            "summary": "Grounded observations " + json.dumps(row["answer"]),
        },
        "report_authoring": {"status": "accepted"},
    } for row in realize_scene(scene)["macro_tasks"]]
    episode, _, _, status = build_episode(scene, reports)
    if status["status"] != "accepted":
        raise AssertionError("invalid_clean_fixture:" + str(status))
    return scene, reports, episode



def source_pair_scene():
    scene = deepcopy(sample(0)[0])
    source_a = {
        "source_id": "parent_src", "access_policy": {"mode": "browse"},
        "schema": ["case_id", "entity_final"],
        "observations": [{
            "values": {"case_id": "Case-001",
                       "entity_final": "Lakehurst District"},
            "fact_refs": ["parent_src:F001"],
            "observation_id": "parent_src.OBS001",
            "evidence_origin_id": "synthetic:parent001",
        }],
    }
    source_b = {
        "source_id": "lookup_src",
        "access_policy": {"mode": "keyed_lookup",
                          "key_field": "entity_final"},
        "schema": ["entity_final", "observation_year"],
        "observations": [{
            "values": {"entity_final": "Lakehurst District",
                       "observation_year": 2025},
            "fact_refs": ["lookup_src:F001"],
            "observation_id": "lookup_src.OBS001",
            "evidence_origin_id": "synthetic:lookup001",
        }, {
            "values": {"entity_final": "Dunhaven District",
                       "observation_year": 2024},
            "fact_refs": ["lookup_src:F002"],
            "observation_id": "lookup_src.OBS002",
            "evidence_origin_id": "synthetic:lookup002",
        }],
    }
    parent = {
        "task_id": "parent", "source_ids": ["parent_src"],
        "depends_on": [], "external_inputs": [], "input_schema": {},
        "operator_chain": [
            {"id": "parent_scan", "op": "scan", "inputs": [],
             "source_id": "parent_src"},
            {"id": "parent_project", "op": "project",
             "inputs": ["parent_scan"], "fields": ["entity_final"]},
        ],
        "program_node_id": "parent_project",
        "outputs": {"typed_relation": ["entity_final"]},
    }
    child = {
        "task_id": "child", "source_ids": ["lookup_src"],
        "depends_on": ["parent"], "external_inputs": ["parent_project"],
        "input_schema": {"input_0": ["entity_final"]},
        "operator_chain": [
            {"id": "child_lookup", "op": "lookup_by_key",
             "inputs": ["parent_project"], "source_id": "lookup_src",
             "input_key_field": "entity_final",
             "source_key_field": "entity_final"},
            {"id": "child_project", "op": "project",
             "inputs": ["child_lookup"],
             "fields": ["entity_final", "observation_year"]},
        ],
        "program_node_id": "child_project",
        "outputs": {"typed_relation": ["entity_final", "observation_year"]},
    }
    scene["world"]["sources"] = {
        "parent_src": source_a, "lookup_src": source_b,
    }
    scene["macro_graph"] = {
        **scene["macro_graph"],
        "tasks": [parent, child],
        "reference_waves": [["parent"], ["child"]],
        "terminal_task_id": "child",
    }
    scene["gold_answer"] = [{
        "entity_final": "Lakehurst District",
        "observation_year": 2025,
    }]
    return scene, parent, child


