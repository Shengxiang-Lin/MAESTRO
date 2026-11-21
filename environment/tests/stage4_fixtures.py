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

