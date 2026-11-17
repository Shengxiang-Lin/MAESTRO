"""Regression: current Stage-1 Task Plan must feed Stage 2, 3 and 4.

All evidence is synthetic and locally replayed; no remote LLM or YAGO dump.
"""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from environment.tests.current_stage1 import current_scene
from environment.generate_question.planning.spec import spec_issues
from environment.generate_question.auditing.quality import validate_replay
from environment.implement_subtask.realize import realize_scene
from environment.base_orchestration.trajectory_builder import build_episode
from environment.untrusted_orchestration.trajectory_builder import build_branch
from environment.untrusted_orchestration.faults import Fault


def accepted_reports(scene):
    rows = realize_scene(scene)["macro_tasks"]
    return [{
        **deepcopy(row),
        "finish_result": {
            "status": "done",
            "result": "Observed: " + json.dumps(row["answer"], ensure_ascii=False),
            "summary": "Grounded observations " + json.dumps(row["answer"], ensure_ascii=False),
        },
        "report_authoring": {"status": "accepted"},
    } for row in rows]


class CurrentStagePipelineTests(unittest.TestCase):
    def test_only_task_plan_is_supported(self):
        scene = current_scene(0)
        self.assertEqual(scene["problem_spec"]["reasoning_pattern"], "task_plan")
        self.assertEqual(validate_replay(scene)["status"], "valid")
        retired = deepcopy(scene["problem_spec"])
        retired["reasoning_pattern"] = "typed_knowledge_v4"
        self.assertIn("unsupported_stage1_reasoning_pattern", spec_issues(retired))

    def test_stage2_exposes_source_scoped_workproducts(self):
        scene = current_scene(1, "medium")
        before = deepcopy(scene)
        teacher = realize_scene(scene)
        self.assertEqual(scene, before)
        self.assertEqual(len(teacher["macro_tasks"]), scene["macro_graph"]["macro_task_count"])
        self.assertTrue(all(row["trace_summary"]["steps"] for row in teacher["macro_tasks"]))
        self.assertTrue(all(row["answer_with_provenance"] for row in teacher["macro_tasks"]))

    def test_stage3_and_stage4_consume_frozen_current_scene(self):
        scene = current_scene(3)
        reports = accepted_reports(scene)
        episode, turns, private, status = build_episode(
            scene, reports, freeze_dispatch=True,
        )
        self.assertEqual(status["status"], "accepted", status)
        self.assertTrue(turns)
        self.assertTrue(private)
        self.assertEqual(episode["trajectory"][-1]["action"], "complete")
        target = scene["macro_graph"]["reference_waves"][0][0]
        recovered, damaged_turns, audit = build_branch(
            scene, reports, episode, branch_round=2,
            faults=[Fault(target, "missing")],
        )
        self.assertEqual(audit["outcome"], "recovered", audit)
        self.assertTrue(damaged_turns)
        self.assertEqual(recovered["trajectory"][-1]["action"], "complete")


if __name__ == "__main__":
    unittest.main()
