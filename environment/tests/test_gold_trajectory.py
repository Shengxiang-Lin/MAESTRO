"""No-network two-role gold trajectory simulation and grounding tests."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from environment.trajectory_simulation.backend import (
    PassthroughExecutor, PassthroughPlanner,
)
from environment.trajectory_simulation.compiler import (
    compile_gold_solution, public_plan, ready_subtasks,
)
from environment.trajectory_simulation.pipeline import (
    GoldTrajectoryError, plan_trajectories_jsonl,
    synthesize_one, synthesize_trajectories_jsonl,
)
from environment.trajectory_simulation.validator import validate_subagent_result


def fixture(dependent=False):
    record = {
        "program": {
            "nodes": [
                {"node_id": 0, "op": "FIND", "dependencies": [], "inputs": ["entity:A"]},
                {"node_id": 1, "op": "RELATE", "dependencies": [0],
                 "inputs": ["r:first", "forward"]},
                {"node_id": 2, "op": "RELATE", "dependencies": [0],
                 "inputs": ["r:second", "forward"]},
                {"node_id": 3, "op": "INTERSECT", "dependencies": [1, 2], "inputs": []},
                {"node_id": 4, "op": "QUERY_NAME", "dependencies": [3], "inputs": []},
            ],
            "metadata": {},
        },
        "world": {
            "entities": {"A": "schema:Person", "B": "schema:Person"},
            "facts": [
                {"fact_id": "f1", "subject": "A", "predicate": "r:first",
                 "object": "B", "object_kind": "entity", "source_id": "s1"},
                {"fact_id": "f2", "subject": "A", "predicate": "r:second",
                 "object": "B", "object_kind": "entity", "source_id": "s2"},
            ],
            "sources": {
                "s1": {"source_id": "s1", "source_type": "text",
                       "fact_ids": ["f1"], "metadata": {}},
                "s2": {"source_id": "s2", "source_type": "text",
                       "fact_ids": ["f2"], "metadata": {}},
            },
            "metadata": {},
        },
    }
    realized = {
        "task_id": "world-test",
        "question": "Who is connected to A by both relations?",
        "capability_world": {
            "resources": [
                {"resource_id": "s1", "source_type": "text", "visibility": "provided",
                 "role": "entry", "required_capability_ids": ["text_cap"],
                 "anchor_dependency_resource_ids": []},
                {"resource_id": "s2", "source_type": "text",
                 "visibility": "discoverable" if dependent else "provided",
                 "role": "discovery", "required_capability_ids": ["text_cap"],
                 "anchor_dependency_resource_ids": ["s1"] if dependent else []},
            ],
            "capabilities": [
                {"capability_id": "text_cap", "specialist": "DocumentAgent"}
            ],
        },
    }
    rendered = SimpleNamespace(
        task_id="world-test", answer=["B"],
        sources=[
            SimpleNamespace(source_id="s1", source_type="text",
                            title="Source 1", content="A has r:first B"),
            SimpleNamespace(source_id="s2", source_type="text",
                            title="Source 2", content="A has r:second B"),
        ],
    )
    return realized, record, rendered


class GoldTrajectoryTests(unittest.TestCase):
    def test_compiler_reuses_symbolic_proof_and_program_dependency(self):
        realized, record, rendered = fixture()
        plan = compile_gold_solution(realized, record, rendered)
        self.assertEqual(plan["symbolic_answer"], ["B"])
        self.assertEqual(plan["final_answer"], ["B"])
        self.assertEqual(plan["proof_source_ids"], ["s1", "s2"])
        self.assertEqual(plan["gold_parallel_waves"],
                         [["subtask:s1", "subtask:s2"]])
        self.assertEqual(len(ready_subtasks(plan, set())), 2)
        public = public_plan(plan)
        self.assertNotIn("final_answer", public)
        self.assertNotIn("gold_answer_claims", json.dumps(public))
        self.assertEqual(plan["symbolic_trace"]["3"], ["B"])

    def test_gold_claims_use_public_rendered_aliases(self):
        realized, record, rendered = fixture()
        rendered.hidden_metadata = {
            "entity_aliases": {"A": "Alice", "B": "Bob"},
            "literal_surfaces_by_predicate": {},
            "literal_surfaces": {},
        }
        rendered.answer = ["Bob"]
        rendered.sources[0].content = "Alice — first: Bob."
        rendered.sources[1].content = "Alice — second: Bob."
        gold = compile_gold_solution(realized, record, rendered)
        claims = gold["tasks"][0]["gold_answer_claims"]
        self.assertEqual(claims, [{
            "subject": "Alice", "predicate": "first",
            "object": "Bob", "object_kind": "entity",
        }])
        self.assertNotIn("fact_id", json.dumps(claims))
        output = synthesize_one(
            realized, record, rendered,
            PassthroughPlanner(), PassthroughExecutor(),
        )
        self.assertTrue(output["validation"]["accepted"])
        self.assertEqual(output["final"]["answer"], ["Bob"])

    def test_dependency_forces_two_rounds(self):
        realized, record, rendered = fixture(dependent=True)
        plan = compile_gold_solution(realized, record, rendered)
        self.assertEqual(plan["gold_parallel_waves"],
                         [["subtask:s1"], ["subtask:s2"]])
        self.assertEqual(
            [task["subtask_id"] for task in ready_subtasks(plan, {"subtask:s1"})],
            ["subtask:s2"],
        )

    def test_two_role_offline_synthesis_preserves_all_gold_answers(self):
        realized, record, rendered = fixture(dependent=True)
        output = synthesize_one(
            realized, record, rendered,
            PassthroughPlanner(), PassthroughExecutor(),
            subagent_workers=2,
        )
        self.assertTrue(output["validation"]["accepted"])
        self.assertEqual(output["final"]["answer"], ["B"])
        self.assertEqual(len(output["steps"]), 2)
        self.assertEqual([len(r["subagent_results"]) for r in output["steps"]],
                         [1, 1])
        self.assertEqual(output["steps"][1]["completed_before"],
                         ["subtask:s1"])
        self.assertEqual(output["metadata"]["clean_trajectory"], True)

    def test_parallel_subagent_calls_and_no_future_evidence_in_main(self):
        realized, record, rendered = fixture()
        class InspectorPlanner(PassthroughPlanner):
            def plan(self, payload, prompt):
                self_outer.assertNotIn("gold_answer_claims", json.dumps(payload["public_plan"]))
                self_outer.assertEqual(payload["completed_subtask_ids"], [])
                self_outer.assertNotIn("A has r:second B", prompt)
                self_outer.assertNotIn("expected_answer", prompt)
                self_outer.assertNotIn("gold_answer_claims", prompt)
                return super().plan(payload, prompt)
        self_outer = self
        class SlowExecutor(PassthroughExecutor):
            def __init__(self):
                self.lock = threading.Lock()
                self.active = 0
                self.peak = 0
            def execute(self, payload, prompt):
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(.025)
                response = super().execute(payload, prompt)
                with self.lock:
                    self.active -= 1
                return response
        worker = SlowExecutor()
        trajectory = synthesize_one(
            realized, record, rendered, InspectorPlanner(), worker,
            subagent_workers=2,
        )
        self.assertTrue(trajectory["validation"]["accepted"])
        self.assertGreaterEqual(worker.peak, 2)

    def test_wrong_grounded_subtask_answer_is_rejected(self):
        realized, record, rendered = fixture()
        class WrongExecutor(PassthroughExecutor):
            def execute(self, payload, prompt):
                value = super().execute(payload, prompt)
                value["answer"] = []
                return value
        with self.assertRaisesRegex(GoldTrajectoryError,
                                    "answer_claims_not_grounded"):
            synthesize_one(
                realized, record, rendered,
                PassthroughPlanner(), WrongExecutor(), max_attempts=2,
            )

    def test_wrong_final_answer_is_rejected(self):
        realized, record, rendered = fixture()
        class WrongMain(PassthroughPlanner):
            def finalize(self, payload, prompt):
                value = super().finalize(payload, prompt)
                value["answer"] = ["C"]
                return value
        with self.assertRaisesRegex(GoldTrajectoryError, "final_answer_mismatch"):
            synthesize_one(
                realized, record, rendered,
                WrongMain(), PassthroughExecutor(), max_attempts=2,
            )

    def test_parallel_plan_cannot_drop_or_invent_tasks(self):
        realized, record, rendered = fixture()
        class WrongMain(PassthroughPlanner):
            def plan(self, payload, prompt):
                result = super().plan(payload, prompt)
                result["parallel_subtasks"].pop()
                return result
        with self.assertRaisesRegex(GoldTrajectoryError, "parallel_group_mismatch"):
            synthesize_one(
                realized, record, rendered,
                WrongMain(), PassthroughExecutor(), max_attempts=2,
            )

    def test_plan_only_and_synthesis_jsonl(self):
        realized, record, rendered = fixture()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            question_file = folder / "questions.jsonl"
            question_file.write_text(json.dumps(realized) + "\n", encoding="utf-8")
            mapping = {"world-test": (record, rendered)}
            with patch("environment.trajectory_simulation.pipeline._world_map",
                       return_value=mapping):
                summary = plan_trajectories_jsonl(
                    folder/"worlds.jsonl", question_file, folder/"plan.jsonl"
                )
                self.assertEqual(summary["ready"], 1)
                output = synthesize_trajectories_jsonl(
                    folder/"worlds.jsonl", question_file, folder/"trajectories.jsonl",
                    planner=PassthroughPlanner(), executor=PassthroughExecutor(),
                    progress=False,
                )
                self.assertEqual(output["accepted"], 1)
                row = json.loads((folder/"trajectories.jsonl").read_text().strip())
                self.assertTrue(row["validation"]["final_answer_verified"])
                self.assertEqual(row["final"]["answer"], ["B"])


if __name__ == "__main__":
    unittest.main()
