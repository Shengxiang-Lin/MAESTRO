"""End-to-end alignment smoke test against the repository's actual O1 renderer."""
import unittest

from environment.base_orchestration.o1_compat import o1_history, o1_prompt_from_history
from train_maestro_8b.data.export_risk import render_samples, history_spans


class RuntimeAlignmentTests(unittest.TestCase):
    def test_real_history_renderer_and_stage4_round_export(self):
        report = {
            "task_instruction": "Read the project record",
            "model": "model_1",
            "steps_taken": 2,
            "statistics": {"max_steps": 30},
            "cost": 0.0,
            "finish_result": {
                "status": "done",
                "result": "Case-011: completed_count=7",
                "summary": "Read Case-011, completed count seven.",
            },
            "trace_summary": "Looked up an assigned record.",
        }
        scene = {
            "task_id": "integration-1", "question": "What was the count?",
            "problem_spec": {"macro_plan": {"domain": "general"}},
            "public": {"available_source_descriptions": []},
        }
        episode = {
            "task_id": scene["task_id"], "branch_round": 2,
            "round_sample_id": "integration-1-fork-round-02-candidate-001",
            "trajectory": [{"attempt": 1, "action": "delegate_task",
                            "result": {"subtask_results": [report]}}],
        }
        state = {
            2: {
                "main_can_act_from_round": 2,
                "report_events": [{"macro_task_id": "a", "status": "done",
                                   "visible_claim_count": 1}],
                "claim_nodes": [{"claim_id": "x", "report_task_id": "a"}],
                "claim_states": {"key": {"status": "unverified",
                                          "supporting_claim_ids": ["x"]}},
                "claim_dependency_edges": [],
            },
        }
        items = list(render_samples(scene, episode, state, max_attempts=16))
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertIn("==== QUESTION ====", item["prompt"])
        self.assertEqual((item["reports"][0]["risk"], item["reports"][0]["mask"]),
                         (0, 1))
        report_text = item["prompt"][item["reports"][0]["start"]:
                                     item["reports"][0]["end"]]
        self.assertIn("Case-011", report_text)
        self.assertIn("Trace summary:", report_text)
        self.assertNotIn("==== AVAILABLE TOOLS", report_text)
        history = o1_history([{
            "attempt": 1, "is_parallel_batch": True,
            "subtask_results": [{
                "status": "done", "instruction": report["task_instruction"],
                "model": "model_1", "steps_taken": 2, "max_steps": 30,
                "cost": 0.0, "result": report["finish_result"]["result"],
                "summary": report["finish_result"]["summary"],
                "trace_summary": report["trace_summary"],
            }],
        }])
        self.assertEqual(len(history_spans(history, [(1, 1)])), 1)
        self.assertIn(history, o1_prompt_from_history(
            scene, history, attempt=2, max_attempts=16, sub_model="qwen3-8b",
        ))


if __name__ == "__main__":
    unittest.main()
