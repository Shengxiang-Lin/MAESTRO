"""Minimal offline converter contract checks (no pandas, pyarrow or GPU required)."""
import unittest
from unittest.mock import patch

from dataset.prepare_da_grpo import make_examples, check_order


class TestExportContract(unittest.TestCase):
    def setUp(self):
        self.scenes = {"x": {
            "task_id": "x", "question": "What is the answer?",
            "gold_answer": [{"answer": 42}],
        }}
        self.settings = {"max_attempts": 16, "sub_model": "qwen3-8b", "model_alias": "model_1"}

    def turn(self, sample_id="x-round-01", round_number=1):
        return {
            "sample_id": sample_id,
            "task_id": "x",
            "round": round_number,
            "input": {"subtask_history": "No subtasks completed yet."},
            "output": {"action": "complete", "reasoning": "sufficient",
                       "params": {"answer": "42"}},
            "generation_mode": "llm_reviewed",
        }

    @patch("dataset.prepare_da_grpo.o1_prompt_from_turn", return_value="O1 prompt")
    def test_policy_and_reward_separated(self, prompt_builder):
        rows, experts, modes = make_examples(
            self.scenes, [self.turn()], source="stage3",
            settings=self.settings, seen=set(),
        )
        self.assertEqual(rows[0]["prompt"], [{"role": "user", "content": "O1 prompt"}])
        extra = rows[0]["extra_info"]
        self.assertEqual(extra["answer"], "42")
        self.assertEqual(extra["expert_action"], "complete")
        self.assertEqual(extra["expert_params"]["answer"], "42")
        self.assertEqual(len(experts["x"]), 1)
        self.assertEqual(modes["llm_reviewed"], 1)
        prompt_builder.assert_called_once()

    @patch("dataset.prepare_da_grpo.o1_prompt_from_turn", return_value="O1 prompt")
    def test_branch_identity_and_duplicate_guard(self, _):
        seen = set()
        turn = self.turn("x-branch-02-round-03", 3)
        rows, experts, _ = make_examples(
            self.scenes, [turn], source="stage4", settings=self.settings,
            seen=seen,
        )
        self.assertEqual(len(rows), 1)
        self.assertIn("x-branch-02", experts)
        with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
            make_examples(
                self.scenes, [turn], source="stage4", settings=self.settings,
                seen=seen,
            )

    def test_order_validation(self):
        with self.assertRaisesRegex(ValueError, "Out-of-order"):
            check_order({"x": [{"step_index": 2}, {"step_index": 1}]})


if __name__ == "__main__":
    unittest.main()
