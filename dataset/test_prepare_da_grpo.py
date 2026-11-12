"""Minimal offline converter contract checks (no pandas, pyarrow or GPU required)."""
import unittest
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dataset.prepare_da_grpo import make_examples, check_order, check_renderer_compatibility


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

    def test_historical_o1_docstring_only_change_is_safe(self):
        current = ('"""The on-disk per-turn reviewed-turn input stores"""\n'
                   'def render():\n'
                   '    """Rebuild a full baseline O1 prompt for an exported compact reviewed-turn turn."""\n'
                   '    return "prompt"\n')
        old = current.replace(
            "The on-disk per-turn reviewed-turn input stores",
            "The on-disk per-turn SFT input stores",
        ).replace(
            "Rebuild a full baseline O1 prompt for an exported compact reviewed-turn turn.",
            "Rebuild a full baseline O1 prompt for an exported compact SFT turn.",
        )
        old_hash = hashlib.sha256(old.encode("utf-8")).hexdigest()
        with TemporaryDirectory() as temp:
            path = Path(temp) / "o1_compat.py"
            path.write_text(current, encoding="utf-8")
            check_renderer_compatibility(old_hash, path, "Stage 3")
            check_renderer_compatibility(old_hash, path, "Stage 4")
            # Changing executable content cannot be waived by the two
            # historical documentation substitutions.
            path.write_text(current.replace('return "prompt"', 'return "other"'),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot be verified"):
                check_renderer_compatibility(old_hash, path, "Stage 3")

    @patch("dataset.prepare_da_grpo.o1_prompt_from_turn", return_value="O1 prompt")
    def test_reviewed_modes_keep_completion_and_contract_frozen(self, _):
        turns = []
        for index, mode in enumerate((
            "llm_reviewed", "llm_reviewed_completion",
            "llm_reviewed_contract_frozen", "teacher_grounded_completion_fallback",
        ), 1):
            turn = self.turn(f"x-round-{index:02d}", index)
            turn["generation_mode"] = mode
            turns.append(turn)
        rows, experts, modes = make_examples(
            self.scenes, turns, source="stage3",
            settings=self.settings, seen=set(), require_llm=True,
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(experts["x"]), 3)
        self.assertEqual(modes["teacher_grounded_completion_fallback"], 1)

    def test_order_validation(self):
        with self.assertRaisesRegex(ValueError, "Out-of-order"):
            check_order({"x": [{"step_index": 2}, {"step_index": 1}]})


if __name__ == "__main__":
    unittest.main()
