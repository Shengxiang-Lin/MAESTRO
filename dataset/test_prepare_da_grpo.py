"""Minimal offline converter contract checks (no pandas, pyarrow or GPU required)."""
import unittest
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dataset.prepare_da_grpo import make_examples, check_order, check_renderer_compatibility, validate_prompt_lengths
from dataset.tokenization import extract_token_ids


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



class TestTokenizationContracts(unittest.TestCase):
    def test_extract_unbatched_input_ids_from_supported_transformers_outputs(self):
        class FakeArray:
            def __init__(self, data):
                self.data = data

            def tolist(self):
                return self.data

        for result in (
            [7, 8, 9],
            FakeArray([[7, 8, 9]]),
            {"input_ids": FakeArray([7, 8, 9]), "attention_mask": [1, 1, 1]},
            {"input_ids": [[7, 8, 9]]},
        ):
            with self.subTest(type=type(result).__name__):
                self.assertEqual(extract_token_ids(result), [7, 8, 9])
        for bad in (
            {"attention_mask": [1, 1]},
            {"input_ids": [[7, 8], [9, 10]]},
            "not token IDs",
        ):
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                extract_token_ids(bad)

    def test_length_check_uses_input_ids_not_batch_encoding_key_count(self):
        class FakeTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return {"input_ids": [[7, 8, 9, 10, 11]], "attention_mask": [[1]*5]}

        rows = [{
            "prompt": [{"role": "user", "content": "Five tokens"}],
            "extra_info": {"sample_id": "x-round-01"},
        }]
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
            validate_prompt_lengths(rows, "fake-qwen-tokenizer", limit=5)
            with self.assertRaisesRegex(ValueError, "1 prompts exceed 4 tokens"):
                validate_prompt_lengths(rows, "fake-qwen-tokenizer", limit=4)


if __name__ == "__main__":
    unittest.main()
