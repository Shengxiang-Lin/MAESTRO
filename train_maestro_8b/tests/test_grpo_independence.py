"""CPU-only safeguards for independent MAESTRO Stage2 GRPO training."""
from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from train_maestro_8b.policy import grpo_runtime
from train_maestro_8b.policy.da_reward_adapter import completion_text, make_reward_function


class StandaloneGRPOTests(unittest.TestCase):
    def test_config_and_batch_match_reference_hyperparameters(self):
        cfg = grpo_runtime.load_config()
        self.assertTrue(grpo_runtime.DEFAULT_CONFIG.is_file())
        self.assertEqual(grpo_runtime.DEFAULT_CONFIG.parent.name, "policy")
        self.assertTrue(str(cfg["paths"]["train_parquet"]).endswith(
            "workspace/maestro_8b/stage2_da/train.parquet"))
        plan = grpo_runtime.batch_plan(cfg)
        self.assertEqual(plan["num_generations"], 8)
        self.assertEqual(plan["gradient_accumulation_steps"], 8)
        kwargs = grpo_runtime.build_grpo_kwargs(cfg)
        self.assertEqual(kwargs["loss_type"], "dapo")
        self.assertEqual(kwargs["beta"], 0.01)
        self.assertEqual(kwargs["vllm_mode"], "colocate")
        self.assertEqual(kwargs["generation_batch_size"], 8)
        self.assertTrue(kwargs["use_vllm"])

    def test_all_stage2_modules_are_free_of_baseline_imports_and_paths(self):
        root = Path(__file__).resolve().parents[2]
        for module in (
            "train_maestro_8b/policy/da_backend.py",
            "train_maestro_8b/policy/grpo_runtime.py",
            "train_maestro_8b/policy/trl_reward_adapter.py",
            "train_maestro_8b/policy/da_reward_adapter.py",
            "train_maestro_8b/policy/da_reward_fn.py",
        ):
            with self.subTest(module=module):
                text = (root / module).read_text(encoding="utf-8")
                # The only permitted occurrence is the negative assertion
                # in explanatory module comments/docstrings.
                for line in text.splitlines():
                    if "baseline/da-grpo" in line and (
                        'import' in line or 'Path(' in line or 'open(' in line
                    ):
                        self.fail(f"Direct baseline source dependency in {module}: {line}")
                self.assertNotIn("from reward_adapter import", text)
                self.assertNotIn("from reward_fn import", text)
                self.assertNotIn("spec_from_file_location(", text)

    def test_completion_normalization_is_in_project(self):
        self.assertEqual(completion_text(
            [{"role": "assistant", "content": '{"action":"complete"}'}]),
            '{"action":"complete"}')
        with self.assertRaises(ValueError):
            completion_text([{"role": "user", "content": "not a completion"}])

    def test_legacy_da_reward_ablation_keeps_four_dimension_adapter(self):
        metadata = json.dumps({"has_expert": True, "sample_id": "sample-1"})
        scorer = make_reward_function(workers=2, num_generations=2)
        with patch("train_maestro_8b.policy.da_reward_fn.compute_score",
                   return_value={"score": 0.8}) as mocked:
            rewards = scorer(
                completions=['{"action":"complete"}', '{"action":"delegate_task"}'],
                data_source=["maestro", "maestro"],
                ground_truth=["answer", "answer"],
                extra_info_json=[metadata, metadata],
            )
        self.assertEqual(rewards, [0.8, 0.8])
        self.assertEqual(mocked.call_count, 2)


if __name__ == "__main__":
    unittest.main()
