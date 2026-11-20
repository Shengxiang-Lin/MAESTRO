"""CPU contract tests for the *main* MAESTRO Stage2 DA training route."""
from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

from train_maestro_8b.policy.da_backend import build_stage2_config
from train_maestro_8b.policy import grpo_runtime as runtime


class Stage2DABackend(unittest.TestCase):
    def args(self):
        return argparse.Namespace(
            config=str(runtime.DEFAULT_CONFIG),
            model="base_models/Qwen3-8B",
            train_file="workspace/maestro_8b/stage2_da/train.parquet",
            output_dir="workspace/maestro_8b/checkpoints/stage2-da",
            epochs=3.0, max_steps=3, lr=5e-5, weight_decay=.01,
            max_grad_norm=1.0, save_steps=50, logging_steps=1,
            optim="adamw_8bit", lora_rank=32, lora_alpha=64,
            temperature=1.0, kl_coef=0.01, clip_epsilon=0.2,
            warmup_ratio=0.03, seed=42,
            num_generations=8, gradient_accumulation_steps=8,
            micro_batch_size=1,
            max_length=12288, max_new_tokens=1536,
            vllm_memory_utilization=.70, judge_workers=4,
            judge_model="qwen3.7-flash",
            judge_api_key_env="DASHSCOPE_API_KEY",
            judge_registry="config/model_config.yaml",
        )

    def test_independent_grpo_trainer_contract(self):
        cfg = build_stage2_config(self.args())
        kwargs = runtime.build_grpo_kwargs(cfg)
        self.assertEqual(kwargs["loss_type"], "dapo")
        self.assertEqual(kwargs["beta"], 0.01)
        self.assertEqual(kwargs["num_generations"], 8)
        self.assertEqual(kwargs["per_device_train_batch_size"], 1)
        self.assertEqual(kwargs["gradient_accumulation_steps"], 8)
        self.assertTrue(kwargs["use_vllm"])
        self.assertFalse(kwargs["vllm_enable_sleep_mode"])
        self.assertEqual(kwargs["vllm_mode"], "server")
        self.assertIsNone(kwargs["generation_batch_size"])
        self.assertEqual(cfg["judge"]["model"], "qwen3.7-flash")
        self.assertEqual(cfg["judge"]["workers"], 4)
        self.assertEqual(cfg["rollout"]["max_completion_tokens"], 1536)
        self.assertTrue(str(cfg["paths"]["train_parquet"]).endswith("stage2_da/train.parquet"))
        self.assertFalse(cfg["model"]["load_in_4bit"])

    def test_reject_context_overflow(self):
        args = self.args()
        args.max_new_tokens = 4096
        args.max_length = 13000
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_stage2_config(args)

    @patch("train_maestro_8b.policy.da_backend.main")
    def test_main_trainer_dispatch(self, fake):
        from train_maestro_8b.policy.train import main
        main(["da", "--model", "qwen", "--train_file", "data.parquet"])
        fake.assert_called_once_with([
            "--model", "qwen", "--train_file", "data.parquet",
        ])


if __name__ == "__main__":
    unittest.main()
