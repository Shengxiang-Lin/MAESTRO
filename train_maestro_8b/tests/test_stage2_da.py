"""CPU contract tests for the *main* MAESTRO Stage2 DA training route."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from train_maestro_8b.data.export_da_grpo import convert, export
from train_maestro_8b.policy.da_backend import build_stage2_config, get_da_baseline


def scene():
    return {
        "task_id": "scene_A", "question": "Which project was approved?",
        "gold_answer": [{"answer": "Project A"}],
    }


def turn(candidate="scene_A-fork-round-02", round_num=3):
    return {
        "task_id": "scene_A", "round": round_num,
        "sample_id": f"scene_A-branch-02-round-{round_num:02d}",
        "round_sample_id": candidate,
        "input": {"subtask_history": "Sub-Agent 1: conflicting status evidence."},
        "output": {
            "action": "complete", "reasoning": "The reviewed evidence suffices.",
            "params": {"answer": "Project A"},
        },
        "generation_mode": "llm_reviewed",
    }


class Stage2DAExport(unittest.TestCase):
    def setUp(self):
        self.settings = {"max_attempts": 16, "sub_model": "qwen3-8b",
                         "model_alias": "model_1"}

    @patch("train_maestro_8b.data.export_da_grpo.o1_prompt_from_turn",
           return_value="exact native O1 prompt")
    def test_stage4_stable_keys_and_reward_only_gold(self, renderer):
        first, second = turn(), turn(candidate="scene_A-fork-round-04")
        rows, experts, modes = convert([scene()], [first, second], self.settings)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["prompt"], [{"role": "user", "content": "exact native O1 prompt"}])
        self.assertEqual(rows[0]["extra_info"]["answer"], "Project A")
        self.assertEqual(rows[0]["extra_info"]["expert_action"], "complete")
        self.assertNotIn("Project A", rows[0]["prompt"][0]["content"])
        self.assertNotIn("reference_action", rows[0]["prompt"][0])
        self.assertNotIn("decision_state", rows[0])
        self.assertEqual(len(experts), 2)
        self.assertEqual(modes["llm_reviewed"], 2)
        self.assertNotEqual(rows[0]["extra_info"]["sample_id"], rows[1]["extra_info"]["sample_id"])
        self.assertEqual(renderer.call_count, 2)

    @patch("train_maestro_8b.data.export_da_grpo.o1_prompt_from_turn", return_value="prompt")
    def test_duplicate_candidate_round_rejected(self, _):
        with self.assertRaisesRegex(ValueError, "duplicate Stage4 candidate"):
            convert([scene()], [turn(), turn()], self.settings)

    @patch("train_maestro_8b.data.export_da_grpo.o1_prompt_from_turn", return_value="prompt")
    def test_reviewed_filter(self, _):
        bad = turn(candidate="scene_A-fork-round-04")
        bad["generation_mode"] = "contract_frozen"
        rows, _, modes = convert(
            [scene()], [turn(), bad], self.settings, reviewed_only=True,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(modes["contract_frozen"], 1)

    @patch("train_maestro_8b.data.export_da_grpo.validate_prompt_lengths")
    @patch("train_maestro_8b.data.export_da_grpo.settings_for")
    @patch("train_maestro_8b.data.export_da_grpo.read_jsonl")
    @patch("train_maestro_8b.data.export_da_grpo.o1_prompt_from_turn", return_value="prompt")
    def test_export_dry_run_without_gpu_or_parquet(self, _, read, settings, check_len):
        settings.return_value = self.settings
        def loader(path):
            return [scene()] if str(path).endswith("scenes.jsonl") else [turn()]
        read.side_effect = loader
        report = export(
            Path("scenes.jsonl"), Path("main_turns.jsonl"), Path("/tmp/unused"),
            tokenizer=None, dry_run=True,
        )
        self.assertEqual(report["rows"], 1)
        self.assertEqual(report["trainer"], "unsloth_trl_grpo_da")
        self.assertEqual(report["episodes"], 1)
        check_len.assert_called_once()


class Stage2DABackend(unittest.TestCase):
    def args(self):
        return argparse.Namespace(
            config=str(Path("baseline/da-grpo/config.yaml").resolve()),
            model="base_models/Qwen3-8B",
            train_file="workspace/maestro_8b/stage2_da/train.parquet",
            output_dir="workspace/maestro_8b/checkpoints/stage2-da",
            epochs=3.0, max_steps=3, lr=5e-5, weight_decay=.01,
            max_grad_norm=1.0, save_steps=50, logging_steps=1,
            optim="adamw_8bit", lora_rank=32, lora_alpha=64,
            temperature=1.0, kl_coef=0.01, clip_epsilon=0.2,
            warmup_ratio=0.03, seed=42,
            num_generations=8, gradient_accumulation_steps=1,
            max_length=12288, max_new_tokens=1536,
            vllm_memory_utilization=.70, judge_workers=4,
            judge_model="qwen3.7-flash",
            judge_api_key_env="DASHSCOPE_API_KEY",
            judge_registry="config/model_config.yaml",
        )

    def test_exact_reference_trainer_contract(self):
        baseline = get_da_baseline()
        cfg = build_stage2_config(self.args(), baseline)
        kwargs = baseline.build_grpo_kwargs(cfg)
        self.assertEqual(kwargs["loss_type"], "dapo")
        self.assertEqual(kwargs["beta"], 0.01)
        self.assertEqual(kwargs["num_generations"], 8)
        self.assertEqual(kwargs["per_device_train_batch_size"], 1)
        self.assertEqual(kwargs["gradient_accumulation_steps"], 8)
        self.assertTrue(kwargs["use_vllm"])
        self.assertTrue(kwargs["vllm_enable_sleep_mode"])
        self.assertEqual(cfg["judge"]["model"], "qwen3.7-flash")
        self.assertEqual(cfg["judge"]["workers"], 4)
        self.assertEqual(cfg["rollout"]["max_completion_tokens"], 1536)
        self.assertTrue(str(cfg["paths"]["train_parquet"]).endswith("stage2_da/train.parquet"))
        self.assertFalse(cfg["model"]["load_in_4bit"])

    def test_reject_context_overflow(self):
        baseline = get_da_baseline()
        args = self.args()
        args.max_new_tokens = 4096
        args.max_length = 13000
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_stage2_config(args, baseline)

    @patch("train_maestro_8b.policy.da_backend.main")
    def test_main_trainer_dispatch(self, fake):
        from train_maestro_8b.policy.train import main
        main(["da", "--model", "qwen", "--train_file", "data.parquet"])
        fake.assert_called_once_with([
            "--model", "qwen", "--train_file", "data.parquet",
        ])


if __name__ == "__main__":
    unittest.main()
