"""CPU contract for two-GPU and 1+3 GPU GRPO Server Mode launch."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
import yaml

from train_maestro_8b.policy import grpo_runtime
from train_maestro_8b.run.launch_grpo import command, topology


class Stage2DistributedTests(unittest.TestCase):
    def setUp(self):
        self.cfg = grpo_runtime.load_config()

    def test_default_dual_gpu_topology_and_batch(self):
        self.assertEqual(self.cfg["distributed"]["vllm_mode"], "server")
        self.assertEqual(self.cfg["distributed"]["train_gpus"], 1)
        rollout, _ = topology(self.cfg, "serve")
        trainer, _ = topology(self.cfg, "train")
        self.assertEqual(rollout, ["1"])
        self.assertEqual(trainer, ["0"])
        self.assertTrue(set(rollout).isdisjoint(trainer))
        plan = grpo_runtime.batch_plan(self.cfg)
        self.assertEqual(plan["per_device_train_batch_size"], 1)
        self.assertEqual(plan["gradient_accumulation_steps"], 8)
        self.assertEqual(plan["prompts_per_step"], 1)
        self.assertIsNone(plan["generation_batch_size"])

    def test_original_gpu_order_can_still_be_configured(self):
        # Other machines can still deploy vLLM on GPU 0 and trainer on GPU 1.
        raw = yaml.safe_load(Path(grpo_runtime.DEFAULT_CONFIG).read_text())
        raw["distributed"]["rollout_gpu"] = 0
        raw["distributed"]["train_gpu_start"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            cfg_file = Path(tmp) / "original-order.yaml"
            cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
            cfg = grpo_runtime.load_config(cfg_file)
        self.assertEqual(topology(cfg, "serve")[0], ["0"])
        self.assertEqual(topology(cfg, "train")[0], ["1"])
        self.assertEqual(topology(cfg, "train")[1]["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(grpo_runtime.batch_plan(cfg)["prompts_per_step"], 1)

    def test_swapped_roles_still_reject_overlapping_devices(self):
        raw = yaml.safe_load(Path(grpo_runtime.DEFAULT_CONFIG).read_text())
        raw["distributed"]["rollout_gpu"] = 1
        raw["distributed"]["train_gpu_start"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            cfg_file = Path(tmp) / "overlap.yaml"
            cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                grpo_runtime.load_config(cfg_file)

    def test_4gpu_three_trainer_topology_and_microbatch_scaling(self):
        cfg = deepcopy(self.cfg)
        # With four physical cards use rollout 0 + trainers 1..3.
        # The default two-card swap is intentionally not suitable for 1+3.
        cfg["distributed"].update({
            "rollout_gpu": 0, "train_gpu_start": 1, "train_gpus": 3,
        })
        self.assertEqual(topology(cfg, "train")[0], ["1", "2", "3"])
        self.assertEqual(grpo_runtime.batch_plan(cfg)["prompts_per_step"], 3)
        cfg["training"]["per_device_train_batch_size"] = 2
        self.assertEqual(grpo_runtime.batch_plan(cfg)["prompts_per_step"], 6)
        cfg["training"]["gradient_accumulation_steps"] = 16
        self.assertEqual(grpo_runtime.batch_plan(cfg)["prompts_per_step"], 12)
        args = grpo_runtime.build_grpo_kwargs(cfg)
        self.assertEqual(args["gradient_accumulation_steps"], 16)
        self.assertEqual(args["per_device_train_batch_size"], 2)
        self.assertEqual(args["ddp_find_unused_parameters"], False)

    def test_invalid_generation_group_divisibility(self):
        cfg = deepcopy(self.cfg)
        cfg["training"]["gradient_accumulation_steps"] = 3
        with self.assertRaisesRegex(ValueError, "divisible"):
            grpo_runtime.batch_plan(cfg)

    def test_server_command_contains_required_weight_transfer_config(self):
        cmd = command(self.cfg, "serve")
        self.assertEqual(cmd[:2], ["vllm", "serve"])
        self.assertIn("--weight-transfer-config", cmd)
        self.assertIn('{"backend": "nccl"}', cmd)
        self.assertIn("--logprobs-mode", cmd)
        self.assertIn("processed_logprobs", cmd)
        self.assertIn("--max-model-len", cmd)
        self.assertIn("--enforce-eager", cmd)
        _, env = topology(self.cfg, "serve")
        self.assertEqual(env["VLLM_SERVER_DEV_MODE"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")

    def test_training_command_is_torchrun_with_three_ranks(self):
        cfg = deepcopy(self.cfg)
        cfg["distributed"].update({
            "rollout_gpu": 0, "train_gpu_start": 1, "train_gpus": 3,
        })
        args = ["--model", "base_models/Qwen3-8B", "--max_steps", "2"]
        cmd = command(cfg, "train", args)
        self.assertIn("torch.distributed.run", cmd)
        self.assertEqual(cmd[cmd.index("--nproc_per_node") + 1], "3")
        self.assertEqual(cmd[-len(args):], args)
        _, env = topology(cfg, "train")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1,2,3")
        self.assertNotIn("LOCAL_RANK", env)

    def test_reward_diagnostics_are_isolated_per_rank(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "3", "RANK": "2"}):
            self.assertEqual(
                str(grpo_runtime.ranked_path("/tmp/reward_diagnostics.jsonl")),
                "/tmp/reward_diagnostics.rank02.jsonl",
            )
        with patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0"}):
            self.assertEqual(
                str(grpo_runtime.ranked_path("/tmp/reward_diagnostics.jsonl")),
                "/tmp/reward_diagnostics.jsonl",
            )

    def test_training_parameters_not_overridden_by_stale_baseline_cli(self):
        # Server mode keeps the input prompt and the reward unchanged.
        kwargs = grpo_runtime.build_grpo_kwargs(self.cfg)
        self.assertEqual(kwargs["loss_type"], "dapo")
        self.assertEqual(kwargs["num_generations"], 8)
        self.assertTrue(kwargs["use_vllm"])
        self.assertEqual(kwargs["vllm_mode"], "server")
        self.assertFalse(kwargs["vllm_enable_sleep_mode"])
        self.assertIsNone(kwargs["generation_batch_size"])


if __name__ == "__main__":
    unittest.main()
