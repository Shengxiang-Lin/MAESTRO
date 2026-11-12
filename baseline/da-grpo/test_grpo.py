"""CPU-only contract tests for the Unsloth/TRL MAESTRO GRPO pipeline.

Run: python -m unittest discover -s baseline/da-grpo -p 'test_*.py'
These do not load the 8B model or make any Judge API requests.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock, patch
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train
from reward_adapter import completion_text, make_reward_function


class TestConfig(unittest.TestCase):
    def test_official_four_prompts_eight_generations(self):
        cfg = train.load_config()
        plan = train.batch_plan(cfg)
        self.assertEqual(cfg["version"], 2)
        self.assertEqual(plan["num_generations"], 8)
        self.assertEqual(plan["prompts_per_step"], 4)
        self.assertEqual(plan["per_device_train_batch_size"], 1)
        self.assertEqual(plan["gradient_accumulation_steps"], 32)
        self.assertEqual(plan["generation_batch_size"], 32)
        self.assertEqual(cfg["training"]["epochs"], 3)
        self.assertEqual(cfg["rollout"]["kv_cache_dtype"], "auto")
        self.assertEqual(cfg["rollout"]["gpu_memory_utilization"], 0.70)
        self.assertEqual(cfg["rollout"]["max_num_seqs"], 16)
        self.assertTrue(cfg["rollout"]["standby_util_override"])

    def test_framework_options_preserve_o1_behavior(self):
        cfg = train.load_config()
        opts = train.build_grpo_kwargs(cfg)
        self.assertTrue(opts["use_vllm"])
        self.assertEqual(opts["vllm_mode"], "colocate")
        self.assertTrue(opts["vllm_enable_sleep_mode"])
        self.assertEqual(opts["beta"], 0.01)
        self.assertEqual(opts["epsilon"], 0.2)
        self.assertEqual(opts["loss_type"], "dapo")
        self.assertEqual(opts["scale_rewards"], "group")
        self.assertEqual(opts["num_iterations"], 1)
        self.assertEqual(opts["num_generations"], 8)
        self.assertEqual(opts["per_device_train_batch_size"], 1)
        self.assertEqual(opts["gradient_accumulation_steps"], 32)
        self.assertEqual(opts["generation_batch_size"], 32)
        self.assertNotIn("max_prompt_length", opts)
        self.assertEqual(opts["warmup_steps"], 0.03)
        self.assertEqual(opts["max_completion_length"], 4096)
        self.assertFalse(opts["vllm_importance_sampling_correction"])

    def test_sm120_flash_attn_disallows_fp8_kv_cache(self):
        import tempfile
        import yaml
        cfg = yaml.safe_load(train.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        cfg["rollout"]["kv_cache_dtype"] = "fp8"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "fp8-bad.yaml"
            p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "FP8 KV cache unsupported"):
                train.load_config(p)

    def test_flashinfer_disabled_before_unsloth_import(self):
        import os
        cfg = train.load_config()
        self.assertTrue(cfg["rollout"]["disable_flashinfer"])
        with patch.dict(os.environ, {
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
            "VLLM_USE_FLASHINFER_SAMPLER": "1",
            "UNSLOTH_VLLM_NO_FLASHINFER": "0",
        }, clear=False):
            train.configure_runtime(cfg)
            self.assertEqual(os.environ["UNSLOTH_VLLM_NO_FLASHINFER"], "1")
            self.assertEqual(os.environ["VLLM_USE_FLASHINFER_SAMPLER"], "0")
            self.assertNotIn("VLLM_ATTENTION_BACKEND", os.environ)
            self.assertEqual(
                os.environ["UNSLOTH_VLLM_STANDBY_UTIL_OVERRIDE"], "1"
            )

    def test_flashinfer_autotune_disabled_in_all_vllm_presets(self):
        # Simulate the vLLM 0.32 internal optimization levels without
        # importing CUDA packages during the CPU-only tests.
        from types import ModuleType
        fake_module = ModuleType("vllm.config.vllm")
        fake_module.OPTIMIZATION_LEVEL_TO_CONFIG = {
            "O0": {"kernel_config": {"enable_flashinfer_autotune": False}},
            "O1": {"kernel_config": {"enable_flashinfer_autotune": True}},
            "O2": {"kernel_config": {"enable_flashinfer_autotune": True}},
            "O3": {"kernel_config": {"enable_flashinfer_autotune": True}},
        }
        with patch.dict(sys.modules, {"vllm.config.vllm": fake_module}):
            train.disable_vllm_flashinfer_autotune()
        self.assertTrue(all(
            not preset["kernel_config"]["enable_flashinfer_autotune"]
            for preset in fake_module.OPTIMIZATION_LEVEL_TO_CONFIG.values()
        ))

    def test_flashinfer_autotune_unknown_preset_rejected(self):
        fake_module = ModuleType("vllm.config.vllm")
        fake_module.OPTIMIZATION_LEVEL_TO_CONFIG = {
            "O2": {"kernel_config": {}},
        }
        with patch.dict(sys.modules, {"vllm.config.vllm": fake_module}):
            with self.assertRaisesRegex(RuntimeError, "optimization preset changed"):
                train.disable_vllm_flashinfer_autotune()

    def test_vllm_memory_override_cli_and_sequence_limit(self):
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()) as capture:
            train.main(["--dry-run", "--vllm-memory-utilization", "0.65",
                        "--max-num-seqs", "8"])
        data = json.loads(capture.getvalue())
        self.assertEqual(data["vllm_gpu_memory_utilization"], 0.65)
        self.assertEqual(data["vllm_max_num_seqs"], 8)
        self.assertEqual(data["batch_plan"]["num_generations"], 8)
        self.assertEqual(data["batch_plan"]["prompts_per_step"], 4)

    def test_invalid_batch_or_context_rejected(self):
        import tempfile
        import yaml
        original = yaml.safe_load(train.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            cfg = yaml.safe_load(train.DEFAULT_CONFIG.read_text(encoding="utf-8"))
            cfg["rollout"]["num_generations"] = 1
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "GRPO requires"):
                train.load_config(path)

            original["rollout"]["max_completion_tokens"] = 32000
            path.write_text(yaml.safe_dump(original), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exceeds Unsloth model context"):
                train.load_config(path)

    def test_prompt_validation_no_truncation(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return {"input_ids": [[1, 2, 3, 4]]}
        rows = [{
            "prompt": [{"role": "user", "content": "example"}],
            "extra_info_json": json.dumps({"sample_id": "sample-1"}),
        }]
        train.check_prompt_lengths(rows, Tokenizer(), 4)
        with self.assertRaisesRegex(ValueError, "exceed 3 tokens"):
            train.check_prompt_lengths(rows, Tokenizer(), 3)

    def test_judge_secret_never_stored_in_yaml(self):
        yaml_text = train.DEFAULT_CONFIG.read_text(encoding="utf-8")
        self.assertNotIn("sk-", yaml_text)
        self.assertNotIn("OPENAI_API_KEY:", yaml_text)


class TestJudgeAdapter(unittest.TestCase):
    def test_text_and_conversational_outputs(self):
        self.assertEqual(completion_text("hello"), "hello")
        self.assertEqual(
            completion_text([{"role": "assistant", "content": '{"action":"complete"}'}]),
            '{"action":"complete"}',
        )
        with self.assertRaisesRegex(ValueError, "Expected assistant"):
            completion_text([{"role": "user", "content": "bad"}])

    def test_uses_correct_expert_for_each_completion(self):
        recorded = []
        fake = ModuleType("reward_fn")

        def score(data_source, generated, answer, metadata):
            recorded.append((data_source, generated, answer, metadata["sample_id"]))
            return {"score": 1.0 if generated == answer else 0.2}

        fake.compute_score = score
        info = [json.dumps({"has_expert": True, "sample_id": sid}) for sid in ("A", "A", "B", "B")]
        with patch.dict(sys.modules, {"reward_fn": fake}):
            rewards = make_reward_function(2)(
                completions=[
                    [{"role": "assistant", "content": "yes"}],
                    [{"role": "assistant", "content": "no"}],
                    [{"role": "assistant", "content": "no"}],
                    [{"role": "assistant", "content": "yes"}],
                ],
                data_source=["s3", "s3", "s4", "s4"],
                ground_truth=["yes", "yes", "no", "no"],
                extra_info_json=info,
            )
        self.assertEqual(rewards, [1.0, 0.2, 1.0, 0.2])
        self.assertEqual({record[-1] for record in recorded}, {"A", "B"})
        self.assertEqual(len(recorded), 4)

    def test_mismatched_reward_metadata_is_not_silently_broadcast(self):
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            make_reward_function(4)(
                completions=["A", "B"],
                data_source=["s3"],
                ground_truth=["A", "B"],
                extra_info_json=["{}", "{}"],
            )
        with patch.dict(sys.modules, {"reward_fn": ModuleType("reward_fn")}):
            with self.assertRaisesRegex(ValueError, "Missing reward-only expert"):
                make_reward_function(1)(
                    completions=["A"],
                    data_source=["s3"],
                    ground_truth=["A"],
                    extra_info_json=[json.dumps({"sample_id": "x"})],
                )


if __name__ == "__main__":
    unittest.main()
