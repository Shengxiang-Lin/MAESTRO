"""CPU contracts for DA-SFT/DA-GRPO online LoRA vLLM integration."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from base.engine.vllm_service import VLLMServiceManager
from base.engine.async_llm import LLMConfig
from baseline.evaluate_lora import MODEL_NAMES, get_reference, valid_o1_action

ROOT = Path(__file__).resolve().parents[1]


class TestOnlineLoRA(unittest.TestCase):
    def test_both_configs_serve_a_peft_adapter_with_thinking(self):
        configs = yaml.safe_load(
            (ROOT / "config/model_config.yaml").read_text(encoding="utf-8")
        )["models"]
        for method, name in MODEL_NAMES.items():
            with self.subTest(method=method):
                cfg = configs[name]
                self.assertEqual(cfg["model_path"], "${QWEN3_8B_MODEL_PATH}")
                self.assertTrue(cfg["lora_adapter_path"].startswith("${"))
                self.assertEqual(cfg["served_model_name"], name)
                self.assertEqual(cfg["max_lora_rank"], 32)
                self.assertEqual(cfg["reasoning_parser"], "qwen3")
                self.assertTrue(cfg["extra_body"]["chat_template_kwargs"]["enable_thinking"])
                self.assertEqual(cfg["max_completion_tokens"], 4096)

    def test_pretest_and_final_benchmarks_choose_the_correct_adapter(self):
        for phase in ("pretest", "final"):
            for method, name in MODEL_NAMES.items():
                path = ROOT / (
                    f"baseline/orchestra-o1/config/omnigaia/{phase}/"
                    f"orchestra_o1_{method.replace('-', '_')}_lora.yaml"
                )
                cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(cfg["main_model"], name)
                self.assertEqual(cfg["sub_models"], ["qwen3.8-max"])
                self.assertEqual(
                    cfg["task_ids_file"],
                    f"data/OmniGAIA/splits/{'pretest_60' if phase == 'pretest' else 'final_100'}_ids.txt",
                )

    def test_maestro_global_grpo_uses_shared_model_registry(self):
        """Final benchmark selects a registered LoRA without local overrides."""
        registry = yaml.safe_load(
            (ROOT / "config/model_config.yaml").read_text(encoding="utf-8")
        )["models"]
        name = "qwen3-8b-maestro-grpo-lora"
        service = registry[name]
        self.assertEqual(service["model_path"], "base_models/Qwen3-8B")
        self.assertEqual(
            service["lora_adapter_path"],
            "workspace/maestro_8b/checkpoints/stage2-global-grpo/checkpoint-200",
        )
        self.assertEqual(service["gpus"], [2])
        self.assertEqual(service["port"], 8812)
        self.assertEqual(service["gpu_memory_utilization"], 0.90)
        self.assertEqual(service["max_model_len"], 16384)
        self.assertTrue(
            service["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        )
        benchmark_path = ROOT / (
            "baseline/orchestra-o1/config/omnigaia/final/"
            "orchestra_o1_omnigaia_maestro_grpo_qwen37_flash.yaml"
        )
        benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
        self.assertEqual(benchmark["main_model"], name)
        self.assertEqual(benchmark["sub_models"], ["qwen3.7-flash"])
        self.assertNotIn("models", benchmark)
        self.assertEqual(
            benchmark["task_ids_file"],
            "data/OmniGAIA/splits/final_100_ids.txt",
        )

    def test_adapter_rank_validation_and_launch_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "Qwen3-8B"
            base.mkdir()
            adapter = root / "final_adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"fake-test-weights")
            (adapter / "adapter_config.json").write_text(
                json.dumps({"r": 32, "base_model_name_or_path": str(base)}),
                encoding="utf-8",
            )
            manager = VLLMServiceManager()
            config = {
                "model_path": str(base),
                "lora_adapter_path": str(adapter),
                "max_lora_rank": 32,
                "lora_base_served_name": "sft-base",
                "gpus": [2],
                "tensor_parallel_size": 1,
                "reasoning_parser": "qwen3",
            }
            resolved = manager._validate_lora_adapter(
                "qwen3-8b-da-sft-lora", config, str(base),
            )
            self.assertEqual(resolved, str(adapter))
            command, env = manager._build_command(
                name="qwen3-8b-da-sft-lora", config=config,
                model_path=str(base), served_name="qwen3-8b-da-sft-lora",
                host="127.0.0.1", port=8810, api_key="EMPTY",
            )
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(command[command.index("--model") + 1], str(base))
            self.assertEqual(command[command.index("--served-model-name") + 1], "sft-base")
            self.assertIn("--enable-lora", command)
            self.assertEqual(command[command.index("--max-lora-rank") + 1], "32")
            self.assertEqual(
                command[command.index("--lora-modules") + 1],
                f"qwen3-8b-da-sft-lora={adapter}",
            )
            self.assertEqual(command[command.index("--reasoning-parser") + 1], "qwen3")
            with self.assertRaisesRegex(ValueError, "exceeds"):
                manager._validate_lora_adapter(
                    "model", {**config, "max_lora_rank": 16}, str(base),
                )
            with self.assertRaisesRegex(FileNotFoundError, "missing trained"):
                manager._validate_lora_adapter(
                    "model", {**config, "lora_adapter_path": str(root / "missing")},
                    str(base),
                )

    def test_adapter_training_base_inaccessible_on_another_host(self):
        """An inaccessible old training root must not block portable LoRA use."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "Qwen3-8B"
            base.mkdir()
            adapter = root / "final_adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"test-weights")
            training_base = Path("/root/autodl-tmp/MAESTRO/base_models/Qwen3-8B")
            (adapter / "adapter_config.json").write_text(
                json.dumps({
                    "r": 32,
                    "base_model_name_or_path": str(training_base),
                }),
                encoding="utf-8",
            )
            config = {
                "model_path": str(base),
                "lora_adapter_path": str(adapter),
                "max_lora_rank": 32,
            }
            original_exists = Path.exists

            def simulated_exists(path):
                if path == training_base:
                    raise PermissionError("Permission denied: " + str(path))
                return original_exists(path)

            with patch("base.engine.vllm_service.Path.exists",
                       autospec=True, side_effect=simulated_exists):
                self.assertEqual(
                    VLLMServiceManager._validate_lora_adapter(
                        "maestro-lora", config, str(base)
                    ),
                    str(adapter),
                )

    def test_adapter_training_base_mismatch_still_rejected(self):
        """Keep rejecting a different accessible training base directory."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "Qwen3-8B"
            base.mkdir()
            other_base = root / "different-Qwen3-8B"
            other_base.mkdir()
            adapter = root / "final_adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"test-weights")
            (adapter / "adapter_config.json").write_text(
                json.dumps({
                    "r": 32,
                    "base_model_name_or_path": str(other_base),
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "adapter was trained for"):
                VLLMServiceManager._validate_lora_adapter(
                    "maestro-lora",
                    {
                        "model_path": str(base),
                        "lora_adapter_path": str(adapter),
                        "max_lora_rank": 32,
                    },
                    str(base),
                )

    def test_stale_lora_model_card_is_not_reused(self):
        class FakeResponse:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return json.dumps({"data": [
                    {"id": "sft", "root": "/weights/experiment-A"}
                ]}).encode("utf-8")
        with patch("base.engine.vllm_service.urllib.request.urlopen",
                   return_value=FakeResponse()):
            self.assertTrue(VLLMServiceManager._server_matches(
                "http://localhost:8810/v1", "sft", "EMPTY",
                expected_lora_root="/weights/experiment-A",
            ))
            self.assertFalse(VLLMServiceManager._server_matches(
                "http://localhost:8810/v1", "sft", "EMPTY",
                expected_lora_root="/weights/experiment-B",
            ))

    def test_llm_keeps_reasoning_parser_kwargs_and_completion_budget(self):
        cfg = LLMConfig({
            "backend": "vllm", "model": "qwen3-8b-da-sft-lora",
            "max_completion_tokens": 4096,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        })
        self.assertEqual(cfg.max_completion_tokens, 4096)
        self.assertTrue(cfg.extra_body["chat_template_kwargs"]["enable_thinking"])

    def test_evaluator_reads_both_label_formats(self):
        completion = {
            "action": "complete", "reasoning": "Verified.",
            "params": {"answer": "42"},
        }
        sft = {"completion": [{"role": "assistant",
                               "content": json.dumps(completion)}]}
        grpo = {"extra_info": {"expert_action": "complete",
                                "expert_params": {"answer": "42"}}}
        self.assertEqual(get_reference(sft, "da-sft")["action"], "complete")
        self.assertEqual(get_reference(grpo, "da-grpo")["action"], "complete")
        self.assertTrue(valid_o1_action(completion))
        self.assertFalse(valid_o1_action({"action": "complete",
                                           "reasoning": "", "params": {"answer": "42"}}))


if __name__ == "__main__":
    unittest.main()
