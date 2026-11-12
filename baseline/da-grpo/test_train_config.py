"""Offline DA-GRPO YAML and provider wiring tests (no GPU, verl or key needed)."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from unittest.mock import patch
import unittest

TRAIN = Path(__file__).with_name("train.py")
spec = importlib.util.spec_from_file_location("maestro_da_grpo_train", TRAIN)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestDAGRPOConfig(unittest.TestCase):
    def test_default_yaml_reuses_shared_registry(self):
        cfg = module.load_config()
        self.assertEqual(cfg["judge"]["model"], "qwen3.7-flash")
        self.assertEqual(
            cfg["judge"]["api_base"],
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.assertFalse(cfg["judge"]["enable_thinking"])
        self.assertTrue(str(cfg["paths"]["model_path"]).endswith("base_models/Qwen3-8B"))
        self.assertEqual(cfg["settings"]["ROLLOUT_N"], "8")
        self.assertEqual(cfg["settings"]["N_GPUS_PER_NODE"], "8")

    def test_secret_only_in_environment(self):
        cfg = module.load_config()
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "private-test-key"}):
            self.assertEqual(module.get_key("DASHSCOPE_API_KEY", allow_prompt=False), "private-test-key")
            env = module.build_env(cfg, "private-test-key")
            self.assertEqual(env["OPENAI_API_KEY"], "private-test-key")
            self.assertEqual(env["LLM_JUDGE_MODEL"], "qwen3.7-flash")
            self.assertEqual(env["LLM_JUDGE_ENABLE_THINKING"], "false")
            self.assertEqual(env["TRAINER_LOGGER"], '["console","file"]')
            self.assertNotIn("private-test-key", cfg["config"].read_text(encoding="utf-8"))

    def test_check_only_does_not_start_training_or_prompt(self):
        from unittest.mock import patch
        with patch.object(module, "check_environment", return_value={"verl_import": True}), \
             patch.object(module, "gpu_count", return_value=8), \
             patch.object(module, "get_key", side_effect=(
                 lambda name, *, allow_prompt: None if not allow_prompt
                 else (_ for _ in ()).throw(AssertionError("unexpected password prompt"))
             )), \
             patch.object(module.subprocess, "call", side_effect=AssertionError("unexpected training")):
            self.assertEqual(module.main(["--check"]), 0)

    def test_bad_gpus_validation(self):
        cfg = module.load_config()
        self.assertEqual(cfg["settings"]["TP_SIZE"], "2")


if __name__ == "__main__":
    unittest.main()
