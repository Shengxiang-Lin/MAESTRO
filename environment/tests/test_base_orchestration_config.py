"""Configuration-only Main trajectory workflow and isolated config validation."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import yaml

from environment.__main__ import main, parser
from environment.base_orchestration.config import (
    DEFAULT_CONFIG, generate_from_config, load_config,
)


class TrajectoryConfigurationTests(TestCase):
    def test_functional_package_and_yaml_names(self):
        from environment import base_orchestration
        from environment.base_orchestration.config import DEFAULT_CONFIG
        from environment.base_orchestration.trajectory_builder import build_episode
        from environment.base_orchestration.generate_turns import generate_turns
        self.assertIsNotNone(base_orchestration)
        self.assertTrue(callable(build_episode))
        self.assertTrue(callable(generate_turns))
        self.assertEqual(DEFAULT_CONFIG.name, "main_orchestration.yaml")
        self.assertTrue(DEFAULT_CONFIG.is_file())

    def test_configs_are_independent_and_root_relative(self):
        cfg = load_config()
        self.assertEqual(DEFAULT_CONFIG.name, "main_orchestration.yaml")
        self.assertTrue(cfg["scenes_path"].is_absolute())
        self.assertTrue(cfg["reports_path"].is_absolute())
        self.assertTrue(cfg["output_dir"].is_absolute())
        self.assertTrue(str(cfg["scenes_path"]).endswith(
            "/questions/scenes.jsonl"
        ))
        self.assertTrue(str(cfg["reports_path"]).endswith(
            "/subtasks.jsonl"
        ))
        self.assertEqual(cfg["mode"], "llm")
        self.assertEqual(
            cfg["workers"],
            yaml.safe_load(DEFAULT_CONFIG.read_text())["generation"]["workers"],
        )
        self.assertEqual(cfg["author_model"], "local-qwen3.5-9b")

        from environment.config import DEFAULT_CONFIG as QUESTIONS
        from environment.implement_subtask.config import DEFAULT_CONFIG as SUBAGENTS
        self.assertEqual(QUESTIONS.name, "questions.yaml")
        self.assertEqual(SUBAGENTS.name, "subagent_reports.yaml")
        self.assertEqual({QUESTIONS.parent, SUBAGENTS.parent,
                          DEFAULT_CONFIG.parent}, {DEFAULT_CONFIG.parent})
        self.assertEqual(set(parser()._subparsers._group_actions[0].choices), {
            "generate-questions", "quality", "naturalize-subtasks",
            "generate-main-turns", "generate-untrusted-turns",
        })

    def test_runner_uses_yaml_values_without_hidden_defaults(self):
        with patch(
            "environment.generate_question.authoring.backend.OpenAICompatibleBackend"
        ) as backend, patch(
            "environment.base_orchestration.generate_turns.generate_turns",
            return_value={"status": "done"},
        ) as runner:
            result = generate_from_config()
        self.assertEqual(result["status"], "done")
        self.assertEqual(backend.call_count, 2)
        self.assertEqual(backend.call_args_list[0].kwargs["max_tokens"], 2800)
        self.assertEqual(backend.call_args_list[1].kwargs["max_tokens"], 1200)
        kwargs = runner.call_args.kwargs
        self.assertEqual(
            kwargs["workers"],
            yaml.safe_load(DEFAULT_CONFIG.read_text())["generation"]["workers"],
        )
        self.assertEqual(kwargs["max_revisions"], 2)
        self.assertEqual(
            kwargs["max_attempts"],
            yaml.safe_load(DEFAULT_CONFIG.read_text())["generation"]["max_attempts"],
        )
        self.assertEqual(kwargs["author_model"], "local-qwen3.5-9b")
        self.assertEqual(kwargs["critic_model"], "local-qwen3.5-9b")
        self.assertTrue(kwargs["scenes_path"].is_absolute())

    def test_removed_fallback_option_is_not_accepted(self):
        with self.assertRaisesRegex(ValueError, "trajectory_unknown_override"):
            generate_from_config(allow_delegate_fallback=True)
        with self.assertRaises(SystemExit):
            parser().parse_args(["generate-main-turns", "--allow-delegate-fallback"])

    def test_local_model_cost_is_zero_without_pricing_warning(self):
        from base.engine.async_llm import (
            AsyncLLM, LLMConfig, ModelPricing, TokenUsageTracker,
        )
        from orchestra.base_main_agent import build_model_pricing_table

        with patch("base.engine.async_llm.logger.warning") as warning:
            for name in ("local-qwen3.5-9b", "local-qwen3-8b"):
                self.assertEqual(ModelPricing.get_price(name, "input"), 0.0)
                self.assertEqual(ModelPricing.get_price(name, "output"), 0.0)
            table = build_model_pricing_table(
                ["model_1"], {"local-qwen3.5-9b": "model_1"},
            )
            self.assertIn("| model_1 | $0.00000 | $0.00000 |", table)
            warning.assert_not_called()

        # Both local vLLM and OpenAI-compatible local endpoints have zero
        # provider API cost, without losing the original token statistics.
        config = LLMConfig({
            "model": "local-qwen3.5-9b", "backend": "api",
            "base_url": "http://127.0.0.1:8807/v1",
            "key": "test-only",
        })
        llm = AsyncLLM(config)
        self.assertFalse(llm.usage_tracker.pricing_enabled)
        result = llm.usage_tracker.add_usage(
            config.model, 1000, 200,
        )
        self.assertEqual(result["total_cost"], 0.0)
        self.assertEqual(result["input_cost"], 0.0)
        self.assertEqual(result["output_cost"], 0.0)
        self.assertTrue(result["coverage_complete"])
        self.assertEqual(llm.usage_tracker.total_input_tokens, 1000)

        # Paid and unknown non-local models retain their existing behavior.
        self.assertGreater(ModelPricing.get_price("qwen3-8b", "input"), 0)
        unknown = "unknown-remote-for-cost-regression-test"
        ModelPricing._warned_unknown.discard(unknown)
        with patch("base.engine.async_llm.logger.warning") as warning:
            self.assertEqual(ModelPricing.get_price(unknown, "input"), 0)
            self.assertEqual(ModelPricing.get_price(unknown, "output"), 0)
            warning.assert_called_once()

    def test_template_mode_does_not_construct_llm_backends(self):
        with patch(
            "environment.generate_question.authoring.backend.OpenAICompatibleBackend"
        ) as backend, patch(
            "environment.base_orchestration.generate_turns.generate_turns",
            return_value={"status": "done"},
        ) as runner:
            generate_from_config(mode="template", max_scenes=4)
        backend.assert_not_called()
        self.assertIsNone(runner.call_args.kwargs["author"])
        self.assertIsNone(runner.call_args.kwargs["critic"])
        self.assertEqual(runner.call_args.kwargs["max_scenes"], 4)

    def test_invalid_unknown_or_mismatched_settings_fail(self):
        template = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        with TemporaryDirectory() as folder:
            target = Path(folder) / "config.yaml"
            invalid = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
            invalid["generation"]["unused_field"] = 5
            target.write_text(yaml.safe_dump(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown_generation"):
                load_config(target)
            invalid = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
            invalid["generation"]["allow_delegate_fallback"] = "false"
            target.write_text(yaml.safe_dump(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown_generation"):
                load_config(target)
            invalid = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
            invalid["generation"]["workers"] = 0
            target.write_text(yaml.safe_dump(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid_workers"):
                load_config(target)
            invalid = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
            invalid["model"]["author_max_tokens"] = -1
            target.write_text(yaml.safe_dump(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid_author_max_tokens"):
                load_config(target)
            invalid = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
            invalid["input"]["scenes"] = ""
            target.write_text(yaml.safe_dump(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "path_must_be_nonempty"):
                load_config(target)
