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
            "generate-questions", "quality", "implement-subtasks",
            "naturalize-subtasks", "generate-main-turns", "generate-untrusted-turns", "run",
        })

    def test_public_cli_runs_stage1_to_stage2_to_stage4(self):
        with patch("environment.__main__.run_pipeline",
                   return_value={
                       "status": "done", "published_main_turns": 4,
                   }) as runner:
            self.assertEqual(main([
                "run", "--mode", "template", "--max-scenes", "5",
                "--no-progress",
            ]), 0)
            self.assertEqual(runner.call_args.kwargs["mode"], "template")
            self.assertEqual(runner.call_args.kwargs["max_scenes"], 5)
            self.assertTrue(runner.call_args.kwargs["no_progress"])

    def test_runner_uses_yaml_values_without_hidden_defaults(self):
        with patch(
            "environment.generate_question.backend.OpenAICompatibleBackend"
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
        self.assertEqual(kwargs["max_attempts"], 16)
        self.assertEqual(kwargs["author_model"], "local-qwen3.5-9b")
        self.assertEqual(kwargs["critic_model"], "local-qwen3.5-9b")
        self.assertTrue(kwargs["scenes_path"].is_absolute())

    def test_template_mode_does_not_construct_llm_backends(self):
        with patch(
            "environment.generate_question.backend.OpenAICompatibleBackend"
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
