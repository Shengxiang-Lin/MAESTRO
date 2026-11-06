"""No-network regression tests for environment YAML and model-role routing."""
from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from environment.config import (
    EnvironmentConfigError,
    REPO_DIR,
    load_config,
    parse_with_config,
    resolve_model,
)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "environment.yaml"
        self.config = {
            "version": 1,
            "models": {
                "remote": {
                    "backend": "api", "model": "remote-model",
                    "base_url": "https://example.invalid/v1",
                    "api_key_env": "MAESTRO_TEST_KEY",
                },
                "local": {
                    "backend": "vllm", "model": "local-served",
                    "base_url": "http://127.0.0.1:8803/v1",
                    "api_key_env": None,
                },
            },
            "task_construction": {"candidate_pool_size": 12, "seed": 3},
            "question_generation": {
                "author_profile": "remote",
                "critic_profile": "local",
                "manifest": "environment/data/frozen_manifest_{pool_size}.json",
                "question_workers": 2,
            },
            "trajectory_simulation": {
                "planner_profile": "local",
                "executor_profile": "remote",
                "worlds": "environment/data/typed_worlds_{pool_size}.jsonl",
                "output": "environment/data/trajectories_{pool_size}.jsonl",
            },
        }
        self.path.write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def test_yaml_defaults_path_expansion_and_cli_precedence(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--manifest")
        parser.add_argument("--question-workers", type=int, default=4)
        parser.add_argument("--seed", type=int, default=17)
        parser.add_argument("--provider", default="passthrough")
        parser.add_argument("--model")
        parser.add_argument("--base-url")
        parser.add_argument("--api-key-env", default="OTHER_KEY")
        parser.add_argument("--no-json-mode", action="store_true")
        args = parse_with_config(parser, "question_generation", [
            "--config", str(self.path), "--question-workers", "7"
        ])
        self.assertEqual(args.question_workers, 7)
        self.assertEqual(
            args.manifest,
            str(REPO_DIR / "environment/data/frozen_manifest_12.json"),
        )
        with patch.dict(os.environ, {"MAESTRO_TEST_KEY": "test-secret"}):
            author = resolve_model(args, "question_author")
            critic = resolve_model(args, "question_critic")
        self.assertEqual(author["backend"], "api")
        self.assertEqual(author["api_key"], "test-secret")
        self.assertEqual(critic["backend"], "vllm")
        self.assertEqual(critic["model"], "local-served")
        self.assertIsNone(critic["api_key"])

    def test_explicit_offline_override(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--provider", default="openai-compatible")
        parser.add_argument("--worlds")
        parser.add_argument("--output")
        parser.add_argument("--seed", type=int, default=17)
        args = parse_with_config(parser, "trajectory_simulation", [
            "--provider", "passthrough", "--config", str(self.path)
        ])
        self.assertEqual(resolve_model(args, "trajectory_planner")["backend"], "passthrough")
        self.assertEqual(resolve_model(args, "trajectory_executor")["backend"], "passthrough")

    def test_separate_trajectory_profiles_and_model_overrides(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--provider", default="openai-compatible")
        parser.add_argument("--model")
        parser.add_argument("--planner-model")
        parser.add_argument("--executor-model")
        parser.add_argument("--worlds")
        parser.add_argument("--output")
        parser.add_argument("--seed", type=int, default=17)
        args = parse_with_config(parser, "trajectory_simulation", [
            "--config", str(self.path), "--planner-model", "planner-only"
        ])
        with patch.dict(os.environ, {"MAESTRO_TEST_KEY": "test-secret"}):
            planner = resolve_model(args, "trajectory_planner")
            executor = resolve_model(args, "trajectory_executor")
        self.assertEqual(planner["model"], "planner-only")
        self.assertEqual(planner["backend"], "vllm")
        self.assertEqual(executor["model"], "remote-model")
        self.assertEqual(executor["backend"], "api")
        self.assertEqual(executor["api_key"], "test-secret")

    def test_subparser_required_paths_satisfied_by_yaml(self):
        root = argparse.ArgumentParser()
        subs = root.add_subparsers(dest="command", required=True)
        sub = subs.add_parser("synthesize")
        sub.add_argument("--worlds", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--seed", type=int, default=17)
        args = parse_with_config(root, "trajectory_simulation", [
            "synthesize", "--config", str(self.path)
        ])
        self.assertEqual(args.command, "synthesize")
        self.assertTrue(args.worlds.endswith("typed_worlds_12.jsonl"))
        self.assertTrue(args.output.endswith("trajectories_12.jsonl"))

    def test_missing_remote_api_key_has_helpful_error(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--manifest")
        parser.add_argument("--question-workers", type=int)
        parser.add_argument("--seed", type=int)
        args = parse_with_config(parser, "question_generation", [
            "--config", str(self.path)
        ])
        with patch.dict(os.environ, {"MAESTRO_TEST_KEY": ""}):
            with self.assertRaisesRegex(EnvironmentConfigError, "MAESTRO_TEST_KEY"):
                resolve_model(args, "question_author")

    def test_missing_profile_fails(self):
        self.config["question_generation"]["critic_profile"] = "does_not_exist"
        self.path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        parser = argparse.ArgumentParser()
        parser.add_argument("--manifest")
        parser.add_argument("--question-workers", type=int)
        parser.add_argument("--seed", type=int)
        args = parse_with_config(parser, "question_generation", [
            "--config", str(self.path)
        ])
        with self.assertRaisesRegex(EnvironmentConfigError, "does_not_exist"):
            resolve_model(args, "question_critic")

    def test_thinking_setting_reaches_both_question_clients_and_readback(self):
        from dataclasses import replace
        from environment.question_generation.pipeline import _backends, build_parser
        self.config['models']['remote']['enable_thinking'] = False
        self.config['question_generation']['critic_profile'] = 'remote'
        self.path.write_text(yaml.safe_dump(self.config), encoding='utf-8')
        args = parse_with_config(build_parser(), 'question_generation', ['--config', str(self.path)])
        with patch.dict(os.environ, {'MAESTRO_TEST_KEY': 'test-secret'}):
            author, critic = _backends(args)
        self.assertIs(author.backend.enable_thinking, False)
        self.assertIs(critic.backend.enable_thinking, False)
        self.assertIs(replace(critic.backend, max_tokens=4096).enable_thinking, False)

    def test_thinking_flag_is_sent_only_when_configured(self):
        import json
        from unittest.mock import MagicMock
        from environment.question_generation.backend import OpenAICompatibleBackend
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            'choices': [{'message': {'content': '{"ok":true}'}}],
        }).encode()
        for setting in (None, False, True):
            backend = OpenAICompatibleBackend('http://test.local/v1', 'stub',
                                              enable_thinking=setting)
            with patch('environment.question_generation.backend.urllib.request.urlopen',
                       return_value=response) as request:
                self.assertEqual(backend.complete_json('system', 'question'), {'ok': True})
                body = json.loads(request.call_args.args[0].data)
                if setting is None:
                    self.assertNotIn('enable_thinking', body)
                else:
                    self.assertIs(body['enable_thinking'], setting)


if __name__ == "__main__":
    unittest.main()
