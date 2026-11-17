"""One-command Stage-2 Oracle -> realistic Sub-Agent report regressions."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from environment.__main__ import main, parser
from environment.tests.current_stage1 import current_scene
from environment.implement_subtask.combined import generate_subagent_reports
from environment.streaming import StreamingError


def _scene(i):
    scene = current_scene(i, difficulty="medium")
    scene["question_authoring"] = {"status": "accepted"}
    return scene


class Author:
    def __init__(self):
        self.calls = 0

    def complete_json(self, system, payload):
        self.calls += 1
        contract = json.loads(payload)["frozen_task_and_evidence"]
        return {
            "result": contract["exact_oracle_answer"],
            "summary": (
                "I checked the authorized input and applied the recorded "
                "operations for this specific handoff. The exact records "
                "were returned without inventing any later eligibility test."
            ),
            "status": "done",
        }


class Critic:
    def complete_json(self, system, payload):
        return {
            "verdict": "pass",
            "grounded": True,
            "answer_scope_preserved": True,
            "evidence_scope_respected": True,
            "natural": True,
            "reason": "Matches the exact source-scoped handoff.",
            "repair_hint": "",
        }


class CombinedStage2Tests(unittest.TestCase):
    def _inputs(self, root, count=2):
        file = root / "accepted_scenes.jsonl"
        with file.open("w", encoding="utf-8") as out:
            for i in range(count):
                out.write(json.dumps(_scene(i), ensure_ascii=False) + "\n")
        return file

    @staticmethod
    def _options(root):
        return {
            "source_subtasks": root / "unused_teacher.jsonl",
            "source_scene_index": root / "unused_index.jsonl",
            "output_dir": root / "reports",
            "api_base": "http://unused/v1",
            "model": "mock-model",
            "workers": 2,
            "max_revisions": 0,
            "timeout": 60,
            "flush_every": 1,
            "overwrite": False,
        }

    def test_single_command_materializes_oracle_and_publishes_finish(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._inputs(root)
            author = Author()
            out = root / "reports"
            result = generate_subagent_reports(
                scenes_path=source, output_dir=out,
                reporting_options=self._options(root),
                evidence_limit=12, strict_replay=True,
                clients=(author, Critic()), no_progress=True,
            )
            self.assertEqual(result["status"], "done")
            self.assertEqual(result["oracle_status"], "done")
            self.assertEqual(result["accepted_scenes"], 2)
            self.assertEqual(result["failed_scenes"], 0)
            self.assertGreater(result["oracle_macro_tasks"], 0)
            self.assertEqual(result["published_macro_tasks"],
                             result["oracle_macro_tasks"])
            self.assertEqual(result["completed_macro_tasks"], author.calls)
            self.assertTrue((out / "oracle/manifest.json").is_file())
            oracle = [
                json.loads(line) for line in
                (out / "oracle/subtasks.jsonl").read_text().splitlines()
            ]
            finished = [
                json.loads(line) for line in
                (out / "subtasks.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(oracle), len(finished))
            for before, after in zip(oracle, finished):
                self.assertEqual(before["answer"], after["answer"])
                self.assertEqual(before["trace_summary"]["steps"],
                                 after["trace_summary"]["steps"])
                self.assertEqual(after["finish_result"]["status"], "done")

    def test_finished_oracle_reused_after_report_interruption(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._inputs(root, count=1)
            kwargs = {
                "scenes_path": source, "output_dir": root / "reports",
                "reporting_options": self._options(root),
                "clients": (Author(), Critic()), "no_progress": True,
            }
            with patch(
                "environment.implement_subtask.combined.naturalize_subtasks",
                side_effect=RuntimeError("report interrupted"),
            ):
                with self.assertRaisesRegex(RuntimeError, "report interrupted"):
                    generate_subagent_reports(**kwargs)
            self.assertTrue(
                (root / "reports/oracle/manifest.json").is_file()
            )
            with patch(
                "environment.implement_subtask.combined.materialize_subtasks",
                side_effect=AssertionError("should reuse completed Oracle"),
            ):
                resumed = generate_subagent_reports(**kwargs)
            self.assertEqual(resumed["status"], "done")
            self.assertEqual(resumed["accepted_scenes"], 1)

    def test_oracle_manifest_invalidates_changed_scene_or_export(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._inputs(root, count=1)
            kwargs = {
                "scenes_path": source, "output_dir": root / "reports",
                "reporting_options": self._options(root),
                "clients": (Author(), Critic()), "no_progress": True,
            }
            generate_subagent_reports(**kwargs)
            with source.open("a", encoding="utf-8") as writer:
                writer.write(json.dumps(_scene(20)) + "\n")
            with self.assertRaisesRegex(StreamingError, "does not match"):
                generate_subagent_reports(**kwargs)

    def test_existing_legacy_teacher_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._inputs(root, count=1)
            oracle = root / "reports/oracle"
            oracle.mkdir(parents=True)
            (oracle / "subtasks.jsonl").write_text('{"stale":true}\n')
            with self.assertRaisesRegex(StreamingError, "without a manifest"):
                generate_subagent_reports(
                    scenes_path=source, output_dir=root / "reports",
                    reporting_options=self._options(root),
                    clients=(Author(), Critic()), no_progress=True,
                )

    def test_cli_scenes_dispatches_combined_and_rejects_double_inputs(self):
        parsed = parser().parse_args([
            "naturalize-subtasks", "--config",
            "environment/configs/subagent_reports.yaml",
            "--scenes", "environment/data/example/questions/scenes.jsonl",
            "--output-dir", "environment/data/out",
        ])
        self.assertIsNotNone(parsed.scenes)
        with patch("environment.__main__.generate_subagent_reports",
                   return_value={"status": "done"}) as called:
            self.assertEqual(main([
                "naturalize-subtasks",
                "--config", "environment/configs/subagent_reports.yaml",
                "--scenes", "environment/data/example/questions/scenes.jsonl",
                "--output-dir", "environment/data/out",
            ]), 0)
            self.assertEqual(called.call_count, 1)
            self.assertEqual(
                called.call_args.kwargs["evidence_limit"], 12
            )
        # A config-only command now takes implementation.scenes and
        # performs both Oracle and Sub-Agent report generation.
        from environment.implement_subtask.config import DEFAULT_CONFIG, load_config
        with patch("environment.__main__.generate_subagent_reports",
                   return_value={"status": "done"}) as from_config:
            self.assertEqual(main([
                "naturalize-subtasks", "--config", str(DEFAULT_CONFIG),
            ]), 0)
            self.assertEqual(
                from_config.call_args.kwargs["scenes_path"],
                load_config(DEFAULT_CONFIG)["scenes_path"],
            )
        with self.assertRaises(SystemExit):
            main([
                "naturalize-subtasks",
                "--scenes", "scenes.jsonl",
                "--source-subtasks", "old.jsonl",
            ])


if __name__ == "__main__":
    unittest.main()
