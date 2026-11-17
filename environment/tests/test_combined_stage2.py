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
        rows = contract["exact_oracle_answer"]
        def flatten(value):
            if isinstance(value, dict):
                return ", ".join(flatten(x) for x in value.values())
            if isinstance(value, list):
                return ", ".join(flatten(x) for x in value)
            return str(value)
        details = "; ".join(
            ", ".join(flatten(value) for value in row.values())
            for row in rows
        ) or "no matching entries"
        return {
            "answer": (
                f"The assigned evidence returned {len(rows)} records: "
                f"{details}. Subsequent eligibility has not been checked."
            ),
            "trace_summary": (
                "- Reviewed the named source observations supplied for "
                "the current delegated inquiry.\n"
                "- Compared their documented clues and identifying "
                "values within the local task scope.\n"
                "- Returned the supported records for the next stage, "
                "without asserting that later checks had been performed."
            ),
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
            "result_policy": "code_owned",
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
            self.assertFalse((out / "oracle/manifest.json").exists())
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
                from environment.implement_subtask.reporting import (
                    code_owned_result, _o1_answer_issues,
                    _o1_answer_requirements,
                )
                self.assertEqual(after["answer_text"], code_owned_result(before))
                self.assertEqual(_o1_answer_issues(
                    after["finish_result"]["result"],
                    {"answer_required_literals":
                     _o1_answer_requirements(before)},
                ), [])
                self.assertFalse(after["finish_result"]["result"].startswith("[{"))
                self.assertEqual(after["report_authoring"]["result_policy"],
                                 "code_owned")
                self.assertGreater(
                    len(after["trace_summary"]["narrative"]), 60
                )
                self.assertIn(
                    "- Reviewed the named source observations",
                    after["trace_summary"]["narrative"],
                )

    def test_oracle_rebuilt_after_report_interruption(self):
        from environment.implement_subtask.pipeline import materialize_subtasks
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
            self.assertFalse((root / "reports/oracle/manifest.json").exists())
            # Re-entry must recompile Oracle, not trust a previous manifest.
            with patch(
                "environment.implement_subtask.combined.materialize_subtasks",
                wraps=materialize_subtasks,
            ) as materialize:
                resumed = generate_subagent_reports(**kwargs)
            self.assertEqual(materialize.call_count, 1)
            self.assertTrue(materialize.call_args.kwargs["overwrite"])
            self.assertEqual(resumed["status"], "done")
            self.assertEqual(resumed["accepted_scenes"], 1)

    def test_oracle_rebuilds_when_scene_or_parameters_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._inputs(root, count=1)
            kwargs = {
                "scenes_path": source, "output_dir": root / "reports",
                "reporting_options": self._options(root),
                "clients": (Author(), Critic()), "no_progress": True,
            }
            # Avoid creating a completed LLM report bank; test Oracle only.
            with patch(
                "environment.implement_subtask.combined.naturalize_subtasks",
                return_value={"status": "done"},
            ):
                first = generate_subagent_reports(**kwargs)
                with source.open("a", encoding="utf-8") as writer:
                    writer.write(json.dumps(_scene(20)) + "\n")
                second = generate_subagent_reports(
                    **kwargs, evidence_limit=3, max_scenes=2,
                )
            self.assertEqual(first["oracle_accepted_scenes"], 1)
            self.assertEqual(second["oracle_accepted_scenes"], 2)
            oracle = root / "reports/oracle"
            meta = json.loads((oracle / "summary.json").read_text())
            self.assertEqual(meta["evidence_atoms_per_step_limit"], 3)
            self.assertEqual(
                len((oracle / "scene_index.jsonl").read_text().splitlines()),
                2,
            )

    def test_stale_oracle_manifest_and_teacher_are_replaced(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._inputs(root, count=1)
            oracle = root / "reports/oracle"
            oracle.mkdir(parents=True)
            (oracle / "subtasks.jsonl").write_text('{"stale":true}\n')
            (oracle / "manifest.json").write_text(
                '{"signature":"incompatible-old-version"}'
            )
            result = generate_subagent_reports(
                scenes_path=source, output_dir=root / "reports",
                reporting_options=self._options(root),
                clients=(Author(), Critic()), no_progress=True,
            )
            self.assertEqual(result["status"], "done")
            self.assertFalse((oracle / "manifest.json").exists())
            actual = [
                json.loads(line)
                for line in (oracle / "subtasks.jsonl").read_text().splitlines()
            ]
            self.assertTrue(actual)
            self.assertTrue(all("macro_task_id" in item for item in actual))
            self.assertTrue(all("stale" not in item for item in actual))

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
