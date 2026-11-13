"""Stage 1 -> 2 -> 4 public workflow and deterministic bootstrap tests."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from environment.untrusted_orchestration.trusted_prefix import (
    compile_trusted_prefixes,
)
from environment.workflow import run_pipeline


class DirectPipelineTests(unittest.TestCase):
    def test_stage_order_and_output_handoffs_without_stage3(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions_dir = root / "questions_run"
            reports_dir = root / "reports_run"
            call_order = []

            def generate_questions(options):
                call_order.append("1")
                target = questions_dir / "questions" / "scenes.jsonl"
                target.parent.mkdir(parents=True)
                target.write_text('{"task_id":"Q1"}\n')
                return {"status": "done"}

            def generate_reports(**kwargs):
                call_order.append("2")
                self.assertEqual(
                    kwargs["scenes_path"],
                    questions_dir / "questions" / "scenes.jsonl",
                )
                reports_dir.mkdir(parents=True)
                (reports_dir / "subtasks.jsonl").write_text(
                    '{"scene_task_id":"Q1"}\n'
                )
                return {"status": "done"}

            def generate_stage4(path, **kwargs):
                call_order.append("4")
                self.assertEqual(
                    kwargs["scenes_path"],
                    questions_dir / "questions" / "scenes.jsonl",
                )
                self.assertEqual(
                    kwargs["reports_path"],
                    reports_dir / "subtasks.jsonl",
                )
                self.assertEqual(kwargs["mode"], "template")
                self.assertNotIn("base_episodes_path", kwargs)
                return {"status": "done", "main_training_turns": 7}

            with (
                patch("environment.workflow.load_questions_config",
                      return_value={}),
                patch("environment.workflow.args_from_config",
                      return_value=SimpleNamespace(output_dir=questions_dir)),
                patch("environment.workflow.generate_questions",
                      side_effect=generate_questions),
                patch("environment.workflow.load_subtask_config",
                      return_value={
                          "evidence_limit": 40, "strict_replay": True,
                      }),
                patch("environment.workflow.load_reporting_config",
                      return_value={"output_dir": reports_dir}),
                patch("environment.workflow.generate_subagent_reports",
                      side_effect=generate_reports),
                patch("environment.workflow.generate_untrusted_from_config",
                      side_effect=generate_stage4),
                patch("environment.workflow.load_untrusted_output",
                      return_value=root / "final"),
            ):
                result = run_pipeline(mode="template", max_scenes=1)

            self.assertEqual(call_order, ["1", "2", "4"])
            self.assertEqual(result["published_main_turns"], 7)
            self.assertEqual(result["stages"], {
                "1": "done", "2": "done", "4": "done",
            })

    def test_bad_upstream_stops_before_stage4(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            with (
                patch("environment.workflow.load_questions_config",
                      return_value={}),
                patch("environment.workflow.args_from_config",
                      return_value=SimpleNamespace(output_dir=path)),
                patch("environment.workflow.generate_questions",
                      return_value={"status": "failed"}),
                patch("environment.workflow.generate_subagent_reports"
                      ) as reports,
                patch("environment.workflow.generate_untrusted_from_config"
                      ) as stage4,
            ):
                with self.assertRaisesRegex(RuntimeError, "stage1_not_accepted"):
                    run_pipeline(mode="template")
            reports.assert_not_called()
            stage4.assert_not_called()


class TrustedPrefixBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from environment.tests.test_untrusted_orchestration import sample
        cls.scene, cls.reports, _ = sample(0)

    def test_stage1_stage2_material_is_sufficient(self):
        prefixes, reviews = compile_trusted_prefixes(
            [self.scene], self.reports, max_rounds=16,
            sub_model="qwen3-8b", model_alias="model_1",
        )
        self.assertEqual(len(prefixes), 1, reviews)
        self.assertEqual(reviews[0]["status"], "accepted")
        trajectory = prefixes[self.scene["task_id"]]["trajectory"]
        self.assertTrue(any(t["action"] == "delegate_task" for t in trajectory))
        self.assertEqual(trajectory[-1]["action"], "complete")
        self.assertEqual(reviews[0]["mode"],
                         "deterministic_template_not_published")

    def test_missing_stage2_reports_are_explicitly_rejected(self):
        prefixes, reviews = compile_trusted_prefixes(
            [self.scene], [], max_rounds=16,
            sub_model="qwen3-8b", model_alias="model_1",
        )
        self.assertFalse(prefixes)
        self.assertEqual(reviews[0]["status"], "rejected")
        self.assertTrue(reviews[0]["cause"])


if __name__ == "__main__":
    unittest.main()
