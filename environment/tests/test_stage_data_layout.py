"""Canonical Stage1–4 paths and single-fault public CLI contracts."""
from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from environment.__main__ import main
from environment.config import DEFAULT_CONFIG as QUESTIONS, load_config as load_questions
from environment.implement_subtask.config import (
    DEFAULT_CONFIG as REPORTS,
    load_config as load_subtasks,
    load_reporting_config,
)
from environment.base_orchestration.config import (
    DEFAULT_CONFIG as MAIN, load_config as load_main,
)
from environment.untrusted_orchestration.config import (
    DEFAULT_CONFIG as UNTRUSTED, load_config as load_untrusted,
)


class CanonicalStagePathsTests(TestCase):
    def test_paths_are_functional_and_shared_by_every_stage(self):
        stage1 = load_questions(QUESTIONS)
        subtask = load_subtasks(REPORTS)
        reports = load_reporting_config(REPORTS)
        main = load_main(MAIN)
        stage4 = load_untrusted(UNTRUSTED)
        root = Path(__file__).resolve().parents[1] / "data"
        pilot = root / "pilots" / "qwen_stage234_10"
        self.assertEqual(QUESTIONS.name, "questions.yaml")
        self.assertEqual(REPORTS.name, "subagent_reports.yaml")
        self.assertEqual(MAIN.name, "main_orchestration.yaml")
        self.assertEqual(UNTRUSTED.name, "untrusted_orchestration.yaml")
        self.assertEqual(stage1["generation"]["target"], 100)
        self.assertGreaterEqual(stage1["input"]["max_records"], 20)
        self.assertEqual(
            stage1["generation"]["output_dir"],
            "environment/data/stage1",
        )
        self.assertEqual(subtask["output_dir"], pilot / "stage2" / "oracle")
        self.assertEqual(reports["output_dir"], pilot / "stage2")
        self.assertEqual(
            reports["source_subtasks"],
            subtask["output_dir"] / "subtasks.jsonl",
        )
        scenes = root / "stage1" / "questions" / "scenes.jsonl"
        self.assertEqual(subtask["scenes_path"], scenes)
        self.assertEqual(main["scenes_path"], scenes)
        self.assertEqual(stage4["scenes_path"], scenes)
        self.assertEqual(
            main["reports_path"], reports["output_dir"] / "subtasks.jsonl",
        )
        self.assertEqual(stage4["reports_path"], main["reports_path"])
        self.assertEqual(main["output_dir"], pilot / "stage3")
        self.assertEqual(stage4["output_dir"], pilot / "stage4")
        for p in (subtask["output_dir"], reports["output_dir"],
                  main["output_dir"], stage4["output_dir"]):
            self.assertNotRegex(str(p), r"v\d+[_\.]\d+")
        self.assertEqual(stage4["clean_episodes_path"],
                         main["output_dir"] / "episodes.jsonl")
        # The Stage-2 bank may be larger than the Stage-3/4 pilot.
        # Check the handoff capacity, not a stale pilot-specific count.
        self.assertEqual(reports["max_scenes"], subtask["max_records"])
        self.assertGreaterEqual(reports["max_scenes"], main["max_scenes"])
        # Stage 3 may build a larger reviewed bank than the Stage-4 pilot.
        # Avoid coupling this handoff test to an outdated Stage-3 quota.
        self.assertGreaterEqual(main["max_scenes"], stage4["max_scenes"])
        self.assertEqual(stage4["max_scenes"], 10)
        self.assertIsNone(stage4["branches_per_scene"])
        self.assertEqual(stage4["round_sampling_strategy"], "chronological")
        self.assertNotIn("decision_unit", stage4)
        self.assertNotIn("max_post_fork_rounds", stage4)
        self.assertTrue(main["freeze_dispatch"])
        self.assertNotIn("max_fault_atoms", stage4)
        self.assertNotIn("delayed_evidence_probability", stage4)
        self.assertEqual(stage4["mode"], "llm")

    def test_stage3_public_cli_routes_to_config_without_old_runner(self):
        with patch(
            "environment.__main__.generate_main_from_config",
            return_value={"status": "done"},
        ) as generate:
            result = main([
                "generate-main-turns",
                "--config", str(MAIN), "--max-scenes", "12",
                "--output-dir", "environment/data/pilots/stage3",
                "--mode", "template", "--workers", "2",
            ])
        self.assertEqual(result, 0)
        self.assertEqual(generate.call_args.kwargs["max_scenes"], 12)
        self.assertEqual(generate.call_args.kwargs["mode"], "template")
        self.assertEqual(generate.call_args.kwargs["workers"], 2)

    def test_batched_evidence_is_private_not_in_public_sft(self):
        from environment.untrusted_orchestration.inspection.visible_claim_state import (
            audit_visible_claim_states,
        )
        scene = {
            "task_id": "local",
            "world": {"sources": {"s": {
                "evidence_origin_id": "private-root"
            }}},
            "macro_graph": {"tasks": [{
                "task_id": "a", "source_ids": ["s"], "depends_on": [],
                "outputs": {"typed_relation": ["case_id", "metric_a"]},
            }]},
        }
        report = {
            "finish_result": {"status": "done",
                              "result": '[{"case_id":"Case-001","metric_a":4}]',
                              "summary": "Case-001: metric_a=4"}
        }
        audit = audit_visible_claim_states(
            scene, [{
                "attempt": 1, "action": "delegate_task",
                "result": {"subtask_results": [report]},
            }], task_ids_by_round={1: ["a"]},
        )
        self.assertTrue(audit["private_only"])
        self.assertEqual(audit["rounds"][0]["main_can_act_from_round"], 2)
        self.assertIn("private-root", str(audit))
        self.assertNotIn("private-root", str(report))


if __name__ == "__main__":
    import unittest
    unittest.main()
