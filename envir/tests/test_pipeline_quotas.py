"""Accepted difficulty quotas and auditable Scenario field repairs."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from envir.generate_question.quota_pipeline import (
    quota_targets, deficits, percentage_mix, generate_with_quotas,
    write_jsonl, write_json,
)
from envir.generate_question.scenario_fields import repair_scenario_fields
from envir.generate_question.scenario import _narrative_issues


class ScenarioFieldRepairTests(unittest.TestCase):
    def test_long_motivation_repairs_without_changing_frozen_semantics(self):
        spec = {
            "scenario": {"topic": "Premier League"},
            "macro_plan": {"goal": {"public_label": "match result"}},
        }
        original = {
            "scenario": {
                "setting": "Investigation of a match event",
                "user_role": "research analyst",
                "purpose": "Confirm the requested match result",
                "motivation": "Compare records for the final match result. " * 18,
                "entity_kind": "match events",
            },
            "source_titles": [{"source_id": "visual", "title": "Visual annotations"}],
        }
        modified, repairs = repair_scenario_fields(spec, original)
        self.assertGreater(len(original["scenario"]["motivation"]), 250)
        self.assertLessEqual(len(modified["scenario"]["motivation"]), 250)
        self.assertIn("Premier League", modified["scenario"]["setting"])
        self.assertEqual(original["source_titles"], modified["source_titles"])
        self.assertEqual(original["scenario"]["purpose"],
                         modified["scenario"]["purpose"])
        self.assertIn("too_long", {r["reason"] for r in repairs})
        self.assertIn("topic_prefix", {r["reason"] for r in repairs})

    def test_short_and_non_string_fields_have_explicit_reasons(self):
        spec = {"scenario": {"topic": "Some League"},
                "macro_plan": {"goal": {"public_label": "match result"}}}
        draft = {"scenario": {"setting": "ok", "user_role": None,
                              "purpose": "Find the match result",
                              "motivation": "", "entity_kind": 99}}
        repaired, flags = repair_scenario_fields(spec, draft)
        self.assertTrue(all(isinstance(repaired["scenario"][k], str)
                            for k in repaired["scenario"]))
        self.assertTrue(any(x["reason"] == "too_short" for x in flags))
        self.assertTrue(any(x["reason"] == "invalid_type_or_missing"
                            for x in flags))


class AcceptedQuotaTests(unittest.TestCase):
    def test_quota_allocation_exact_one_two_three(self):
        goals = quota_targets(100, (17, 33, 50), seed=17)
        self.assertEqual(goals, {"easy": 17, "medium": 33, "hard": 50})
        self.assertEqual(quota_targets(6, (17, 33, 50), seed=17),
                         {"easy": 1, "medium": 2, "hard": 3})
        self.assertEqual(percentage_mix({
            "easy": 0, "medium": 20, "hard": 30
        }), (0, 40, 60))
        self.assertEqual(sum(percentage_mix({
            "easy": 0, "medium": 1, "hard": 1
        })), 100)

    def test_bounded_rounds_fill_real_accepted_quotas(self):
        from envir.generate_question.spec import difficulty_sequence
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "mmqa.jsonl"
            raw.write_text('{"fixture":true}\n')
            args = argparse.Namespace(
                output_dir=root / "output", target=60, seed=17,
                difficulty_mix="17,33,50", max_candidate_multiplier=3,
                overwrite=False, no_progress=True, mmqa=raw,
                scenario_model="mock", scenario_critic_model="mock",
                question_model="mock", question_critic_model="mock",
            )
            calls = []
            def mock_batch(batch):
                calls.append(batch.target)
                seq = difficulty_sequence(
                    target=batch.target,
                    mix=tuple(map(int, batch.difficulty_mix.split(","))),
                    run_seed=batch.seed,
                )
                rows = []
                for i, difficulty in enumerate(seq):
                    # Hard/Medium have lower Scenario acceptance than Easy.
                    if difficulty != "easy" and i % 4 == 0:
                        continue
                    key = f"{batch.seed}-{i}"
                    rows.append({
                        "task_id": key, "question": f"Question {key}?",
                        "problem_spec": {"difficulty_target": difficulty},
                        "public": {"question": f"Question {key}?"},
                        "difficulty_audit": {},
                        "question_authoring": {"status": "accepted"},
                    })
                write_jsonl(batch.output_dir / "questions" / "scenes.jsonl",
                            rows)
                return {
                    "attempted": batch.target, "generated": len(rows),
                    "scenario_authoring": {"accepted": len(rows)},
                    "question_authoring": {"accepted": len(rows)},
                }
            def fake_quality(scenes, **_):
                return {
                    "status": "valid", "checked": len(scenes),
                    "replay_pass": len(scenes), "replay_fail": 0,
                    "macro_graph_diversity": {},
                    "reasoning": {"unique_structural_program_signatures": 1},
                    "question_surface": {"unique_abstract_shapes": len(scenes)},
                    "interpretation_limits": [],
                }
            result = generate_with_quotas(
                args, generate_batch=mock_batch, quality_report=fake_quality
            )
            self.assertEqual(result["status"], "done")
            self.assertEqual(result["generated"], 60)
            self.assertEqual(result["difficulty_quotas"]["met"], True)
            self.assertEqual(result["difficulty_quotas"]["deficit"],
                             {"easy": 0, "medium": 0, "hard": 0})
            self.assertGreater(len(calls), 1)
            self.assertLessEqual(sum(calls), 180)
            self.assertEqual(
                len((args.output_dir / "questions" / "scenes.jsonl")
                    .read_text().splitlines()), 60,
            )
            with self.assertRaises(FileExistsError):
                generate_with_quotas(
                    args, generate_batch=mock_batch, quality_report=fake_quality
                )


if __name__ == "__main__":
    unittest.main()
