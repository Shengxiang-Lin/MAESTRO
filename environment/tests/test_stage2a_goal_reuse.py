"""Stage2A audited Goal reuse is contract-bound and never silently approves stale data."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from environment.question_generation.goal_cache import load_reusable_fragments
from environment.question_generation.goals_pipeline import generate_goal_record
from environment.question_generation.goal_review import GOAL_REVIEW_METHOD
from environment.tests.test_stage2a_stage2b_split import (
    GoalWriter, GoalCritic, SplitPipelineTests,
)


class Stage2AReuseTests(unittest.TestCase):
    def test_reuse_only_passed_fragments_with_same_outline_and_review_contract(self):
        payload, world, capability, _, outline = SplitPipelineTests().context()
        writer, critic = GoalWriter(), GoalCritic()
        ok, record, audit = generate_goal_record(
            payload, world, capability, writer, critic,
            max_goal_revisions=0,
        )
        self.assertTrue(ok, (record, audit))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "goal_review_audit.jsonl"
            path.write_text(json.dumps({
                "task_id": payload["task_id"], "audit": audit,
            }) + "\n", encoding="utf-8")
            models = {
                "question_author": {"model": "local-test"},
                "question_critic": {"model": "local-test"},
            }
            (path.parent / "goal_generation_summary.json").write_text(
                json.dumps({
                    "goal_review_contract": GOAL_REVIEW_METHOD,
                    "model_roles": models,
                }), encoding="utf-8",
            )
            reused = load_reusable_fragments(path, model_roles=models)
            self.assertEqual(len(reused), len(outline["tasks"]))
            author2, critic2 = GoalWriter(), GoalCritic()
            ok, reused_record, audit2 = generate_goal_record(
                payload, world, capability, author2, critic2,
                max_goal_revisions=0, reused_fragments=reused,
            )
            self.assertTrue(ok, (reused_record, audit2))
            self.assertEqual(author2.calls, [])
            self.assertEqual(critic2.calls, [])
            self.assertEqual(
                record["delegation_blueprint"],
                reused_record["delegation_blueprint"],
            )
            self.assertTrue(all(
                row.get("reused") is True
                for row in audit2 if row.get("stage") == "fragment_goal"
            ))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_reusable_fragments(path, model_roles={})
            (path.parent / "goal_generation_summary.json").write_text(
                json.dumps({
                    "goal_review_contract": "stale-review-v1",
                    "model_roles": models,
                }), encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "stale"):
                load_reusable_fragments(path, model_roles=models)

    def test_cache_does_not_reuse_mismatched_outline(self):
        payload, world, capability, _, _ = SplitPipelineTests().context()
        writer, critic = GoalWriter(), GoalCritic()
        ok, record, audit = generate_goal_record(
            payload, world, capability, writer, critic,
            max_goal_revisions=0,
        )
        self.assertTrue(ok)
        reused = {
            (payload["task_id"], "wrong_outline_hash", row["fragment_id"]):
                {"goal": row["goal"], "review": record["goal_reviews"][i]["review"]}
            for i, row in enumerate(record["delegation_blueprint"]["goals"])
        }
        writer2, critic2 = GoalWriter(), GoalCritic()
        ok, _, audit2 = generate_goal_record(
            payload, world, capability, writer2, critic2,
            max_goal_revisions=0, reused_fragments=reused,
        )
        self.assertTrue(ok)
        self.assertGreater(len(writer2.calls), 0)
        self.assertEqual(
            len(writer2.calls), len(record["delegation_blueprint"]["goals"]),
        )


if __name__ == "__main__":
    unittest.main()
