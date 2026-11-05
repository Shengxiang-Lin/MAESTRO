"""Round-centric gold SFT: independent states with causal public teacher history."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from environment.tests.test_gold_trajectory import fixture
from environment.trajectory_simulation.backend import PassthroughPlanner, PassthroughExecutor
from environment.trajectory_simulation.eligibility import digest
from environment.trajectory_simulation.exporter import validate_public_episode
from environment.trajectory_simulation.pipeline import (
    GoldTrajectoryError, synthesize_one,
)
from environment.trajectory_simulation.rounds import (
    RoundSink, make_round_record, synthesize_rounds_jsonl, o1_input,
)


def reviewed(gold, record, role, *, review=True):
    return {
        "eligible": True, "reviewed": True,
        "question_sha256": digest(gold["question"]),
        "program_sha256": digest(record["program"]),
        "method": "test_approved_exact_question",
        "issues": [],
    }


class RoundCentricTests(unittest.TestCase):
    def test_o1_input_reuses_real_baseline_prompt_and_only_public_history(self):
        state = {
            "question": "Who wrote the research report?",
            "subtask_history": "No subtasks completed yet.",
            "available_tools": ["GoogleSearchAction", "ExecuteCodeAction"],
            "evidence_store": {},
        }
        text = o1_input(state, [], question_record={}, round_index=1)
        self.assertIn("==== QUESTION ====", text)
        self.assertIn("==== SUBTASK HISTORY ====", text)
        self.assertIn("==== AVAILABLE TOOLS", text)
        self.assertIn("[Attempt 1/10]", text)
        self.assertIn("Who wrote the research report?", text)
        self.assertNotIn("Gold", text)
        self.assertNotIn("expected_answer", text)

    def test_single_selected_round_builds_prior_clean_feedback_without_future(self):
        q, record, rendered = fixture()
        planner, executor = PassthroughPlanner(), PassthroughExecutor()
        with tempfile.TemporaryDirectory() as temp, patch(
            "environment.trajectory_simulation.pipeline.check_question", reviewed
        ):
            out = Path(temp) / "rounds.jsonl"
            sink = RoundSink(out, question=q, include_fallback_rounds=True)
            try:
                episode = synthesize_one(
                    q, record, rendered, planner, executor,
                    schedule="baseline",
                    round_sink=sink,
                    selected_rounds={2},
                    history_style="teacher",
                    stop_after_round=2,
                )
            finally:
                sink.close()
            self.assertTrue(episode["partial"])
            self.assertEqual(episode["last_round"], 2)
            self.assertEqual(sink.written, 1)
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["round_index"], 2)
            self.assertEqual(row["history_origin"], "teacher_reconstructed")
            self.assertEqual(row["macro_label"], "advance")
            self.assertTrue(row["history_valid"])
            self.assertTrue(row["current_action_valid"])
            self.assertIn("[Attempt 2/10]", row["input"])
            self.assertIn("[Attempt 1]", row["input"])
            self.assertIn("Bob", row["input"])
            self.assertNotIn("gold_supervision", row["input"])
            self.assertEqual(json.loads(row["target_json"])["action"], "delegate_task")

    def test_natural_main_failure_does_not_block_later_clean_round(self):
        class BadFirstMain(PassthroughPlanner):
            offline = False
            style_provider = "natural"

            def naturalize_action(self, payload, canonical):
                if canonical["action"] == "complete":
                    return dict(canonical)
                # Language writer fails both rounds; continue with valid
                # canonical execution but withhold strict natural SFT labels.
                raise ValueError("intentionally_bad_natural_language")

        q, record, rendered = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "environment.trajectory_simulation.pipeline.check_question", reviewed
        ):
            path = Path(tmp) / "strict.jsonl"
            sink = RoundSink(path, question=q, include_fallback_rounds=False)
            try:
                synthesize_one(
                    q, record, rendered, BadFirstMain(), PassthroughExecutor(),
                    schedule="baseline",
                    round_sink=sink, selected_rounds={1,2},
                    history_style="teacher", stop_after_round=2,
                )
            finally:
                sink.close()
            self.assertEqual(sink.considered, 2)
            self.assertEqual(sink.skipped, 2)
            self.assertEqual(sink.written, 0)
            self.assertEqual(sink.skip_reasons["incomplete_naturalization"], 2)
            self.assertEqual(path.read_text(), "")

    def test_successful_round_persists_if_later_main_action_raises(self):
        class BadLaterMain(PassthroughPlanner):
            calls = 0

            def decide(self, payload, prompt):
                self.calls += 1
                if self.calls > 1:
                    raise ValueError("later_main_failed")
                return super().decide(payload,prompt)

        q, record, rendered = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "environment.trajectory_simulation.pipeline.check_question", reviewed
        ):
            out = Path(tmp) / "rounds.jsonl"
            sink = RoundSink(out, question=q, include_fallback_rounds=True)
            try:
                with self.assertRaisesRegex(GoldTrajectoryError, "later_main_failed"):
                    synthesize_one(
                        q, record, rendered, BadLaterMain(), PassthroughExecutor(),
                        max_attempts=1, schedule="baseline", round_sink=sink,
                    )
            finally:
                sink.close()
            self.assertEqual(sink.written, 1)
            self.assertEqual(json.loads(out.read_text().strip())["round_index"], 1)

    def test_full_trajectory_prefix_validator_accepts_partial_clean_history(self):
        q, record, rendered = fixture()
        episode = synthesize_one(
            q, record, rendered, PassthroughPlanner(), PassthroughExecutor(),
        )
        prefix = episode["trajectory"][:1]
        validate_public_episode({
            "task_id": episode["task_id"],
            "question": episode["question"],
            "input_eligibility": episode["input_eligibility"],
            "validation": episode["validation"],
            "trajectory": prefix,
        }, require_training=False, require_terminal=False)
        with self.assertRaisesRegex(ValueError, "missing terminal"):
            validate_public_episode({
                "task_id": episode["task_id"],
                "question": episode["question"],
                "input_eligibility": episode["input_eligibility"],
                "validation": episode["validation"],
                "trajectory": prefix,
            }, require_training=False)
        with self.assertRaisesRegex(ValueError, "not a history prefix"):
            validate_public_episode(episode, require_training=False,
                                    require_terminal=False)

    def test_streaming_round_synthesis_retains_samples_after_later_failure(self):
        q, record, rendered = fixture()
        record["task_id"] = q["task_id"]
        with tempfile.TemporaryDirectory() as tmp, patch(
            "environment.trajectory_simulation.pipeline.check_question", reviewed
        ), patch(
            "environment.trajectory_simulation.pipeline._rows", return_value=[q],
        ), patch(
            "environment.trajectory_simulation.pipeline._world_map",
            return_value={q["task_id"]:(record, rendered)},
        ):
            src = Path(tmp) / "worlds.jsonl"
            realized = Path(tmp) / "realized.jsonl"
            src.write_text("{}\n")
            realized.write_text("{}\n")
            dst = Path(tmp) / "rounds.jsonl"
            result = synthesize_rounds_jsonl(
                src, realized, dst,
                planner=PassthroughPlanner(), executor=PassthroughExecutor(),
                progress=False, schedule="baseline",
                history_style="teacher", include_fallback_rounds=True,
                round_indices={1,2},
            )
            self.assertEqual(result["rounds_considered"], 2)
            self.assertEqual(result["eligible_rounds"], 2)
            self.assertEqual(result["tasks_processed"], 1)
            self.assertEqual(result["history_origins"], {
                "natural":1, "teacher_reconstructed":1,
            })
            lines=[json.loads(row) for row in dst.read_text().splitlines()]
            self.assertEqual(len(lines),2)
            self.assertEqual([row["round_index"] for row in lines],[1,2])
            self.assertTrue(all(row["history_valid"] for row in lines))

    def test_round_record_rejects_private_or_missing_history_evidence(self):
        q, record, rendered = fixture()
        episode = synthesize_one(q, record, rendered,
                                  PassthroughPlanner(), PassthroughExecutor())
        history = episode["trajectory"][:1]
        step = episode["trajectory"][1]
        fake = json.loads(json.dumps(history))
        fake[0]["result"]["subtask_results"][0]["result_value"] = ["NotReally"]
        for prior in (fake,):
            with self.assertRaises(ValueError):
                make_round_record(
                    task_id=episode["task_id"], round_index=2,
                    question=episode["question"],
                    eligibility=episode["input_eligibility"],
                    state=step["state_before"],prefix=prior,action=step,
                    question_record=q, natural_task_count=0,
                    assigned_task_count=len(step["params"].get("tasks",[])),
                    history_origin="teacher_reconstructed",require_review=False,
                )


if __name__ == "__main__":
    unittest.main()
