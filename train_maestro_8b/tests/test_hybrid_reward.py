"""CPU-only tests: hard gate, semantic scoring, privacy, cache and API failures."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from environment.untrusted_orchestration.rewards.decision_reward_state import SCHEMA
from train_maestro_8b.policy.hybrid_reward import HybridReward
from train_maestro_8b.policy.semantic_judge import (
    DIMENSIONS, SemanticJudge, parse_semantic_response,
)
from train_maestro_8b.policy.train import parse_args, validate_args


def action():
    return json.dumps({
        "action": "delegate_task",
        "reasoning": "Collect the required documented evidence.",
        "params": {"tasks": [{
            "task_instruction": "Retrieve project_id from source_A.",
            "context": "Needed for the user question.",
            "model": "model_1", "tools": ["GoogleSearchAction"],
        }]},
    })


def record():
    return {
        "prompt": "Question and visible Sub-Agent evidence ONLY.",
        "reference_action": {"private_teacher": "DO NOT SEND"},
        "decision_state": {
            "schema": SCHEMA, "scorable": True, "can_complete": False,
            "max_parallel_tasks": 4,
            "secret_oracle": "DO NOT SEND",
            "actionable": [{
                "task_id": "A", "operation": "retrieve_required_evidence",
                "kind": "new", "target_fields": ["project_id"],
                "source_ids": ["source_A"], "known_identifiers": [],
                "required_parent_fields": {}, "depends_on": [],
                "descendants": [], "tools": ["GoogleSearchAction"],
                "model": "model_1", "priority": 0.67,
                "evidence_units": ["A::project_id"],
            }],
        },
    }


def response(values=(4, 3, 2, 1)):
    return json.dumps({
        **dict(zip(DIMENSIONS, values)), "reason": "Supported next action.",
    })


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


class FakeJudge:
    def __init__(self, reward=0.25):
        self.reward = reward
        self.calls = []

    def score_many(self, prompt, candidates):
        self.calls.append((prompt, list(candidates)))
        return [{
            "reward": self.reward,
            "dimensions": {key: 2 for key in DIMENSIONS},
            "reason": "acceptable", "cached": False,
        } for _ in candidates]


class HybridRewardTests(unittest.TestCase):
    def test_hard_invalid_gate_and_mixed_scores(self):
        judge = FakeJudge(0.25)
        scores = HybridReward(judge, mode="hybrid", judge_weight=0.4).score_group(
            ["not JSON", action()], record(),
        )
        self.assertEqual(scores[0]["reward"], -1)
        self.assertFalse(scores[0]["valid"])
        self.assertEqual(scores[0]["judge_reward"], None)
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(judge.calls[0][1], [action()])
        self.assertEqual(judge.calls[0][0], record()["prompt"])
        self.assertAlmostEqual(scores[1]["env_reward"], 1.0)
        self.assertAlmostEqual(scores[1]["reward"], 0.7)
        self.assertEqual(scores[1]["judge_dimensions"]["evidence_grounding"], 2)

    def test_judge_only_retains_hard_gate(self):
        scores = HybridReward(FakeJudge(0.6), mode="judge_only").score_group(
            ["bad", action()], record(),
        )
        self.assertEqual([s["reward"] for s in scores], [-1.0, 0.6])

    def test_all_invalid_never_calls_api(self):
        judge = FakeJudge()
        scored = HybridReward(judge).score_group(["bad", "{}"], record())
        self.assertEqual([x["reward"] for x in scored], [-1, -1])
        self.assertEqual(judge.calls, [])

    def test_rubric_response_is_strict(self):
        r = parse_semantic_response(response())
        self.assertAlmostEqual(r["reward"], (4*.3 + 3*.3 + 2*.2 + 1*.2)/4)
        with self.assertRaises(ValueError):
            parse_semantic_response('{"evidence_grounding": 3}')
        with self.assertRaises(ValueError):
            parse_semantic_response(response(("4", 3, 2, 1)))

    def test_api_payload_excludes_private_state_and_caches(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ, {"TEST_JUDGE_KEY": "test-token"},
        ):
            path = str(Path(temp) / "judge.jsonl")
            judge = SemanticJudge(
                model="test-model", base_url="https://localhost/api/v1",
                api_key_env="TEST_JUDGE_KEY", workers=2, retries=0,
                cache_path=path,
            )
            seen = []
            def mock_post(url, *, headers, json, timeout):
                seen.append(json)
                self.assertEqual(headers["Authorization"], "Bearer test-token")
                return FakeResponse(response())
            with patch(
                "train_maestro_8b.policy.semantic_judge.requests.post",
                side_effect=mock_post,
            ):
                scored = HybridReward(judge).score_group(
                    ["invalid", action(), action()], record(),
                )
            self.assertEqual(len(seen), 1)  # duplicate valid Action cached/deduped
            data = seen[0]["messages"][1]["content"]
            self.assertIn(record()["prompt"], data)
            self.assertIn("source_A", data)
            self.assertNotIn("DO NOT SEND", data)
            self.assertEqual(scored[0]["reward"], -1)
            self.assertEqual(scored[1]["reward"], scored[2]["reward"])
            self.assertEqual(len(Path(path).read_text().splitlines()), 1)
            # Cache persists across training resumes.
            judge_again = SemanticJudge(
                model="test-model", base_url="https://localhost/api/v1",
                api_key_env="TEST_JUDGE_KEY", workers=2, retries=0,
                cache_path=path,
            )
            with patch(
                "train_maestro_8b.policy.semantic_judge.requests.post",
                side_effect=AssertionError("cache should prevent request"),
            ):
                cached = judge_again.score_many(record()["prompt"], [action()])
            self.assertTrue(cached[0]["cached"])

    def test_judge_failure_is_not_a_zero_reward(self):
        import requests
        with patch.dict(os.environ, {"TEST_JUDGE_KEY": "test-token"}):
            judge = SemanticJudge(
                model="test-model", base_url="https://localhost/v1",
                api_key_env="TEST_JUDGE_KEY", retries=0,
            )
            with patch(
                "train_maestro_8b.policy.semantic_judge.requests.post",
                side_effect=requests.Timeout("timed out"),
            ):
                with self.assertRaisesRegex(RuntimeError, "refusing to invent a reward"):
                    HybridReward(judge).score_group([action()], record())

    def test_cli_modes_and_custom_reward_are_mutually_exclusive(self):
        a = parse_args([
            "--model", "/tmp/qwen", "--risk_checkpoint", "/tmp/risk",
            "--train_file", "/tmp/train", "--output_dir", "/tmp/out",
            "--reward_mode", "hybrid",
        ])
        validate_args(a)
        self.assertEqual(a.judge_weight, 0.4)
        a.reward_callable = "some.path:reward"
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            validate_args(a)


if __name__ == "__main__":
    unittest.main()
