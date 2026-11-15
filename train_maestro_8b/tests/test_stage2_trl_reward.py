"""CPU-only contract tests for native TRL Stage2 hybrid reward and state join."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from train_maestro_8b.policy.trl_reward_adapter import (
    load_private_states, make_stage2_reward_function, normalize_action,
    private_sample_id, prompt_lookup, select_scorable_rows,
)
from train_maestro_8b.tests.test_hybrid_reward import action, record, FakeJudge


SAMPLE = "scene_A-fork-round-02-decision-round-03"
VISIBLE = "O1 Question: verify project status from visible reports."
PRIVATE_SECRET = "PRIVATE_ORACLE_NEVER_SEND"


def state():
    item = record()["decision_state"]
    item["decision_round"] = 3
    item["private_secret"] = PRIVATE_SECRET
    return item


def row(sample_id=SAMPLE):
    return {
        "prompt": [{"role": "user", "content": VISIBLE}],
        "data_source": "maestro_stage4",
        "ground_truth": "secret gold answer",
        "extra_info_json": json.dumps({
            "sample_id": sample_id, "has_expert": True,
            "expert_params": {"secret": PRIVATE_SECRET},
        }),
    }


def entry(sample_id="scene_A-fork-round-02", scorable=True):
    s = state()
    s["scorable"] = scorable
    return {
        "task_id": "scene_A", "round_sample_id": sample_id,
        "decision_round": 3, "state": s, "private_only": True,
    }


class Stage2StateJoinTests(unittest.TestCase):
    def test_stable_id_reconstructs_exporter_key(self):
        self.assertEqual(private_sample_id(entry()), SAMPLE)
        self.assertEqual(private_sample_id(entry("scene_A-fork-round-04")),
                         "scene_A-fork-round-04-decision-round-03")

    def test_exact_alignment_and_filter_without_private_leak(self):
        states = {
            SAMPLE: state(),
            "scene_A-fork-round-04-decision-round-03": {
                **state(), "scorable": False,
            },
        }
        rows = [
            row(SAMPLE),
            row("scene_A-fork-round-04-decision-round-03"),
            row("scene_A-fork-round-05-decision-round-03"),
        ]
        selected, report = select_scorable_rows(rows, states)
        self.assertEqual(len(selected), 1)
        self.assertEqual(report["excluded"], {
            "unscorable_private_state": 1, "missing_private_state": 1,
        })
        self.assertNotIn(PRIVATE_SECRET, str(selected[0]["prompt"]))
        self.assertEqual(prompt_lookup(selected), {SAMPLE: VISIBLE})

    def test_fail_closed_on_unmatched_rows(self):
        with self.assertRaisesRegex(ValueError, "No aligned scorable"):
            select_scorable_rows([row()], {})
        with self.assertRaisesRegex(ValueError, "Duplicate Stage2"):
            select_scorable_rows([row(), row()], {SAMPLE: state()})

    def test_loading_private_snapshot_checks_schema_and_round(self):
        with TemporaryDirectory() as root:
            p = Path(root) / "states.jsonl"
            p.write_text(json.dumps(entry()) + "\n", encoding="utf-8")
            loaded = load_private_states(p)
            self.assertEqual(list(loaded), [SAMPLE])
            bad = entry()
            bad["state"]["decision_round"] = 4
            p.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatched decision round"):
                load_private_states(p)
            p.write_text(json.dumps(entry()) + "\n" + json.dumps(entry()) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate private"):
                load_private_states(p)


class Stage2JSONNormalizationTests(unittest.TestCase):
    def test_qwen_think_and_code_fences_keep_exactly_one_action(self):
        valid = action()
        for text in (
            valid,
            "<think>Analyze source conflicts, then delegate.</think>\n" + valid,
            "<think>Think about \\n braces { in text.</think>\n```json\n" + valid + "\n```",
        ):
            parsed, transformed = normalize_action(text)
            self.assertEqual(json.loads(parsed), json.loads(valid))
        self.assertFalse(normalize_action(valid)[1])
        self.assertTrue(normalize_action("<think>Reasoning</think>\n" + valid)[1])

    def test_multiple_json_and_unfinished_thinking_are_never_rescued(self):
        sample = action()
        invalid = [
            "<think>unfinished" + sample,
            sample + "\n" + sample,
            "Answer: " + sample,
            "<think>Good</think>\n" + sample + "\nAnother action",
        ]
        for text in invalid:
            with self.subTest(raw=text[:30]):
                result, stripped = normalize_action(text)
                self.assertFalse(stripped)
                with self.assertRaises(json.JSONDecodeError):
                    json.loads(result)


class Stage2TRLRewardTests(unittest.TestCase):
    def setUp(self):
        self.states = {SAMPLE: state()}
        self.prompts = {SAMPLE: VISIBLE}

    def score(self, candidate_actions, mode="hybrid", judge=None, generations=2, path=None):
        fn = make_stage2_reward_function(
            private_states=self.states, prompts=self.prompts,
            mode=mode, judge=judge, num_generations=generations,
            diagnostics_path=path,
        )
        n = len(candidate_actions)
        return fn(
            completions=[[{"role": "assistant", "content": t}] for t in candidate_actions],
            data_source=["maestro_stage4"] * n,
            ground_truth=["secret gold answer"] * n,
            extra_info_json=[row()["extra_info_json"]] * n,
        )

    def test_hard_invalid_gate_and_weighted_judge(self):
        fake = FakeJudge(0.25)
        scores = self.score([
            "this is not JSON",
            "<think>Find status</think>\n" + action(),
        ], judge=fake)
        self.assertEqual(scores, [-1, 0.7])
        self.assertEqual(fake.calls, [(VISIBLE, [json.dumps(
            json.loads(action()), ensure_ascii=False, separators=(",", ":"),
        )])])
        self.assertNotIn(PRIVATE_SECRET, fake.calls[0][0])
        self.assertNotIn("secret gold answer", str(fake.calls))

    def test_deterministic_mode_never_invokes_external_judge(self):
        rewards = self.score(["bad", action()], mode="deterministic")
        self.assertEqual(rewards, [-1, 1.0])

    def test_judge_only_still_local_gate(self):
        rewards = self.score(["bad", action()], mode="judge_only", judge=FakeJudge(.6))
        self.assertEqual(rewards, [-1, .6])

    def test_all_invalid_skips_judge_and_records_reasons(self):
        fake = FakeJudge()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "reward_diagnostics.jsonl"
            scores = self.score(["bad", "{}"], judge=fake, path=path)
            self.assertEqual(scores, [-1, -1])
            self.assertEqual(fake.calls, [])
            logs = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(logs), 2)
            self.assertEqual(logs[0]["reason"], "invalid_json")
            self.assertEqual(logs[1]["reason"], "invalid_action_type")
            self.assertIsNone(logs[0]["judge_reward"])
            self.assertNotIn(PRIVATE_SECRET, path.read_text())
            self.assertNotIn("secret gold answer", path.read_text())

    def test_reward_order_and_group_identity_are_enforced(self):
        fake = FakeJudge(0.25)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "rewards.jsonl"
            rewards = self.score(
                ["bad", action(), action(), "invalid"], judge=fake,
                generations=2, path=path,
            )
            self.assertEqual(rewards, [-1, .7, .7, -1])
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r["group_index"] for r in records], [0, 0, 1, 1])
            self.assertEqual([r["candidate_index"] for r in records], [0, 1, 0, 1])
        fn = make_stage2_reward_function(
            private_states=self.states, prompts=self.prompts,
            mode="deterministic", num_generations=2,
        )
        with self.assertRaisesRegex(ValueError, "Mixed Stage4"):
            fn(
                completions=[action(), action()],
                data_source=["maestro_stage4"] * 2,
                ground_truth=["a"] * 2,
                extra_info_json=[
                    row()["extra_info_json"],
                    row("other-id")["extra_info_json"],
                ],
            )

    def test_judge_error_bubbles_up_instead_of_fake_reward(self):
        class FailingJudge:
            def score_many(self, prompt, candidates):
                raise RuntimeError("provider unavailable")
        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            self.score([action(), "bad"], judge=FailingJudge())


if __name__ == "__main__":
    unittest.main()
