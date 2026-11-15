"""Cold-start and output auditing must not leak private scorer state."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from environment.untrusted_orchestration.decision_reward_state import SCHEMA
from train_maestro_8b.data.audit_rollouts import summarize_rollouts
from train_maestro_8b.policy.rewards import score_training_candidate
from train_maestro_8b.policy.train import (
    select_warmup_records, parse_args, validate_args,
)


def row(index, *, valid=True):
    state = {
        "schema": SCHEMA,
        "scorable": True,
        "can_complete": bool(valid),
        "max_parallel_tasks": 4,
        "actionable": [],
    }
    return {
        "sample_id": f"sample-{index}",
        "prompt": f"previous agent history for sample {index}",
        "reports": [],
        "decision_state": state,
        "reference_action": {
            "action": "complete", "params": {"answer": "verified answer"}
        },
    }


class RiskColdStartTests(unittest.TestCase):
    def test_selects_only_legal_reference_actions_from_full_dataset(self):
        all_records = [row(i, valid=i < 5) for i in range(9)]
        subset = select_warmup_records(all_records, seed=42, limit=3)
        self.assertEqual(len(subset), 3)
        self.assertTrue(all(x["decision_state"]["can_complete"] for x in subset))
        self.assertEqual(
            [x["sample_id"] for x in subset],
            [x["sample_id"] for x in select_warmup_records(
                all_records, seed=42, limit=3,
            )],
        )
        self.assertTrue(any(x["sample_id"] != "sample-0" for x in subset))

    def test_never_trains_on_invalid_reference_action(self):
        with self.assertRaisesRegex(ValueError, "No legal reference Actions"):
            select_warmup_records([row(0, valid=False)], seed=42, limit=1)

    def test_parse_compatible_optional_sft_config(self):
        args = parse_args([
            "--model", "base", "--risk_checkpoint", "/tmp/risk",
            "--train_file", "/tmp/train.jsonl", "--output_dir", "/tmp/output",
            "--sft_warmup", "1", "--sft_warmup_samples", "32",
            "--fail_on_zero_adv",
        ])
        validate_args(args)
        self.assertEqual(args.sft_warmup, 1)
        self.assertEqual(args.sft_warmup_samples, 32)
        self.assertTrue(args.fail_on_zero_adv)
        self.assertEqual(args.grpo_reward_mode, "curriculum")
        self.assertEqual(args.zero_adv_strategy, "reference_sft")
        self.assertAlmostEqual(args.zero_adv_sft_weight, 0.25)
        args.sft_warmup_samples = -1
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_curriculum_reward_distinguishes_realistic_invalid_json(self):
        state = row(0, valid=True)["decision_state"]
        samples = [
            "Now let's think. We should delegate and then consider what to do.",
            'Here is my analysis and then {"action":"complete","params":',
            '{"action":"complete","params":',
            '{"action":"complete","params":{"answer":"A"}}\nExtra analysis',
        ]
        outcomes = [
            score_training_candidate(text, state, mode="curriculum")
            for text in samples
        ]
        self.assertTrue(all(not x["valid"] for x in outcomes))
        self.assertTrue(all(x["reason"] == "invalid_json" for x in outcomes))
        self.assertEqual(outcomes[0]["training_reward"], 0.0)
        self.assertGreater(
            outcomes[2]["training_reward"], outcomes[1]["training_reward"]
        )
        self.assertGreater(
            outcomes[3]["training_reward"], outcomes[2]["training_reward"]
        )
        self.assertTrue(all(0 <= x["training_reward"] < 0.35 for x in outcomes))
        self.assertTrue(all(x["strict_reward"] == -1 for x in outcomes))
        self.assertTrue(all(not x["json_valid"] for x in outcomes))

    def test_strict_evaluator_is_not_changed_by_training_shaping(self):
        state = row(0, valid=True)["decision_state"]
        legal = '{"action":"complete","params":{"answer":"verified"}}'
        result = score_training_candidate(legal, state, mode="curriculum")
        self.assertTrue(result["valid"])
        self.assertEqual(result["training_reward"], 1.0)
        premature = score_training_candidate(
            legal, row(0, valid=False)["decision_state"], mode="curriculum",
        )
        self.assertFalse(premature["valid"])
        self.assertEqual(premature["reason"], "premature_completion")
        self.assertEqual(premature["training_reward"], 0.35)
        baseline = score_training_candidate(
            "natural language output", state, mode="strict",
        )
        self.assertEqual(baseline["training_reward"], -1.0)
        with self.assertRaises(ValueError):
            score_training_candidate(legal, state, mode="unknown")

    def test_audit_reports_reward_failures_and_completion_cap(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "rewards.jsonl"
            examples = [
                {
                    "candidate": "plain text without JSON",
                    "completion_tokens": 1536,
                    "reward": -1.0,
                    "components": {"valid": False, "reason": "invalid_json"},
                    "sample_id": "a", "step": 0,
                },
                {
                    "candidate": '{"action":"complete","params":{"answer":"A"}}',
                    "completion_tokens": 20,
                    "reward": 1.0,
                    "components": {"valid": True, "reason": None},
                    "sample_id": "a", "step": 0,
                },
            ]
            with path.open("w", encoding="utf-8") as f:
                for e in examples:
                    f.write(json.dumps(e) + "\n")
            result = summarize_rollouts(
                path, tail=2, token_cap=1536, preview_chars=12,
            )
            self.assertEqual(result["inspected_candidates"], 2)
            self.assertEqual(result["unique_rewards"], 2)
            self.assertEqual(result["reason_distribution"]["invalid_json"], 1)
            self.assertEqual(result["reason_distribution"]["valid"], 1)
            self.assertTrue(result["candidates"][0]["hit_token_cap"])
            self.assertFalse(result["candidates"][1]["hit_token_cap"])
            self.assertEqual(result["candidates"][0]["candidate_start"], "plain text w")
            self.assertNotIn("decision_state", str(result))


if __name__ == "__main__":
    unittest.main()
