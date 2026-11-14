"""Unit tests for custom TRL-configured GRPO math and private-state isolation.

These tests do not need CUDA or the Unsloth runtime. Actual Unsloth Qwen3
DynamicCache replay is checked on a CUDA GPU via --preflight_only.
"""
import unittest
import torch

from train_maestro_8b.policy.trainer import (
    group_advantages, clipped_policy_loss, MainDecisionDataset,
    decision_collator,
)
from train_maestro_8b.policy.train import parse_args, validate_args


def arguments(*extra):
    return parse_args([
        "--model", "Qwen/Qwen3-8B", "--risk_checkpoint", "/tmp/risk",
        "--train_file", "/tmp/data.jsonl", "--output_dir", "/tmp/grpo",
        *extra,
    ])


class TRLRiskGRPOTests(unittest.TestCase):
    def test_group_relative_rewards_and_zero_variance(self):
        adv = group_advantages([-1.0, 0.1, 0.9, 0.9])
        self.assertAlmostEqual(float(adv.mean()), 0, places=6)
        self.assertAlmostEqual(float(adv.std(unbiased=False)), 1, places=6)
        self.assertTrue(torch.equal(group_advantages([-1.0]*4),
                                    torch.zeros(4)))
        with self.assertRaises(ValueError):
            group_advantages([0.5])

    def test_clipped_loss_has_only_new_logprob_gradients(self):
        old = torch.tensor([-1., -2.])
        new = old.clone().requires_grad_(True)
        loss = clipped_policy_loss(new, old, torch.tensor(1.5), 0.2)
        self.assertAlmostEqual(float(loss.detach()), -1.5)
        loss.backward()
        self.assertTrue(torch.isfinite(new.grad).all())
        self.assertTrue((new.grad < 0).all())
        overshoot = clipped_policy_loss(
            torch.tensor([-0.1, -0.1]), torch.tensor([-2., -2.]),
            torch.tensor(1.), 0.2,
        )
        self.assertAlmostEqual(float(overshoot), -1.2, places=5)
        with self.assertRaises(ValueError):
            clipped_policy_loss(new, old[:1], torch.tensor(1.), 0.2)

    def test_private_state_only_goes_to_collator_scorer(self):
        row = {
            "prompt": "Main sees Question and report only",
            "reports": [{"start": 16, "end": 22}],
            "decision_state": {
                "scorable": True,
                "actionable": [{"task_id": "private_01"}],
            },
            "reference_action": {"action": "complete"},
        }
        ds = MainDecisionDataset([row])
        batch = decision_collator([ds[0]])
        self.assertEqual(batch["records"], [row])
        self.assertNotIn("private_01", row["prompt"])
        with self.assertRaises(ValueError):
            MainDecisionDataset([{**row, "decision_state": {}}])

    def test_default_unsloth_trl_and_safe_gradient_accum(self):
        a = arguments()
        self.assertEqual(a.backend, "unsloth")
        self.assertTrue(a.load_in_4bit)
        self.assertEqual(a.num_generations, 4)
        self.assertEqual(a.gradient_accumulation_steps, 4)
        validate_args(a)
        with self.assertRaises(ValueError):
            validate_args(arguments("--gradient_accumulation_steps", "2"))
        with self.assertRaises(ValueError):
            validate_args(arguments("--num_generations", "1"))
        with self.assertRaises(ValueError):
            validate_args(arguments("--no_4bit"))
        self.assertEqual(arguments("--backend", "hf").backend, "hf")


if __name__ == "__main__":
    unittest.main()
