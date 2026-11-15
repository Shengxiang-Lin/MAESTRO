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
from train_maestro_8b.policy.backend import assess_probability_contract


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

    def test_numerical_mismatch_within_ppo_clip_is_valid(self):
        sampled = torch.tensor([-4.0, -2.0, -1.5])
        incremental = sampled.clone()
        # Single-token NF4 and batched NF4 kernels may give different
        # likelihoods. The old sampling distribution remains the denominator.
        batched = sampled + torch.tensor([0.124283, -0.044, 0.03])
        result = assess_probability_contract(
            sampled, incremental, batched, batched.clone(),
            rollout_engine="transformers", clip_epsilon=0.2,
            causal_future_sensitivity=0.0,
        )
        self.assertEqual(
            result["probability_contract"],
            "bounded_behavior_importance_ratio",
        )
        self.assertEqual(result["initial_importance_ratio_clipped_fraction"], 0.0)
        self.assertGreater(result["max_abs_sample_vs_batched_train"], 0.12)
        self.assertLess(result["initial_importance_ratio_max"], 1.2)
        self.assertGreater(result["initial_importance_ratio_min"], 0.8)

    def test_strict_gate_remains_available(self):
        sampled = torch.tensor([-4.0, -2.0, -1.5])
        batched = sampled + 0.124283
        with self.assertRaisesRegex(RuntimeError, "strict rollout/replay"):
            assess_probability_contract(
                sampled, sampled, batched, batched,
                rollout_engine="transformers", clip_epsilon=0.2,
                strict_max_drift=0.01,
            )

    def test_mismatch_exceeding_initial_ppo_clip_still_blocks(self):
        sampled = torch.tensor([-4.0, -2.0, -1.5])
        batched = sampled + 0.35
        with self.assertRaisesRegex(RuntimeError, "already exceeds the PPO clip"):
            assess_probability_contract(
                sampled, sampled, batched, batched,
                rollout_engine="transformers", clip_epsilon=0.2,
            )

    def test_cache_and_train_invariants_remain_strict(self):
        sampled = torch.tensor([-4.0, -2.0, -1.5])
        with self.assertRaisesRegex(RuntimeError, "local sampling"):
            assess_probability_contract(
                sampled, sampled + 0.03, sampled, sampled,
                rollout_engine="transformers", clip_epsilon=0.2,
            )
        with self.assertRaisesRegex(RuntimeError, "identical batched forwards"):
            assess_probability_contract(
                sampled, sampled, sampled, sampled + 0.03,
                rollout_engine="transformers", clip_epsilon=0.2,
            )
        with self.assertRaisesRegex(RuntimeError, "future-token influence"):
            assess_probability_contract(
                sampled, sampled, sampled, sampled,
                rollout_engine="transformers", clip_epsilon=0.2,
                causal_future_sensitivity=0.04,
            )

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
        self.assertEqual(a.num_generations, 8)
        self.assertEqual(a.gradient_accumulation_steps, 1)
        self.assertEqual(a.max_length, 12288)
        self.assertEqual(a.max_new_tokens, 4096)
        self.assertEqual(a.temperature, 1.0)
        self.assertEqual(a.epochs, 3)
        self.assertAlmostEqual(a.lr, 5e-5)
        self.assertAlmostEqual(a.weight_decay, 0.01)
        self.assertAlmostEqual(a.max_grad_norm, 1.0)
        self.assertEqual(a.optim, "adamw_8bit")
        self.assertEqual(a.lora_rank, 32)
        self.assertEqual(a.lora_alpha, 64)
        self.assertEqual(a.logging_steps, 1)
        self.assertEqual(a.save_steps, 50)
        self.assertAlmostEqual(a.vllm_memory_utilization, 0.70)
        self.assertEqual(a.zero_adv_strategy, "skip")
        self.assertEqual(a.sft_warmup, 0)
        self.assertEqual(a.sft_warmup_samples, 0)
        validate_args(a)
        self.assertIsNone(a.preflight_max_logprob_drift)
        strict = arguments("--preflight_max_logprob_drift", "0.01")
        validate_args(strict)
        self.assertAlmostEqual(strict.preflight_max_logprob_drift, 0.01)
        # Our custom Trainer owns an entire G-completion group per prompt.
        # One gradient accumulation step therefore means ONE prompt x G,
        # unlike stock DA-GRPO where G completion-microsteps are required.
        validate_args(arguments("--gradient_accumulation_steps", "2"))
        with self.assertRaises(ValueError):
            validate_args(arguments("--gradient_accumulation_steps", "0"))
        with self.assertRaisesRegex(ValueError, "16384"):
            validate_args(arguments("--max_length", "12289"))
        with self.assertRaises(ValueError):
            validate_args(arguments("--num_generations", "1"))
        with self.assertRaises(ValueError):
            validate_args(arguments("--no_4bit"))
        self.assertEqual(arguments("--backend", "hf").backend, "hf")


if __name__ == "__main__":
    unittest.main()
