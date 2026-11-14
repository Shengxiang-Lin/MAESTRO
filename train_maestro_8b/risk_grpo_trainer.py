"""TRL TrainingArguments/Trainer loop for MAESTRO's nonstandard GRPO policy.

Uses the TRL GRPOConfig to configure scheduling/logging/checkpointing, and
HF Trainer/Accelerate for optimizer lifecycle. It deliberately overrides the
step because stock TRL GRPOTrainer cannot replay MAESTRO's frozen-prefix,
Risk-Head-steered KV cache as the SAME distribution in rollout and loss.

One sample/prompt per microbatch; per prompt G sequential completions.
Each completion's gradient is accumulated *immediately* to avoid holding
G suffix graphs simultaneously in GPU memory. No vLLM, no policy/ref KL,
no DDP, and no invented environment rollout.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch


def group_advantages(rewards):
    values = torch.as_tensor(rewards, dtype=torch.float32)
    if values.ndim != 1 or values.numel() < 2 or not torch.isfinite(values).all():
        raise ValueError("GRPO reward group must contain at least two finite scores")
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-4)


def clipped_policy_loss(new_logps, sampled_logps, advantage, epsilon):
    """Clipped policy gradient for a single generated Action sequence.

    The sampled log probabilities include the same sampling temperature as
    the new replay distribution. No KL or artificial teacher likelihood.
    """
    if new_logps.shape != sampled_logps.shape or not new_logps.numel():
        raise ValueError("completion logprob lengths differ")
    if not 0 < epsilon < 1:
        raise ValueError("invalid PPO clip epsilon")
    ratio = (new_logps - sampled_logps.detach()).clamp(-20, 20).exp()
    return -torch.minimum(
        ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage,
    ).mean()


class MainDecisionDataset(torch.utils.data.Dataset):
    """Keep private reward state exclusively on the scorer side."""

    def __init__(self, records):
        self.records = list(records)
        if not self.records:
            raise ValueError("empty Stage2 training dataset")
        for row in self.records:
            if not isinstance(row.get("prompt"), str) or not isinstance(row.get("reports"), list):
                raise ValueError("missing Stage2 prompt or historical report spans")
            if not isinstance(row.get("decision_state"), dict):
                raise ValueError(
                    "Stage2 requires private deterministic decision_state; "
                    "re-export with current Stage4 scorer"
                )
            if row["decision_state"].get("scorable") is not True:
                raise ValueError("unscorable private state was not filtered")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def decision_collator(rows):
    """Do not convert private textual fields into model input tensors."""
    return {"records": rows}


def make_trainer_class():
    """Lazy import so reward tests and old HF path work without Unsloth/TRL."""
    from transformers import Trainer

    class RiskSteeredGRPOTrainer(Trainer):
        def __init__(self, *, engine, reward_fn, num_generations,
                     temperature, max_new_tokens, clip_epsilon,
                     debug_path=None, **kwargs):
            self.engine = engine
            self.reward_fn = reward_fn
            self.num_generations = num_generations
            self.temperature = temperature
            self.max_new_tokens = max_new_tokens
            self.clip_epsilon = clip_epsilon
            self.debug_path = Path(debug_path) if debug_path else None
            self.seen_groups = 0
            self.zero_adv_groups = 0
            self.invalid_total = 0
            self.generated_total = 0
            self.last_components = {}
            if self.debug_path:
                self.debug_path.parent.mkdir(parents=True, exist_ok=True)
            super().__init__(
                model=engine.policy, data_collator=decision_collator,
                processing_class=engine.tokenizer, **kwargs,
            )
            if self.accelerator.num_processes != 1:
                raise ValueError(
                    "RiskKV reference rollout currently supports ONE GPU "
                    "only; use one process and gradient accumulation"
                )

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            raise NotImplementedError(
                "RiskKV uses streaming per-completion backwards in training_step "
                "to avoid holding all rollout suffix graphs in GPU memory"
            )

        def training_step(self, model, inputs, num_items_in_batch=None):
            if len(inputs["records"]) != 1:
                raise ValueError("RiskKV currently uses one prompt per microbatch")
            # In the supported single-GPU execution model, Trainer and engine
            # share this actual policy module. Never send a separate model
            # to scoring: that would silently compute the wrong logprobs.
            if model is not self.engine.policy:
                raise RuntimeError(
                    "RiskAwarePolicy and TRL Trainer were wrapped separately; "
                    "cannot guarantee rollout/gradient equivalence"
                )
            model.eval()
            row = inputs["records"][0]
            prefix = self.engine.prefill(row["prompt"], row["reports"])
            actions = []
            rewards = []
            metadata = []
            from environment.untrusted_orchestration.orchestration_reward import score_decision
            with torch.no_grad():
                for _ in range(self.num_generations):
                    ids, sampled = self.engine.sample(
                        prefix, max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                    )
                    candidate = self.engine.decode(ids)
                    if self.reward_fn is None:
                        info = score_decision(candidate, row["decision_state"])
                        reward = info["reward"]
                    else:
                        info = None
                        reward = float(self.reward_fn(candidate, row))
                    if not torch.isfinite(torch.tensor(reward)):
                        raise ValueError("nonfinite deterministic GRPO reward")
                    actions.append((ids, sampled))
                    rewards.append(float(reward))
                    metadata.append((candidate, info))
                    if self.debug_path:
                        with self.debug_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps({
                                "sample_id": row.get("sample_id"),
                                "step": int(self.state.global_step),
                                "candidate": candidate,
                                "reward": reward, "components": info,
                            }, ensure_ascii=False) + "\n")
            self.seen_groups += 1
            self.generated_total += len(actions)
            self.invalid_total += sum(info is not None and not info["valid"]
                                      for _, info in metadata)
            advantages = group_advantages(rewards).to(self.engine.device)
            model.train()
            if float(advantages.abs().max()) < 1e-6:
                self.zero_adv_groups += 1
                return torch.tensor(0.0, device=self.engine.device)
            group_losses = []
            for i, (ids, old_logps) in enumerate(actions):
                # The Accelerate .backward() method automatically accounts
                # for Trainer's gradient_accumulation_steps. Divide here
                # only by the number of completions in this GRPO group.
                with self.compute_loss_context_manager():
                    new_logps = self.engine.log_probs(
                        prefix, ids, temperature=self.temperature
                    )
                    loss = clipped_policy_loss(
                        new_logps, old_logps, advantages[i], self.clip_epsilon
                    ) / self.num_generations
                self.accelerator.backward(loss)
                group_losses.append(loss.detach())
            self.last_components = {
                "reward": sum(rewards) / len(rewards),
                "loss": float(torch.stack(group_losses).sum()),
                "invalid_fraction": sum(
                    info is not None and not info["valid"] for _, info in metadata
                ) / len(actions),
                "zero_adv_groups": self.zero_adv_groups,
            }
            for metric in ("coverage", "priority", "parallel", "nonredundancy"):
                valid_infos = [info for _, info in metadata if info is not None]
                if valid_infos:
                    self.last_components[metric] = sum(
                        x[metric] for x in valid_infos
                    ) / len(valid_infos)
            if self.seen_groups % max(1, self.args.logging_steps) == 0:
                self.log({f"risk_grpo/{k}": v
                          for k, v in self.last_components.items()})
            return torch.stack(group_losses).sum().detach()

    return RiskSteeredGRPOTrainer
