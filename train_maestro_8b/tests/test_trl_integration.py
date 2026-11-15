"""One real tiny-Qwen3 CPU Trainer step using TRL GRPOConfig.

This checks that a full HF Trainer optimizer/update cycle actually works
with our streaming KV-steered policy. CUDA-only Unsloth kernels are
separately exercised via --preflight_only on the training machine.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "<eos>"
    padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        return [3 + (ord(x) % 80) for x in text]

    def __call__(self, text, *, add_special_tokens=False,
                 return_offsets_mapping=False, truncation=False):
        ids = self.encode(text)
        d = {"input_ids": ids}
        if return_offsets_mapping:
            d["offset_mapping"] = [(i,i+1) for i in range(len(text))]
        return d

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(x) - 3 + 32) for x in ids)

    def save_pretrained(self, folder):
        return (str(folder),)


class TRLTrainerTinyQwen3Tests(unittest.TestCase):
    def test_complete_trl_trainer_optimizer_step_with_frozen_cache(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from peft import LoraConfig, TaskType, get_peft_model
        from trl import GRPOConfig

        from train_maestro_8b.risk.model import WindowRiskHead
        from train_maestro_8b.policy.model import RiskAwarePolicy
        from train_maestro_8b.policy.trainer import (
            MainDecisionDataset, make_trainer_class,
        )

        torch.manual_seed(37)
        config = Qwen3Config(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8,
            max_position_embeddings=256,
        )
        model = get_peft_model(
            Qwen3ForCausalLM(config),
            LoraConfig(task_type=TaskType.CAUSAL_LM, r=4,
                       lora_alpha=8, lora_dropout=0.0,
                       target_modules=["q_proj", "v_proj"]),
        )
        head = WindowRiskHead(32, head_size=12, top_k=2)
        head.eval().requires_grad_(False)
        engine = RiskAwarePolicy(
            model, DummyTokenizer(), head,
            {"window_size":4, "window_stride":3},
            max_length=64, last_n_layers=1, anchor_text="\n>",
            strength=0.2, threshold=0.5,
        )
        row = {
            "sample_id": "toy-0001",
            "prompt": "abcde fghij",
            "reports": [{"start":1,"end":6}],
            "decision_state": {"scorable":True, "actionable":[]},
        }
        count = [0]
        def deterministic_sample(prefix, *, temperature, **kwargs):
            # Two distinct rollout actions, scored by exactly the same
            # underlying policy logprob distribution as gradient replay.
            token = 11 if count[0] % 2 == 0 else 13
            count[0] += 1
            ids = torch.tensor([token], dtype=torch.long)
            old = engine.log_probs(prefix, ids, temperature=temperature)
            return ids, old.detach()

        engine.sample = deterministic_sample
        first = engine.decode(torch.tensor([11]))
        def reward(candidate, record):
            return 1.0 if candidate == first else 0.0

        before = {
            n: p.detach().clone() for n, p in model.named_parameters()
            if "lora_" in n and p.requires_grad
        }
        with TemporaryDirectory() as tmp:
            args = GRPOConfig(
                output_dir=tmp, max_steps=1,
                num_generations=2, per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                generation_batch_size=2,
                max_completion_length=4,
                learning_rate=1e-3, beta=0.0,
                loss_type="grpo", use_vllm=False,
                bf16=False, fp16=False,
                logging_steps=50, save_strategy="no",
                remove_unused_columns=False, report_to="none",
                disable_tqdm=True,
            )
            trainer = make_trainer_class()(
                engine=engine, reward_fn=reward,
                num_generations=2, temperature=1.0,
                max_new_tokens=2, clip_epsilon=0.2,
                args=args, train_dataset=MainDecisionDataset([row,row]),
            )
            output = trainer.train()
            self.assertEqual(trainer.state.global_step, 1)
            self.assertEqual(trainer.seen_groups, 1)
            self.assertEqual(trainer.generated_total, 2)
            self.assertGreater(trainer.completion_tokens_total, 0)
            self.assertEqual(trainer.prefill_tokens_total, len(row["prompt"]))
            self.assertTrue(any(
                not torch.equal(p.detach(), before[n])
                for n, p in model.named_parameters() if n in before
            ), "Trainer did not update any LoRA weight")
            self.assertTrue(all(p.grad is None for p in head.parameters()))
            self.assertTrue(all(
                p.grad is None for n, p in model.named_parameters()
                if "lora_" not in n
            ))


if __name__ == "__main__":
    unittest.main()
