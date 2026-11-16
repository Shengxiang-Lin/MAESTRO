"""CPU-only Stage1 belief supervision: no hidden truth, exact O1 spans,
per-scene split, causally informed per-report features and LoRA gradients.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from train_maestro_8b.data.export_belief import (
    CLASSES, UNKNOWN, annotate_beliefs, validate_belief, SCHEMA,
)


def snapshot():
    return {
        "main_can_act_from_round": 3,
        "claim_nodes": [
            {"claim_id": "cA", "report_task_id": "A"},
            {"claim_id": "cB", "report_task_id": "B"},
        ],
        "claim_states": {
            "project|year": {
                "status": "disputed", "supporting_claim_ids": ["cA", "cB"],
                "answer_supporting_subject_private": True,
            },
        },
        "transitions": [{
            "claim_key": "project|year", "before": "unverified",
            "after": "disputed", "next_main_decision_round": 3,
        }],
        "private_only": True,
    }


def raw_risk_row():
    return {
        "schema": "maestro.window_risk.v1",
        "sample_id": "S-fork-round-02-decision-03",
        "scene_id": "S", "round_sample_id": "S-fork-round-02",
        "decision_round": 3,
        "prompt": "The report A says 2021. The report B says 2023.",
        "reports": [
            {"report_id": "1:0:A", "start": 0, "end": 23,
             "risk": 1, "mask": 1, "label_reason": "visible_conflict"},
            {"report_id": "2:0:B", "start": 24, "end": 47,
             "risk": 1, "mask": 1, "label_reason": "visible_conflict"},
        ],
    }


class BeliefExportTests(unittest.TestCase):
    def test_conflict_and_current_round_update(self):
        row = annotate_beliefs(raw_risk_row(), snapshot())
        self.assertEqual(row["schema"], SCHEMA)
        self.assertEqual([r["belief_label"] for r in row["reports"]],
                         [CLASSES.index("disputed")] * 2)
        self.assertEqual([r["update_label"] for r in row["reports"]], [UNKNOWN, 1])
        self.assertNotIn("answer_supporting_subject_private", str(row))
        self.assertNotIn("gold_answer", str(row))

    def test_mismatched_round_and_ambiguous_status_are_not_silently_labeled(self):
        state = snapshot()
        state["main_can_act_from_round"] = 4
        with self.assertRaisesRegex(ValueError, "round"):
            annotate_beliefs(raw_risk_row(), state)
        state = snapshot()
        state["claim_states"]["other"] = {
            "status": "unverified", "supporting_claim_ids": ["cA"],
        }
        row = annotate_beliefs(raw_risk_row(), state)
        self.assertEqual(row["reports"][0]["belief_label"], UNKNOWN)

    def test_unknown_reports_masked_not_negative(self):
        row = raw_risk_row()
        row["reports"][0]["mask"] = 0
        row["reports"][0]["label_reason"] = "superseded"
        result = annotate_beliefs(row, snapshot())
        self.assertEqual(result["reports"][0]["belief_label"], UNKNOWN)
        self.assertEqual(result["reports"][1]["belief_label"], 2)

    def test_offset_validation_rejects_overlap(self):
        row = annotate_beliefs(raw_risk_row(), snapshot())
        row["reports"][1]["start"] = 10
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_belief(row)


class FakeTokenizer:
    is_fast = True
    pad_token_id = 0
    def apply_chat_template(self, messages, *, tokenize=False,
                            add_generation_prompt=False):
        assert not tokenize and add_generation_prompt
        return "<usr>" + messages[0]["content"] + "</usr><assistant>"
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]
    def __call__(self, text, *, add_special_tokens=False,
                 return_offsets_mapping=False, truncation=False):
        output = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            output["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return output


class BeliefCollationTests(unittest.TestCase):
    def test_offsets_shift_under_stage2_chat_template(self):
        import torch
        from train_maestro_8b.belief.train import (
            BeliefCollator, BeliefDataset, chat_wrapped_prompt,
        )
        row = annotate_beliefs(raw_risk_row(), snapshot())
        render, shift = chat_wrapped_prompt(FakeTokenizer(), row["prompt"])
        self.assertEqual(shift, 5)
        batch = BeliefCollator(FakeTokenizer(), 300)([row])
        self.assertEqual(int(batch["report_starts"][0, 0]), shift)
        self.assertEqual(int(batch["report_ends"][0, 1]), 47 + shift)
        self.assertEqual(batch["state_labels"].tolist(), [[2, 2]])
        with TemporaryDirectory() as root:
            file = Path(root) / "samples.jsonl"
            file.write_text(json.dumps(row) + "\n", encoding="utf-8")
            ds = BeliefDataset(file, FakeTokenizer(), max_length=300)
            self.assertEqual(len(ds), 1)
            self.assertEqual(ds.class_counts[2], 2)
            tiny = BeliefDataset(file, FakeTokenizer(), max_length=12)
            self.assertEqual(len(tiny), 0)
            self.assertEqual(tiny.overlong, 1)

    def test_head_propagates_gradients_to_global_and_earlier_reports(self):
        import torch
        from train_maestro_8b.belief.model import EvidenceBeliefHead, report_embeddings
        torch.manual_seed(7)
        hidden = torch.randn(1, 16, 24, requires_grad=True)
        attention = torch.ones((1, 16), dtype=torch.long)
        starts = torch.tensor([[1, 7]])
        ends = torch.tensor([[5, 11]])
        mask = torch.tensor([[True, True]])
        local, global_ = report_embeddings(hidden, attention, starts, ends, mask)
        head = EvidenceBeliefHead(24, head_size=16, dropout=0)
        out = head(local, global_, mask, state_labels=torch.tensor([[0, 2]]),
                   update_labels=torch.tensor([[UNKNOWN, 1]]))
        out["loss"].backward()
        self.assertGreater(hidden.grad[0, 2].abs().sum().item(), 0)
        self.assertGreater(hidden.grad[0, 15].abs().sum().item(), 0)
        self.assertGreater(head.state_classifier.weight.grad.abs().sum().item(), 0)


class TinyLoRABeliefTests(unittest.TestCase):
    def test_live_qwen3_lora_receives_auxiliary_head_gradient(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from peft import LoraConfig, TaskType, get_peft_model
        from train_maestro_8b.belief.model import EvidenceBeliefHead, report_embeddings
        torch.manual_seed(8)
        cfg = Qwen3Config(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
        )
        policy = get_peft_model(
            Qwen3ForCausalLM(cfg),
            LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
                       target_modules=["q_proj", "v_proj"]),
        )
        head = EvidenceBeliefHead(32, head_size=12, dropout=0)
        ids = torch.randint(2, 128, (1, 18))
        out = policy.get_base_model().model(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            use_cache=False, return_dict=True,
        )
        local, global_ = report_embeddings(
            out.last_hidden_state, torch.ones_like(ids),
            torch.tensor([[2, 7]]), torch.tensor([[5, 12]]),
            torch.tensor([[True, True]]),
        )
        loss = head(local, global_, torch.tensor([[True, True]]),
                    state_labels=torch.tensor([[1, 2]]),
                    update_labels=torch.tensor([[UNKNOWN, 1]]))["loss"]
        loss.backward()
        grads = [p.grad for name, p in policy.named_parameters()
                 if "lora_" in name and p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(any(g.abs().sum().item() > 0 for g in grads))
        self.assertTrue(all(p.grad is None for name, p in policy.named_parameters()
                            if "lora_" not in name))


if __name__ == "__main__":
    unittest.main()
