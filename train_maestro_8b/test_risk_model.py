"""Small CPU tests for one-pass window risk and backbone LoRA gradient flow."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from train_maestro_8b.risk_model import WindowRiskModel


class FakeTransformer(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.emb = nn.Embedding(32, dim)
        self.proj = nn.Linear(dim, dim)
        self.forwards = 0

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        self.forwards += 1
        # Each token has a locally encoded feature; global state is final token.
        hidden = self.proj(self.emb(input_ids))
        return SimpleNamespace(last_hidden_state=hidden)


class FakePolicy(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.model = FakeTransformer(dim)

    def get_base_model(self):
        return self


class RiskModelTests(unittest.TestCase):
    def test_one_forward_multiple_windows_and_backbone_gradient(self):
        torch.manual_seed(7)
        policy = FakePolicy()
        model = WindowRiskModel(policy, hidden_size=8, top_k=2)
        batch = dict(
            input_ids=torch.tensor([[1, 2, 3, 4, 5, 0],
                                    [6, 7, 8, 9, 0, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 1, 0],
                                         [1, 1, 1, 1, 0, 0]]),
            window_starts=torch.tensor([[0, 2, 3], [0, 1, 0]]),
            window_ends=torch.tensor([[2, 4, 5], [2, 4, 0]]),
            window_owner=torch.tensor([[0, 0, 1], [0, 0, -1]]),
            report_labels=torch.tensor([[1., 0.], [0., 0.]]),
            report_mask=torch.tensor([[1., 1.], [1., 0.]]),
        )
        result = model(**batch)
        self.assertEqual(policy.model.forwards, 1)
        self.assertEqual(tuple(result["report_logits"].shape), (2, 2))
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        self.assertIsNotNone(policy.model.emb.weight.grad)
        self.assertGreater(policy.model.emb.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.risk_head[-1].weight.grad.abs().sum().item(), 0)

    def test_all_masked_refused(self):
        model = WindowRiskModel(FakePolicy(), hidden_size=8)
        with self.assertRaises(ValueError):
            model(
                input_ids=torch.tensor([[1, 2]]),
                attention_mask=torch.tensor([[1, 1]]),
                window_starts=torch.tensor([[0]]),
                window_ends=torch.tensor([[2]]),
                window_owner=torch.tensor([[0]]),
                report_labels=torch.tensor([[0.]]),
                report_mask=torch.tensor([[0.]]),
            )


if __name__ == "__main__":
    unittest.main()
