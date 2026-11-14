"""Stage1 unit tests: only Risk Head gets gradients; frozen features reused."""
import unittest
import torch

from train_maestro_8b.risk_model import WindowRiskHead, extract_window_features


class RiskHeadTests(unittest.TestCase):
    def test_risk_head_gradient_and_report_pooling(self):
        torch.manual_seed(7)
        hidden = torch.randn(2, 8, 16, requires_grad=False)
        attention = torch.tensor([
            [1,1,1,1,1,1,0,0], [1,1,1,1,0,0,0,0],
        ])
        starts = torch.tensor([[0,2,4], [0,2,0]])
        ends = torch.tensor([[2,4,6], [2,4,0]])
        owners = torch.tensor([[0,0,1], [0,1,-1]])
        window, global_ = extract_window_features(
            hidden, attention, starts, ends, owners
        )
        self.assertEqual(tuple(window.shape), (2,3,16))
        self.assertEqual(tuple(global_.shape), (2,16))
        head = WindowRiskHead(16, head_size=12, top_k=2)
        output = head(
            window, global_, owners,
            report_labels=torch.tensor([[1.,0.], [0.,1.]]),
            report_mask=torch.tensor([[1.,1.], [1.,0.]]),
        )
        self.assertEqual(tuple(output["report_logits"].shape), (2,2))
        output["loss"].backward()
        self.assertIsNone(hidden.grad)
        self.assertGreater(head.risk_head[-1].weight.grad.abs().sum().item(), 0)

    def test_inference_without_report_labels(self):
        head = WindowRiskHead(4)
        result = head(
            torch.randn(1,2,4), torch.randn(1,4),
            torch.tensor([[0,1]]),
        )
        self.assertEqual(tuple(result["window_logits"].shape), (1,2))
        self.assertEqual(tuple(result["report_logits"].shape), (1,2))
        self.assertNotIn("loss", result)

    def test_masked_labels_rejected(self):
        head = WindowRiskHead(4)
        with self.assertRaises(ValueError):
            head(torch.randn(1,1,4), torch.randn(1,4),
                 torch.tensor([[0]]),
                 report_labels=torch.zeros(1,1),
                 report_mask=torch.zeros(1,1))


if __name__ == "__main__":
    unittest.main()
