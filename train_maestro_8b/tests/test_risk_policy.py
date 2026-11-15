"""Tiny real-Qwen3 CPU integration tests for cache-steered LoRA decoding.

No model download. Test the *same policy* for generation and logprobs.
"""
import unittest
from types import SimpleNamespace
import torch

from train_maestro_8b.policy.cache import (
    clone_cache, steer_value_cache, token_risk_from_windows,
)
from train_maestro_8b.risk.model import WindowRiskHead
from train_maestro_8b.policy.backend import _probe_nf4_projection_shape


class CacheSteeringTests(unittest.TestCase):
    def test_only_selected_layer_value_and_report_positions_change(self):
        torch.manual_seed(2)
        layer = SimpleNamespace(
            keys=torch.randn(1, 2, 5, 8),
            values=torch.ones(1, 2, 5, 8),
        )
        original = SimpleNamespace(layers=[
            SimpleNamespace(keys=layer.keys.clone(), values=layer.values.clone()),
            layer,
        ])
        before = clone_cache(original)
        risk = torch.tensor([[0.0, 0.9, 0.9, 0.0, 0.0]])
        steer_value_cache(original, risk, last_n_layers=1,
                          strength=0.2, threshold=0.5)
        self.assertTrue(torch.equal(original.layers[0].values,
                                    before.layers[0].values))
        self.assertTrue(torch.equal(original.layers[1].keys,
                                    before.layers[1].keys))
        self.assertTrue(torch.equal(original.layers[1].values[:, :, 0],
                                    before.layers[1].values[:, :, 0]))
        self.assertGreater(original.layers[1].values[0,0,1,0].item(), 1.0)
        self.assertTrue(torch.equal(original.layers[1].values[:, :, 3:],
                                    before.layers[1].values[:, :, 3:]))

    def test_overlapping_windows_and_non_report_tokens(self):
        probs = torch.tensor([0.2,0.8])
        risk = token_risk_from_windows(
            probs, torch.tensor([1,3]), torch.tensor([4,6]),
            torch.tensor([0,0]), 8
        )
        self.assertEqual(risk[0].item(), 0)
        self.assertAlmostEqual(risk[3].item(), 0.5)
        self.assertEqual(risk[7].item(), 0)
        with self.assertRaises(TypeError):
            steer_value_cache(SimpleNamespace(layers=[]), risk[None])


class NF4ShapeProbeTests(unittest.TestCase):
    def test_linear_quantization_shape_probe_detects_mocked_gemm_bias(self):
        from types import SimpleNamespace
        class Fake4Bit(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(4))
                self.weight.quant_state = object()

            def forward(self, x):
                # Simulate an NF4 path returning slightly different numbers
                # at M=1 versus M=3 without changing the input vector.
                return torch.nn.functional.linear(x, self.weight) + (
                    0.125 if x.shape[1] >= 3 else 0.0
                )

        class FakePolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Module()])
                self.layers[0].self_attn = torch.nn.Module()
                self.layers[0].self_attn.q_proj = torch.nn.Module()
                self.layers[0].self_attn.q_proj.base_layer = Fake4Bit()

        class FakeEngine:
            def __init__(self):
                self.policy = FakePolicy()
                self.device = torch.device("cpu")
                self.anchor_ids = [8]

            def _forward_suffix(self, _state, suffix):
                x = torch.ones(1, suffix.shape[1], 4)
                return self.policy.layers[0].self_attn.q_proj.base_layer(x)

        results = _probe_nf4_projection_shape(
            FakeEngine(), None, torch.tensor([11, 12, 13])
        )
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["max_abs"], 0.125)
        self.assertAlmostEqual(results[0]["mean_abs"], 0.125)
        self.assertIn("q_proj", results[0]["module"])


class DummyTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [3 + (ord(c) % 80) for c in text]

    def __call__(self, text, *, add_special_tokens=False,
                 return_offsets_mapping=False, truncation=False):
        ids = self.encode(text)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i,i+1) for i in range(len(text))]
        return result

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr((int(i)-3) % 80 + 32) for i in ids)


class TinyModelTests(unittest.TestCase):
    def test_qwen3_cache_replay_and_only_lora_gradients(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from peft import LoraConfig, TaskType, get_peft_model
        from train_maestro_8b.policy.model import RiskAwarePolicy

        torch.manual_seed(13)
        cfg = Qwen3Config(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8,
            max_position_embeddings=256,
        )
        base = Qwen3ForCausalLM(cfg)
        lora = get_peft_model(base, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=4,
            lora_alpha=8, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        ))
        lora.eval()
        head = WindowRiskHead(32, head_size=12, top_k=2)
        with torch.no_grad():
            head.risk_head[-1].weight.zero_()
            head.risk_head[-1].bias.fill_(3)
        engine = RiskAwarePolicy(
            lora, DummyTokenizer(), head,
            {"window_size": 4, "window_stride": 3},
            max_length=64, last_n_layers=1,
            strength=0.2, threshold=0.5,
            anchor_text="\n>",
        )
        prompt = "abcde fghij"
        prefix = engine.prefill(
            prompt, [{"start": 1, "end": 6}, {"start": 7, "end": 11}]
        )
        self.assertEqual(prefix.length, len(prompt))
        self.assertGreater(prefix.risk.max().item(), 0.5)
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        for p in lora.parameters():
            p.grad = None
        ids, old_logps = engine.sample(
            prefix, max_new_tokens=3, temperature=0.9,
            eos_token_id=-1,
        )
        incremental = engine.log_probs_incremental(prefix, ids, temperature=0.9)
        new_logps = engine.log_probs(prefix, ids, temperature=0.9)
        self.assertEqual(new_logps.shape, old_logps.shape)
        self.assertTrue(torch.allclose(incremental, old_logps, atol=1e-5),
                        f"forced incremental replay differs from sample: "
                        f"{old_logps} vs {incremental}")
        self.assertTrue(torch.allclose(new_logps.detach(), incremental, atol=1e-4),
                        f"incremental/batched replay drift: "
                        f"{incremental} vs {new_logps}")
        # Change the SECOND suffix input, not the gathered target tokens.
        # Earlier logits must remain stable under true causal attention.
        forced = torch.cat([
            torch.tensor(engine.anchor_ids, dtype=torch.long, device=ids.device),
            ids[:-1],
        ])[None]
        changed_suffix = forced.clone()
        changed_suffix[0, -1] = (changed_suffix[0, -1] + 1) % cfg.vocab_size
        with torch.no_grad():
            changed_out = engine._forward_suffix(prefix, changed_suffix)
            changed_logps = changed_out.logits[:, -ids.numel():].float().div(
                0.9
            ).log_softmax(dim=-1).gather(
                2, ids.view(1, -1, 1)
            ).squeeze(0).squeeze(-1)
        torch.testing.assert_close(
            new_logps.detach()[:2], changed_logps[:2],
            atol=1e-4, rtol=1e-4,
        )
        # Full preflight exercises autoregressive, SDPA batched and Eager
        # batched evaluation on an identical 3-token sequence.
        from train_maestro_8b.policy.backend import check_risk_cache_contract
        metrics = check_risk_cache_contract(
            engine, {"prompt": prompt, "reports": [
                {"start": 1, "end": 6}, {"start": 7, "end": 11},
            ]}, temperature=0.9, max_logprob_drift=1e-3,
        )
        self.assertLess(metrics["max_abs_sample_vs_incremental"], 1e-4)
        self.assertLess(metrics["max_abs_batched_causal_future_sensitivity"], 1e-4)
        self.assertIsNotNone(metrics["max_abs_batched_eager_vs_sdpa"])
        (-new_logps.mean()).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item() > 0
                            for n, p in lora.named_parameters() if "lora_" in n))
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        self.assertTrue(all(p.grad is None for n,p in lora.named_parameters()
                            if "lora_" not in n))
        self.assertEqual(
            prefix.cache.layers[0].keys.shape[-2], len(prompt),
            "sampling mutated shared frozen prefix cache"
        )


if __name__ == "__main__":
    unittest.main()
