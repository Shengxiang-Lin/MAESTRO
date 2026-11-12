"""CPU-only contract tests for the single-GPU O1 GRPO trainer."""
from __future__ import annotations

import json

import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load(name: str):
    spec = importlib.util.spec_from_file_location("da_grpo_" + name, HERE / (name + ".py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer = load("train")
rollout = load("rollout")
losses = load("losses")
policy = load("policy")

import torch
import torch.nn.functional as F


class TestConfig(unittest.TestCase):
    def test_one_config_only_and_shared_judge(self):
        cfg = trainer.load_config()
        self.assertEqual(cfg["model"]["lora_rank"], 32)
        self.assertEqual(cfg["rollout"]["group_size"], 8)
        self.assertEqual(cfg["training"]["micro_batch_size"], 1)
        self.assertEqual(cfg["training"]["kl_coef"], 0.01)
        self.assertEqual(cfg["judge"]["model"], "qwen3.7-flash")
        self.assertEqual(
            cfg["judge"]["base_url"],
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.assertEqual(cfg["rollout"]["max_prompt_tokens"], 24576)
        self.assertEqual(cfg["rollout"]["max_response_tokens"], 4096)

    def test_no_provider_key_is_stored_in_yaml(self):
        cfg = trainer.load_config()
        self.assertNotIn("sk-", cfg["_source"].read_text(encoding="utf-8"))
        self.assertNotIn("OPENAI_API_KEY", cfg["_source"].read_text(encoding="utf-8"))
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "unit-test-secret"}):
            self.assertEqual(trainer.key_from_tty("DASHSCOPE_API_KEY"), "unit-test-secret")

    def test_only_single_gpu_is_valid(self):
        cfg = trainer.load_config()
        with patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.is_bf16_supported", return_value=True), \
             patch("torch.cuda.get_device_properties") as properties:
            properties.return_value.total_memory = 96 * (1024 ** 3)
            checks = trainer.local_checks(cfg)["checks"]
        self.assertTrue(checks["one_cuda_gpu"])
        self.assertTrue(checks["gpu_memory"])


class TestGRPO(unittest.TestCase):
    def test_group_advantages_zero_mean_and_constant(self):
        a = losses.group_advantages([0.0, 0.5, 1.0])
        self.assertAlmostEqual(sum(a), 0.0, places=5)
        self.assertEqual(losses.group_advantages([0.4] * 8), [0.0] * 8)

    def test_streamed_gradient_is_equivalent_to_grouped_loss(self):
        a = torch.tensor([-0.4, 0.6, -0.2], requires_grad=True)
        old = torch.tensor([-0.3, 0.5, -0.1])
        b = torch.tensor([-0.6, -0.1], requires_grad=True)
        old_b = torch.tensor([-0.5, -0.2])
        denom = 5
        la = losses.clipped_grpo_loss(
            a, old, 1.0, clip_range=0.2, normalization_tokens=denom,
        )
        lb = losses.clipped_grpo_loss(
            b, old_b, -1.0, clip_range=0.2, normalization_tokens=denom,
        )
        (la + lb).backward()
        simultaneous = (a.grad.clone(), b.grad.clone())
        a.grad = b.grad = None
        la = losses.clipped_grpo_loss(
            a, old, 1.0, clip_range=0.2, normalization_tokens=denom,
        )
        lb = losses.clipped_grpo_loss(
            b, old_b, -1.0, clip_range=0.2, normalization_tokens=denom,
        )
        la.backward()
        lb.backward()
        torch.testing.assert_close(a.grad, simultaneous[0])
        torch.testing.assert_close(b.grad, simultaneous[1])

    def test_low_var_kl_nonnegative(self):
        old = torch.tensor([-1.0, -1.0])
        self.assertAlmostEqual(
            float(losses.low_var_kl(old, old, normalization_tokens=2)), 0.0
        )
        new = torch.tensor([-1.4, -0.6])
        self.assertGreaterEqual(
            float(losses.low_var_kl(new, old, normalization_tokens=2)), 0.0
        )


class ToyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyDecoder()
        self.lm_head = torch.nn.Linear(6, 15, bias=False)


class ToyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(15, 6)

    def forward(self, *, input_ids, use_cache, return_dict):
        return type("Output", (), {"last_hidden_state": self.embed(input_ids)})


class ToyPEFT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ToyBase()

    def get_base_model(self):
        return self.backbone


class TestActionLogprob(unittest.TestCase):
    def test_shift_and_lm_head_chunked_projection(self):
        """The action logprob must predict token t from hidden state t-1."""
        model = ToyPEFT()
        with patch("torch.tensor", wraps=torch.tensor):
            # CPU fixture: replace hardcoded CUDA device in the module for test.
            with patch.object(policy.torch, "tensor", side_effect=(
                lambda data, **kwargs: torch.as_tensor(
                    data, dtype=kwargs.get("dtype", torch.long),
                )
            )):
                got = policy.action_log_probs(
                    model, [1, 2], [3, 4], logits_chunk_size=1,
                )
        full_ids = torch.tensor([[1, 2, 3, 4]])
        hidden = model.backbone.model(
            input_ids=full_ids, use_cache=False, return_dict=True,
        ).last_hidden_state
        expected = F.log_softmax(
            model.backbone.lm_head(hidden[0, 1:3]), dim=-1,
        ).gather(-1, torch.tensor([[3], [4]])).squeeze(-1)
        torch.testing.assert_close(got, expected)
        got.sum().backward()
        self.assertIsNotNone(model.backbone.lm_head.weight.grad)


class TestRolloutTokens(unittest.TestCase):
    def test_exact_sampled_logprob(self):
        self.assertAlmostEqual(
            rollout.sampled_logprob({15: -1.25}, 15), -1.25
        )
        with self.assertRaises(ValueError):
            rollout.sampled_logprob({10: -1.0}, 15)

    def test_json_prefix_ignores_string_braces(self):
        class FakeTokenizer:
            def decode(self, token_ids, **kwargs):
                return "".join(chr(x) for x in token_ids)
        # Build genuinely valid escaped JSON instead of a hand-escaped
        # Python literal. The old fixture contained two backslashes before
        # an unescaped quote, so the JSON root never legally closed.
        value = "} escaped " + chr(92) + chr(34) + " ok"
        payload = {"action": "complete", "params": {"answer": value}}
        text = "noise " + json.dumps(payload) + " trailing"
        end = rollout.json_prefix_end(FakeTokenizer(), list(map(ord, text)))
        self.assertIsNotNone(end)
        self.assertEqual(json.loads(text[text.index("{"):end]), payload)
        self.assertFalse(text[:end].endswith("trailing"))
        self.assertIsNone(
            rollout.json_prefix_end(
                FakeTokenizer(),
                list(map(ord, '{"action":"complete","params":{"answer":"unfinished"')),
            )
        )


if __name__ == "__main__":
    unittest.main()
