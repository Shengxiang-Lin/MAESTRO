"""CPU-only contract tests for the single-GPU O1 GRPO trainer."""
from __future__ import annotations

import json
from types import ModuleType
from tempfile import TemporaryDirectory

import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch, Mock
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

    def test_batched_two_prompt_generation_preserves_group_boundaries(self):
        from types import SimpleNamespace

        class Tokenizer:
            def decode(self, ids, **kwargs):
                return "".join(chr(x) for x in ids)

        fake = Mock()
        sample = lambda: SimpleNamespace(
            token_ids=[123, 125, 32],
            logprobs=[{123: SimpleNamespace(logprob=-0.1)},
                      {125: SimpleNamespace(logprob=-0.2)},
                      {32: SimpleNamespace(logprob=-0.3)}],
        )
        fake.generate.return_value = [
            SimpleNamespace(outputs=[sample() for _ in range(8)]),
            SimpleNamespace(outputs=[sample() for _ in range(8)]),
        ]
        vllm = ModuleType("vllm")
        vllm.SamplingParams = Mock()
        lora = ModuleType("vllm.lora")
        lora_request = ModuleType("vllm.lora.request")
        lora_request.LoRARequest = Mock()
        with TemporaryDirectory() as tmp, patch.dict(
            sys.modules, {
                "vllm": vllm,
                "vllm.lora": lora,
                "vllm.lora.request": lora_request,
            },
        ):
            engine = rollout.ColocatedRollout(
                "Qwen3-8B", Tokenizer(), {}, Path(tmp) / "sync",
            )
            engine.llm = fake
            prompts = [[11, 12], [21, 22]]
            groups = engine.sample_batch(
                Mock(), prompts, n=8, max_tokens=32,
                temperature=1.0, top_p=1.0,
            )
        self.assertEqual(len(groups), 2)
        self.assertEqual([len(x) for x in groups], [8, 8])
        self.assertEqual(groups[0][0]["prompt_input_ids"], prompts[0])
        self.assertEqual(groups[1][0]["prompt_input_ids"], prompts[1])
        self.assertEqual(groups[0][0]["action_input_ids"], [123, 125])
        self.assertEqual(groups[0][0]["old_log_probs"], [-0.1, -0.2])
        self.assertTrue(groups[0][0]["json_complete"])
        self.assertEqual(fake.generate.call_count, 1)
        self.assertEqual(fake.generate.call_args.args[0], [
            {"prompt_token_ids": prompts[0]},
            {"prompt_token_ids": prompts[1]},
        ])
        self.assertEqual(fake.sleep.call_count, 1)

    def test_legacy_config_still_uses_triton_eager_and_sleep_mode(self):
        raw = trainer.yaml.safe_load(trainer.DEFAULT_CONFIG.read_text())
        raw["rollout"].pop("compilation_config")
        raw["rollout"].pop("prompts_per_batch")
        raw["rollout"]["max_num_seqs"] = 8
        raw["rollout"]["enforce_eager"] = True
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "legacy.yaml"
            config.write_text(trainer.yaml.safe_dump(raw), encoding="utf-8")
            cfg = trainer.load_config(config)
        self.assertEqual(cfg["rollout"]["prompts_per_batch"], 1)
        vllm_cfg = {**cfg["rollout"], "lora_rank": cfg["model"]["lora_rank"]}
        module = ModuleType("vllm")
        module.LLM = Mock()
        with TemporaryDirectory() as tmp, patch.dict(sys.modules, {"vllm": module}):
            engine = rollout.ColocatedRollout(
                "local-Qwen3-8B", None, vllm_cfg, Path(tmp) / "adapter",
            )
            engine.initialize()
            kwargs = module.LLM.call_args.kwargs
            self.assertEqual(kwargs["attention_backend"], "TRITON_ATTN")
            self.assertTrue(kwargs["enforce_eager"])
            self.assertNotIn("compilation_config", kwargs)
            self.assertEqual(kwargs["gpu_memory_utilization"], 0.35)
            self.assertEqual(kwargs["kv_cache_dtype"], "fp8")
            self.assertTrue(kwargs["enable_sleep_mode"])
            self.assertEqual(kwargs["max_lora_rank"], 32)

    def test_default_decode_cuda_graph_config_reaches_vllm(self):
        with patch.object(trainer, "show_config") as show_config:
            self.assertEqual(trainer.main(["--dry-run"]), 0)
            cfg = show_config.call_args.args[0]
            self.assertEqual(trainer.main(["--dry-run", "--decode-cuda-graph"]), 0)
            self.assertEqual(show_config.call_args.args[0]["rollout"], cfg["rollout"])
        vllm_cfg = {**cfg["rollout"], "lora_rank": cfg["model"]["lora_rank"]}
        module = ModuleType("vllm")
        module.LLM = Mock()
        with TemporaryDirectory() as tmp, patch.dict(sys.modules, {"vllm": module}):
            engine = rollout.ColocatedRollout("Qwen3-8B", None, vllm_cfg, Path(tmp))
            engine.initialize()
        kwargs = module.LLM.call_args.kwargs
        self.assertEqual(cfg["rollout"]["prompts_per_batch"], 4)
        self.assertEqual(kwargs["max_num_seqs"], 32)
        self.assertFalse(kwargs["enforce_eager"])
        self.assertEqual(kwargs["compilation_config"], {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
        })
        self.assertEqual(kwargs["attention_backend"], "TRITON_ATTN")
        self.assertTrue(kwargs["enable_sleep_mode"])

    def test_optional_two_prompt_mode_does_not_change_default_config(self):
        import io
        from contextlib import redirect_stdout

        with redirect_stdout(io.StringIO()):
            self.assertEqual(trainer.main(["--dry-run"]), 0)
        # The experimental mode is opt-in and can use a separate output dir.
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(trainer.main([
                "--dry-run", "--rollout-prompts", "2",
                "--max-num-seqs", "16",
                "--output-dir", "baseline/da-grpo/output_batch2_smoke",
            ]), 0)
        displayed = json.loads(out.getvalue())
        self.assertEqual(displayed["prompts_per_batch"], 2)
        self.assertEqual(displayed["max_num_seqs"], 16)
        self.assertTrue(displayed["output_dir"].endswith("output_batch2_smoke"))

    def test_four_prompt_mode_enables_32_sequence_graph_capture(self):
        with patch.object(trainer, "show_config") as show_config:
            self.assertEqual(trainer.main([
                "--dry-run", "--rollout-prompts", "4", "--max-num-seqs", "32",
                "--decode-cuda-graph",
            ]), 0)
        cfg = show_config.call_args.args[0]["rollout"]
        self.assertEqual(cfg["prompts_per_batch"], 4)
        self.assertEqual(cfg["max_num_seqs"], 32)
        self.assertEqual(cfg["compilation_config"]["cudagraph_capture_sizes"],
                         [1, 2, 4, 8, 16, 32])

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
    def test_two_groups_keep_distinct_judge_targets(self):
        judge = ModuleType("reward_fn")
        judge.compute_score = Mock(
            side_effect=lambda ds, text, answer, info: {
                "score": 1.0 if text == answer else 0.0
            },
        )
        groups = [[{"text": "X"}, {"text": "Y"}],
                  [{"text": "Y"}, {"text": "Y"}]]
        rows = [
            {"data_source": "a", "reward_model": {"ground_truth": "X"},
             "extra_info": {"sample_id": "p1"}},
            {"data_source": "b", "reward_model": {"ground_truth": "Y"},
             "extra_info": {"sample_id": "p2"}},
        ]
        with patch.dict(sys.modules, {"reward_fn": judge}):
            scores = trainer.score_groups(groups, rows, 2)
        self.assertEqual([[x["score"] for x in group] for group in scores],
                         [[1.0, 0.0], [1.0, 1.0]])
        self.assertAlmostEqual(sum(losses.group_advantages([1.0, 0.0])), 0.0)
        self.assertEqual(losses.group_advantages([1.0, 1.0]), [0.0, 0.0])

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
    def test_prompt_ids_accepts_hf_tensor_and_batch_encoding(self):
        class FakeTokenizer:
            def __init__(self, result):
                self.result = result

            def apply_chat_template(self, messages, **kwargs):
                self._messages = messages
                self._kwargs = kwargs
                return self.result

        messages = [{"role": "user", "content": "question"}]
        cases = (
            [100, 101, 102],
            torch.tensor([100, 101, 102], dtype=torch.long),
            torch.tensor([[100, 101, 102]], dtype=torch.long),
            {"input_ids": [[100, 101, 102]], "attention_mask": [[1, 1, 1]]},
            {"input_ids": torch.tensor([[100, 101, 102]])},
        )
        for encoded in cases:
            with self.subTest(type=type(encoded).__name__):
                tokenizer = FakeTokenizer(encoded)
                self.assertEqual(policy.prompt_ids(tokenizer, messages, 3), [100, 101, 102])
                self.assertTrue(tokenizer._kwargs["add_generation_prompt"])
                with self.assertRaisesRegex(ValueError, "exceeding 2"):
                    policy.prompt_ids(tokenizer, messages, 2)

    def test_prompt_ids_rejects_missing_or_batched_ids(self):
        class FakeTokenizer:
            def __init__(self, ids):
                self.ids = ids

            def apply_chat_template(self, *args, **kwargs):
                return self.ids

        for bad in ({"attention_mask": [1, 1]}, [[1, 2], [3, 4]], "text", []):
            with self.subTest(value=repr(bad)):
                with self.assertRaises(ValueError):
                    policy.prompt_ids(FakeTokenizer(bad), [{"role": "user", "content": "x"}], 9)

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
