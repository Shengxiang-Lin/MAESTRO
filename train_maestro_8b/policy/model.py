"""Single frozen 8B prefill -> risk-steered cache -> LoRA-only action decoding.

The same policy code is used for sampling and for teacher-forced log-probs.
Prefix KV and Risk Head are detached; only the action suffix has LoRA grads.
This initial backend deliberately supports ONE unpadded prompt at a time.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from train_maestro_8b.policy.cache import clone_cache, steer_value_cache, token_risk_from_windows
from train_maestro_8b.risk.model import WindowRiskHead, extract_window_features
from train_maestro_8b.data.contracts import span_to_tokens, windows


@dataclass
class RiskPrefix:
    cache: Any
    length: int
    risk: torch.Tensor
    window_probs: torch.Tensor
    token_ids: list[int] | None = None
    # Keep no completed generation's mutable cache in this shared object.


def prompt_windows(tokenizer, prompt: str, reports: list[dict],
                   *, window_size: int, window_stride: int,
                   max_length: int):
    """Use the same nontruncating token/character alignment as Stage 1."""
    encoding = tokenizer(
        prompt, add_special_tokens=False, return_offsets_mapping=True,
        truncation=False,
    )
    ids = encoding["input_ids"]
    if not ids or len(ids) > max_length:
        raise ValueError(f"prompt has {len(ids)} tokens, exceeds {max_length}")
    ranges = []
    owners = []
    for rid, report in enumerate(reports):
        start = report["start"]
        end = report["end"]
        left, right = span_to_tokens(encoding["offset_mapping"], start, end)
        for x, y in windows(left, right, window_size, window_stride):
            ranges.append((x, y))
            owners.append(rid)
    return ids, ranges, owners


class RiskAwarePolicy:
    """Frozen prefix features and persistent Risk Head, with LoRA suffix.

    Adapter must implement .disable_adapter() to guarantee frozen prefix
    uses *original* base weights. Full prefix never enters autograd graph.
    """

    def __init__(self, policy, tokenizer, risk_head: WindowRiskHead,
                 risk_config: dict, *, max_length=16384,
                 last_n_layers=8, strength=0.10, threshold=0.50,
                 anchor_text="\n"):
        self.policy = policy
        self.tokenizer = tokenizer
        self.risk_head = risk_head
        self.config = risk_config
        self.rollout_attention = None
        self.rollout_engine = None
        self.max_length = max_length
        self.last_n_layers = last_n_layers
        self.strength = strength
        self.threshold = threshold
        self.anchor_ids = tokenizer.encode(anchor_text, add_special_tokens=False)
        if not self.anchor_ids:
            raise ValueError("decision anchor must tokenize to at least one token")
        if not hasattr(policy, "disable_adapter"):
            raise TypeError("Stage 2 policy requires PEFT.disable_adapter")
        self.risk_head.requires_grad_(False).eval()

    @property
    def device(self):
        return next(self.policy.parameters()).device

    @torch.no_grad()
    def prefill(self, prompt, reports):
        """ONE frozen full prompt pass; use final H and exact prefix cache."""
        from transformers import DynamicCache
        ids, ranges, owners = prompt_windows(
            self.tokenizer, prompt, reports,
            window_size=self.config["window_size"],
            window_stride=self.config["window_stride"],
            max_length=self.max_length,
        )
        if not ranges:
            # No completed feedback in the first Main round: plain prefix.
            owners = []
        device = self.device
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        cache = DynamicCache(config=self.policy.get_base_model().config)
        # Explicit no-grad, not torch.inference_mode: suffix autograd is legal.
        with self.policy.disable_adapter():
            backbone = self.policy.get_base_model().model
            output = backbone(
                input_ids=input_ids, attention_mask=attention_mask,
                past_key_values=cache, use_cache=True, return_dict=True,
            )
        cache = output.past_key_values
        if not ranges:
            risk = torch.zeros(len(ids), device=device, dtype=torch.float32)
            probs = torch.empty(0, device=device, dtype=torch.float32)
        else:
            start = torch.tensor([[x for x, _ in ranges]], device=device)
            end = torch.tensor([[y for _, y in ranges]], device=device)
            owner = torch.tensor([owners], device=device)
            w, g = extract_window_features(
                output.last_hidden_state, attention_mask, start, end, owner
            )
            head_device = next(self.risk_head.parameters()).device
            if head_device != w.device:
                raise ValueError("risk head and policy must share device")
            scores = self.risk_head(w, g, owner)["window_logits"][0]
            probs = scores.sigmoid()
            risk = token_risk_from_windows(
                probs, start[0], end[0], owner[0], len(ids),
            )
        steered = steer_value_cache(
            cache, risk[None], last_n_layers=self.last_n_layers,
            strength=self.strength, threshold=self.threshold,
        )
        # No reference to output H is kept in RiskPrefix.
        return RiskPrefix(steered, len(ids), risk.detach(), probs.detach(), ids)

    def _forward_suffix(self, prefix: RiskPrefix, suffix_ids: torch.Tensor):
        if suffix_ids.ndim != 2 or suffix_ids.shape[0] != 1:
            raise ValueError("one unpadded completion per suffix forward")
        cache = clone_cache(prefix.cache)
        input_len = suffix_ids.shape[1]
        mask = torch.ones((1, prefix.length + input_len),
                          dtype=torch.long, device=suffix_ids.device)
        out = self.policy(
            input_ids=suffix_ids, attention_mask=mask,
            past_key_values=cache, use_cache=True, return_dict=True,
            cache_position=torch.arange(
                prefix.length, prefix.length + input_len,
                device=suffix_ids.device
            ),
        )
        return out

    def log_probs(self, prefix: RiskPrefix, completion_ids: torch.Tensor,
                  temperature: float = 1.0):
        """Log-prob for each sampled token; identical anchor and KV semantics."""
        if temperature <= 0 or completion_ids.ndim != 1 or not completion_ids.numel():
            raise ValueError("invalid completion or temperature")
        anchor = torch.tensor(self.anchor_ids, dtype=torch.long, device=self.device)
        suffix = torch.cat([anchor, completion_ids[:-1]], dim=0)[None]
        output = self._forward_suffix(prefix, suffix)
        positions = output.logits[:, -completion_ids.numel():, :].float()
        return positions.div(temperature).log_softmax(dim=-1).gather(
            2, completion_ids.view(1, -1, 1)
        ).squeeze(0).squeeze(-1)

    def sample_group(self, prefix, *, count, max_new_tokens, temperature, eos_token_id=None):
        if self.rollout_engine is not None:
            eos = self.tokenizer.eos_token_id if eos_token_id is None else eos_token_id
            return self.rollout_engine.sample(
                self.policy, prefix, self.anchor_ids, count=count,
                max_new_tokens=max_new_tokens, temperature=temperature, eos_token_id=eos,
            )
        kwargs = {"max_new_tokens": max_new_tokens, "temperature": temperature}
        if eos_token_id is not None:
            kwargs["eos_token_id"] = eos_token_id
        return [self.sample(prefix, **kwargs) for _ in range(count)]

    @torch.no_grad()
    def sample(self, prefix: RiskPrefix, *, max_new_tokens=256,
               temperature=1.0, eos_token_id=None):
        """Autoregressive LoRA suffix with the selected inference attention."""
        config = self.policy.config
        original = config._attn_implementation if self.rollout_attention else None
        if self.rollout_attention:
            config._attn_implementation = self.rollout_attention
        try:
            return self._sample(prefix, max_new_tokens, temperature, eos_token_id)
        finally:
            if self.rollout_attention:
                config._attn_implementation = original

    def _next_step_logits(self, prefix: RiskPrefix, cache, input_ids, generated: int):
        """One exact autoregressive decoding step for sampling and preflight.

        `generated` counts completed tokens. Keeping this one shared path
        isolates token-by-token decoding from batched teacher-forced replay.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or generated < 0:
            raise ValueError("expected one token sequence and valid generated count")
        consumed = 0 if generated == 0 else len(self.anchor_ids) + generated - 1
        n = input_ids.shape[1]
        mask = torch.ones((1, prefix.length + consumed + n),
                          dtype=torch.long, device=self.device)
        end = prefix.length + len(self.anchor_ids) + generated
        if end - n != prefix.length + consumed:
            raise ValueError("autoregressive RiskKV position mismatch")
        out = self.policy(
            input_ids=input_ids, attention_mask=mask,
            past_key_values=cache, use_cache=True, return_dict=True,
            cache_position=torch.arange(end - n, end, device=self.device),
        )
        return out.logits[:, -1, :].float()

    @torch.no_grad()
    def log_probs_incremental(self, prefix: RiskPrefix,
                              completion_ids: torch.Tensor,
                              temperature: float = 1.0):
        """Diagnostic replay: force the same tokens through the sample path.

        Never used for optimizer backward. This tells whether a discrepancy
        originates in the autoregressive/model engine or the different-shape
        teacher-forced forward used by GRPO.
        """
        if (temperature <= 0 or completion_ids.ndim != 1
                or not completion_ids.numel()):
            raise ValueError("invalid diagnostic tokens or temperature")
        cache = clone_cache(prefix.cache)
        inputs = torch.tensor([self.anchor_ids], device=self.device, dtype=torch.long)
        values = []
        for i, target in enumerate(completion_ids):
            logits = self._next_step_logits(prefix, cache, inputs, generated=i)
            distribution = logits.div(temperature).log_softmax(dim=-1)
            token_id = int(target.item())
            values.append(distribution[0, token_id])
            inputs = target.view(1, 1).to(device=self.device, dtype=torch.long)
        return torch.stack(values)

    def _sample(self, prefix, max_new_tokens, temperature, eos_token_id):
        if max_new_tokens < 1 or temperature <= 0:
            raise ValueError("invalid generation settings")
        eos = self.tokenizer.eos_token_id if eos_token_id is None else eos_token_id
        cache = clone_cache(prefix.cache)
        ids = torch.tensor([self.anchor_ids], device=self.device, dtype=torch.long)
        seq = []
        old_logps = []
        for _ in range(max_new_tokens):
            logits = self._next_step_logits(prefix, cache, ids, generated=len(seq))
            distribution = logits.div(temperature).log_softmax(dim=-1)
            next_id = torch.multinomial(distribution.exp(), 1)
            seq.append(next_id.item())
            old_logps.append(distribution.gather(-1, next_id).squeeze().detach())
            if seq[-1] == eos:
                break
            ids = next_id
        return (torch.tensor(seq, dtype=torch.long, device=self.device),
                torch.stack(old_logps))

    def decode(self, ids):
        return self.tokenizer.decode(ids.detach().cpu().tolist(), skip_special_tokens=True)
