"""Qwen causal-LM LoRA backbone + disposable latent window-risk MIL head."""
from __future__ import annotations
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class WindowRiskModel(nn.Module):
    """A single full-history backbone pass; report-wise top-k window pooling.

    This only learns OBSERVABLE risk. The head must not receive hidden Gold,
    fault IDs, private origin roots, or future reports.
    """
    def __init__(self, policy: nn.Module, hidden_size: int, *,
                 head_size: int = 256, top_k: int = 2):
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.policy = policy
        self.top_k = top_k
        self.risk_head = nn.Sequential(
            nn.LayerNorm(3 * hidden_size),
            nn.Linear(3 * hidden_size, head_size),
            nn.GELU(),
            nn.Linear(head_size, 1),
        )

    def forward(self, input_ids, attention_mask, window_starts, window_ends,
                window_owner, report_labels, report_mask):
        base = self.policy.get_base_model()
        backbone = getattr(base, "model", None)
        if backbone is None:
            raise ValueError("expected a Qwen-style CausalLM with a .model backbone")
        # Calling the transformer directly avoids the large vocab LM logits.
        hidden = backbone(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False, return_dict=True).last_hidden_state
        batch = hidden.shape[0]
        last = attention_mask.long().sum(dim=1) - 1
        global_context = hidden[torch.arange(batch, device=hidden.device), last]
        logits = hidden.new_zeros(report_labels.shape, dtype=torch.float32)
        for b in range(batch):
            valid = (window_owner[b] >= 0).nonzero(as_tuple=True)[0]
            if valid.numel() == 0:
                continue
            local = torch.stack([
                hidden[b, int(window_starts[b, j]):int(window_ends[b, j])].mean(dim=0)
                for j in valid.tolist()
            ], dim=0)
            global_ = global_context[b].unsqueeze(0).expand_as(local)
            features = torch.cat([local, global_, local * global_], dim=-1)
            window_logits = self.risk_head(features).float().view(-1)
            for rid in torch.unique(window_owner[b, valid]).tolist():
                group = window_logits[window_owner[b, valid] == rid]
                logits[b, rid] = group.topk(min(group.numel(), self.top_k)).values.mean()
        if not bool(report_mask.any()):
            raise ValueError("no supervised report in batch")
        bce = F.binary_cross_entropy_with_logits(
            logits.float(), report_labels.float(), reduction="none"
        )
        # Normalize within each history so long histories do not dominate.
        weights = report_mask.float()
        per_history = (bce * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        active = (weights.sum(dim=1) > 0).float()
        loss = (per_history * active).sum() / active.sum().clamp_min(1)
        return {"loss": loss, "report_logits": logits, "n_labels": weights.sum()}

    def save_pretrained(self, path, tokenizer, metadata):
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        # Native PEFT adapter: usable by TRL/GRPO with the original CausalLM.
        self.policy.save_pretrained(str(out))
        tokenizer.save_pretrained(str(out))
        torch.save(self.risk_head.state_dict(), out / "window_risk_head.pt")
        (out / "window_risk_config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
