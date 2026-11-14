"""Frozen-backbone features + trainable window-level weak-supervision head.

Risk is an OBSERVABLE warning, not probability of private fault/correctness.
The head is retained, frozen, and reused by the Stage-2 policy.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def extract_window_features(hidden, attention_mask, window_starts, window_ends,
                            window_owner):
    """Pool actual report windows from ONE full-history forward.

    Returns [B,M,D] and [B,D]. Padding windows are zero. No report boundaries
    cross task feedback blocks. The global token can see the full history.
    """
    bsz, _, dim = hidden.shape
    windows = hidden.new_zeros((bsz, window_owner.shape[1], dim))
    for b in range(bsz):
        for j in (window_owner[b] >= 0).nonzero(as_tuple=True)[0].tolist():
            start = int(window_starts[b, j])
            end = int(window_ends[b, j])
            if not 0 <= start < end <= int(attention_mask[b].sum()):
                raise ValueError("invalid window bounds or padding overlap")
            windows[b, j] = hidden[b, start:end].mean(dim=0)
    last = attention_mask.long().sum(dim=1) - 1
    if (last < 0).any():
        raise ValueError("empty context")
    global_ = hidden[torch.arange(bsz, device=hidden.device), last]
    return windows, global_


class WindowRiskHead(nn.Module):
    def __init__(self, hidden_size: int, head_size: int = 256, top_k: int = 2):
        super().__init__()
        if hidden_size < 1 or head_size < 1 or top_k < 1:
            raise ValueError("invalid head dimensions")
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.risk_head = nn.Sequential(
            nn.LayerNorm(3 * hidden_size),
            nn.Linear(3 * hidden_size, head_size),
            nn.GELU(),
            nn.Linear(head_size, 1),
        )

    def forward(self, window_features, global_feature, window_owner,
                report_labels=None, report_mask=None, report_count=None):
        if window_features.ndim != 3:
            raise ValueError("window_features must be [B,M,D]")
        bsz, _, dim = window_features.shape
        if dim != self.hidden_size or global_feature.shape != (bsz, dim):
            raise ValueError("feature dimension mismatch")
        z = F.layer_norm(window_features.float(), (dim,))
        g = F.layer_norm(global_feature.float(), (dim,)).unsqueeze(1).expand_as(z)
        scores = self.risk_head(torch.cat([z, g, z * g], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(window_owner < 0, -1e4)
        if report_labels is not None:
            report_count = report_labels.shape[1]
        elif report_count is None:
            report_count = int(window_owner.clamp_min(0).max().item()) + 1
        pooled = []
        for b in range(bsz):
            per_report = []
            for i in range(report_count):
                eligible = scores[b][window_owner[b] == i]
                if eligible.numel():
                    per_report.append(eligible.topk(min(self.top_k, eligible.numel())).values.mean())
                else:
                    per_report.append(scores[b].new_zeros(()))
            pooled.append(torch.stack(per_report))
        report_logits = torch.stack(pooled)
        result = {"window_logits": scores, "report_logits": report_logits}
        if report_labels is not None:
            if report_mask is None:
                raise ValueError("risk supervision must be explicitly masked")
            weights = report_mask.float()
            if not bool(weights.any()):
                raise ValueError("all labels masked")
            bce = F.binary_cross_entropy_with_logits(
                report_logits, report_labels.float(), reduction="none"
            )
            per_history = (bce * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            active = (weights.sum(dim=1) > 0).float()
            result["loss"] = (per_history * active).sum() / active.sum()
            result["n_labels"] = weights.sum()
        return result
