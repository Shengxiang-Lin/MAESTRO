"""Minimal Stage1 belief head over live Qwen3+LoRA hidden representations.

A report token alone is causally blind to feedback arriving AFTER that
report. Therefore every report embedding is jointly conditioned on the
LAST full-prompt hidden state, which has observed all received reports.
This is not the old frozen-feature Risk Head.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

NUM_STATES = 4
IGNORE = -100


def report_embeddings(hidden, attention_mask, report_starts, report_ends,
                      report_mask):
    """Mean report features + FINAL global history feature, preserving autograd."""
    if hidden.ndim != 3:
        raise ValueError("Expected [batch, tokens, hidden] representations")
    batch_size, seq_len, dim = hidden.shape
    if (attention_mask.shape != (batch_size, seq_len)
            or report_starts.shape != report_ends.shape
            or report_mask.shape != report_starts.shape
            or report_starts.shape[0] != batch_size):
        raise ValueError("Belief report/token shape mismatch")
    end_index = attention_mask.long().sum(dim=1) - 1
    if (end_index < 0).any():
        raise ValueError("Empty evidence history")
    global_feature = hidden[torch.arange(batch_size, device=hidden.device), end_index]
    reports = []
    for batch in range(batch_size):
        pooled = []
        for r in range(report_mask.shape[1]):
            if not bool(report_mask[batch, r]):
                pooled.append(hidden.new_zeros((dim,)))
                continue
            left = int(report_starts[batch, r])
            right = int(report_ends[batch, r])
            if not 0 <= left < right <= int(end_index[batch]) + 1:
                raise ValueError("Report span invalid or overlaps padding")
            pooled.append(hidden[batch, left:right].mean(dim=0))
        reports.append(torch.stack(pooled))
    return torch.stack(reports), global_feature


class EvidenceBeliefHead(nn.Module):
    """One shared predictor trunk, state classification and belief-update probe."""

    def __init__(self, hidden_size: int, head_size: int = 256, dropout: float = 0.1):
        super().__init__()
        if hidden_size < 1 or head_size < 1 or not 0 <= dropout < 1:
            raise ValueError("Bad belief head dimensions/dropout")
        self.hidden_size = hidden_size
        self.trunk = nn.Sequential(
            nn.LayerNorm(3 * hidden_size),
            nn.Linear(3 * hidden_size, head_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.state_classifier = nn.Linear(head_size, NUM_STATES)
        self.update_classifier = nn.Linear(head_size, 2)

    def forward(self, report_features, global_feature, report_mask,
                state_labels=None, update_labels=None, *,
                state_weight=None, update_loss_weight=0.3):
        if report_features.ndim != 3:
            raise ValueError("Expected [B,Reports,H]")
        bsz, reports, dim = report_features.shape
        if (dim != self.hidden_size or global_feature.shape != (bsz, dim)
                or report_mask.shape != (bsz, reports)):
            raise ValueError("Belief features/mask mismatch")
        local = F.layer_norm(report_features.float(), (dim,))
        global_ = F.layer_norm(global_feature.float(), (dim,)).unsqueeze(1)
        context = global_.expand_as(local)
        features = self.trunk(torch.cat((local, context, local * context), dim=-1))
        status_logits = self.state_classifier(features)
        update_logits = self.update_classifier(features)
        result = {"state_logits": status_logits, "update_logits": update_logits}
        if state_labels is None:
            return result
        if (state_labels.shape != (bsz, reports)
                or update_labels is None or update_labels.shape != (bsz, reports)):
            raise ValueError("Missing belief supervision")
        labelled = (state_labels != IGNORE) & report_mask.bool()
        if not bool(labelled.any()):
            raise ValueError("Belief batch contains no valid state supervision")
        state_loss = F.cross_entropy(
            status_logits[labelled], state_labels[labelled],
            weight=state_weight,
        )
        updated = (update_labels != IGNORE) & report_mask.bool()
        if bool(updated.any()) and update_loss_weight > 0:
            update_loss = F.cross_entropy(
                update_logits[updated], update_labels[updated],
            )
        else:
            update_loss = state_loss.detach().new_zeros(())
        result.update({
            "loss": state_loss + update_loss_weight * update_loss,
            "state_loss": state_loss,
            "update_loss": update_loss,
            "state_count": int(labelled.sum()),
            "update_count": int(updated.sum()),
        })
        return result
