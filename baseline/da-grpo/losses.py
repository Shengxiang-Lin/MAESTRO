"""Single-card streamed GRPO objective; no expert or auxiliary losses."""
from __future__ import annotations

import torch


def group_advantages(scores: list[float]) -> list[float]:
    if len(scores) < 2:
        raise ValueError("GRPO requires at least two candidates")
    values = torch.tensor(scores, dtype=torch.float32)
    sigma = values.std(unbiased=False)
    if not torch.isfinite(sigma) or sigma < 1e-6:
        return [0.0] * len(scores)
    return ((values - values.mean()) / (sigma + 1e-6)).tolist()


def clipped_grpo_loss(
    current: torch.Tensor, old: torch.Tensor, advantage: float,
    *, clip_range: float, normalization_tokens: int,
) -> torch.Tensor:
    """One candidate's contribution to the full-group token-normalized loss.

    Sum backward gradients for all candidates BEFORE optimizer.step().
    """
    if current.ndim != 1 or old.shape != current.shape:
        raise ValueError("Candidate logprob shape mismatch")
    if normalization_tokens <= 0 or current.numel() == 0:
        raise ValueError("Empty GRPO action tokens")
    ratio = (current.float() - old.float()).clamp(-20, 20).exp()
    bounded = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    gain = torch.minimum(ratio * advantage, bounded * advantage)
    return -gain.sum() / normalization_tokens


def low_var_kl(
    current: torch.Tensor, reference: torch.Tensor,
    *, normalization_tokens: int,
) -> torch.Tensor:
    """Nonnegative k3 KL estimator, same low-var family as O1/verl.

    k3 = exp(log p_ref - log p_policy) - (log p_ref - log p_policy) - 1.
    The KL penalty is separate from the four-dimensional reward.
    """
    if current.shape != reference.shape:
        raise ValueError("Reference/current logprob shape mismatch")
    delta = (reference.float() - current.float()).clamp(-20, 20)
    return (delta.exp() - delta - 1).sum() / normalization_tokens
