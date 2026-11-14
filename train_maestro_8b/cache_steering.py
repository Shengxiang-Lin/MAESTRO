"""Risk-conditioned edits of a *frozen* Hugging Face DynamicCache.

Only the report-associated prefix Value tensors change; Keys, unrelated
positions and future decoding tokens are untouched. Never mutate a shared
cache while another rollout is using it.
"""
from __future__ import annotations

from copy import deepcopy
import torch


def token_risk_from_windows(probs, starts, ends, owners, seq_len: int):
    """Average probabilities over overlapping windows; 0 outside reports.

    probs, starts, ends and owners are 1D for one prompt, with -1 owners
    representing padding windows. The returned tensor is on probs.device.
    """
    if seq_len < 1:
        raise ValueError("empty prefix")
    risk = torch.zeros(seq_len, device=probs.device, dtype=torch.float32)
    count = torch.zeros_like(risk)
    for j in (owners >= 0).nonzero(as_tuple=True)[0].tolist():
        start, end = int(starts[j]), int(ends[j])
        if not (0 <= start < end <= seq_len):
            raise ValueError("window points outside actual prefix")
        risk[start:end] += probs[j].float()
        count[start:end] += 1.0
    return (risk / count.clamp_min(1)).clamp_(0, 1)


def clone_cache(cache):
    """Each candidate gets a private growing cache; no shared in-place writes."""
    return deepcopy(cache)


def steer_value_cache(cache, token_risk, *, last_n_layers: int = 8,
                      strength: float = 0.10, threshold: float = 0.50):
    """Return cache with historical Value entries adjusted, in place.

    Expects a standard DynamicCache (list of layer.keys/.values tensors).
    Does not support offloaded, quantized, sliding, static or hybrid caches.
    The caller is responsible for ensuring prefix KV tensors are frozen.
    """
    if last_n_layers < 1 or strength < 0 or not 0 <= threshold <= 1:
        raise ValueError("invalid steering hyperparameters")
    if token_risk.ndim != 2:
        raise ValueError("token_risk must have shape [B,T]")
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, list) or not layers:
        raise TypeError("requires a nonempty DynamicCache with .layers")
    if last_n_layers > len(layers):
        raise ValueError("requested more steer layers than the model has")
    if not torch.isfinite(token_risk).all():
        raise ValueError("nonfinite risk scores")
    if token_risk.min() < 0 or token_risk.max() > 1:
        raise ValueError("risk scores must lie in [0,1]")
    mask = (token_risk.float() - threshold).clamp_min(0.0)
    scale = 1.0 + strength * mask
    # Clone outside this function if the source cache must remain reusable.
    for layer in layers[-last_n_layers:]:
        k, v = getattr(layer, "keys", None), getattr(layer, "values", None)
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise TypeError("unsupported non-tensor KV cache layer")
        if k.shape != v.shape or v.ndim != 4:
            raise ValueError("KV tensor layout mismatch")
        if v.shape[0] != token_risk.shape[0] or v.shape[-2] != token_risk.shape[1]:
            raise ValueError("risk tokens and cached prefix differ")
        layer.values = v * scale[:, None, :, None].to(v.device, dtype=v.dtype)
    return cache
