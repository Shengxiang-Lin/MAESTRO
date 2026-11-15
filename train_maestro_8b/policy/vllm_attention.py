"""vLLM FlashAttention for unpadded RiskKV rollout on the training GPU."""

import torch

IMPLEMENTATION = "maestro_vllm_flash"


def register_vllm_attention():
    from transformers import AttentionInterface
    from vllm.vllm_flash_attn import flash_attn_varlen_func, is_fa_version_supported

    if not torch.cuda.is_available() or not is_fa_version_supported(2):
        raise RuntimeError("RiskKV vLLM rollout requires the CUDA vLLM FA2 extension")

    def attention(module, query, key, value, attention_mask, *,
                  dropout=0.0, scaling=None, sliding_window=None, **kwargs):
        # Transformers already updated the cache with the current suffix.
        # FA2's bottom-right causal alignment lets each query see the prefix
        # and only the suffix tokens at or before its own position.
        if (query.shape[0] != 1 or key.shape[0] != 1 or attention_mask is not None
                or sliding_window is not None or dropout != 0.0 or torch.is_grad_enabled()):
            raise ValueError("vLLM attention requires one unpadded causal inference sequence")
        q_len, kv_len = query.shape[-2], key.shape[-2]
        cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=query.device)
        cu_kv = torch.tensor([0, kv_len], dtype=torch.int32, device=query.device)
        output = flash_attn_varlen_func(
            query[0].transpose(0, 1).contiguous(),
            key[0].transpose(0, 1).contiguous(),
            value[0].transpose(0, 1).contiguous(),
            max_seqlen_q=q_len, cu_seqlens_q=cu_q,
            max_seqlen_k=kv_len, cu_seqlens_k=cu_kv,
            softmax_scale=scaling, causal=True, fa_version=2,
        )
        return output.unsqueeze(0), None

    AttentionInterface.register(IMPLEMENTATION, attention)
    return IMPLEMENTATION
