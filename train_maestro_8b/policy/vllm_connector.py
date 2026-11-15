"""Import the frozen steered Qwen3 prefix into vLLM paged KV."""
from dataclasses import dataclass, field
from pathlib import Path
import re

import torch
from safetensors.torch import load_file
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata,
)


def write_triton_kv(target, values, blocks, block_size):
    """Copy only the real prefix slots, including a partially filled page."""
    if target.ndim != 4 or target.shape[2] != block_size:
        raise ValueError(f"Unsupported Triton paged KV layout: {target.shape}")
    length = len(values)
    heads, width = target.shape[1], target.shape[-1]
    if values.shape != (length, 2, heads, width // 2) or values.dtype != target.dtype:
        raise ValueError("HF and vLLM KV shapes or dtypes differ")
    if len(blocks) * block_size < length:
        raise ValueError("vLLM block allocation does not cover the RiskKV prefix")
    positions = torch.arange(length, device=target.device)
    block_table = torch.tensor(blocks, dtype=torch.long, device=target.device)
    # vLLM 0.32 Triton layout: [block, head, slot, K | V].
    packed = values.to(target.device).transpose(1, 2).reshape(length, heads, width)
    target.transpose(1, 2)[
        block_table[positions // block_size], positions % block_size
    ] = packed


@dataclass
class RiskKVMetadata(KVConnectorMetadata):
    loads: list[tuple[str, int, list[int]]] = field(default_factory=list)


class RiskKVConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.path = Path(self._kv_transfer_config.kv_connector_extra_config["directory"])
        self.block_size = vllm_config.cache_config.block_size
        self.pending = RiskKVMetadata()

    def _source(self, request):
        params = request.kv_transfer_params or {}
        source, length = params.get("risk_id"), params.get("prefix_length")
        if not isinstance(source, str) or not re.fullmatch(r"[a-f0-9]{32}", source):
            raise ValueError("vLLM rollout requires a RiskKV snapshot ID")
        if type(length) is not int or not 0 < length < len(request.prompt_token_ids):
            raise ValueError("RiskKV prefix must leave the decision anchor for vLLM")
        if not (self.path / f"{source}.safetensors").is_file():
            raise FileNotFoundError("RiskKV snapshot is unavailable")
        return source, length

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        _, length = self._source(request)
        if num_computed_tokens:
            raise RuntimeError("RiskKV requires prefix caching disabled")
        return length, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        source, length = self._source(request)
        if num_external_tokens != length:
            raise RuntimeError("vLLM did not allocate the complete edited RiskKV prefix")
        (block_ids,) = blocks.get_block_ids()
        self.pending.loads.append((source, length, block_ids))

    def build_connector_meta(self, scheduler_output):
        pending, self.pending = self.pending, RiskKVMetadata()
        return pending

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        saved_snapshots = {}
        for snapshot, length, blocks in metadata.loads:
            if snapshot not in saved_snapshots:
                saved_snapshots[snapshot] = load_file(self.path / f"{snapshot}.safetensors")
            saved = saved_snapshots[snapshot]
            filled = 0
            for name, layer in forward_context.no_compile_layers.items():
                target = getattr(layer, "kv_cache", None)
                if target is None:
                    continue
                match = re.fullmatch(r"model\.layers\.(\d+)\.self_attn\.attn", name)
                if match is None:
                    raise RuntimeError(f"Unrecognized Qwen3 attention layer {name}")
                values = saved[match[1]]
                if len(values) != length:
                    raise RuntimeError(f"RiskKV snapshot token count differs for {name}")
                write_triton_kv(target, values, blocks, self.block_size)
                filled += 1
            if filled != len(saved):
                raise RuntimeError(f"Imported {filled}/{len(saved)} RiskKV layers")

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass
