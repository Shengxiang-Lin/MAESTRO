"""Full vLLM decoding using an external RiskKV prefix and live PEFT snapshots."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import torch


def export_frozen_weights(policy, model_path, directory, dtype):
    """Materialize the same frozen NF4 weights for native vLLM linear layers."""
    from bitsandbytes.functional import dequantize_4bit
    from huggingface_hub import save_torch_state_dict

    base = policy.get_base_model()
    tensors = {}
    for name, param in base.named_parameters(remove_duplicate=False):
        if ".lora_" in name:
            continue
        tensor = param.detach()
        quant_state = getattr(param, "quant_state", None)
        if quant_state is not None:
            tensor = dequantize_4bit(tensor, quant_state)
        tensors[name.replace(".base_layer.", ".")] = tensor.to(
            device="cpu", dtype=dtype
        ).contiguous()
    config = json.loads((Path(model_path) / "config.json").read_text())
    config.pop("quantization_config", None)
    (directory / "config.json").write_text(json.dumps(config))
    save_torch_state_dict(
        tensors, directory, max_shard_size="2GB", metadata={"format": "pt"},
        shared_tensors_to_discard=list(getattr(base, "_tied_weights_keys", ()) or ()),
    )


class VLLMRiskRollout:
    def __init__(self, policy, args):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        from vllm import LLM

        parent = Path(args.output_dir).resolve().parent
        parent.mkdir(parents=True, exist_ok=True)
        self.workspace = TemporaryDirectory(prefix="riskkv-vllm-", dir=parent)
        self.directory = Path(self.workspace.name)
        self.engine = None
        self.version = 0
        self.cached_tokens = None
        model_dir = self.directory / "model"
        model_dir.mkdir()
        dtype = policy.get_input_embeddings().weight.dtype
        print("Materializing frozen NF4 backbone for vLLM", flush=True)
        export_frozen_weights(policy, args.model, model_dir, dtype)
        torch.cuda.empty_cache()
        print("Starting vLLM RiskKV engine", flush=True)
        self.engine = LLM(
            model=str(model_dir), tokenizer=args.model, dtype=dtype,
            skip_tokenizer_init=True, generation_config="vllm",
            max_model_len=args.max_length + args.max_new_tokens + 8,
            gpu_memory_utilization=args.vllm_memory_utilization,
            max_num_seqs=args.num_generations,
            max_num_batched_tokens=4096, block_size=16,
            enable_prefix_caching=False, enforce_eager=True,
            attention_backend="TRITON_ATTN", logprobs_mode="processed_logprobs",
            enable_lora=True, max_lora_rank=args.lora_rank,
            kv_transfer_config={
                "kv_connector": "RiskKVConnector",
                "kv_connector_module_path": "train_maestro_8b.policy.vllm_connector",
                "kv_role": "kv_consumer",
                "kv_load_failure_policy": "fail",
                "kv_connector_extra_config": {"directory": str(self.directory)},
            },
        )

    def close(self):
        try:
            if self.engine is not None:
                self.engine.llm_engine.engine_core.shutdown()
                self.engine = None
        finally:
            self.workspace.cleanup()

    @torch.no_grad()
    def sample(self, policy, prefix, anchor_ids, *, count, max_new_tokens,
               temperature, eos_token_id):
        from safetensors.torch import save_file
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        snapshot_id = uuid4().hex
        snapshot = self.directory / f"{snapshot_id}.safetensors"
        tensors = {}
        for index, layer in enumerate(prefix.cache.layers):
            # Convert HF [batch, head, token, dim] to per-token [K/V, head, dim].
            key = layer.keys[0].transpose(0, 1).cpu()
            value = layer.values[0].transpose(0, 1).cpu()
            tensors[str(index)] = torch.stack((key, value), dim=1)
        save_file(tensors, snapshot)
        self.version += 1
        adapter = self.directory / "adapter"
        try:
            policy.save_pretrained(adapter)
            params = SamplingParams(
                n=count, max_tokens=max_new_tokens, temperature=temperature,
                top_p=1.0, top_k=0, logprobs=0, detokenize=False, ignore_eos=True,
                stop_token_ids=[eos_token_id] if eos_token_id is not None and eos_token_id >= 0 else [],
                extra_args={"kv_transfer_params": {
                    "risk_id": snapshot_id, "prefix_length": prefix.length,
                }},
            )
            (result,) = self.engine.generate(
                [{"prompt_token_ids": prefix.token_ids + list(anchor_ids)}],
                params, use_tqdm=False,
                lora_request=LoRARequest(f"maestro-{self.version}", self.version, str(adapter)),
            )
            self.cached_tokens = result.num_cached_tokens
            if self.cached_tokens != prefix.length or len(result.outputs) != count:
                raise RuntimeError(
                    f"vLLM reported {self.cached_tokens}/{prefix.length} imported tokens "
                    f"and {len(result.outputs)}/{count} candidates"
                )
            device = next(policy.parameters()).device
            candidates = []
            for output in result.outputs:
                ids = list(output.token_ids)
                if not ids or not output.logprobs or len(ids) != len(output.logprobs):
                    raise RuntimeError("vLLM rollout did not return aligned sampled logprobs")
                old = [entry[token].logprob for token, entry in zip(ids, output.logprobs)]
                candidates.append((torch.tensor(ids, device=device), torch.tensor(old, device=device)))
            return candidates
        finally:
            snapshot.unlink(missing_ok=True)
