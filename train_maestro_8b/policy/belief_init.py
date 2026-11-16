"""Safely warm-start Stage2 DA-style GRPO's existing PEFT LoRA from Stage1.

No Risk Head, modified KV cache, or extra prompt is used at Stage2.
Do not silently ignore unmatched Stage1 adapter keys.
"""
from __future__ import annotations

import json
from pathlib import Path


def normalized_adapter_key(key: str, adapter_name: str = "default") -> str:
    for part in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
        key = key.replace(f".{part}.{adapter_name}.", f".{part}.")
    return key


def validate_adapter_keys(current: dict, source: dict,
                          adapter_name: str = "default") -> dict:
    actual = {
        normalized_adapter_key(key, adapter_name): key
        for key in current
        if ".lora_A." in key or ".lora_B." in key
    }
    source_keys = {
        key for key in source
        if ".lora_A." in key or ".lora_B." in key
    }
    if not source_keys or set(source) != source_keys:
        raise ValueError("Stage1 adapter contains unexpected/non-LoRA tensors")
    missing = sorted(set(actual) - source_keys)
    unexpected = sorted(source_keys - set(actual))
    if missing or unexpected:
        raise ValueError(
            f"Stage1/Stage2 LoRA architecture mismatch: "
            f"missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    for key, loaded in source.items():
        target = current[actual[key]]
        if target.shape != loaded.shape:
            raise ValueError(f"Stage1 LoRA shape mismatch for {key}")
    return actual


def load_stage1_belief_adapter(policy, adapter_directory, *,
                               stage2_model: str, rank: int,
                               alpha: int) -> dict:
    import torch
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    directory = Path(adapter_directory).expanduser().resolve()
    manifest_path = directory.parent / "manifest.json"
    weights = directory / "adapter_model.safetensors"
    if not manifest_path.is_file() or not weights.is_file():
        raise FileNotFoundError("Stage1 pilot manifest/PEFT adapter missing")
    config = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config.get("schema") != "maestro.stage1_belief_checkpoint.v1":
        raise ValueError("Not a Stage1 evidence-belief checkpoint")
    if (config.get("lora_rank") != rank or config.get("lora_alpha") != alpha
            or Path(config.get("base_model", "")).expanduser().resolve()
            != Path(stage2_model).expanduser().resolve()):
        raise ValueError("Stage1 LoRA base/rank/alpha incompatible with Stage2")
    tensors = load_file(weights, device="cpu")
    current = policy.state_dict()
    key_map = validate_adapter_keys(current, tensors)
    set_peft_model_state_dict(policy, tensors, adapter_name="default")
    updated = policy.state_dict()
    for key, value in tensors.items():
        loaded = updated[key_map[key]]
        if not torch.equal(loaded.detach().cpu(), value.to(dtype=loaded.dtype)):
            raise RuntimeError("Stage1 LoRA weights did not actually load: " + key)
    return {
        "adapter": str(directory),
        "loaded_tensors": len(tensors),
        "stage1_steps": config.get("steps"),
        "stage1_validation": (config.get("validation") or [])[-1:]
    }
