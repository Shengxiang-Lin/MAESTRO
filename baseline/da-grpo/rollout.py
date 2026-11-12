"""O3-inspired colocated vLLM rollout with LoRA snapshots and sleep/wake.

The environment and latent layers from O2/O3 are intentionally not imported.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch


def sampled_logprob(entry: Any, token_id: int) -> float:
    if entry is None:
        raise ValueError("vLLM did not return token logprobs")
    if not hasattr(entry, "get"):
        raise TypeError(f"Unexpected vLLM logprob entry: {type(entry)}")
    item = entry.get(token_id, entry.get(str(token_id)))
    if item is None:
        raise ValueError(f"vLLM sampled-token logprob missing: {token_id}")
    return float(getattr(item, "logprob", item))


def json_prefix_end(tokenizer: Any, token_ids: list[int]) -> int | None:
    """Find first completed JSON root, respecting JSON strings and escapes."""
    seen, braces, string, escaped = False, 0, False, False
    for i, token_id in enumerate(token_ids):
        fragment = tokenizer.decode(
            [token_id], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for char in fragment:
            if not seen:
                if char == "{":
                    seen, braces = True, 1
                continue
            if string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    string = False
            elif char == '"':
                string = True
            elif char == "{":
                braces += 1
            elif char == "}":
                braces -= 1
                if braces == 0:
                    return i + 1
    return None


class ColocatedRollout:
    """vLLM shares the same GPU with the BF16 LoRA trainer in phases."""

    def __init__(self, model_path: str, tokenizer: Any, cfg: dict, sync_dir: Path):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.sync_dir = Path(sync_dir)
        self.sync_dir.mkdir(parents=True, exist_ok=True)
        self.llm = None
        self.sleeping = False
        self.version = 0

    def initialize(self):
        if self.llm is None:
            from vllm import LLM
            self.llm = LLM(
                model=self.model_path,
                dtype="bfloat16",
                tensor_parallel_size=1,
                gpu_memory_utilization=self.cfg["gpu_memory_utilization"],
                max_model_len=self.cfg["max_model_len"],
                kv_cache_dtype=self.cfg["kv_cache_dtype"],
                max_num_seqs=self.cfg["max_num_seqs"],
                enable_lora=True,
                max_loras=1,
                max_lora_rank=int(self.cfg["lora_rank"]),
                enable_sleep_mode=self.cfg["enable_sleep_mode"],
                enable_prefix_caching=self.cfg["enable_prefix_caching"],
                generation_config="vllm",
                trust_remote_code=True,
            )

    def sleep(self):
        if self.llm is not None and not self.sleeping:
            self.llm.sleep(level=1)
            self.sleeping = True
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def sample(self, model: Any, prompt: list[int], *, n: int,
               max_tokens: int, temperature: float, top_p: float) -> list[dict]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        if temperature != 1.0 or top_p != 1.0:
            raise ValueError("GRPO behavior logprobs require temperature=top_p=1")
        gc.collect()
        torch.cuda.empty_cache()
        self.initialize()
        self.version += 1
        adapter_dir = self.sync_dir / "current_adapter"
        model.save_pretrained(adapter_dir)
        request = LoRARequest(f"maestro_o1_grpo_{self.version}",
                              self.version, str(adapter_dir))
        if self.sleeping:
            self.llm.wake_up()
            self.sleeping = False
        try:
            params = SamplingParams(
                n=n, temperature=temperature, top_p=top_p,
                max_tokens=max_tokens, logprobs=1,
                skip_special_tokens=True,
            )
            results = self.llm.generate(
                {"prompt_token_ids": prompt},
                sampling_params=params,
                lora_request=request,
                use_tqdm=False,
            )
            generated = list(results[0].outputs)
            if len(generated) != n:
                raise RuntimeError(f"vLLM generated {len(generated)} != {n}")
            out = []
            for candidate in generated:
                ids = list(map(int, candidate.token_ids))
                entries = list(candidate.logprobs or [])
                if len(ids) != len(entries):
                    raise ValueError("vLLM logprobs do not align with token ids")
                old = [sampled_logprob(item, token)
                       for item, token in zip(entries, ids)]
                end = json_prefix_end(self.tokenizer, ids)
                if end is not None:
                    ids, old = ids[:end], old[:end]
                if not ids:
                    raise ValueError("vLLM returned an empty completion")
                out.append({
                    "text": self.tokenizer.decode(
                        ids, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                    "prompt_input_ids": prompt,
                    "action_input_ids": ids,
                    "old_log_probs": old,
                    "json_complete": end is not None,
                })
            return out
        finally:
            self.sleep()

    def shutdown(self):
        self.sleep()
        self.llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
