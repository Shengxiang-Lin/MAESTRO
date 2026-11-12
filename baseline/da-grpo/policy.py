"""Plain Qwen3-8B LoRA policy for O1 JSON actions (NO future latent)."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F


def load_policy(model_path: str, cfg: dict, *, resume_adapter: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model

    if not torch.cuda.is_available():
        raise RuntimeError("Single-GPU LoRA-GRPO requires CUDA")
    dtype = torch.bfloat16 if cfg["dtype"] == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    if resume_adapter:
        model = PeftModel.from_pretrained(
            backbone, resume_adapter, is_trainable=True,
        )
    else:
        lora_cfg = LoraConfig(
            r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["lora_target_modules"],
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(backbone, lora_cfg)
    model.to("cuda")
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False
    model.print_trainable_parameters()
    return model, tokenizer


def prompt_ids(tokenizer: Any, messages: list[dict], max_length: int) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )
    if not isinstance(ids, list) or not ids:
        raise ValueError("Qwen chat template did not produce token IDs")
    if len(ids) > max_length:
        raise ValueError(
            f"Training prompt is {len(ids)} tokens, exceeding {max_length}; "
            "refusing to truncate evidence/history"
        )
    return [int(v) for v in ids]


def action_log_probs(
    model: Any,
    prompt: list[int],
    completion: list[int],
    *, logits_chunk_size: int,
) -> torch.Tensor:
    """Differentiable token logprobs without allocating full [T, vocab] logits.

    The PEFT-wrapped causal LM exposes the *same adapter-aware* decoder through
    get_base_model().model. Calling that backbone avoids a giant all-token
    vocabulary projection for 24K-token context. The LM head is applied only
    to the selected completion-token positions, in small chunks.
    """
    if not prompt or not completion:
        raise ValueError("Empty prompt/completion")
    ids = torch.tensor(prompt + completion, dtype=torch.long, device="cuda")
    base = model.get_base_model()
    hidden = base.model(
        input_ids=ids.unsqueeze(0),
        use_cache=False,
        return_dict=True,
    ).last_hidden_state[0]
    start = len(prompt) - 1
    selected = hidden[start : start + len(completion)]
    if selected.size(0) != len(completion):
        raise RuntimeError("Action logprob shift mismatch")
    labels = ids[len(prompt):]
    chunks = []
    for offset in range(0, len(completion), logits_chunk_size):
        part = selected[offset : offset + logits_chunk_size]
        logits = base.lm_head(part).float()
        token_logp = F.log_softmax(logits, dim=-1).gather(
            -1, labels[offset:offset + logits_chunk_size].unsqueeze(-1)
        ).squeeze(-1)
        chunks.append(token_logp)
    return torch.cat(chunks)


def reference_log_probs(
    model: Any, prompt: list[int], completion: list[int],
    *, logits_chunk_size: int,
) -> torch.Tensor:
    """Frozen base model via temporarily disabling the LoRA adapter.

    No second 8B reference-model copy is kept on GPU.
    """
    with torch.no_grad(), model.disable_adapter():
        return action_log_probs(
            model, prompt, completion, logits_chunk_size=logits_chunk_size,
        ).detach()
