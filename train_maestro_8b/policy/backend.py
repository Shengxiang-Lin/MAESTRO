"""Stage-2 Unsloth + PEFT backend for risk-steered Qwen3 decoding.

Unsloth is intentionally loaded lazily, *before* transformers/peft imports.
The Transformers forward owns the risk-edited DynamicCache. An optional
vLLM attention kernel can read it in the custom rollout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def load_risk_engine(args):
    """Load the frozen Stage-1 Risk Head plus trainable Unsloth QLoRA model."""
    if args.backend != "unsloth":
        raise ValueError("this loader requires --backend unsloth")
    if not getattr(args, "load_in_4bit", False):
        raise ValueError("first Unsloth integration currently requires --load_in_4bit")
    # This must be set before importing Unsloth so custom GRPO can always
    # access the real per-token LM logits for exact policy log probabilities.
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    os.environ["UNSLOTH_BLOCK_COMPILE_LEGACY"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        from unsloth import FastLanguageModel
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Unsloth CUDA installation missing. Install compatible "
            "unsloth / unsloth_zoo / trl packages (see README)."
        ) from exc

    import torch
    from train_maestro_8b.risk.model import WindowRiskHead
    from train_maestro_8b.policy.model import RiskAwarePolicy

    checkpoint = Path(args.risk_checkpoint)
    cfg = json.loads((checkpoint / "risk_config.json").read_text(encoding="utf-8"))
    if cfg.get("schema") != "maestro.frozen_window_risk.v1":
        raise ValueError("Stage-1 Risk Head version is incompatible")
    if cfg.get("base_model") != args.model:
        raise ValueError(
            "Stage1 Risk Head must use exactly the same Qwen3-8B model ID "
            "as the Stage2 frozen backbone"
        )
    if cfg.get("trained_parameters") != "risk_head_only":
        raise ValueError("Stage1 checkpoint did not freeze the backbone")
    precision = cfg.get("feature_model_quantization")
    if precision != "nf4":
        raise ValueError(
            "Risk Head frozen features were not verified as NF4. "
            "Re-extract Stage1 features with --load_in_4bit, retrain Risk Head, "
            "and use the same base model and tokenizer; Stage2 Unsloth is NF4."
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_length + args.max_new_tokens + 8,
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=args.lora_rank,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Intentionally disable gradient checkpointing here: this research
    # policy *requires* use_cache=True throughout the LoRA suffix.
    # Checkpointing and use_cache can conflict in patched model forwards.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing=False,
        random_state=args.seed,
    )
    if not hasattr(model, "disable_adapter") or not hasattr(model, "get_base_model"):
        raise RuntimeError("Unsloth returned a policy without PEFT adapter controls")
    if any(p.requires_grad and "lora_" not in name
           for name, p in model.named_parameters()):
        raise RuntimeError("non-LoRA parameter is trainable in Stage2")
    if not any(p.requires_grad and "lora_" in name
               for name, p in model.named_parameters()):
        raise RuntimeError("Unsloth failed to create trainable LoRA weights")
    # Unsloth's patched forwards expect a tuple cache. The original Qwen3
    # forward works with the same quantized weights and PEFT adapters.
    for module in model.modules():
        if getattr(module, "_unsloth_fast_path", False):
            module._unsloth_fast_path = False
    model.config._attn_implementation = "sdpa"
    model.config.use_cache = True

    head = WindowRiskHead(cfg["hidden_size"], cfg["head_size"], cfg["top_k"])
    head.load_state_dict(torch.load(
        checkpoint / "risk_head.pt", map_location="cpu", weights_only=True,
    ))
    device = next(model.parameters()).device
    head.to(device).eval().requires_grad_(False)
    engine = RiskAwarePolicy(
        model, tokenizer, head, cfg,
        max_length=args.max_length,
        last_n_layers=args.steer_last_n,
        strength=args.steer_strength,
        threshold=args.steer_threshold,
    )
    return engine


def _probe_nf4_projection_shape(engine, state, ids):
    """Compare SAME first-token activation through NF4 at M=1 vs M=3.

    Three-token forward captures realistic layer inputs, then computes
    an identical vector through one-token and three-token 4-bit kernels.
    This isolates quantized linear matmul from attention and KV caching.
    """
    import torch
    from contextlib import ExitStack

    if ids.ndim != 1 or ids.numel() < 3:
        return []
    targets = (
        ".layers.0.self_attn.q_proj.base_layer",
        ".layers.0.self_attn.k_proj.base_layer",
        ".layers.0.self_attn.v_proj.base_layer",
        ".layers.0.mlp.up_proj",
        ".layers.0.mlp.down_proj",
    )
    modules = [
        (name, module) for name, module in engine.policy.named_modules()
        if any(name == suffix[1:] or name.endswith(suffix) for suffix in targets)
        and getattr(getattr(module, "weight", None), "quant_state", None) is not None
    ]
    if not modules:
        return []
    x = torch.cat(
        (torch.tensor(engine.anchor_ids, device=engine.device, dtype=torch.long),
         ids[:-1]),
    ).unsqueeze(0)
    captured = {}
    with torch.no_grad(), ExitStack() as stack:
        for name, module in modules:
            def capture(_module, inputs, name=name):
                if name not in captured:
                    captured[name] = inputs[0].detach().clone()
            hook = module.register_forward_pre_hook(capture)
            stack.callback(hook.remove)
        engine._forward_suffix(state, x)

    differences = []
    with torch.no_grad():
        for name, module in modules:
            activations = captured.get(name)
            if activations is None or activations.ndim != 3 or activations.shape[1] < 3:
                continue
            single = module(activations[:, :1, :])[:, 0, :].float()
            multiple = module(activations)[:, 0, :].float()
            if single.shape != multiple.shape or not torch.isfinite(single).all() or not torch.isfinite(multiple).all():
                raise RuntimeError("invalid 4-bit shape comparison")
            delta = (single - multiple).abs()
            differences.append({
                "module": name,
                "max_abs": round(delta.max().item(), 8),
                "mean_abs": round(delta.mean().item(), 8),
                "relative_max": round((delta.max() / multiple.abs().max().clamp_min(1e-8)).item(), 8),
                "output_features": int(single.shape[-1]),
            })
    return differences


def check_risk_cache_contract(engine, record, *, temperature=0.8, max_logprob_drift=0.01):
    """Fail early if a particular Unsloth version breaks cache continuation.

    This is a semantic consistency probe, not a 16K throughput benchmark.
    The frozen RiskHead is never modified by the test.
    """
    import torch
    policy = engine.policy
    policy.eval()
    state = engine.prefill(record["prompt"], record["reports"])
    steered_tokens = int((state.risk > engine.threshold).sum().item()) if engine.strength else 0
    [(ids, old)] = engine.sample_group(
        state, count=1, max_new_tokens=3, temperature=temperature, eos_token_id=-1,
    )
    with torch.no_grad():
        incremental = engine.log_probs_incremental(state, ids, temperature=temperature)
        replay = engine.log_probs(state, ids, temperature=temperature)

        # To test causal independence, modify the ACTUAL future input
        # token while keeping the TARGET ids in gather() unchanged. Calling
        # log_probs(prefix, mutated_targets) would be a false-positive:
        # it also changes the target token at that position, so its own
        # log probability naturally differs even with a correct mask.
        causal_future_sensitivity = None
        if ids.numel() >= 3:
            anchor = torch.tensor(
                engine.anchor_ids, dtype=torch.long, device=engine.device,
            )
            suffix = torch.cat((anchor, ids[:-1]), dim=0)[None]
            changed_suffix = suffix.clone()
            vocab_size = int(policy.get_input_embeddings().weight.shape[0])
            changed_suffix[0, -1] = (changed_suffix[0, -1] + 1) % vocab_size
            changed_logits = engine._forward_suffix(state, changed_suffix).logits
            changed_logps = changed_logits[:, -ids.numel():].float().div(
                temperature
            ).log_softmax(dim=-1).gather(
                2, ids.view(1, -1, 1)
            ).squeeze(0).squeeze(-1)
            # The last suffix input affects the final prediction, not the
            # first two. Compare probabilities of the SAME target tokens.
            causal_future_sensitivity = float(
                (replay[:-1] - changed_logps[:-1]).abs().max().item()
            )

        # Distinguish the SDPA attention implementation from NF4 GEMV/GEMM
        # shape-specific numerical effects. A 3-token Eager teacher-forced
        # replay uses the SAME model, LoRA weights and steered KV, but the
        # explicit eager attention path. Restore both wrappers' settings
        # unconditionally, including if this optional diagnostic fails.
        eager = None
        eager_probe_error = None
        configs = []
        for cfg in (policy.config, policy.get_base_model().config):
            if not any(cfg is x for x, _ in configs):
                configs.append((cfg, cfg._attn_implementation))
        try:
            for cfg, _ in configs:
                cfg._attn_implementation = "eager"
            eager = engine.log_probs(state, ids, temperature=temperature)
        except Exception as exc:
            # Optional diagnostic only; the strict consistency gate below
            # still fails closed regardless of eager backend support.
            eager_probe_error = f"{type(exc).__name__}: {str(exc)[:240]}"
        finally:
            for cfg, original in configs:
                cfg._attn_implementation = original

    if any(
        not torch.isfinite(v).all()
        for v in (old, incremental, replay)
    ):
        raise RuntimeError("RiskKV probe returned nonfinite log probabilities")
    if not state.window_probs.numel() and record["reports"]:
        raise RuntimeError("reported feedback was not assigned any risk windows")
    # This also verifies that the accelerated suffix remains differentiable
    # and freezes all non-LoRA weights, not merely sampling correctly.
    policy.train()
    for param in policy.parameters():
        param.grad = None
    replay_with_grad = engine.log_probs(state, ids, temperature=temperature)
    if not torch.isfinite(replay_with_grad).all():
        raise RuntimeError("RiskKV differentiable replay has nonfinite log probabilities")
    grad_values = replay_with_grad.detach()
    drift = float((old - grad_values).abs().max().item())
    def max_drift(a, b):
        return float((a - b).abs().max().item())
    diagnostic = {
        "prefix_tokens": state.length,
        "probe_tokens": len(ids),
        "rollout_engine": (
            "vllm" if engine.rollout_engine is not None
            else "transformers"
        ),
        "max_abs_sample_vs_incremental": max_drift(old, incremental),
        "max_abs_incremental_vs_batched_eval": max_drift(incremental, replay),
        "max_abs_batched_eval_vs_train": max_drift(replay, grad_values),
        "max_abs_sample_vs_batched_train": drift,
        "max_abs_batched_causal_future_sensitivity": causal_future_sensitivity,
        "max_abs_batched_eager_vs_incremental": (
            max_drift(eager, incremental) if eager is not None else None
        ),
        "max_abs_batched_eager_vs_sdpa": (
            max_drift(eager, replay) if eager is not None else None
        ),
        "eager_probe_error": eager_probe_error,
        "nf4_projection_shape_probe": _probe_nf4_projection_shape(engine, state, ids),
        "per_token_sample_vs_incremental": [
            round(float(v), 7) for v in (old - incremental).abs().tolist()
        ],
        "per_token_incremental_vs_batched_eval": [
            round(float(v), 7) for v in (incremental - replay).abs().tolist()
        ],
        "per_token_batched_eval_vs_train": [
            round(float(v), 7) for v in (replay - grad_values).abs().tolist()
        ],
        "vllm_cached_tokens": getattr(engine.rollout_engine, "cached_tokens", None),
        "steered_tokens": steered_tokens,
        "allowed_drift": max_logprob_drift,
    }
    import json
    print("risk_steering_preflight_diagnostics=" +
          json.dumps(diagnostic, ensure_ascii=False), flush=True)
    if drift > max_logprob_drift:
        raise RuntimeError(
            f"RiskKV rollout vs differentiable replay drift {drift:.6g} "
            f"exceeds the configured bound {max_logprob_drift}. "
            "Inspect risk_steering_preflight_diagnostics: "
            "sample_vs_incremental isolates inference engine/LoRA/weight drift; "
            "incremental_vs_batched_eval isolates one-token vs batched "
            "teacher-forcing (attention masks/kernels or NF4 matmul shapes); "
            "batched_eval_vs_train isolates training-mode kernel behavior. "
            "Causal-future sensitivity checks whether a future token changes "
            "earlier logits; Eager-vs-SDPA helps isolate attention kernels "
            "from NF4 GEMV/GEMM shape-specific numerical effects. "
            "Do not increase the tolerance or start GRPO until diagnosed."
        )
    (-replay_with_grad.mean()).backward()
    if not any(p.grad is not None and bool((p.grad != 0).any())
               for name, p in policy.named_parameters() if "lora_" in name):
        raise RuntimeError("no nonzero LoRA gradients through RiskKV suffix")
    if any(p.grad is not None for name, p in policy.named_parameters()
           if "lora_" not in name):
        raise RuntimeError("frozen backbone unexpectedly received gradients")
    if any(p.grad is not None for p in engine.risk_head.parameters()):
        raise RuntimeError("frozen Risk Head unexpectedly received gradients")
    policy.zero_grad(set_to_none=True)
    return {
        "prefix_tokens": state.length,
        "probe_tokens": len(ids),
        "max_abs_logprob_drift": drift,
        "max_abs_no_grad_replay_drift": max_drift(old, replay),
        "max_abs_sample_vs_incremental": diagnostic["max_abs_sample_vs_incremental"],
        "max_abs_incremental_vs_batched_eval":
            diagnostic["max_abs_incremental_vs_batched_eval"],
        "max_abs_batched_eval_vs_train": diagnostic["max_abs_batched_eval_vs_train"],
        "max_abs_batched_causal_future_sensitivity":
            diagnostic["max_abs_batched_causal_future_sensitivity"],
        "max_abs_batched_eager_vs_incremental":
            diagnostic["max_abs_batched_eager_vs_incremental"],
        "max_abs_batched_eager_vs_sdpa":
            diagnostic["max_abs_batched_eager_vs_sdpa"],
        "nf4_projection_shape_probe": diagnostic["nf4_projection_shape_probe"],
        "risk_windows": int(state.window_probs.numel()),
        "steered_tokens": steered_tokens,
        "vllm_cached_tokens": getattr(engine.rollout_engine, "cached_tokens", None),
        "max_window_risk": float(state.window_probs.max().item())
        if state.window_probs.numel() else None,
    }
