"""Train 8B Main-Agent latent evidence-risk representations (no text generation).

The output is a standard PEFT CausalLM adapter for a subsequent GRPO stage.
The window-risk probe is saved separately and is not part of the Main prompt.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import random

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from .risk_data import RiskCollator, RiskDataset
from .risk_model import WindowRiskModel
from .risk_schema import VERSION


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--eval_file")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_length", type=int, default=16384)
    p.add_argument("--window_size", type=int, default=128)
    p.add_argument("--window_stride", type=int, default=96)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--head_size", type=int, default=256)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--head_lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--mixed_precision", choices=("no", "bf16", "fp16"), default="bf16")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, accelerator):
    model.eval()
    total = torch.zeros(2, device=accelerator.device)
    for batch in loader:
        result = model(**batch)
        labels = result["n_labels"].detach().float()
        total += torch.stack([result["loss"].detach() * labels, labels])
    total = accelerator.reduce(total, reduction="sum")
    model.train()
    return (total[0] / total[1].clamp_min(1)).item()


def main():
    args = arguments()
    if min(args.epochs, args.batch_size, args.grad_accum, args.max_length,
           args.window_size, args.window_stride, args.top_k) <= 0:
        raise ValueError("training sizes and epoch counts must be positive")
    if args.window_stride > args.window_size:
        raise ValueError("window_stride cannot exceed window_size")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "no": torch.float32}[args.mixed_precision]
    kwargs = dict(torch_dtype=dtype, attn_implementation="sdpa")
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise ValueError("4-bit QLoRA requires CUDA")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": accelerator.local_process_index}
    policy = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.load_in_4bit:
        policy = prepare_model_for_kbit_training(
            policy, use_gradient_checkpointing=args.gradient_checkpointing
        )
    elif args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
    policy.config.use_cache = False
    policy = get_peft_model(policy, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    hidden_size = int(policy.get_base_model().config.hidden_size)
    model = WindowRiskModel(policy, hidden_size, head_size=args.head_size,
                            top_k=args.top_k)
    collator = RiskCollator(
        tokenizer, max_length=args.max_length,
        window_size=args.window_size, window_stride=args.window_stride,
    )
    training = RiskDataset(args.train_file, tokenizer, args.max_length)
    train_loader = DataLoader(
        training, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=0,
    )
    validation = (RiskDataset(args.eval_file, tokenizer, args.max_length)
                  if args.eval_file else None)
    val_loader = (DataLoader(validation, batch_size=args.batch_size,
                             collate_fn=collator, shuffle=False)
                  if validation is not None else None)
    optimizer = AdamW([
        {"params": [p for p in model.policy.parameters() if p.requires_grad],
         "lr": args.lr},
        {"params": list(model.risk_head.parameters()), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    if val_loader is None:
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader,
        )
    else:
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader,
        )
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    scheduler = accelerator.prepare(scheduler)
    if accelerator.is_main_process:
        print(json.dumps({
            "train_examples": len(training), "dropped_overlong": training.overlong,
            "eval_examples": len(validation) if validation is not None else 0,
            "total_optimizer_steps": total_steps,
            "trainable_parameters": sum(p.numel() for p in model.parameters()
                                        if p.requires_grad),
        }, indent=2), flush=True)
    step = 0
    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
                result = model(**batch)
                accelerator.backward(result["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                step += 1
                if step % args.log_every == 0 and accelerator.is_main_process:
                    print(f"epoch={epoch + 1} step={step} "
                          f"loss={result['loss'].detach().float().item():.4f} "
                          f"labels={int(result['n_labels'].item())}",
                          flush=True)
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, accelerator)
            if accelerator.is_main_process:
                print(f"epoch={epoch + 1} val_bce={val_loss:.5f}", flush=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        metadata = {
            "schema": VERSION, "base_model": args.model,
            "window_size": args.window_size, "window_stride": args.window_stride,
            "top_k": args.top_k, "head_size": args.head_size,
            "max_length": args.max_length,
            "training_target": "observable_report_warning_not_hidden_fault_truth",
            "grpo_handoff": "load adapter_model.safetensors onto original CausalLM",
        }
        accelerator.unwrap_model(model).save_pretrained(
            Path(args.output_dir), tokenizer, metadata,
        )
        print(f"saved PEFT adapter and diagnostic risk head to {args.output_dir}")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
