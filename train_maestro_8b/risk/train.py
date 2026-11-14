"""Stage 1: freeze ALL Qwen3-8B weights; train ONLY the persistent Risk Head.

Modes:
  (1) --train_file: extract frozen features online each epoch.
  (2) --features_dir: use precomputed compact frozen hidden-state features.
Neither route creates, loads, or updates a LoRA adapter.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from train_maestro_8b.data.dataset import RiskCollator, RiskDataset
from train_maestro_8b.data.features import (
    CachedRiskDataset, collate_cached, create_frozen_backbone, features_from_batch,
)
from train_maestro_8b.risk.model import WindowRiskHead
from train_maestro_8b.data.contracts import VERSION


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument("--train_file")
    sources.add_argument("--features_dir")
    p.add_argument("--eval_file")
    p.add_argument("--eval_features_dir")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_length", type=int, default=16384)
    p.add_argument("--window_size", type=int, default=128)
    p.add_argument("--window_stride", type=int, default=96)
    p.add_argument("--head_size", type=int, default=256)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--mixed_precision", choices=("no", "bf16", "fp16"), default="bf16")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def _extract(backbone, batch):
    if backbone is None:
        return batch
    device = next(backbone.parameters()).device
    packed = features_from_batch(
        backbone, {k: v.to(device) for k, v in batch.items()}
    )
    return packed


@torch.no_grad()
def evaluate(head, loader, backbone, accelerator):
    head.eval()
    totals = torch.zeros(2, device=accelerator.device)
    for batch in loader:
        out = head(**_extract(backbone, batch))
        # Equal weight per history, not per long report list.
        n = (batch["report_mask"].sum(dim=1) > 0).sum()
        totals += torch.stack([out["loss"].detach() * n, n])
    totals = accelerator.reduce(totals, reduction="sum")
    head.train()
    return (totals[0] / totals[1].clamp_min(1)).item()


def main():
    args = parse_args()
    if (min(args.epochs, args.batch_size, args.grad_accum, args.head_size,
            args.top_k, args.max_length) <= 0 or not
            1 <= args.window_stride <= args.window_size):
        raise ValueError("invalid training shape")
    if bool(args.eval_file) and args.features_dir:
        raise ValueError("use --eval_features_dir with --features_dir")
    if bool(args.eval_features_dir) and args.train_file:
        raise ValueError("use --eval_file with --train_file")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    accelerator = Accelerator(
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum,
    )
    dtype = {"no": torch.float32, "bf16": torch.bfloat16,
             "fp16": torch.float16}[args.mixed_precision]
    backbone = None
    if args.features_dir:
        train_set = CachedRiskDataset(args.features_dir)
        if (train_set.model_name != args.model
                or train_set.window_size != args.window_size
                or train_set.window_stride != args.window_stride):
            raise ValueError("cached features do not match base model or window settings")
        eval_set = CachedRiskDataset(args.eval_features_dir) if args.eval_features_dir else None
        if eval_set is not None and (
            train_set.hidden_size != eval_set.hidden_size
            or train_set.model_name != eval_set.model_name
            or train_set.window_size != eval_set.window_size
            or train_set.window_stride != eval_set.window_stride
            or train_set.quantization != eval_set.quantization
        ):
            raise ValueError("incompatible train/eval features")
        hidden_size = train_set.hidden_size
        collator = collate_cached
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        collator = RiskCollator(
            tokenizer, args.max_length, args.window_size, args.window_stride
        )
        train_set = RiskDataset(args.train_file, tokenizer, args.max_length)
        eval_set = RiskDataset(args.eval_file, tokenizer, args.max_length) if args.eval_file else None
        backbone = create_frozen_backbone(
            args.model, dtype=dtype, load_in_4bit=args.load_in_4bit,
            device=accelerator.device,
        )
        hidden_size = backbone.config.hidden_size
        if any(p.requires_grad for p in backbone.parameters()):
            raise AssertionError("stage 1 backbone must be fully frozen")
    head = WindowRiskHead(
        hidden_size, head_size=args.head_size, top_k=args.top_k
    ).to(accelerator.device)
    train = DataLoader(train_set, batch_size=args.batch_size,
                       shuffle=True, collate_fn=collator)
    val = (DataLoader(eval_set, batch_size=args.batch_size,
                      collate_fn=collator) if eval_set is not None else None)
    optimizer = AdamW(head.parameters(), lr=args.lr)
    if val is not None:
        head, optimizer, train, val = accelerator.prepare(head, optimizer, train, val)
    else:
        head, optimizer, train = accelerator.prepare(head, optimizer, train)
    steps = math.ceil(len(train) / args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(steps * args.warmup_ratio), max(1, steps)
    )
    scheduler = accelerator.prepare(scheduler)
    global_step = 0
    for epoch in range(args.epochs):
        head.train()
        for batch in train:
            with accelerator.accumulate(head):
                output = head(**_extract(backbone, batch))
                accelerator.backward(output["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.log_every == 0 and accelerator.is_main_process:
                    print(f"epoch={epoch+1} step={global_step} "
                          f"risk_loss={output['loss'].detach().item():.5f}", flush=True)
        if val is not None:
            metric = evaluate(head, val, backbone, accelerator)
            if accelerator.is_main_process:
                print(f"epoch={epoch+1} val_risk_bce={metric:.5f}", flush=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        directory = Path(args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(head)
        torch.save(unwrapped.state_dict(), directory / "risk_head.pt")
        config = {
            "schema": "maestro.frozen_window_risk.v1", "data_schema": VERSION,
            "base_model": args.model, "hidden_size": hidden_size,
            "head_size": args.head_size, "top_k": args.top_k,
            "window_size": args.window_size,
            "window_stride": args.window_stride,
            "max_length": args.max_length,
            "trained_parameters": "risk_head_only",
            "feature_model_quantization": (
                train_set.quantization if args.features_dir else
                "nf4" if args.load_in_4bit else "none"
            ),
            "target": "observable_warning_not_oracle_correctness",
        }
        (directory / "risk_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(f"saved frozen-backbone risk head to {directory}", flush=True)


if __name__ == "__main__":
    main()
