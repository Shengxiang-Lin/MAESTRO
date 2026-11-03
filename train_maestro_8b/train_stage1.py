"""Stage-1 MAESTRO training: O1 JSON generation + macro supervision."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from train_maestro_8b.data import Stage1Collator, Stage1Dataset
from train_maestro_8b.modeling import MaestroOrchestratorModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_length", type=int, default=16384)
    parser.add_argument("--lambda_macro", type=float, default=0.25)
    parser.add_argument("--mixed_precision", choices=["no", "bf16", "fp16"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )
    dtype = None
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16

    model, tokenizer = MaestroOrchestratorModel.from_pretrained(
        args.model,
        torch_dtype=dtype,
        lambda_macro=args.lambda_macro,
    )
    if args.gradient_checkpointing:
        model.base_model.gradient_checkpointing_enable()
        if hasattr(model.base_model.config, "use_cache"):
            model.base_model.config.use_cache = False

    dataset = Stage1Dataset(args.train_file)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=Stage1Collator(tokenizer, max_length=args.max_length),
    )
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    updates_per_epoch = math.ceil(len(loader) / max(1, args.grad_accum))
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        for batch in loader:
            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output["loss"]
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
                if (
                    accelerator.is_main_process
                    and global_step % max(1, args.log_every) == 0
                ):
                    lm = output["lm_loss"]
                    macro = output["macro_loss"]
                    print(
                        f"step={global_step} "
                        f"loss={float(loss.detach()):.4f} "
                        f"lm={float(lm.detach()) if lm is not None else float('nan'):.4f} "
                        f"macro={float(macro.detach()) if macro is not None else float('nan'):.4f}"
                    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir, tokenizer=tokenizer)
        print(f"saved Stage-1 checkpoint to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
