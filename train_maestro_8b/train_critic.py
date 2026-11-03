"""Stage-2 MAESTRO training: consequence critic on sparse macro values."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader

from train_maestro_8b.data import CriticCollator, CriticDataset
from train_maestro_8b.modeling import MaestroOrchestratorModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Stage-1 checkpoint directory")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=16384)
    parser.add_argument("--lambda_critic", type=float, default=1.0)
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=0)
    parser.add_argument("--mixed_precision", choices=["no", "bf16", "fp16"], default="bf16")
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
        lambda_critic=args.lambda_critic,
    )
    model.freeze_for_critic(
        unfreeze_last_n_layers=args.unfreeze_last_n_layers
    )

    dataset = CriticDataset(args.train_file)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=CriticCollator(tokenizer, max_length=args.max_length),
    )
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        for batch in loader:
            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output["loss"]
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
                if (
                    accelerator.is_main_process
                    and global_step % max(1, args.log_every) == 0
                ):
                    print(
                        f"step={global_step} "
                        f"critic_loss={float(output['critic_loss'].detach()):.4f}"
                    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_heads(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(
            "saved critic/macro heads only; reuse the Stage-1 model weights with "
            f"{output_dir / 'maestro_heads.pt'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
