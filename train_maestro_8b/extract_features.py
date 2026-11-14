"""Cache only frozen 8B window/global features for fast repeated Risk Head fitting."""
from __future__ import annotations
import argparse
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .risk_data import RiskCollator, RiskDataset
from .risk_features import create_frozen_backbone, write_feature_cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_length", type=int, default=16384)
    p.add_argument("--window_size", type=int, default=128)
    p.add_argument("--window_stride", type=int, default=96)
    p.add_argument("--shard_size", type=int, default=128)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = p.parse_args()
    if args.shard_size < 1:
        raise ValueError("shard_size must be positive")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ds = RiskDataset(args.input, tok, args.max_length)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        collate_fn=RiskCollator(tok, args.max_length,
                              args.window_size, args.window_stride),
    )
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_frozen_backbone(
        args.model, dtype=dtype, load_in_4bit=args.load_in_4bit,
        device=device,
    )
    manifest = write_feature_cache(
        model, loader, args.output_dir, model_name=args.model,
        window_size=args.window_size, window_stride=args.window_stride,
        max_length=args.max_length, shard_size=args.shard_size,
        quantization="nf4" if args.load_in_4bit else "none",
    )
    print(json.dumps({**manifest, "overlong_skipped": ds.overlong}, indent=2))


if __name__ == "__main__":
    main()
