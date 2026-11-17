"""Stage1 Evidence Belief Pilot: train Qwen3-8B LoRA AND one auxiliary head.

The original Qwen3 backbone stays frozen. Unlike the legacy Risk Head,
loss flows through the live LoRA-enabled backbone; no frozen-feature cache.
Only the LoRA adapter is carried forward to Stage2 GRPO.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader

from train_maestro_8b.data.contracts import iter_jsonl, span_to_tokens
from train_maestro_8b.data.export_belief import (
    CLASSES, UNKNOWN, validate_belief, SCHEMA,
)
from train_maestro_8b.belief.model import EvidenceBeliefHead, report_embeddings

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def chat_wrapped_prompt(tokenizer, prompt: str) -> tuple[str, int]:
    """Match Stage2's chat-template input; keep exact original report offsets."""
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Stage1 requires Qwen3 chat-template-compatible tokenizer")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise ValueError("Chat template did not render text")
    if rendered.count(prompt) != 1:
        raise ValueError("O1 prompt not uniquely embedded by chat template")
    return rendered, rendered.index(prompt)


class BeliefDataset(Dataset):
    """Index JSONL offsets; fail closed for malformed/misaligned examples."""

    def __init__(self, source, tokenizer, max_length: int):
        if max_length < 1:
            raise ValueError("Invalid max_length")
        self.path = Path(source)
        self.offsets = []
        self.overlong = 0
        self.class_counts = Counter()
        self.token_lengths = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                record = validate_belief(json.loads(line))
                prompt, _ = chat_wrapped_prompt(tokenizer, record["prompt"])
                tokens = tokenizer.encode(prompt, add_special_tokens=False)
                if len(tokens) > max_length:
                    self.overlong += 1
                    continue  # Never truncate the earlier evidence.
                self.offsets.append(offset)
                self.token_lengths.append(len(tokens))
                self.class_counts.update(
                    report["belief_label"] for report in record["reports"]
                    if report["belief_label"] != UNKNOWN
                )

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return validate_belief(json.loads(handle.readline()))


class BeliefCollator:
    def __init__(self, tokenizer, max_length):
        if not tokenizer.is_fast:
            raise ValueError("Fast tokenizer is required for report span alignment")
        if tokenizer.pad_token_id is None:
            raise ValueError("A pad token is required")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        parsed = []
        for row in batch:
            prompt, offset = chat_wrapped_prompt(self.tokenizer, row["prompt"])
            enc = self.tokenizer(
                prompt, add_special_tokens=False, return_offsets_mapping=True,
                truncation=False,
            )
            if len(enc["input_ids"]) > self.max_length:
                raise ValueError("Overlong Stage1 example; never truncate evidence")
            spans = [
                span_to_tokens(enc["offset_mapping"],
                               r["start"] + offset, r["end"] + offset)
                for r in row["reports"]
            ]
            parsed.append((enc["input_ids"], spans, row["reports"]))
        bsz = len(parsed)
        length = max(len(ids) for ids, _, _ in parsed)
        n_reports = max(len(spans) for _, spans, _ in parsed)
        pad = self.tokenizer.pad_token_id
        ids = torch.full((bsz, length), pad, dtype=torch.long)
        attention = torch.zeros_like(ids)
        starts = torch.zeros((bsz, n_reports), dtype=torch.long)
        ends = torch.zeros_like(starts)
        seen = torch.zeros((bsz, n_reports), dtype=torch.bool)
        state_labels = torch.full((bsz, n_reports), UNKNOWN, dtype=torch.long)
        update_labels = torch.full_like(state_labels, UNKNOWN)
        for b, (tokens, spans, reports) in enumerate(parsed):
            ids[b, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            attention[b, :len(tokens)] = 1
            for j, ((start, end), report) in enumerate(zip(spans, reports)):
                starts[b, j], ends[b, j] = start, end
                seen[b, j] = True
                state_labels[b, j] = report["belief_label"]
                update_labels[b, j] = report["update_label"]
        return {
            "input_ids": ids, "attention_mask": attention,
            "report_starts": starts, "report_ends": ends, "report_mask": seen,
            "state_labels": state_labels, "update_labels": update_labels,
        }


def safe_attention_context(attention_backend: str):
    """Fail closed instead of using quadratic-memory math SDPA at long length."""
    if attention_backend != "sdpa":
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION])


def check_checkpointing(backbone):
    """Verify checkpointing on the decoder ACTUALLY called by Stage1."""
    enabled = [
        module for module in backbone.modules()
        if getattr(module, "gradient_checkpointing", False)
    ]
    if not enabled:
        raise RuntimeError(
            "No gradient-checkpointed Qwen decoder modules found in "
            "the live LoRA backbone. Long-context training would OOM."
        )
    return len(enabled)


def make_lora_backbone(model_name: str, *, load_in_4bit: bool,
                       rank: int, alpha: int, max_length: int,
                       attention_backend: str = "sdpa"):
    """PEFT CausalLM architecture matches the Stage2 LoRA target modules."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if attention_backend not in ("sdpa", "flash_attention_2"):
        raise ValueError("Unsupported attention backend")
    kwargs = {"dtype": torch.bfloat16,
              "attn_implementation": attention_backend}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    base = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    base.config.use_cache = False
    if load_in_4bit:
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=True,
        )
    else:
        base.requires_grad_(False)
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    config = LoraConfig(
        task_type="CAUSAL_LM", r=rank, lora_alpha=alpha,
        lora_dropout=0.0, bias="none", target_modules=TARGET_MODULES,
    )
    model = get_peft_model(base, config)
    # Stage1 calls the inner decoder directly, bypassing CausalLM.forward.
    # Verify that gradient checkpointing was propagated after PEFT wrapping.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if not load_in_4bit:
        model.to("cuda")
    # Avoid computing giant vocabulary logits in belief-only Stage1.
    # The LoRA projections are injected INTO Qwen's underlying model.
    backbone = model.get_base_model().model
    if not hasattr(backbone, "forward"):
        raise RuntimeError("Cannot access LoRA-enabled Qwen3 base model")
    count = check_checkpointing(backbone)
    print(f"[MAESTRO/Stage1] gradient_checkpointing_enabled: {count} modules",
          flush=True)
    return model, backbone


def belief_forward(backbone, head, batch, *, device,
                   state_weight=None, update_loss_weight=0.3):
    output = backbone(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        use_cache=False, return_dict=True,
    )
    if not hasattr(output, "last_hidden_state"):
        raise RuntimeError("Belief backbone must expose last_hidden_state")
    local, global_ = report_embeddings(
        output.last_hidden_state, batch["attention_mask"].to(device),
        batch["report_starts"], batch["report_ends"], batch["report_mask"],
    )
    return head(
        local, global_, batch["report_mask"].to(device),
        state_labels=batch["state_labels"].to(device),
        update_labels=batch["update_labels"].to(device),
        state_weight=state_weight,
        update_loss_weight=update_loss_weight,
    )


def inverse_sqrt_class_weights(counts: Counter, device):
    # Unseen classes get neutral weight for validation, not zero: otherwise
    # validation batches containing a class absent from train can yield NaN.
    raw = [1.0 / math.sqrt(counts[i]) if counts[i] else 1.0
           for i in range(len(CLASSES))]
    active = [v for v in raw if v > 0]
    if not active:
        raise ValueError("No observed Stage1 belief classes")
    scale = len(active) / sum(active)
    return torch.tensor([x * scale for x in raw], device=device)


@torch.no_grad()
def evaluate(backbone, head, loader, *, device, state_weight,
             update_loss_weight: float):
    head.eval()
    confusion = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.long)
    updates = torch.zeros((2, 2), dtype=torch.long)
    weighted = 0.0
    count = 0
    for batch in loader:
        result = belief_forward(backbone, head, batch, device=device,
                                state_weight=state_weight,
                                update_loss_weight=update_loss_weight)
        weighted += float(result["loss"]) * result["state_count"]
        count += result["state_count"]
        valid = batch["state_labels"] != UNKNOWN
        predicted = result["state_logits"].argmax(-1).cpu()
        for y, p in zip(batch["state_labels"][valid], predicted[valid]):
            confusion[int(y), int(p)] += 1
        active = batch["update_labels"] != UNKNOWN
        guesses = result["update_logits"].argmax(-1).cpu()
        for y, p in zip(batch["update_labels"][active], guesses[active]):
            updates[int(y), int(p)] += 1
    head.train()
    if not count:
        raise ValueError("Empty supervised Stage1 validation")
    correct = int(confusion.diagonal().sum())
    f1s = []
    per_class = {}
    for i, label in enumerate(CLASSES):
        tp = int(confusion[i, i])
        support = int(confusion[i].sum())
        fp = int(confusion[:, i].sum()) - tp
        fn = support - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        per_class[label] = {"support": support, "f1": round(f1, 5)}
        if support:
            f1s.append(f1)
    n_updates = int(updates.sum())
    return {
        "loss": round(weighted / count, 6),
        "state_accuracy": round(correct / count, 6),
        "macro_f1_present_classes": round(sum(f1s) / max(1, len(f1s)), 6),
        "state_count": count,
        "per_class": per_class,
        "update_accuracy": (round(int(updates.diagonal().sum()) / n_updates, 6)
                            if n_updates else None),
        "update_count": n_updates,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="base_models/Qwen3-8B")
    p.add_argument("--train_file", default="environment/data/stage4/private/evidence_belief/train.jsonl")
    p.add_argument("--val_file", default="environment/data/stage4/private/evidence_belief/val.jsonl")
    p.add_argument("--output_dir", default="workspace/maestro_8b/checkpoints/belief-stage1-pilot")
    p.add_argument("--max_length", type=int, default=4096,
                   help="No-truncation context max; 8192 only after GPU pilot.")
    p.add_argument("--attention_backend", choices=("sdpa", "flash_attention_2"),
                   default="sdpa", help="SDPA math fallback disabled.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--head_lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--update_loss_weight", type=float, default=0.3)
    p.add_argument("--head_size", type=int, default=256)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--bf16_full", action="store_false", dest="load_in_4bit")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--no_class_balance", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if (min(args.max_length, args.epochs, args.batch_size, args.grad_accum,
            args.head_size, args.lora_rank, args.lora_alpha, args.log_every) < 1
            or args.max_steps == 0 or args.lr <= 0 or args.head_lr <= 0
            or not 0 <= args.update_loss_weight <= 5):
        raise ValueError("Invalid Stage1 belief training configuration")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage1 requires a BF16 capable CUDA GPU")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_set = BeliefDataset(args.train_file, tokenizer, args.max_length)
    val_set = (BeliefDataset(args.val_file, tokenizer, args.max_length)
               if Path(args.val_file).is_file() else None)
    if not len(train_set):
        raise ValueError("No Stage1 belief rows within context length")
    if val_set is not None and not len(val_set):
        val_set = None
    lengths = sorted(train_set.token_lengths)
    print("[MAESTRO/Stage1] data " + json.dumps({
        "schema": SCHEMA, "train": len(train_set),
        "validation": len(val_set) if val_set else 0,
        "train_overlong": train_set.overlong,
        "val_overlong": val_set.overlong if val_set else 0,
        "train_class_counts": dict(train_set.class_counts),
        "token_length_p50": lengths[len(lengths) // 2],
        "token_length_p90": lengths[min(len(lengths) - 1, int(len(lengths) * 0.9))],
        "token_length_max": lengths[-1],
    }), flush=True)
    missing = [CLASSES[i] for i in (1, 2) if not train_set.class_counts[i]]
    if missing:
        print("[MAESTRO/Stage1] WARNING: no examples for " +
              ", ".join(missing) +
              "; this dataset cannot test independent corroboration or "
              "disputed beliefs. Inspect and diversify Stage4 evidence.",
              flush=True)
    collator = BeliefCollator(tokenizer, args.max_length)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collator,
    )
    val_loader = (DataLoader(val_set, batch_size=1, shuffle=False,
                             collate_fn=collator) if val_set else None)
    policy, backbone = make_lora_backbone(
        args.model, load_in_4bit=args.load_in_4bit,
        rank=args.lora_rank, alpha=args.lora_alpha, max_length=args.max_length,
        attention_backend=args.attention_backend,
    )
    device = next(backbone.parameters()).device
    if torch.cuda.is_available():
        allocated = round(torch.cuda.memory_allocated() / 2**30, 2)
        reserved = round(torch.cuda.memory_reserved() / 2**30, 2)
        print(f"[MAESTRO/Stage1] GPU after load: allocated={allocated}GiB "
              f"reserved={reserved}GiB; attention={args.attention_backend}",
              flush=True)
        torch.cuda.reset_peak_memory_stats()
    head = EvidenceBeliefHead(
        policy.config.hidden_size, head_size=args.head_size
    ).to(device)
    lora_params = [p for p in policy.parameters() if p.requires_grad]
    if not lora_params or any("lora_" not in name for name, param in
                              policy.named_parameters() if param.requires_grad):
        raise RuntimeError("Stage1 must update ONLY LoRA model parameters")
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr},
        {"params": list(head.parameters()), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    max_steps = (
        args.max_steps if args.max_steps > 0 else
        math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(max_steps * 0.03)),
        num_training_steps=max_steps,
    )
    class_weight = (None if args.no_class_balance else
                    inverse_sqrt_class_weights(train_set.class_counts, device))
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Nonempty output directory: choose a fresh Stage1 checkpoint path")
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    steps = 0
    accumulation = 0
    optimizer.zero_grad(set_to_none=True)
    head.train()
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_loader):
            try:
                with safe_attention_context(args.attention_backend):
                    result = belief_forward(
                        backbone, head, batch, device=device,
                        state_weight=class_weight,
                        update_loss_weight=args.update_loss_weight,
                    )
                    (result["loss"] / args.grad_accum).backward()
            except torch.OutOfMemoryError:
                print("[MAESTRO/Stage1] CUDA OOM: try BELIEF_MAX_LENGTH=2048 "
                      "or 4096, check nvidia-smi, or use "
                      "--attention_backend flash_attention_2 if supported. "
                      "No prompt truncation is used.", flush=True)
                raise
            accumulation += 1
            if accumulation == args.grad_accum or i + 1 == len(train_loader):
                # If no LoRA gradient exists, Head-only fitting is a bug.
                if steps == 0 and not any(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    and p.grad.abs().sum() > 0 for p in lora_params
                ):
                    raise RuntimeError("No Stage1 LoRA gradient: auxiliary head disconnected")
                torch.nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                accumulation = 0
                if steps % args.log_every == 0 or steps == 1:
                    print("[MAESTRO/Stage1] " + json.dumps({
                        "epoch": epoch + 1, "step": steps,
                        "loss": round(float(result["loss"].detach()), 6),
                        "state_loss": round(float(result["state_loss"].detach()), 6),
                        "update_loss": round(float(result["update_loss"].detach()), 6),
                        "n_states": result["state_count"],
                        "n_updates": result["update_count"],
                        "gpu_peak_allocated_GiB": round(
                            torch.cuda.max_memory_allocated() / 2**30, 2),
                    }), flush=True)
                if steps >= max_steps:
                    break
        if val_loader is not None:
            with safe_attention_context(args.attention_backend):
                score = evaluate(backbone, head, val_loader, device=device,
                                 state_weight=class_weight,
                                 update_loss_weight=args.update_loss_weight)
            metrics.append({"epoch": epoch + 1, "step": steps, **score})
            print("[MAESTRO/Stage1] validation: " +
                  json.dumps(metrics[-1], ensure_ascii=False), flush=True)
        if steps >= max_steps:
            break
    # Save the transferable Peft LoRA AND separate probe weights.
    adapter_dir = output / "adapter"
    policy.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    torch.save(head.state_dict(), output / "belief_head.pt")
    manifest = {
        "schema": "maestro.stage1_belief_checkpoint.v1",
        "base_model": args.model, "data_schema": SCHEMA,
        "adapter": str(adapter_dir), "head": "belief_head.pt",
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "lora_target_modules": TARGET_MODULES,
        "head_size": args.head_size, "classes": list(CLASSES),
        "max_length": args.max_length,
        "attention_backend": args.attention_backend,
        "trained_parameters": "lora_plus_head",
        "quantization": "nf4" if args.load_in_4bit else "bf16",
        "steps": steps, "train_samples": len(train_set),
        "val_samples": len(val_set) if val_set else 0,
        "validation": metrics,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print("[MAESTRO/Stage1] checkpoint " + str(adapter_dir), flush=True)


if __name__ == "__main__":
    main()
