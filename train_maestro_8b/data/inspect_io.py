"""Inspect EXACT Stage-2 model-visible input and reference Action output.

No GPU, no private reward-state leakage, no hidden O1 prompt rewriting.
MAESTRO consumes a RAW full O1 prompt string via tokenizer(...,
add_special_tokens=False). DA-GRPO consumes a chat-template-wrapped
one-message user prompt. Both are counted to expose the difference.

The JSON report contains lengths and small visible-text previews only.
Full original prompts/references remain in the existing private train
JSONL. The future generated outputs appear in reward_debug_jsonl.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from train_maestro_8b.data.contracts import iter_jsonl


def _stats(values):
    if not values:
        return {"count": 0}
    a = sorted(values)
    def quantile(p):
        return a[int((len(a)-1)*p)]
    return {
        "count": len(a), "min": a[0], "p50": quantile(.50),
        "p90": quantile(.90), "p95": quantile(.95),
        "p99": quantile(.99), "max": a[-1],
        "mean": round(statistics.mean(a), 2),
    }


def _ids(x):
    if isinstance(x, dict) and "input_ids" in x:
        return _ids(x["input_ids"])
    if not isinstance(x, list):
        raise TypeError("Tokenizer returned unexpected non-list token ids")
    if x and isinstance(x[0], list):
        if len(x) != 1:
            raise ValueError("Expected one input")
        return x[0]
    return x


def inspect(records, tokenizer, *, max_prompt_tokens=12288,
            max_completion_tokens=4096, num_generations=8,
            previews=2, preview_chars=500):
    prompt_lengths, chat_lengths, reference_lengths = [], [], []
    action_types = Counter()
    overlong, over_reference, examples = [], [], []
    for row in records:
        text = row.get("prompt")
        action = row.get("reference_action")
        if not isinstance(text, str) or not text:
            raise ValueError("Missing model-visible O1 prompt")
        if not isinstance(action, dict):
            raise ValueError("Missing separate reference_action")
        if "decision_state" not in row or "reports" not in row:
            raise ValueError("Missing private scoring or report-span metadata")
        # Matches RiskAwarePolicy.prompt_windows; NEVER truncate.
        raw = _ids(tokenizer(text, add_special_tokens=False,
                             truncation=False)["input_ids"])
        chat = _ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True,
        ))
        target = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
        reference = _ids(tokenizer(target, add_special_tokens=False)["input_ids"])
        prompt_lengths.append(len(raw))
        chat_lengths.append(len(chat))
        reference_lengths.append(len(reference))
        action_types[action.get("action", "unknown")] += 1
        if len(raw) > max_prompt_tokens:
            overlong.append({"sample_id": row.get("sample_id"), "raw_tokens": len(raw)})
        if len(reference) > max_completion_tokens:
            over_reference.append({"sample_id": row.get("sample_id"),
                                   "reference_tokens": len(reference)})
        if len(examples) < previews:
            examples.append({
                "sample_id": row.get("sample_id"),
                "report_spans": len(row["reports"]),
                "raw_prompt_tokens": len(raw),
                "da_chat_wrapped_tokens": len(chat),
                "reference_action_tokens": len(reference),
                "action_type": action.get("action"),
                "prompt_head": text[:preview_chars],
                "prompt_tail": text[-preview_chars:],
                # Only reference action (not future/private scorer state).
                "reference_action_preview": target[:preview_chars],
            })
    anchor = _ids(tokenizer("\n", add_special_tokens=False)["input_ids"])
    return {
        "schema": "maestro.grpo_io_audit.v1",
        "model_input_format": "raw O1 prompt string (no chat template)",
        "baseline_input_format": "one user message with Qwen chat template",
        "sample_count": len(prompt_lengths),
        "reference_action_types": dict(action_types),
        "prompt_raw_tokens": _stats(prompt_lengths),
        "prompt_da_chat_template_tokens": _stats(chat_lengths),
        "reference_action_tokens": _stats(reference_lengths),
        "candidate_group_size": num_generations,
        "completion_token_cap": max_completion_tokens,
        "raw_prompt_token_cap": max_prompt_tokens,
        "riskkv_anchor_token_count": len(anchor),
        "notes": (
            "DA-GRPO uses chat-template-wrapped prompts, BF16 LoRA and shared "
            "vLLM. MAESTRO uses raw O1 strings, NF4 training + custom "
            "RiskKV and optionally a separate vLLM engine. Matching budget "
            "does not imply matching memory, throughput, or model likelihood."
        ),
        "overlong_prompt_count": len(overlong),
        "overlong_prompt_examples": overlong[:10],
        "overlong_reference_count": len(over_reference),
        "overlong_reference_examples": over_reference[:10],
        "examples": examples,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--max_prompt_tokens", type=int, default=12288)
    p.add_argument("--max_completion_tokens", type=int, default=4096)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--previews", type=int, default=2)
    p.add_argument("--preview_chars", type=int, default=500)
    p.add_argument("--out")
    args = p.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    result = inspect(
        iter_jsonl(args.data), tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_completion_tokens=args.max_completion_tokens,
        num_generations=args.num_generations,
        previews=args.previews, preview_chars=args.preview_chars,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    if result["overlong_prompt_count"]:
        raise SystemExit(
            f"{result['overlong_prompt_count']} O1 prompts exceed "
            f"{args.max_prompt_tokens} tokens; no silent truncation allowed. "
            "Inspect the report and explicitly choose a larger budget "
            "or filter these examples."
        )


if __name__ == "__main__":
    main()
