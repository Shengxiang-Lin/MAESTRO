#!/usr/bin/env python3
"""Evaluate a trained DA-SFT or DA-GRPO PEFT LoRA through the real O1 vLLM client.

This is a held-out *single-decision* sanity/evaluation check. It does not
execute Sub-Agents or claim a multi-round task success rate.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_NAMES = {
    "da-sft": "qwen3-8b-da-sft-lora",
    "da-grpo": "qwen3-8b-da-grpo-lora",
}


def get_reference(row: dict, method: str) -> dict:
    if method == "da-sft":
        return json.loads(row["completion"][0]["content"])
    info = row["extra_info"]
    return {"action": info["expert_action"], "params": info["expert_params"]}


def valid_o1_action(action: object) -> bool:
    if not isinstance(action, dict) or set(action) != {"action", "reasoning", "params"}:
        return False
    if not isinstance(action["reasoning"], str) or not action["reasoning"].strip():
        return False
    params = action["params"]
    if not isinstance(params, dict):
        return False
    if action["action"] == "delegate_task":
        return isinstance(params.get("tasks"), list) and bool(params["tasks"])
    if action["action"] == "complete":
        return isinstance(params.get("answer"), str)
    return False


def load_test_rows(path: Path, method: str, limit: int) -> list[dict]:
    import pyarrow.parquet as pq

    if not path.is_file():
        raise FileNotFoundError(f"Missing {method} held-out test data: {path}")
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError("Empty test dataset")
    chosen = rows if limit == 0 else rows[:limit]
    for index, row in enumerate(chosen):
        prompt = row.get("prompt")
        if (not isinstance(prompt, list) or len(prompt) != 1
                or prompt[0].get("role") != "user"
                or not isinstance(prompt[0].get("content"), str)):
            raise ValueError(f"Invalid single-turn O1 prompt at test row {index}")
        expert = get_reference(row, method)
        if expert.get("action") not in ("delegate_task", "complete"):
            raise ValueError(f"Invalid expert action at test row {index}")
    return chosen


def configure_model(method: str, base: Path, adapter: Path,
                    gpu: int | None, port: int | None):
    from base.engine.async_llm import LLMsConfig

    registry = LLMsConfig.default()
    name = MODEL_NAMES[method]
    config = dict(registry.configs[name])
    config["model_path"] = str(base)
    config["lora_adapter_path"] = str(adapter)
    if gpu is not None:
        config["gpus"] = [gpu]
    if port is not None:
        config["port"] = port
        config.pop("base_url", None)
    if (config.get("extra_body") or {}).get("chat_template_kwargs", {}).get(
        "enable_thinking"
    ) is not True or config.get("reasoning_parser") != "qwen3":
        raise ValueError("DA LoRA evaluation requires Qwen3 thinking+reasoning parser")
    registry.add_config(name, config)
    return name, registry


async def run(args) -> dict:
    from base.engine.async_llm import create_llm_instance
    from orchestra.common.utils import parse_json_response
    from tqdm.auto import tqdm

    name, registry = configure_model(
        args.method, args.base_model, args.adapter, args.gpu, args.port,
    )
    rows = load_test_rows(args.test_parquet, args.method, args.max_samples)
    llm = create_llm_instance(registry.get(name))
    results = []
    for index, row in enumerate(tqdm(rows, desc=f"Evaluate {args.method}",
                                      unit="decision")):
        expert = get_reference(row, args.method)
        prediction, error, raw = None, None, None
        try:
            raw = await llm(
                row["prompt"][0]["content"], max_tokens=args.max_tokens,
                temperature=args.temperature, seed=args.seed,
            )
            prediction = parse_json_response(raw)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        valid = valid_o1_action(prediction)
        results.append({
            "index": index,
            "expert_action": expert["action"],
            "predicted_action": prediction.get("action") if valid else None,
            "parseable_json": isinstance(prediction, dict),
            "valid_o1_action": valid,
            "action_match": bool(valid and prediction["action"] == expert["action"]),
            "error": error,
            "response": raw,
        })
    total = len(results)
    summary = {
        "model": name, "method": args.method, "split": "test",
        "samples": total,
        "parseable_json": sum(x["parseable_json"] for x in results),
        "valid_o1_actions": sum(x["valid_o1_action"] for x in results),
        "action_matches": sum(x["action_match"] for x in results),
        "json_rate": sum(x["parseable_json"] for x in results) / total,
        "valid_action_rate": sum(x["valid_o1_action"] for x in results) / total,
        "action_accuracy": sum(x["action_match"] for x in results) / total,
        "enable_thinking": True,
        "note": "Offline one-step action agreement, NOT multi-round task success",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "results": results},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved evaluation: {args.output}")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(MODEL_NAMES), required=True)
    parser.add_argument("--base-model", type=Path, default=ROOT / "base_models/Qwen3-8B")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--test-parquet", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=10,
                        help="0 evaluates all test decisions")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, help="Override config/model_config.yaml GPU")
    parser.add_argument("--port", type=int, help="Override vLLM service port")
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect inputs without starting vLLM")
    args = parser.parse_args(argv)
    if args.max_samples < 0 or args.max_tokens <= 0 or args.temperature < 0:
        parser.error("Invalid sample, token or temperature limit")
    if args.gpu is not None and args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("--port out of range")
    if args.adapter is None:
        args.adapter = ROOT / f"baseline/{args.method}/output/final_adapter"
    if args.test_parquet is None:
        args.test_parquet = ROOT / f"baseline/{args.method}/data/test.parquet"
    if args.output is None:
        args.output = ROOT / f"baseline/{args.method}/output/eval_lora_test.json"
    for field in ("base_model", "adapter", "test_parquet", "output"):
        value = getattr(args, field)
        setattr(args, field, value.expanduser().resolve())
    if args.dry_run:
        print(json.dumps({
            "method": args.method, "model": MODEL_NAMES[args.method],
            "base": str(args.base_model), "adapter": str(args.adapter),
            "test_parquet": str(args.test_parquet),
            "output": str(args.output), "max_samples": args.max_samples,
            "max_tokens": args.max_tokens, "thinking": True,
        }, indent=2))
        return 0
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
