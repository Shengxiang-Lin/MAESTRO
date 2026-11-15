"""Audit recent RiskKV GRPO candidates and explain why no gradient was obtained.

Uses only recorded candidate text and public scorer diagnostics. Does NOT load
private reference states, model weights, GPUs, or regenerate trajectories.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path


def summarize_rollouts(path, *, tail=8, preview_chars=200, token_cap=0):
    if tail < 1 or preview_chars < 0 or token_cap < 0:
        raise ValueError("invalid audit parameters")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = deque(maxlen=tail)
    with source.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("empty rollout log")
    reasons = Counter()
    rewards = Counter()
    summaries = []
    for i, row in enumerate(rows, 1):
        info = row.get("components") or {}
        reason = info.get("reason") or ("valid" if info.get("valid") else "unknown")
        reward = row.get("reward")
        if not isinstance(reward, (int, float)):
            raise ValueError("rollout has no numeric reward")
        text = row.get("candidate")
        if not isinstance(text, str):
            raise ValueError("rollout has no generated text")
        tokens = row.get("completion_tokens")
        if not isinstance(tokens, int) or tokens < 1:
            raise ValueError("rollout lacks completion_tokens")
        reasons[reason] += 1
        rewards[str(reward)] += 1
        summaries.append({
            "candidate_index": i,
            "sample_id": row.get("sample_id"),
            "step": row.get("step"),
            "reward": reward,
            "valid": bool(info.get("valid")),
            "reason": reason,
            "completion_tokens": tokens,
            "hit_token_cap": tokens == token_cap if token_cap else None,
            "candidate_start": text[:preview_chars],
            "candidate_end": text[-preview_chars:] if preview_chars else "",
        })
    return {
        "inspected_candidates": len(summaries),
        "reward_distribution": dict(rewards),
        "reason_distribution": dict(reasons),
        "unique_rewards": len(rewards),
        "token_cap": token_cap or None,
        "candidates": summaries,
        "note": (
            "The log may contain several different pilots; use --tail with "
            "the most recent G candidates. Long generations can be clipped "
            "before a complete JSON Action is emitted. A valid_json failure "
            "is distinct from evidence binding failures."
        ),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--tail", type=int, default=8)
    p.add_argument("--preview_chars", type=int, default=180)
    p.add_argument("--token_cap", type=int, default=0)
    args = p.parse_args(argv)
    print(json.dumps(summarize_rollouts(
        args.input, tail=args.tail,
        preview_chars=args.preview_chars, token_cap=args.token_cap,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
