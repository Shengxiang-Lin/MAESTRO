"""Audit deterministic Stage-2 reward quality before spending 8B GPU time.

Scores the PRIVATE reference decisions only as a sanity check; Teacher
text matching is not used as a reward. The reference may legitimately
score less than 1 when it misses legal parallel opportunities.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from environment.untrusted_orchestration.orchestration_reward import score_decision
from train_maestro_8b.data.contracts import iter_jsonl


def audit_dataset(path, output=None, max_rows=0):
    scores, failures = [], Counter()
    observations = []
    for row in iter_jsonl(path):
        result = score_decision(
            row["reference_action"], row["decision_state"]
        )
        result["sample_id"] = row["sample_id"]
        scores.append(result)
        if not result["valid"]:
            failures[result["reason"]] += 1
        observations.append(result)
        if max_rows and len(scores) >= max_rows:
            break
    if not scores:
        raise ValueError("empty reward inspection dataset")
    columns = ("reward", "coverage", "priority", "parallel", "nonredundancy")
    summary = {
        "samples": len(scores),
        "reference_valid": sum(row["valid"] for row in scores),
        "reference_invalid": sum(not row["valid"] for row in scores),
        "reference_invalid_reasons": dict(failures),
        "reference_mean_components": {
            key: round(mean(row[key] for row in scores), 5)
            for key in columns
        },
        "interpretation": (
            "Sanity check of binding/available choices ONLY; not task "
            "success nor an oracle optimality claim"
        ),
    }
    if output is not None:
        file = Path(output)
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("w", encoding="utf-8") as handle:
            for row in observations:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_file", required=True)
    p.add_argument("--output_jsonl")
    p.add_argument("--max_rows", type=int, default=0)
    args = p.parse_args()
    print(json.dumps(audit_dataset(
        args.train_file, args.output_jsonl, args.max_rows
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
