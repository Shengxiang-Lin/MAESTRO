"""Join accepted Stage-4 Main decisions with risk-window historical inputs.

References are private training labels, never appended to the model prompt.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from .risk_schema import iter_jsonl, validate


def reference_key(turn):
    task, decision_round = turn["task_id"], turn["round"]
    match = re.fullmatch(
        re.escape(task) + r"-branch-(\d+)-round-(\d+)",
        turn.get("sample_id", ""),
    )
    if match is None or int(match.group(2)) != decision_round:
        raise ValueError("unrecognized accepted Stage-4 Main turn ID")
    return task, int(match.group(1)), int(decision_round)


def export(risk_dir, main_turns, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reference = {}
    for turn in iter_jsonl(main_turns):
        key = reference_key(turn)
        if key in reference:
            raise ValueError("duplicate reference decision")
        action = turn["output"]
        if action.get("action") not in ("complete", "delegate_task"):
            raise ValueError("invalid reference action")
        reference[key] = action
    stats = {"reference_turns": len(reference), "train": 0, "val": 0,
             "missing_reference": 0}
    for split in ("train", "val"):
        path = Path(risk_dir) / (split + ".jsonl")
        if not path.exists():
            if split == "val":
                continue
            raise FileNotFoundError(path)
        with (out / (split + ".jsonl")).open("w", encoding="utf-8") as dest:
            for sample in iter_jsonl(path):
                validate(sample)
                key = (sample["scene_id"], sample["branch_round"], sample["decision_round"])
                if key not in reference:
                    stats["missing_reference"] += 1
                    continue
                row = {
                    "schema": "maestro.risk_grpo.v1",
                    "sample_id": sample["sample_id"], "scene_id": sample["scene_id"],
                    "prompt": sample["prompt"],
                    "reports": [
                        {"start": r["start"], "end": r["end"], "report_id": r["report_id"]}
                        for r in sample["reports"]
                    ],
                    "reference_action": reference[key],
                }
                dest.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats[split] += 1
    (out / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk_dir", default="environment/data/stage4/private/window_risk")
    parser.add_argument("--main_turns", default="environment/data/stage4/main_turns.jsonl")
    parser.add_argument("--output_dir", default="environment/data/stage4/private/risk_grpo")
    args = parser.parse_args()
    print(json.dumps(export(args.risk_dir, args.main_turns, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
