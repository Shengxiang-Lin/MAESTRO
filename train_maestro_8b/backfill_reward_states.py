"""Backfill private deterministic decision-score states from old Stage4 journals.

Replays audited logged attempts and finalized Main-visible reports; NO LLM.
Core replay lives in environment.untrusted_orchestration.decision_reward_replay
and is identical to the logic used after Stage4C language review.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from environment.untrusted_orchestration.decision_reward_replay import (
    replay_reward_states,
)
from .risk_schema import iter_jsonl

def backfill(stage4, scenes_path, *, max_parallel_tasks=4, max_retries=1,
             model_alias="model_1", output=None, max_journals=0):
    stage4 = Path(stage4)
    output = Path(output or stage4 / "private" / "decision_reward_states.jsonl")
    scenes = {row["task_id"]: row for row in iter_jsonl(scenes_path)}
    journals = sorted((stage4 / "journal").glob("*.json"))
    if max_journals:
        journals = journals[:max_journals]
    if not journals:
        raise ValueError("Stage4 journal missing; no reconstruction possible")
    result, failures, accepted = [], [], 0
    for path in journals:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "recovered" or not data.get("turns"):
            continue
        accepted += 1
        try:
            scene = scenes[data["task_id"]]
            episode = data["episode"]
            states = replay_reward_states(
                scene, episode, data["private_audit"],
                max_parallel_tasks=max_parallel_tasks, max_retries=max_retries,
                model_alias=model_alias,
            )
            by_round = {r["decision_round"]: r for r in states}
            sample_id = data.get("round_sample_id")
            if not sample_id:
                raise ValueError("accepted journal has no candidate ID")
            for turn in data["turns"]:
                snapshot = by_round.get(turn["round"])
                if snapshot is None:
                    raise ValueError("accepted Main turn missing reward snapshot")
                result.append({
                    "task_id": data["task_id"],
                    "round_sample_id": sample_id,
                    "decision_round": turn["round"],
                    "state": snapshot, "private_only": True,
                })
        except (KeyError, ValueError, TypeError) as exc:
            failures.append({"journal": path.name, "reason": str(exc)[:250]})
    if failures:
        failure_path = output.parent / "reward_backfill_failures.jsonl"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open("w", encoding="utf-8") as handle:
            for row in failures:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        raise ValueError(
            f"{len(failures)} / {accepted} accepted journals failed closed; "
            f"see {failure_path}. Original reward sidecar was NOT overwritten."
        )
    if not result:
        raise ValueError("no accepted candidate journals to backfill")
    keys = [(x["task_id"], x["round_sample_id"], x["decision_round"])
            for x in result]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate decision reward state during backfill")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in result:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(output)
    return {
        "accepted_journals": accepted, "decision_states": len(result),
        "unscorable": sum(not row["state"]["scorable"] for row in result),
        "output": str(output),
        "algorithm": "logged_attempt_replay_without_llm",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage4", default="environment/data/stage4")
    p.add_argument("--scenes", required=True,
                   help="Exact Stage1 scenes.jsonl used to build this Stage4")
    p.add_argument("--output")
    p.add_argument("--max_parallel_tasks", type=int, default=4)
    p.add_argument("--max_retries", type=int, default=1)
    p.add_argument("--model_alias", default="model_1")
    p.add_argument("--max_journals", type=int, default=0)
    args = p.parse_args()
    print(json.dumps(backfill(
        args.stage4, args.scenes,
        max_parallel_tasks=args.max_parallel_tasks,
        max_retries=args.max_retries, model_alias=args.model_alias,
        output=args.output, max_journals=args.max_journals,
    ), indent=2))


if __name__ == "__main__":
    main()
