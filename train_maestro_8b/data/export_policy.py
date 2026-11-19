"""Join public Stage-4 prompts with private deterministic decision reward states.

References remain optional teacher-warmup labels, NEVER the default reward
and NEVER part of the student-visible prompt.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from train_maestro_8b.data.contracts import iter_jsonl, validate


def reference_key(turn):
    """Join Stage4 reference actions with risk/reward state by stable identity.

    ``sample_id`` is a display/trajectory identifier whose formatting is not
    a data contract: Stage4 may produce branch-round, fault-candidate, or
    reviewed-turn variants. The publisher already exports the exact
    ``round_sample_id`` candidate key on every accepted turn, which is also
    used by risk labels and private reward states.
    """
    task = turn.get("task_id")
    decision_round = turn.get("round")
    scenario_id = turn.get("round_sample_id")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Stage4 Main turn lacks a valid task_id")
    if type(decision_round) is not int or decision_round < 1:
        raise ValueError("Stage4 Main turn lacks a valid integer round")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ValueError(
            "Stage4 main_turns lacks round_sample_id; republish from "
            "existing Stage4 journals using the updated writer"
        )
    return task, scenario_id, decision_round

def export(risk_dir, main_turns, output_dir, decision_states=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    private_path = Path(decision_states or
                        "environment/data/stage4/private/decision_reward_states.jsonl")
    if not private_path.exists():
        raise FileNotFoundError(
            str(private_path) + ": rerun Stage4 with the deterministic "
            "decision-reward snapshot writer; old journals lack this state"
        )
    from environment.untrusted_orchestration.rewards.decision_reward_state import SCHEMA
    private = {}
    for entry in iter_jsonl(private_path):
        key = (entry["task_id"], entry["round_sample_id"], entry["decision_round"])
        state = entry.get("state")
        if not isinstance(state, dict) or state.get("schema") != SCHEMA:
            raise ValueError("wrong private decision reward state schema")
        if key in private:
            raise ValueError("duplicate private decision reward state")
        private[key] = state
    reference = {}
    for turn in iter_jsonl(main_turns):
        key = reference_key(turn)
        if key in reference:
            raise ValueError("duplicate reference decision")
        action = turn["output"]
        if action.get("action") not in ("complete", "delegate_task"):
            raise ValueError("invalid reference action")
        reference[key] = action
    stats = {
        "reference_turns": len(reference), "private_states": len(private),
        "train": 0, "val": 0, "missing_reference": 0,
        "missing_decision_state": 0, "unscorable": 0,
    }
    for split in ("train", "val"):
        path = Path(risk_dir) / (split + ".jsonl")
        if not path.exists():
            if split == "val":
                continue
            raise FileNotFoundError(path)
        with (out / (split + ".jsonl")).open("w", encoding="utf-8") as dest:
            for sample in iter_jsonl(path):
                validate(sample)
                key = (sample["scene_id"], sample["round_sample_id"], sample["decision_round"])
                if key not in reference:
                    stats["missing_reference"] += 1
                    continue
                if key not in private:
                    stats["missing_decision_state"] += 1
                    continue
                state = private[key]
                if state.get("scorable") is not True:
                    stats["unscorable"] += 1
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
                    "decision_state": state,
                }
                dest.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats[split] += 1
    if stats["train"] == 0:
        raise ValueError(
            "no scorable Stage-2 examples: inspect candidate IDs, "
            "private decision_reward_states.jsonl, and summary"
        )
    (out / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk_dir", default="environment/data/stage4/private/window_risk")
    parser.add_argument("--main_turns", default="environment/data/stage4/main_turns.jsonl")
    parser.add_argument("--output_dir", default="environment/data/stage4/private/risk_grpo")
    parser.add_argument("--decision_states", default=(
        "environment/data/stage4/private/decision_reward_states.jsonl"
    ))
    args = parser.parse_args()
    print(json.dumps(export(
        args.risk_dir, args.main_turns, args.output_dir, args.decision_states
    ), indent=2))


if __name__ == "__main__":
    main()
