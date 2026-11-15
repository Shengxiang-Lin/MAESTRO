"""Stage-2 DA-style Main-Agent data export from accepted Stage-4 turns.

This export is owned by train_maestro_8b, NOT baseline/da-grpo. It produces
the exact DA dataset schema; expert/gold stay in reward-only columns. Stage-1
Risk Head, fault labels and private decision states are not needed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from dataset.prepare_da_grpo import (
    read_jsonl, settings_for, validate_prompt_lengths,
)
from environment.base_orchestration.o1_compat import o1_prompt_from_turn
from environment.base_orchestration.trajectory_builder import _answer_text


ROOT = Path(__file__).resolve().parents[2]
REVIEWED_MODES = {
    "llm_reviewed", "llm_reviewed_completion", "llm_reviewed_contract_frozen",
}


def convert(scenes: list[dict], turns: list[dict], settings: dict, *,
            reviewed_only: bool = False):
    scene_by_id = {s["task_id"]: s for s in scenes}
    if len(scene_by_id) != len(scenes):
        raise ValueError("duplicate Stage1 scenes")
    rows, experts, modes = [], defaultdict(list), Counter()
    seen = set()
    for turn in turns:
        scene_id, round_number = turn.get("task_id"), turn.get("round")
        if scene_id not in scene_by_id:
            raise ValueError(f"Stage4 turn references unknown Scene: {scene_id}")
        if type(round_number) is not int or not 1 <= round_number <= settings["max_attempts"]:
            raise ValueError("invalid decision round")
        mode = turn.get("generation_mode", "unknown")
        modes[mode] += 1
        if reviewed_only and mode not in REVIEWED_MODES:
            continue
        candidate_id = turn.get("round_sample_id")
        display_id = turn.get("sample_id")
        if candidate_id is None:
            # Older approved Stage4 data had only the display branch identity.
            if not isinstance(display_id, str) or "-round-" not in display_id:
                raise ValueError("Stage4 turn missing candidate/sample identifier")
            candidate_id, suffix = display_id.rsplit("-round-", 1)
            if not candidate_id.startswith(f"{scene_id}-branch-") or int(suffix) != round_number:
                raise ValueError("legacy Stage4 sample identity mismatch")
        elif not isinstance(candidate_id, str) or not candidate_id.startswith(f"{scene_id}-"):
            raise ValueError("invalid Stage4 round_sample_id")
        sample_id = f"{candidate_id}-decision-round-{round_number:02d}"
        if sample_id in seen:
            raise ValueError(f"duplicate Stage4 candidate/round: {sample_id}")
        seen.add(sample_id)

        action = turn.get("output")
        if not isinstance(action, dict) or action.get("action") not in ("delegate_task", "complete"):
            raise ValueError(f"invalid Stage4 reference Action: {sample_id}")
        params = action.get("params")
        if not isinstance(params, dict):
            raise ValueError(f"reference Action params must be object: {sample_id}")
        tasks = params.get("tasks", []) if action["action"] == "delegate_task" else []
        if action["action"] == "delegate_task" and (
            not isinstance(tasks, list) or not tasks or any(
                not isinstance(t, dict) or not isinstance(t.get("task_instruction"), str)
                or not isinstance(t.get("model"), str) or not isinstance(t.get("tools", []), list)
                for t in tasks
            )
        ):
            raise ValueError(f"invalid delegation in reference: {sample_id}")
        if action["action"] == "complete" and "answer" not in params:
            raise ValueError(f"missing reference answer: {sample_id}")
        history = (turn.get("input") or {}).get("subtask_history")
        if not isinstance(history, str):
            raise ValueError(f"Stage4 lacks visible subtask history: {sample_id}")
        scene = scene_by_id[scene_id]
        prompt = o1_prompt_from_turn(
            scene, turn, max_attempts=settings["max_attempts"],
            sub_model=settings["sub_model"], model_alias=settings["model_alias"],
        )
        extra_info = {
            "sample_id": sample_id,
            "question": scene["question"],
            "answer": _answer_text(scene),
            "subtask_history": history,
            "has_expert": True,
            "step_index": round_number,
            "expert_action": action["action"],
            "expert_params": params,
            "expert_tools": [name for t in tasks for name in t.get("tools", [])],
            "expert_task_instructions": [t["task_instruction"] for t in tasks],
            "expert_num_subtasks": len(tasks),
            "expert_answer": params.get("answer", "") if action["action"] == "complete" else "",
        }
        rows.append({
            "data_source": "maestro_stage4",
            "prompt": [{"role": "user", "content": prompt}],
            "reward_model": {"style": "rule", "ground_truth": extra_info["answer"]},
            "extra_info": extra_info,
        })
        experts[candidate_id].append({
            "sample_id": sample_id, "step_index": round_number,
            "expert_action": action["action"], "expert_params": params,
            "subtask_history": history,
        })
    if not rows:
        raise ValueError("No accepted Stage4 turns selected for Stage2 DA-GRPO")
    for cid, decisions in experts.items():
        indices = [x["step_index"] for x in decisions]
        if indices != sorted(indices):
            raise ValueError(f"out-of-order Stage4 candidate: {cid}")
    return rows, experts, modes


def export(scenes_path: Path, turns_path: Path, output_dir: Path, *,
           tokenizer: str | None = None, max_prompt_tokens: int = 12288,
           reviewed_only: bool = False, dry_run: bool = False):
    scenes_path, turns_path, output_dir = map(Path, (scenes_path, turns_path, output_dir))
    settings = settings_for(turns_path, scenes_path, "stage4")
    rows, experts, modes = convert(
        read_jsonl(scenes_path), read_jsonl(turns_path), settings,
        reviewed_only=reviewed_only,
    )
    validate_prompt_lengths(rows, tokenizer, max_prompt_tokens)
    summary = {
        "schema": "maestro.stage2_da_grpo.v1", "rows": len(rows),
        "episodes": len(experts), "generation_modes": dict(modes),
        "reviewed_only": reviewed_only, "tokenizer_checked": bool(tokenizer),
        "output_dir": str(output_dir),
        "trainer": "unsloth_trl_grpo_da",
        "reward": "baseline_da_four_dimension_llm_judge",
    }
    if dry_run:
        return summary
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Export needs pandas and pyarrow") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_parquet(
        output_dir / "train.parquet", engine="pyarrow", index=False,
    )
    with (output_dir / "expert_decisions.jsonl").open("w", encoding="utf-8") as f:
        for cid, steps in sorted(experts.items()):
            f.write(json.dumps({
                "episode_id": cid, "data_source": "maestro_stage4",
                "expert_steps": steps,
            }, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", default="environment/data/stage1/questions/scenes.jsonl")
    p.add_argument("--turns", default="environment/data/stage4/main_turns.jsonl")
    p.add_argument("--output_dir", default="workspace/maestro_8b/stage2_da_data")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--max_prompt_tokens", type=int, default=12288)
    p.add_argument("--reviewed_only", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args(argv)
    if args.max_prompt_tokens < 1:
        p.error("max_prompt_tokens must be >0")
    print(json.dumps(export(
        args.scenes, args.turns, args.output_dir,
        tokenizer=args.tokenizer, max_prompt_tokens=args.max_prompt_tokens,
        reviewed_only=args.reviewed_only, dry_run=args.dry_run,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
