"""Deterministic smoke test for the Stage-3 action/observation boundary.

The demo uses a reference schedule ONLY to test environment mechanics.
It is not an independent Agent solver or a new training/evaluation result.

Usage:
python -m envir.stage3.demo --scenes path/to/scenes.jsonl \
    --reports path/to/stage2/subtasks.jsonl
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from envir.generate_question.stages import read_jsonl
from .simulator import TrustedSimulator


def run_demo(scenes: Path, reports: Path | None = None, *,
             budget: float = 40.0, task_id: str | None = None) -> dict:
    rows = read_jsonl(scenes)
    if not rows:
        raise ValueError("stage3_empty_scene_bank")
    by_scene = defaultdict(list)
    if reports is not None:
        for row in read_jsonl(reports):
            by_scene[row["scene_task_id"]].append(row)
    engine = None
    for scene in rows:
        if task_id and scene["task_id"] != task_id:
            continue
        if reports is not None:
            records = by_scene[scene["task_id"]]
            if len(records) != scene["macro_graph"]["macro_task_count"]:
                if task_id:
                    raise ValueError("stage3_selected_scene_reports_incomplete")
                continue
            engine = TrustedSimulator(scene, records, budget=budget)
        else:
            engine = TrustedSimulator(
                scene, mode="oracle_baseline", budget=budget,
            )
        break
    if engine is None:
        raise ValueError("stage3_no_complete_scene_available")
    session = engine.public_session()
    public = session.reset()
    while not public["done"]:
        ready = public["ready_tasks"]
        if not ready:
            raise ValueError("stage3_reference_schedule_stalled")
        macro_id = ready[0]["macro_task_id"]
        session.step({"type": "delegate", "macro_task_id": macro_id})
        public = session.step({
            "type": "accept", "macro_task_id": macro_id,
        })["observation"]
    return {
        "mode": "trusted_environment_reference_smoke_test",
        "task_id": public["task_id"],
        "public_done": public["done"],
        "public_status": public["status"],
        "private_teacher_evaluation": engine.evaluate(),
        "note": (
            "Reference actions are a mechanics test; the public Main-Agent "
            "never receives this private evaluation or future reports."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", type=Path, required=True)
    p.add_argument("--reports", type=Path)
    p.add_argument("--budget", type=float, default=40.0)
    p.add_argument("--task-id")
    args = p.parse_args(argv)
    result = run_demo(
        args.scenes, args.reports, budget=args.budget,
        task_id=args.task_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["private_teacher_evaluation"]["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
