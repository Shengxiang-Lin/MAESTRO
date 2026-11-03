"""Export MAESTRO trajectories into small-Main training views.

This exporter intentionally uses ``state_before.o1_equivalent_prompt`` rather
than the MAESTRO prospective prompt. The trainable Main therefore sees the
same public observation contract as Orchestra-o1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from orchestra.modes import ORCHESTRATION_MODES
from train_maestro_8b.schema import validate_stage1_record


def iter_trajectory_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    yield from sorted(root.rglob("*.json"))


def export_file(path: Path) -> Iterable[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    task_id = str(data.get("task_id") or path.stem)
    trajectory = data.get("trajectory") or []
    for index, attempt in enumerate(trajectory):
        if not isinstance(attempt, dict):
            continue
        state_before = attempt.get("state_before") or {}
        prompt = str(state_before.get("o1_equivalent_prompt") or "").strip()
        if not prompt:
            continue
        decision_record = attempt.get("decision_record") or {}
        macro = str(
            decision_record.get("orchestration_mode")
            or attempt.get("orchestration_mode")
            or ""
        ).strip().lower()
        if macro not in ORCHESTRATION_MODES:
            continue
        action = str(attempt.get("action") or "").strip()
        params = attempt.get("params") or {}
        if action not in {"delegate_task", "complete"}:
            continue
        target = {
            "action": action,
            "reasoning": str(attempt.get("reasoning") or ""),
            "params": params,
        }
        item = {
            "id": f"{task_id}:{index + 1}",
            "task_id": task_id,
            "step": index + 1,
            "input": prompt,
            "macro_label": macro,
            "target_json": json.dumps(
                target,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "episode_success": bool(data.get("success", False)),
            "episode_reward": float(data.get("total_reward", 0.0) or 0.0),
        }
        yield validate_stage1_record(item)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.trajectory_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for path in iter_trajectory_files(source):
            for item in export_file(path):
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
    print(f"exported {count} stage-1 examples to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
