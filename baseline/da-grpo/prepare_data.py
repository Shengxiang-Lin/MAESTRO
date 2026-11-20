#!/usr/bin/env python3
"""DA-GRPO-only reward-aware Parquet preparation from unified base Decisions.

Main policy prompts are unchanged O1 user messages. Gold and expert action
exist exclusively in reward_model / extra_info columns, never in the prompt.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from baseline.dataset_common import (
    DEFAULT_SOURCE, ROOT, SPLITS,
    audit_selection, check_frozen_source, get_splits, load_base,
    prepare_directory, repo_path, select_rows, write_json, write_jsonl,
)


def convert(decisions: list[dict], teachers: dict[str, dict],
            splits: dict[str, str]) -> dict[str, list[dict]]:
    result = {name: [] for name in SPLITS}
    for decision in decisions:
        scene, identity = decision["scene_id"], decision["decision_id"]
        teacher = teachers.get(identity)
        if teacher is None:
            raise ValueError("grpo_teacher_missing:" + identity)
        answer = teacher["ground_truth"]
        action = decision["reference_action"]
        tasks = action["params"].get("tasks", []) if action["action"] == "delegate_task" else []
        if any(
            not isinstance(t, dict) or
            not isinstance(t.get("task_instruction"), str) or
            not isinstance(t.get("model"), str) or
            not isinstance(t.get("tools", []), list)
            for t in tasks
        ):
            raise ValueError("grpo_invalid_expert_delegation:" + identity)
        result[splits[scene]].append({
            "data_source": (
                "maestro_stage3" if decision["source"] == "clean"
                else "maestro_stage4"
            ),
            "prompt": decision["prompt"],
            "reward_model": {
                "style": "rule",
                "ground_truth": answer,
            },
            "extra_info": {
                "sample_id": identity,
                "decision_id": identity,
                "case_id": decision["case_id"],
                "question": decision["question"],
                "answer": answer,
                "subtask_history": decision["visible_state"]["subtask_history"],
                "has_expert": True,
                "step_index": decision["decision_round"],
                "expert_action": action["action"],
                "expert_params": action["params"],
                "expert_tools": [
                    name for task in tasks for name in task["tools"]
                ],
                "expert_task_instructions": [
                    task["task_instruction"] for task in tasks
                ],
                "expert_num_subtasks": len(tasks),
                "expert_answer": (
                    action["params"]["answer"]
                    if action["action"] == "complete" else ""
                ),
            },
        })
    return result


def prepare(
    source_dir: Path = DEFAULT_SOURCE,
    output_dir: Path = ROOT / "baseline/da-grpo/data",
    *, seed: int = 17, train_ratio: float = 0.8, val_ratio: float = 0.1,
    split_map: Path | None = None, include_fallback: bool = False,
    strict_conflicts: bool = False, overwrite: bool = False,
    dry_run: bool = False,
) -> dict:
    decisions, teachers, scene_ids, source_manifest = load_base(source_dir)
    split_data = get_splits(
        scene_ids, seed=seed,
        ratios=(train_ratio, val_ratio, 1 - train_ratio - val_ratio),
        split_map=split_map,
    )
    kept, skipped = select_rows(decisions, include_fallback=include_fallback)
    mapping = split_data["scene_to_split"]
    report = audit_selection(kept, mapping)
    if strict_conflicts and report["conflicting_prompt_groups"]:
        raise ValueError("da_grpo_ambiguous_expert_reference")
    rows = convert(kept, teachers, mapping)
    if any(not rows[s] for s in SPLITS):
        raise ValueError(
            "da_grpo_empty_train_val_or_test_split:" +
            json.dumps(report["decision_counts"], sort_keys=True)
        )
    summary = {
        "schema": "maestro.da_grpo.prepared.v1",
        "input_dir": str(repo_path(source_dir)),
        "source_sha256": source_manifest["base_file_sha256"],
        "data_dir": str(repo_path(output_dir)),
        "scene_split": split_data,
        "generation_mode_skipped": dict(sorted(skipped.items())),
        "include_fallback": include_fallback,
        "quality": report,
        "counts": {s: len(rows[s]) for s in SPLITS},
    }
    if dry_run:
        return summary

    check_frozen_source(source_dir, source_manifest)
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = prepare_directory(output_dir, overwrite=overwrite)
    for split in SPLITS:
        dest = out / (split + ".parquet")
        pq.write_table(pa.Table.from_pylist(rows[split]), dest)
        observed = pq.read_table(dest).to_pylist()
        if len(observed) != len(rows[split]):
            raise RuntimeError("da_grpo_parquet_count_mismatch:" + split)
        for original, loaded in zip(rows[split], observed):
            if (loaded["prompt"] != original["prompt"]
                    or loaded["reward_model"]["ground_truth"] !=
                    original["reward_model"]["ground_truth"]
                    or loaded["extra_info"]["sample_id"] !=
                    original["extra_info"]["sample_id"]):
                raise RuntimeError("da_grpo_parquet_contract_drift:" + split)
    write_json(out / "splits.json", split_data)
    write_json(out / "quality_report.json", report)
    write_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "baseline/da-grpo/data")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-map", type=Path)
    parser.add_argument("--include-fallback", action="store_true")
    parser.add_argument("--strict-conflicts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(
        source_dir=args.source_dir, output_dir=args.output_dir,
        seed=args.seed, train_ratio=args.train_ratio,
        val_ratio=args.val_ratio, split_map=args.split_map,
        include_fallback=args.include_fallback,
        strict_conflicts=args.strict_conflicts,
        overwrite=args.overwrite, dry_run=args.dry_run,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
