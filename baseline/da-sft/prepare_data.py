#!/usr/bin/env python3
"""Prepare *DA-SFT-specific* prompt/completion Parquet from unified base JSONL.

Owns reviewed-mode filtering, Scene-level split, duplication diagnostics, and
training Parquets. Does NOT consume GRPO data or write into environment/data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from tqdm.auto import tqdm

ROOT_PATH = Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from baseline.dataset_common import (
    DEFAULT_SOURCE, ROOT, SPLITS,
    audit_selection, check_frozen_source, get_splits, load_base,
    prepare_directory, repo_path, select_rows, write_json, write_jsonl,
)


def convert(decisions: list[dict], scene_splits: dict[str, str],
            *, progress: bool = False):
    result = {name: [] for name in SPLITS}
    metadata = {name: [] for name in SPLITS}
    bar = tqdm(decisions, desc="SFT conversion", unit="decision",
               disable=not progress)
    by_split = Counter()
    for index, row in enumerate(bar):
        split = scene_splits[row["scene_id"]]
        by_split[split] += 1
        if progress and ((index + 1) % 128 == 0 or index + 1 == len(decisions)):
            bar.set_postfix(train=by_split["train"], val=by_split["val"],
                            test=by_split["test"],
                            refresh=index + 1 == len(decisions))
        result[split].append({
            "prompt": row["prompt"],
            "completion": [{
                "role": "assistant",
                "content": json.dumps(
                    row["reference_action"], ensure_ascii=False,
                ),
            }],
        })
        metadata[split].append({
            "decision_id": row["decision_id"],
            "case_id": row["case_id"],
            "scene_id": row["scene_id"],
            "source": row["source"],
            "action": row["reference_action"]["action"],
            "generation_mode": row["generation_mode"],
        })
    return result, metadata


def prepare(
    source_dir: Path = DEFAULT_SOURCE,
    output_dir: Path = ROOT / "baseline/da-sft/data",
    *, seed: int = 17, train_ratio: float = 0.8, val_ratio: float = 0.1,
    split_map: Path | None = None, include_fallback: bool = False,
    strict_conflicts: bool = False, overwrite: bool = False,
    dry_run: bool = False, progress: bool = False,
) -> dict:
    decisions, _, scene_ids, source_manifest = load_base(
        source_dir, progress=progress,
    )
    mapping = get_splits(
        scene_ids, seed=seed, ratios=(
            train_ratio, val_ratio, 1 - train_ratio - val_ratio,
        ), split_map=split_map,
    )
    kept, skipped = select_rows(
        decisions, include_fallback=include_fallback, progress=progress,
    )
    split_of = mapping["scene_to_split"]
    report = audit_selection(kept, split_of, progress=progress)
    if strict_conflicts and report["conflicting_prompt_groups"]:
        raise ValueError("da_sft_conflicting_supervised_completions")
    rows, metadata = convert(kept, split_of, progress=progress)
    if any(not rows[s] for s in SPLITS):
        raise ValueError(
            "da_sft_empty_train_val_or_test_split:" +
            json.dumps(report["decision_counts"], sort_keys=True)
        )
    summary = {
        "schema": "maestro.da_sft.prepared.v1",
        "input_dir": str(repo_path(source_dir)),
        "source_sha256": source_manifest["base_file_sha256"],
        "data_dir": str(repo_path(output_dir)),
        "scene_split": mapping,
        "generation_mode_skipped": dict(sorted(skipped.items())),
        "include_fallback": include_fallback,
        "quality": report,
        "counts": {s: len(rows[s]) for s in SPLITS},
    }
    if dry_run:
        return summary

    check_frozen_source(source_dir, source_manifest)
    # Parquet is imported only by method-specific preparation.
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = prepare_directory(output_dir, overwrite=overwrite)
    for split in tqdm(SPLITS, desc="Write SFT Parquet", unit="split",
                      disable=not progress):
        name = "eval" if split == "val" else split
        dest = out / (name + ".parquet")
        pq.write_table(pa.Table.from_pylist(rows[split]), dest)
        loaded = pq.read_table(dest).to_pylist()
        if loaded != rows[split]:
            raise RuntimeError("da_sft_parquet_round_trip_failed:" + name)
        write_jsonl(out / (name + "_metadata.jsonl"), metadata[split])
    write_json(out / "splits.json", mapping)
    write_json(out / "quality_report.json", report)
    write_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "baseline/da-sft/data")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-map", type=Path)
    parser.add_argument("--include-fallback", action="store_true")
    parser.add_argument("--strict-conflicts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable tqdm bars for batch logs")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(
        source_dir=args.source_dir, output_dir=args.output_dir,
        seed=args.seed, train_ratio=args.train_ratio,
        val_ratio=args.val_ratio, split_map=args.split_map,
        include_fallback=args.include_fallback,
        strict_conflicts=args.strict_conflicts,
        overwrite=args.overwrite, dry_run=args.dry_run,
        progress=not args.no_progress,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
