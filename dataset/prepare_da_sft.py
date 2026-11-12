#!/usr/bin/env python3
"""Convert existing DA-GRPO prompt rows + original reviewed Main turns into SFT.

Never treat GRPO's reward metadata as a complete assistant label: the original
turn["output"] contains the full O1 JSON action INCLUDING its reasoning.
Every source prompt is regenerated and compared byte-for-byte to the GRPO
Parquet prompt. Train/eval split is by original Scene ID, across all branches.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

from dataset.prepare_da_grpo import read_jsonl, settings_for
from environment.base_orchestration.o1_compat import o1_prompt_from_turn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENES = ROOT / "environment/data/maestro_current_100/questions/scenes.jsonl"
DEFAULT_BASE = ROOT / "environment/data/maestro_base_orchestration_100/main_turns.jsonl"
DEFAULT_GRPO = ROOT / "dataset/data/da-grpo/train.parquet"
DEFAULT_OUTPUT = ROOT / "dataset/data/da-sft"
REVIEWED_MODES = frozenset({
    "llm_reviewed",
    "llm_reviewed_completion",
    "llm_reviewed_contract_frozen",
})


def load_grpo_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    if not path.is_file():
        raise FileNotFoundError(f"Missing DA-GRPO Parquet: {path}")
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError("DA-GRPO Parquet is empty")
    return rows


def index_turns(
    scenes_path: Path,
    base_turns: Path,
    untrusted_turns: Path | None,
) -> tuple[dict[tuple[str, str], dict], dict[str, dict]]:
    """Verify the original generation manifests before joining to GRPO."""
    sources = [("stage3", Path(base_turns))]
    if untrusted_turns is not None:
        sources.append(("stage4", Path(untrusted_turns)))
    turns: dict[tuple[str, str], dict] = {}
    settings: dict[str, dict] = {}
    for source, path in sources:
        settings[source] = settings_for(path, scenes_path, source)
        for turn in read_jsonl(path):
            sample_id = turn.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{source}: missing sample_id")
            key = (f"maestro_{source}", sample_id)
            if key in turns:
                raise ValueError(f"Duplicate original turn: {key}")
            turns[key] = turn
    return turns, settings


def build_sft_rows(
    grpo_rows: list[dict],
    turns: dict[tuple[str, str], dict],
    scenes: dict[str, dict],
    settings: dict[str, dict],
    *,
    include_fallback: bool = False,
    renderer: Callable[..., str] = o1_prompt_from_turn,
) -> tuple[list[dict], list[dict], Counter]:
    """Pure join/validation; return TRL rows, sidecar provenance, skip counts."""
    examples: list[dict] = []
    metadata: list[dict] = []
    skipped: Counter = Counter()
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(grpo_rows):
        source = row.get("data_source")
        info = row.get("extra_info")
        reward = row.get("reward_model")
        prompt = row.get("prompt")
        if source not in {"maestro_stage3", "maestro_stage4"}:
            raise ValueError(f"GRPO row {index}: unknown data_source {source!r}")
        if not isinstance(info, dict) or info.get("has_expert") is not True:
            raise ValueError(f"GRPO row {index}: missing expert metadata")
        if not isinstance(reward, dict):
            raise ValueError(f"GRPO row {index}: missing reward_model")
        if not isinstance(prompt, list) or len(prompt) != 1:
            raise ValueError(f"GRPO row {index}: invalid original prompt")
        if not isinstance(prompt[0], dict) or prompt[0].get("role") != "user":
            raise ValueError(f"GRPO row {index}: expected one O1 user message")
        if not isinstance(prompt[0].get("content"), str):
            raise ValueError(f"GRPO row {index}: prompt must contain text")
        sample_id = info.get("sample_id")
        if not isinstance(sample_id, str):
            raise ValueError(f"GRPO row {index}: missing sample_id")
        key = (source, sample_id)
        if key in seen:
            raise ValueError(f"Duplicate GRPO sample: {key}")
        seen.add(key)
        if key not in turns:
            raise ValueError(
                f"No raw Main turn for {key}; supply --untrusted-turns if "
                "the GRPO Parquet includes Stage 4."
            )
        turn = turns[key]
        scene_id = str(turn.get("task_id"))
        scene = scenes.get(scene_id)
        if scene is None:
            raise ValueError(f"Unknown Scene for {key}: {scene_id}")
        stage = source.removeprefix("maestro_")
        if stage not in settings:
            raise ValueError(f"No original rendering manifest for {source}")
        setting = settings[stage]
        regenerated = renderer(
            scene, turn,
            max_attempts=setting["max_attempts"],
            sub_model=setting["sub_model"],
            model_alias=setting["model_alias"],
        )
        if prompt[0]["content"] != regenerated:
            raise ValueError(f"Prompt drift against original O1 renderer: {key}")
        action = turn.get("output")
        if not isinstance(action, dict) or set(action) != {"action", "reasoning", "params"}:
            raise ValueError(f"Invalid complete O1 action fields: {key}")
        if action["action"] not in {"delegate_task", "complete"}:
            raise ValueError(f"Unknown action type for {key}")
        if not isinstance(action["reasoning"], str) or not action["reasoning"].strip():
            raise ValueError(f"Missing full expert reasoning for {key}")
        params = action["params"]
        if not isinstance(params, dict):
            raise ValueError(f"Invalid O1 action params for {key}")
        if action["action"] == "delegate_task":
            tasks = params.get("tasks")
            if not isinstance(tasks, list) or not tasks:
                raise ValueError(f"Missing delegate tasks for {key}")
        elif not isinstance(params.get("answer"), str):
            raise ValueError(f"Missing complete answer for {key}")
        history = (turn.get("input") or {}).get("subtask_history")
        if (
            info.get("expert_action") != action["action"]
            or info.get("expert_params") != params
            or info.get("subtask_history") != history
            or info.get("step_index") != turn.get("round")
            or info.get("question") != scene.get("question")
            or info.get("answer") != reward.get("ground_truth")
        ):
            raise ValueError(f"Expert / reward metadata drift for {key}")
        mode = turn.get("generation_mode", "unknown")
        if not include_fallback and mode not in REVIEWED_MODES:
            skipped[mode] += 1
            continue
        # This is precisely the full output recorded by Stage 3/4, not a
        # reconstruction from expert_params (which would lose reasoning).
        answer = json.dumps(action, ensure_ascii=False)
        examples.append({
            "prompt": prompt,
            "completion": [{"role": "assistant", "content": answer}],
        })
        metadata.append({
            "sample_id": sample_id,
            "task_id": scene_id,
            "data_source": source,
            "generation_mode": mode,
            "action": action["action"],
        })
    if not examples:
        raise ValueError("All samples filtered out; no reviewed SFT examples")
    return examples, metadata, skipped


def group_split(
    rows: list[dict], metadata: list[dict], ratio: float, seed: int,
) -> dict[str, list[int]]:
    """Stable split over Scene IDs; Stage 4 branches share the same split."""
    if not 0 < ratio < 1:
        raise ValueError("eval_ratio must be between 0 and 1")
    if len(rows) != len(metadata):
        raise ValueError("Rows and metadata lengths differ")
    groups = sorted({m["task_id"] for m in metadata}, key=lambda task: (
        sha256(f"{seed}:{task}".encode("utf-8")).hexdigest(), task,
    ))
    if len(groups) < 2:
        raise ValueError("At least two different Scenes are needed for train/eval")
    n_eval = min(len(groups) - 1, max(1, round(len(groups) * ratio)))
    eval_groups = set(groups[:n_eval])
    indices = {"train": [], "eval": []}
    for i, item in enumerate(metadata):
        indices["eval" if item["task_id"] in eval_groups else "train"].append(i)
    return indices


def export(
    grpo_parquet: Path, scenes_path: Path, base_turns: Path, output_dir: Path,
    *, untrusted_turns: Path | None = None, eval_ratio: float = 0.1,
    seed: int = 42, include_fallback: bool = False, dry_run: bool = False,
) -> dict:
    scenes_list = read_jsonl(Path(scenes_path))
    scenes = {str(item["task_id"]): item for item in scenes_list}
    if len(scenes) != len(scenes_list):
        raise ValueError("Duplicate Scene task_id")
    turns, settings = index_turns(
        Path(scenes_path), Path(base_turns), untrusted_turns,
    )
    rows, metadata, skipped = build_sft_rows(
        load_grpo_rows(Path(grpo_parquet)), turns, scenes, settings,
        include_fallback=include_fallback,
    )
    split = group_split(rows, metadata, eval_ratio, seed)
    summary = {
        "total": len(rows),
        "train": len(split["train"]),
        "eval": len(split["eval"]),
        "train_scenes": len({metadata[i]["task_id"] for i in split["train"]}),
        "eval_scenes": len({metadata[i]["task_id"] for i in split["eval"]}),
        "data_sources": dict(Counter(m["data_source"] for m in metadata)),
        "generation_modes": dict(Counter(m["generation_mode"] for m in metadata)),
        "skipped_modes": dict(skipped),
        "eval_ratio_by_scene": eval_ratio,
        "seed": seed,
        "output_dir": str(output_dir),
    }
    if dry_run:
        return summary

    import pyarrow as pa
    import pyarrow.parquet as pq
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "eval"):
        subset = [rows[i] for i in split[name]]
        pq.write_table(pa.Table.from_pylist(subset), out / f"{name}.parquet")
    with (out / "metadata.jsonl").open("w", encoding="utf-8") as file:
        for name in ("train", "eval"):
            for i in split[name]:
                file.write(json.dumps({"split": name, **metadata[i]}, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    # Verify the actual files (not only the intermediate in-memory objects).
    for name in ("train", "eval"):
        reread = pq.read_table(out / f"{name}.parquet").to_pylist()
        if len(reread) != len(split[name]) or any(
            set(item) != {"prompt", "completion"} for item in reread
        ):
            raise RuntimeError(f"SFT Parquet round-trip failed: {name}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grpo-parquet", type=Path, default=DEFAULT_GRPO)
    parser.add_argument("--scenes", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--base-turns", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--untrusted-turns", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-fallback", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summary = export(
        args.grpo_parquet, args.scenes, args.base_turns, args.output_dir,
        untrusted_turns=args.untrusted_turns, eval_ratio=args.eval_ratio,
        seed=args.seed, include_fallback=args.include_fallback,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
