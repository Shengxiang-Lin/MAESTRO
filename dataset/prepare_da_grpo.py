#!/usr/bin/env python3
"""Export reviewed MAESTRO Main-Agent turns to Orchestra-o1 verl GRPO data.

One published Main decision = one policy prompt. The expert action and gold
answer are reward-only metadata, NEVER included in the policy prompt.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from environment.base_orchestration.o1_compat import o1_prompt_from_turn
from environment.base_orchestration.trajectory_builder import _answer_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENES = ROOT / "environment/data/maestro_current_100/questions/scenes.jsonl"
DEFAULT_BASE = ROOT / "environment/data/maestro_base_orchestration_100/main_turns.jsonl"
DEFAULT_OUTPUT = ROOT / "dataset/data/da-grpo"


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    records = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{number}: expected JSON object")
            records.append(obj)
    return records


def digest(path: Path) -> str:
    sha = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def check_renderer_compatibility(expected_sha256: str | None, renderer: Path, source: str) -> None:
    """Allow a verified historical docstring-only rename, not arbitrary code drift.

    Stage 3/4 manifests hash source *bytes*, so a non-semantic documentation
    change in o1_compat.py otherwise blocks valid, already-reviewed turns.
    The allowed historical source is reconstructed byte-for-byte with only
    the two verified documentation substitutions from PR #137. Comparing its
    SHA256 with the saved manifest ensures no code change is silently ignored.
    """
    if not expected_sha256 or expected_sha256 == digest(renderer):
        return
    current = renderer.read_text(encoding="utf-8")
    historical = current.replace(
        "The on-disk per-turn reviewed-turn input stores",
        "The on-disk per-turn SFT input stores",
        1,
    ).replace(
        "Rebuild a full baseline O1 prompt for an exported compact reviewed-turn turn.",
        "Rebuild a full baseline O1 prompt for an exported compact SFT turn.",
        1,
    )
    if historical != current and sha256(historical.encode("utf-8")).hexdigest() == expected_sha256:
        print(f"[dataset] {source}: verified O1 renderer docstring-only change; reusing existing turns.")
        return
    raise ValueError(
        f"{source} prompt renderer changed semantically or its historical source "
        f"cannot be verified. Expected SHA256={expected_sha256}, "
        f"current SHA256={digest(renderer)}. "
        "Use the matching historical code or regenerate the affected Main turns."
    )


def settings_for(turns_path: Path, scenes_path: Path, source: str) -> dict:
    manifest_path = turns_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing signed generation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source == "stage3":
        expected = manifest.get("scenes_sha256")
        settings = manifest
        # Match the version guard used by Stage 3's existing prompt exporter.
        code_hash = (manifest.get("code_sha256") or {}).get("o1_compat.py")
        renderer = ROOT / "environment/base_orchestration/o1_compat.py"
        check_renderer_compatibility(code_hash, renderer, "Stage 3")
    else:
        hashes = manifest.get("input_sha256") or []
        expected = hashes[0] if hashes else None
        settings = manifest.get("settings") or {}
        code_hash = (manifest.get("source_code_sha256") or {}).get(
            "../base_orchestration/o1_compat.py"
        )
        renderer = ROOT / "environment/base_orchestration/o1_compat.py"
        check_renderer_compatibility(code_hash, renderer, "Stage 4")
    if expected is None:
        raise ValueError(f"Manifest is missing source Scene hash: {manifest_path}")
    if digest(scenes_path) != expected:
        raise ValueError(f"Scene file does not match {source} manifest: {scenes_path}")
    max_attempts = settings.get("max_attempts", settings.get("max_rounds"))
    sub_model = settings.get("sub_model")
    model_alias = settings.get("model_alias")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError(f"Invalid Main round budget in {manifest_path}")
    if not sub_model or not model_alias:
        raise ValueError(f"Missing Sub-Agent model alias in {manifest_path}")
    return {
        "max_attempts": max_attempts,
        "sub_model": sub_model,
        "model_alias": model_alias,
    }


def make_examples(
    scenes: dict[str, dict], turns: list[dict], *, source: str,
    settings: dict, seen: set[str], require_llm: bool = False,
) -> tuple[list[dict], dict[str, list[dict]], Counter]:
    rows = []
    experts = defaultdict(list)
    modes = Counter()
    for turn in turns:
        sample_id = turn.get("sample_id")
        scene_id = turn.get("task_id")
        round_number = turn.get("round")
        if not isinstance(sample_id, str) or "-round-" not in sample_id:
            raise ValueError(f"{source}: invalid sample_id: {sample_id}")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id across inputs: {sample_id}")
        if scene_id not in scenes:
            raise ValueError(f"{source}: unknown Scene: {scene_id}")
        episode_id, suffix = sample_id.rsplit("-round-", 1)
        if type(round_number) is not int or not suffix.isdigit() or int(suffix) != round_number:
            raise ValueError(f"{source}: sample_id/round mismatch: {sample_id}")
        if not 1 <= round_number <= settings["max_attempts"]:
            raise ValueError(f"{source}: invalid round budget for {sample_id}")
        if source == "stage3" and episode_id != str(scene_id):
            raise ValueError(f"{source}: sample_id/task_id mismatch: {sample_id}")
        if source == "stage4" and not episode_id.startswith(str(scene_id) + "-branch-"):
            raise ValueError(f"Stage 4 sample has no matching branch identity: {sample_id}")
        action = turn.get("output")
        if not isinstance(action, dict) or action.get("action") not in {"delegate_task", "complete"}:
            raise ValueError(f"{source}: invalid O1 action for {sample_id}")
        params = action.get("params")
        if not isinstance(params, dict):
            raise ValueError(f"{source}: missing params for {sample_id}")
        if action["action"] == "delegate_task":
            tasks = params.get("tasks")
            if not isinstance(tasks, list) or not tasks or any(
                not isinstance(task, dict)
                or not isinstance(task.get("task_instruction"), str)
                or not isinstance(task.get("model"), str)
                or not isinstance(task.get("tools", []), list)
                for task in tasks
            ):
                raise ValueError(f"{source}: invalid delegated tasks for {sample_id}")
        elif "answer" not in params:
            raise ValueError(f"{source}: complete without answer: {sample_id}")
        mode = turn.get("generation_mode", "unknown")
        modes[mode] += 1
        seen.add(sample_id)
        if require_llm and mode not in {
            "llm_reviewed",
            "llm_reviewed_completion",
            "llm_reviewed_contract_frozen",
        }:
            continue

        scene = scenes[scene_id]
        answer = _answer_text(scene)  # Exact Stage 3 completion serialization.
        history = (turn.get("input") or {}).get("subtask_history")
        if not isinstance(history, str):
            raise ValueError(f"{source}: missing subtask_history for {sample_id}")
        prompt = o1_prompt_from_turn(
            scene, turn, max_attempts=settings["max_attempts"],
            sub_model=settings["sub_model"], model_alias=settings["model_alias"],
        )
        tasks = params.get("tasks", []) if action["action"] == "delegate_task" else []
        extra_info = {
            "sample_id": sample_id,
            "question": scene["question"],
            "answer": answer,
            "subtask_history": history,
            "has_expert": True,
            "step_index": round_number,
            "expert_action": action["action"],
            "expert_params": params,
            "expert_tools": [name for task in tasks for name in task.get("tools", [])],
            "expert_task_instructions": [task["task_instruction"] for task in tasks],
            "expert_num_subtasks": len(tasks),
            "expert_answer": params.get("answer", "") if action["action"] == "complete" else "",
        }
        rows.append({
            "data_source": f"maestro_{source}",
            "prompt": [{"role": "user", "content": prompt}],
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": extra_info,
        })
        experts[episode_id].append({
            "sample_id": sample_id,
            "step_index": round_number,
            "expert_action": action["action"],
            "expert_params": params,
            "subtask_history": history,
        })
    return rows, experts, modes


def check_order(experts: dict[str, list[dict]]) -> None:
    for episode, steps in experts.items():
        rounds = [step["step_index"] for step in steps]
        if len(rounds) != len(set(rounds)):
            raise ValueError(f"Duplicate expert round in {episode}: {rounds}")
        if rounds != sorted(rounds):
            raise ValueError(f"Out-of-order expert rounds in {episode}: {rounds}")


def validate_prompt_lengths(rows: list[dict], tokenizer_path: str | None, limit: int) -> None:
    if tokenizer_path is None:
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    violations = []
    for row in rows:
        messages = row["prompt"]
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        )
        if len(tokens) > limit:
            violations.append((row["extra_info"]["sample_id"], len(tokens)))
    if violations:
        raise ValueError(
            f"{len(violations)} prompts exceed {limit} tokens; examples: {violations[:5]}"
        )


def export(
    scenes_path: Path, base_turns: Path, output_dir: Path, *,
    untrusted_turns: Path | None = None,
    tokenizer: str | None = None,
    max_prompt_tokens: int = 24576,
    require_llm: bool = False,
    dry_run: bool = False,
) -> dict:
    scenes_path, base_turns, output_dir = map(Path, (scenes_path, base_turns, output_dir))
    scenes_list = read_jsonl(scenes_path)
    scenes = {row["task_id"]: row for row in scenes_list}
    if len(scenes) != len(scenes_list):
        raise ValueError("Duplicate task_id in Scenes")
    inputs = [("stage3", base_turns)]
    if untrusted_turns is not None:
        inputs.append(("stage4", Path(untrusted_turns)))
    rows, all_experts, mode_counts, seen = [], defaultdict(list), Counter(), set()
    for source, path in inputs:
        settings = settings_for(path, scenes_path, source)
        batch, expert, modes = make_examples(
            scenes, read_jsonl(path), source=source,
            settings=settings, seen=seen, require_llm=require_llm,
        )
        rows.extend(batch)
        mode_counts.update(modes)
        for episode, steps in expert.items():
            all_experts[(source, episode)].extend(steps)
    if not rows:
        raise ValueError("No reviewed Main-Agent turns available for export")
    check_order(all_experts)
    validate_prompt_lengths(rows, tokenizer, max_prompt_tokens)
    summary = {
        "rows": len(rows),
        "episodes": len(all_experts),
        "by_source": dict(Counter(row["data_source"] for row in rows)),
        "generation_modes_seen": dict(mode_counts),
        "require_llm": require_llm,
        "tokenizer_checked": tokenizer is not None,
        "output_dir": str(output_dir),
    }
    if dry_run:
        return summary

    try:
        import pandas as pd
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Export requires pandas and pyarrow (pip install pandas pyarrow)") from error
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_parquet(
        output_dir / "train.parquet", engine="pyarrow", index=False,
    )
    with (output_dir / "expert_decisions.jsonl").open("w", encoding="utf-8") as file:
        for (source, episode), steps in sorted(all_experts.items()):
            record = {
                "episode_id": episode,
                "data_source": f"maestro_{source}",
                "expert_steps": sorted(steps, key=lambda value: value["step_index"]),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Round-trip structural validation of the actual saved Parquet.
    saved = pd.read_parquet(output_dir / "train.parquet", engine="pyarrow")
    if len(saved) != len(rows) or set(saved.columns) != {
        "data_source", "prompt", "reward_model", "extra_info"
    }:
        raise RuntimeError("Parquet round-trip validation failed")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--base-turns", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--untrusted-turns", type=Path, default=None,
                        help="Optional Stage 4 positive main_turns.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", default=None,
                        help="Optional Qwen3-8B tokenizer path for exact token checks")
    parser.add_argument("--max-prompt-tokens", type=int, default=24576)
    parser.add_argument("--require-llm-expert", action="store_true",
                        help="Exclude template/fallback labels; keep llm_reviewed only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate records without writing files or needing pyarrow")
    args = parser.parse_args(argv)
    if args.max_prompt_tokens < 1:
        parser.error("--max-prompt-tokens must be positive")
    result = export(
        args.scenes, args.base_turns, args.output_dir,
        untrusted_turns=args.untrusted_turns, tokenizer=args.tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        require_llm=args.require_llm_expert, dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
