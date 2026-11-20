"""Prepare MAESTRO Stage2 GRPO from the canonical Environment dataset.

Input is exclusively the audited, method-independent Environment data_export
snapshot. The training method owns reviewed filtering, Scene splits, reward
state selection and Parquet conversion. Policy prompts are copied verbatim.
No baseline/da-grpo code or source data are imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import shutil

from tqdm.auto import tqdm

from environment.data_export.audit import audit_output
from environment.data_export.source_io import digest, read_jsonl
from environment.data_export.splits import load_splits, make_splits
from environment.data_export.private_context import (
    PRIVATE_FILES, verify_private_context,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "environment/data/unified"
DEFAULT_OUTPUT = ROOT / "workspace/maestro_8b/stage2_da"
SPLITS = ("train", "val", "test")
REVIEWED_MODES = frozenset({
    "llm_reviewed", "llm_reviewed_completion", "llm_reviewed_contract_frozen",
})
REWARD_MODES = ("global", "hybrid", "deterministic", "judge_only", "da")
SCHEMA = "maestro.stage2_grpo_prepared.v2"


def absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


def load_unified(source_dir: Path, *, progress: bool = False):
    root = absolute(source_dir)
    location = root / "manifest.json"
    if not location.is_file():
        raise FileNotFoundError(
            f"Canonical dataset missing: {location}; first run "
            "python -m environment.data_export with the Stage1/3/4 inputs"
        )
    manifest = json.loads(location.read_text(encoding="utf-8"))
    if manifest.get("schema") != "maestro.dataset.v1":
        raise ValueError("Invalid canonical Environment dataset schema")
    audit_output(root, verify_manifest=True, progress=progress)
    decisions = read_jsonl(root / "decisions.jsonl", progress=progress)
    labels = read_jsonl(root / "private/decision_labels.jsonl", progress=progress)
    cases = read_jsonl(root / "cases.jsonl", progress=progress)
    teachers = {row["decision_id"]: row for row in labels}
    if len(teachers) != len(labels):
        raise ValueError("Duplicate private teacher decision_id")
    scenes = [row["scene_id"] for row in cases if row["kind"] == "clean"]
    if len(scenes) != len(set(scenes)):
        raise ValueError("Duplicate clean Scene in canonical dataset")
    return decisions, teachers, scenes, manifest


def validate_selected_claims(root: Path, rows: dict[str, list[dict]]) -> None:
    """Selected Global Judge decisions need an exact t-1 claim snapshot."""
    available = {
        (item["round_sample_id"], item["main_can_act_from_round"])
        for item in read_jsonl(root / PRIVATE_FILES[1])
    }
    selected = {
        (row["extra_info"]["case_id"], row["extra_info"]["step_index"])
        for subset in rows.values() for row in subset
        if row["data_source"] == "maestro_stage4"
        and row["extra_info"]["step_index"] > 1
    }
    missing = selected - available
    if missing:
        raise ValueError(
            f"unified_selected_global_decisions_missing_claims:{len(missing)}; "
            f"examples={sorted(missing)[:3]}. Re-export or attach original "
            "Stage4 claim snapshots to the unified dataset."
        )


def sample_id(decision: dict) -> str:
    """Restore the original Stage4 reward-state join key, not decision_id."""
    did = decision["decision_id"]
    case = decision["case_id"]
    scene = decision["scene_id"]
    rnd = decision["decision_round"]
    if type(rnd) is not int or rnd < 1:
        raise ValueError("Invalid canonical decision round: " + str(did))
    if decision["source"] == "untrusted":
        if did != f"stage4:{case}:round:{rnd:02d}":
            raise ValueError("Stage4 decision identity mismatch: " + did)
        if not isinstance(case, str) or not case.startswith(scene + "-"):
            raise ValueError("Stage4 Scene/case mismatch: " + did)
        return f"{case}-decision-round-{rnd:02d}"
    if decision["source"] == "clean":
        if did != f"stage3:{scene}-round-{rnd:02d}":
            raise ValueError("Stage3 decision identity mismatch: " + did)
        return did
    raise ValueError("Unknown canonical Main decision source")


def convert(decisions: list[dict], teachers: dict[str, dict],
            splits: dict[str, str], *, reward_mode: str = "global",
            include_fallback: bool = False, progress: bool = False):
    """Filter on reward compatibility; preserve unified policy prompts."""
    if reward_mode not in REWARD_MODES:
        raise ValueError("Invalid Stage2 reward mode")
    result = {split: [] for split in SPLITS}
    private_states = []
    skipped, sources = Counter(), Counter()
    seen = set()
    prompt_groups = defaultdict(lambda: {"splits": set(), "actions": set(), "ids": []})
    exact = Counter()
    for decision in tqdm(decisions, desc="GRPO method conversion", unit="decision",
                         disable=not progress):
        did = decision["decision_id"]
        scene = decision["scene_id"]
        if scene not in splits:
            raise ValueError("Decision Scene absent from frozen split: " + scene)
        label = teachers.get(did)
        if label is None or label.get("reference_action") != decision["reference_action"]:
            raise ValueError("Private label mismatch: " + did)
        source = decision["source"]
        mode = decision.get("generation_mode", "unknown")
        if not include_fallback and mode not in REVIEWED_MODES:
            skipped["non_reviewed:" + str(mode)] += 1
            continue
        # The current private Environment/Global Judges require Stage4
        # pre-action snapshots. Legacy DA four-dimensional Judge can also
        # evaluate Stage3 clean turns using its expert reference.
        if reward_mode != "da" and source != "untrusted":
            skipped["clean_without_stage4_reward_state"] += 1
            continue
        state = label.get("reward_state") if source == "untrusted" else None
        if reward_mode != "da":
            if not isinstance(state, dict):
                skipped["missing_private_reward_state"] += 1
                continue
            if state.get("scorable") is not True:
                skipped["unscorable_private_reward_state"] += 1
                continue
            if state.get("decision_round") != decision["decision_round"]:
                raise ValueError("Private decision round mismatch: " + did)
        sid = sample_id(decision)
        if sid in seen:
            raise ValueError("Duplicate Stage2 GRPO sample identity: " + sid)
        seen.add(sid)
        action = decision["reference_action"]
        params = action["params"]
        tasks = params.get("tasks", []) if action["action"] == "delegate_task" else []
        if not isinstance(tasks, list) or any(
            not isinstance(t, dict) or
            not isinstance(t.get("task_instruction"), str) or
            not isinstance(t.get("model"), str) or
            not isinstance(t.get("tools", []), list)
            for t in tasks
        ):
            raise ValueError("Invalid expert task specification: " + did)
        if not isinstance(label.get("ground_truth"), str):
            raise ValueError("Missing private ground truth: " + did)
        split = splits[scene]
        row = {
            "data_source": "maestro_stage3" if source == "clean" else "maestro_stage4",
            "prompt": decision["prompt"],
            "reward_model": {"style": "rule", "ground_truth": label["ground_truth"]},
            "extra_info": {
                "sample_id": sid,
                "decision_id": did,
                "scene_id": scene,
                "case_id": decision["case_id"],
                "question": decision["question"],
                "answer": label["ground_truth"],
                "subtask_history": decision["visible_state"]["subtask_history"],
                "has_expert": True,
                "step_index": decision["decision_round"],
                "expert_action": action["action"],
                "expert_params": params,
                "expert_tools": [tool for task in tasks for tool in task.get("tools", [])],
                "expert_task_instructions": [task["task_instruction"] for task in tasks],
                "expert_num_subtasks": len(tasks),
                "expert_answer": params.get("answer", "")
                if action["action"] == "complete" else "",
            },
        }
        result[split].append(row)
        sources[source] += 1
        ph = sha256(decision["prompt"][0]["content"].encode("utf-8")).hexdigest()
        ah = sha256(json.dumps(action, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        prompt_groups[ph]["splits"].add(split)
        prompt_groups[ph]["actions"].add(ah)
        prompt_groups[ph]["ids"].append(did)
        exact[(ph, ah)] += 1
        if source == "untrusted" and isinstance(state, dict):
            private_states.append({
                "task_id": scene,
                "round_sample_id": decision["case_id"],
                "decision_round": decision["decision_round"],
                "private_only": True,
                "state": state,
            })
    leaked = [key for key, group in prompt_groups.items() if len(group["splits"]) > 1]
    if leaked:
        raise ValueError("Cross-split identical model prompts: " + ",".join(leaked[:3]))
    if not result["train"]:
        raise ValueError(
            "No reward-compatible training decisions; excluded=" +
            json.dumps(dict(skipped), sort_keys=True)
        )
    conflicts = [g for g in prompt_groups.values() if len(g["actions"]) > 1]
    quality = {
        "selected_by_source": dict(sorted(sources.items())),
        "selected_counts": {split: len(result[split]) for split in SPLITS},
        "excluded": dict(sorted(skipped.items())),
        "unique_prompt_count": len(prompt_groups),
        "exact_duplicate_rows": sum(n - 1 for n in exact.values()),
        "conflicting_prompt_groups": len(conflicts),
        "conflict_examples": [g["ids"][:8] for g in conflicts[:10]],
        "empty_eval_splits": [split for split in ("val", "test") if not result[split]],
    }
    return result, private_states, quality


def check_prompt_lengths(rows: list[dict], tokenizer_path: str | None,
                         limit: int) -> None:
    if tokenizer_path is None:
        return
    from transformers import AutoTokenizer
    from dataset.tokenization import extract_token_ids
    tokenizer = AutoTokenizer.from_pretrained(
        str(absolute(tokenizer_path)), trust_remote_code=False,
    )
    overlong = []
    for row in rows:
        ids = extract_token_ids(tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True,
        ))
        if len(ids) > limit:
            overlong.append((row["extra_info"]["sample_id"], len(ids)))
    if overlong:
        raise ValueError(f"{len(overlong)} Stage2 prompts exceed {limit}: {overlong[:5]}")


def prepare(*, source_dir: Path = DEFAULT_SOURCE,
            output_dir: Path = DEFAULT_OUTPUT,
            reward_mode: str = "global",
            seed: int = 17, train_ratio: float = 0.8, val_ratio: float = 0.1,
            split_map: Path | None = None,
            include_fallback: bool = False, strict_conflicts: bool = False,
            tokenizer: str | None = None, max_prompt_tokens: int = 12288,
            dry_run: bool = False, overwrite: bool = False,
            reuse_if_unchanged: bool = False, progress: bool = False) -> dict:
    source = absolute(source_dir)
    output = absolute(output_dir)
    if source == output or source in output.parents:
        raise ValueError("Stage2 output must not overlap canonical source")
    if max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must be positive")
    decisions, teachers, scenes, manifest = load_unified(source, progress=progress)
    private_meta = (verify_private_context(source, decisions=decisions,
                  require_complete=False) if reward_mode == "global" else None)
    ratios = (train_ratio, val_ratio, 1 - train_ratio - val_ratio)
    splits = load_splits(absolute(split_map), scenes) if split_map else make_splits(
        scenes, seed=seed, ratios=ratios,
    )
    rows, private, quality = convert(
        decisions, teachers, splits["scene_to_split"],
        reward_mode=reward_mode, include_fallback=include_fallback,
        progress=progress,
    )
    if reward_mode == "global":
        validate_selected_claims(source, rows)
    if strict_conflicts and quality["conflicting_prompt_groups"]:
        raise ValueError("Conflicting expert references for identical prompts")
    for split in SPLITS:
        check_prompt_lengths(rows[split], tokenizer, max_prompt_tokens)
    summary = {
        "schema": SCHEMA,
        "source_dir": str(source),
        "source_manifest_sha256": digest(source / "manifest.json"),
        "source_file_sha256": manifest["base_file_sha256"],
        "private_context_file_sha256": private_meta["file_sha256"] if private_meta else None,
        "output_dir": str(output),
        "reward_mode": reward_mode,
        "include_fallback": include_fallback,
        "reviewed_only": not include_fallback,
        "scene_split": splits,
        "quality": quality,
        "counts": quality["selected_counts"],
        "private_state_rows": len(private),
        "tokenizer_checked": bool(tokenizer),
        "max_prompt_tokens": max_prompt_tokens,
        "trainer": "unsloth_trl_grpo",
    }
    if dry_run:
        return summary
    files = [output / (split + ".parquet") for split in SPLITS] + [
        output / "summary.json", output / "splits.json",
        output / "quality_report.json", output / "private/decision_reward_states.jsonl",
    ] + ([output / name for name in PRIVATE_FILES] if reward_mode == "global" else [])

    if output.exists() and any(output.iterdir()):
        if reuse_if_unchanged and all(f.is_file() for f in files):
            old = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            if old == summary:
                return {**summary, "reused": True}
        if not overwrite:
            raise FileExistsError(
                f"Stage2 data already exists with a different recipe: {output}. "
                "Use a fresh DA_DATA or SKIP_COMPLETED=0 to rebuild."
            )
    for name, expected in manifest["base_file_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError("Canonical dataset mutated during preparation: " + name)
    import pyarrow as pa
    import pyarrow.parquet as pq
    if output.exists() and any(output.iterdir()):
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for split in tqdm(SPLITS, desc="Write Stage2 GRPO Parquet", unit="split",
                      disable=not progress):
        # Preserve a stable Arrow schema even when no reviewed data fall
        # into an evaluation split.
        table = (pa.Table.from_pylist(rows[split]) if rows[split] else
                 pa.Table.from_pylist(rows["train"][:1]).slice(0, 0))
        path = output / (split + ".parquet")
        pq.write_table(table, path)
        if pq.read_metadata(path).num_rows != len(rows[split]):
            raise RuntimeError("GRPO Parquet row count mismatch: " + split)
    if reward_mode == "global":
        # Private evaluator World/claims are copied byte-for-byte from the
        # frozen unified snapshot, not mutable Stage1/4 generation outputs.
        for name, expected in private_meta["file_sha256"].items():
            source_path, destination = source / name, output / name
            if digest(source_path) != expected:
                raise ValueError("Unified private evidence changed during preparation: " + name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination)
            if digest(destination) != expected:
                raise RuntimeError("Private evaluator sidecar copy checksum mismatch: " + name)
    sidecar = output / "private/decision_reward_states.jsonl"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in private),
        encoding="utf-8",
    )
    for name, payload in (
        ("summary.json", summary),
        ("splits.json", splits),
        ("quality_report.json", quality),
    ):
        (output / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--reward-mode", choices=REWARD_MODES, default="global")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--split-map", type=Path)
    p.add_argument("--include-fallback", action="store_true")
    p.add_argument("--strict-conflicts", action="store_true")
    p.add_argument("--tokenizer")
    p.add_argument("--max-prompt-tokens", type=int, default=12288)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--reuse-if-unchanged", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)
    print(json.dumps(prepare(
        source_dir=args.source_dir, output_dir=args.output_dir,
        reward_mode=args.reward_mode, seed=args.seed,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        split_map=args.split_map, include_fallback=args.include_fallback,
        strict_conflicts=args.strict_conflicts, tokenizer=args.tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        dry_run=args.dry_run, overwrite=args.overwrite,
        reuse_if_unchanged=args.reuse_if_unchanged,
        progress=not args.no_progress,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
