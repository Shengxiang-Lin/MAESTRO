"""Shared *baseline-side* dataset input, Scene splitting and audit utilities.

The environment emits method-agnostic records only. Trainers convert, filter,
split and write their own data under baseline/<method>/data/.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import shutil

from tqdm.auto import tqdm

from environment.data_export.audit import audit_output
from environment.data_export.source_io import read_jsonl, digest
from environment.data_export.splits import load_splits, make_splits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "environment/data/unified"

REVIEWED_MODES = frozenset({
    "llm_reviewed", "llm_reviewed_completion",
    "llm_reviewed_contract_frozen",
})
SPLITS = ("train", "val", "test")


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_base(source_dir: Path, *, progress: bool = False) -> tuple[list[dict], dict[str, dict], list[str], dict]:
    """Verify frozen source bytes and exact generic Scene/decision/teacher contract."""
    root = repo_path(source_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("missing_unified_manifest:" + str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "maestro.dataset.v1":
        raise ValueError("invalid_unified_dataset_schema")
    audit_output(root, verify_manifest=True, progress=progress)
    decisions = read_jsonl(root / "decisions.jsonl", progress=progress,
                           desc="Load base decisions")
    labels = read_jsonl(root / "private/decision_labels.jsonl", progress=progress,
                        desc="Load base labels")
    cases = read_jsonl(root / "cases.jsonl", progress=progress,
                       desc="Load base Cases")
    ids = [row["scene_id"] for row in cases if row["kind"] == "clean"]
    teacher = {row["decision_id"]: row for row in labels}
    if len(teacher) != len(labels):
        raise ValueError("duplicate_unified_teacher")
    return decisions, teacher, ids, manifest


def get_splits(scene_ids: list[str], *, seed: int = 17,
               ratios=(0.8, 0.1, 0.1),
               split_map: Path | None = None) -> dict:
    """This function is called ONLY by baseline-specific prepare_data scripts."""
    if split_map is not None:
        return load_splits(repo_path(split_map), scene_ids)
    return make_splits(scene_ids, seed=seed, ratios=ratios)


def select_rows(decisions: list[dict], *, include_fallback: bool = False,
                progress: bool = False):
    kept, skipped = [], Counter()
    bar = tqdm(decisions, desc="Filter Main Decisions", unit="decision",
               disable=not progress)
    for index, decision in enumerate(bar):
        mode = decision.get("generation_mode", "unknown")
        if not include_fallback and mode not in REVIEWED_MODES:
            skipped[str(mode)] += 1
        else:
            kept.append(decision)
        if progress and (index + 1) % 128 == 0:
            bar.set_postfix(accepted=len(kept), filtered=sum(skipped.values()),
                            refresh=False)
    if progress:
        bar.set_postfix(accepted=len(kept), filtered=sum(skipped.values()),
                        refresh=True)
    if not kept:
        raise ValueError("no_reviewed_positive_main_decisions")
    return kept, skipped


def audit_selection(decisions: list[dict], splits: dict[str, str],
                    *, progress: bool = False) -> dict:
    """Within-split duplicate/label conflicts and cross-split prompt overlap.

    Conflicts are counted, not silently discarded; identical model-visible
    prompts must never leak across train/val/test.
    """
    grouped = defaultdict(lambda: {"splits": set(), "answers": set(), "ids": []})
    exact, by_split, by_source, per_scene, actions = (
        Counter(), Counter(), Counter(), Counter(), Counter()
    )
    for row in tqdm(decisions, desc="Audit training selection",
                    unit="decision", disable=not progress, leave=False):
        split = splits[row["scene_id"]]
        prompt = row["prompt"][0]["content"]
        label = row["reference_action"]
        ph = sha256(prompt.encode("utf-8")).hexdigest()
        ah = sha256(json.dumps(
            label, ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")).hexdigest()
        data = grouped[ph]
        data["splits"].add(split)
        data["answers"].add(ah)
        data["ids"].append(row["decision_id"])
        exact[(ph, ah)] += 1
        by_split[split] += 1
        by_source[row["source"]] += 1
        per_scene[row["scene_id"]] += 1
        actions[label["action"]] += 1
    leaked = [k for k, val in grouped.items() if len(val["splits"]) > 1]
    if leaked:
        raise ValueError(
            "cross_split_identical_prompts:" + ",".join(sorted(leaked)[:3])
        )
    conflicts = [(k, val) for k, val in grouped.items()
                 if len(val["answers"]) > 1]
    exact_extras = sum(n - 1 for n in exact.values())
    warnings = []
    if exact_extras:
        warnings.append("duplicate_prompt_and_reference_in_training_view")
    if conflicts:
        warnings.append("identical_prompt_has_multiple_supervised_actions")
    if by_source["untrusted"] > 9 * max(1, by_source["clean"]):
        warnings.append("untrusted_to_clean_decision_ratio_exceeds_9")
    return {
        "decision_counts": {k: by_split[k] for k in SPLITS},
        "source_counts": dict(sorted(by_source.items())),
        "action_counts": dict(sorted(actions.items())),
        "unique_prompts": len(grouped),
        "exact_duplicate_decision_rows": exact_extras,
        "conflicting_prompt_groups": len(conflicts),
        "conflict_examples": [{
            "prompt_sha256": k, "decision_ids": v["ids"][:8],
        } for k, v in sorted(conflicts)[:10]],
        "scenes_with_decisions": len(per_scene),
        "top_scenes": [{
            "scene_id": k, "decisions": n,
        } for k, n in sorted(
            per_scene.items(), key=lambda item: (-item[1], item[0])
        )[:10]],
        "warnings": warnings,
    }


def prepare_directory(output_dir: Path, *, overwrite: bool = False) -> Path:
    out = repo_path(output_dir)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(
                "baseline_data_directory_exists_use_--overwrite:" + str(out)
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ), encoding="utf-8")


def check_frozen_source(source_dir: Path, expected: dict) -> None:
    """Detect source mutation while a baseline is transforming it."""
    root = repo_path(source_dir)
    for relative, hash_value in expected["base_file_sha256"].items():
        if digest(root / relative) != hash_value:
            raise ValueError("source_changed_during_preparation:" + relative)
