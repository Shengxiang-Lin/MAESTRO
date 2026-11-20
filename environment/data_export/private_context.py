"""Frozen, evaluator-only Scene/claim context for unified MAESTRO datasets.

These private sidecars are NOT policy prompts or baseline labels. Once
attached, downstream GRPO never consults mutable Stage1/4 source directories.
The optional attach command upgrades old unified exports without touching
their Cases, Decisions, teacher labels, splits, or training prompts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .source_io import digest, read_jsonl

SCENE_FILE = "private/scenes.jsonl"
CLAIM_FILE = "private/claim_state_supervision.jsonl"
PRIVATE_FILES = (SCENE_FILE, CLAIM_FILE)
CONTEXT_SCHEMA = "maestro.private_evaluator_context.v1"
CLAIM_SCHEMA = "claim_source_dependency"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def _write_json(path: Path, obj: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def collect_private_context(
    scenes: dict[str, dict], decisions: list[dict], claim_path: Path,
) -> tuple[list[dict], list[dict], dict]:
    """Keep only selected Scenes and exact published Stage4 pre-action claims."""
    selected_scene_ids = {row["scene_id"] for row in decisions}
    if not selected_scene_ids <= scenes.keys():
        raise ValueError("private_context_unknown_scene")
    scene_rows = [scenes[sid] for sid in sorted(selected_scene_ids)]
    for scene in scene_rows:
        if not isinstance(scene.get("macro_graph"), dict) or not isinstance(
            scene.get("world"), dict
        ):
            raise ValueError("private_context_scene_missing_graph_or_world")

    required = {
        (row["case_id"], row["decision_round"])
        for row in decisions
        if row["source"] == "untrusted" and row["decision_round"] > 1
    }
    scene_for_case = {
        row["case_id"]: row["scene_id"] for row in decisions
        if row["source"] == "untrusted"
    }
    claims = []
    found = set()
    if claim_path.is_file():
        for record in read_jsonl(claim_path):
            case, rnd = record.get("round_sample_id"), record.get("main_can_act_from_round")
            if (not isinstance(case, str) or type(rnd) is not int
                    or rnd < 2 or record.get("schema") != CLAIM_SCHEMA
                    or record.get("private_only") is not True):
                raise ValueError("private_context_invalid_claim_snapshot")
            key = (case, rnd)
            if key not in required:
                continue
            if key in found:
                raise ValueError("private_context_duplicate_claim_snapshot")
            if record.get("task_id") != scene_for_case[case]:
                raise ValueError("private_context_claim_scene_mismatch")
            for claim in record.get("claim_nodes", []):
                observed = claim.get("observed_in_round")
                if type(observed) is not int or not 1 <= observed < rnd:
                    raise ValueError("private_context_future_claim")
            found.add(key)
            claims.append(record)
    claims.sort(key=lambda x: (x["round_sample_id"], x["main_can_act_from_round"]))
    return scene_rows, claims, {
        "scene_count": len(scene_rows),
        "required_claim_snapshots": len(required),
        "present_claim_snapshots": len(found),
        "missing_claim_snapshots": len(required - found),
        "claim_source_present": claim_path.is_file(),
    }


def publish_private_context(
    output_dir: Path, *, scenes: dict[str, dict], decisions: list[dict],
    claim_path: Path, manifest: dict,
) -> dict:
    """Write private evaluator inputs and bind them to the base manifest."""
    root = Path(output_dir)
    scene_rows, claim_rows, counts = collect_private_context(
        scenes, decisions, Path(claim_path)
    )
    _write_jsonl(root / SCENE_FILE, scene_rows)
    _write_jsonl(root / CLAIM_FILE, claim_rows)
    hashes = {name: digest(root / name) for name in PRIVATE_FILES}
    manifest["private_context"] = {
        "schema": CONTEXT_SCHEMA,
        "file_sha256": hashes,
        "claim_source_sha256": (
            digest(claim_path) if claim_path.is_file() else None
        ),
        **counts,
    }
    return manifest["private_context"]


def verify_private_context(
    source_dir: Path, *, decisions: list[dict] | None = None,
    require_complete: bool = True,
) -> dict:
    """Validate immutable evaluator sidecars; never look at raw Environment."""
    root = Path(source_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    meta = manifest.get("private_context")
    if (not isinstance(meta, dict) or meta.get("schema") != CONTEXT_SCHEMA
            or set(meta.get("file_sha256", {})) != set(PRIVATE_FILES)):
        raise ValueError(
            "unified_missing_private_context: this dataset predates frozen "
            "Global Judge inputs. Use python -m environment.data_export.private_context "
            "--unified-dir ... --scenes ... --stage4-dir ... with the ORIGINAL "
            "matching source files, or re-export the canonical dataset."
        )
    for name, expected in meta["file_sha256"].items():
        path = root / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError("unified_private_context_checksum_changed:" + name)
    scenes = read_jsonl(root / SCENE_FILE)
    if (len(scenes) != len({r.get("task_id") for r in scenes})
            or any(not isinstance(row.get("macro_graph"), dict)
                   or not isinstance(row.get("world"), dict) for row in scenes)):
        raise ValueError("unified_private_scene_integrity")
    if len(scenes) != meta.get("scene_count"):
        raise ValueError("unified_private_scene_count_mismatch")
    claims = read_jsonl(root / CLAIM_FILE)
    if len(claims) != meta.get("present_claim_snapshots"):
        raise ValueError("unified_private_claim_count_mismatch")
    if decisions is None:
        decisions = read_jsonl(root / "decisions.jsonl")
    scene_ids = {row["scene_id"] for row in decisions}
    if scene_ids != {row["task_id"] for row in scenes}:
        raise ValueError("unified_private_scene_decision_mismatch")
    # Reuse the exact join/chronology checks used at export; do not accept
    # extra or out-of-time snapshots not attached to a published decision.
    expected_claims = {
        (row["case_id"], row["decision_round"])
        for row in decisions
        if row["source"] == "untrusted" and row["decision_round"] > 1
    }
    seen = set()
    for claim in claims:
        key = (claim.get("round_sample_id"), claim.get("main_can_act_from_round"))
        if key not in expected_claims or key in seen:
            raise ValueError("unified_private_claim_identity_mismatch")
        if claim.get("schema") != CLAIM_SCHEMA or claim.get("private_only") is not True:
            raise ValueError("unified_private_claim_schema_mismatch")
        for node in claim.get("claim_nodes", []):
            observed = node.get("observed_in_round")
            if type(observed) is not int or not 1 <= observed < key[1]:
                raise ValueError("unified_private_claim_future_information")
        seen.add(key)
    if meta.get("required_claim_snapshots") != len(expected_claims):
        raise ValueError("unified_private_claim_required_count_mismatch")
    if meta.get("missing_claim_snapshots") != len(expected_claims - seen):
        raise ValueError("unified_private_claim_missing_count_mismatch")
    if require_complete and expected_claims - seen:
        raise ValueError(
            f"unified_global_judge_missing_claim_snapshots:{len(expected_claims-seen)}. "
            "Recreate the unified private context from matching Stage4 claim "
            "snapshots; never substitute future or different-run evidence."
        )
    return meta


def attach_to_unified(
    *, unified_dir: Path, scenes_path: Path, stage4_dir: Path,
) -> dict:
    """Backfill a frozen old export after strict byte-for-byte provenance checks."""
    from .audit import audit_output

    root = Path(unified_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "maestro.dataset.v1":
        raise ValueError("unified_dataset_schema_mismatch")
    audit_output(root, verify_manifest=True)
    expected = manifest.get("source_sha256", {})
    source_paths = {
        "scenes": Path(scenes_path),
        "stage4_manifest": Path(stage4_dir) / "manifest.json",
        "stage4_main_turns": Path(stage4_dir) / "main_turns.jsonl",
    }
    for name, path in source_paths.items():
        if not path.is_file() or digest(path) != expected.get(name):
            raise ValueError(
                f"source_mismatch:{name}:{path}. Use the ORIGINAL version "
                "recorded in unified/manifest.json; do not overwrite hashes."
            )
    if "private_context" in manifest:
        verify_private_context(root, require_complete=False)
        raise FileExistsError(
            "unified_already_has_private_context; create a new export to change it"
        )
    scenes = {
        row["task_id"]: row for row in read_jsonl(Path(scenes_path))
    }
    decisions = read_jsonl(root / "decisions.jsonl")
    summary = publish_private_context(
        root, scenes=scenes, decisions=decisions,
        claim_path=Path(stage4_dir) / "private/claim_state_supervision.jsonl",
        manifest=manifest,
    )
    # Commit manifest LAST; if interrupted, the old manifest remains intact.
    _write_json(manifest_path, manifest)
    verify_private_context(root, decisions=decisions, require_complete=False)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Freeze Global Judge private Scene/claim context into an old unified export."
    )
    parser.add_argument("--unified-dir", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--stage4-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = attach_to_unified(
        unified_dir=args.unified_dir, scenes_path=args.scenes,
        stage4_dir=args.stage4_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
