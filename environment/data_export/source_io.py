"""Shared *environment* source reader and generation-manifest checks.

No DA-SFT/DA-GRPO data preparation code is imported here.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid_json") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{number}: expected_json_object")
        result.append(item)
    return result


def digest(path: Path) -> str:
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_renderer_compatibility(expected_sha256: str | None,
                                 renderer: Path, source: str) -> None:
    if not expected_sha256 or expected_sha256 == digest(renderer):
        return
    # Previously approved O1 docstring-only rename (PR #137).
    current = renderer.read_text(encoding="utf-8")
    old = current.replace(
        "The on-disk per-turn reviewed-turn input stores",
        "The on-disk per-turn SFT input stores", 1,
    ).replace(
        "Rebuild a full baseline O1 prompt for an exported compact reviewed-turn turn.",
        "Rebuild a full baseline O1 prompt for an exported compact SFT turn.", 1,
    )
    if old != current and sha256(old.encode("utf-8")).hexdigest() == expected_sha256:
        return
    raise ValueError(f"{source}_o1_prompt_renderer_drift")


def settings_for(turns_path: Path, scenes_path: Path, stage: str) -> dict:
    path = Path(turns_path).parent / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if stage == "stage3":
        expected = manifest.get("scenes_sha256")
        cfg = manifest
        code_hash = (manifest.get("code_sha256") or {}).get("o1_compat.py")
    elif stage == "stage4":
        recorded = manifest.get("input_sha256") or []
        expected = recorded[0] if recorded else None
        cfg = manifest.get("settings") or {}
        code_hash = (manifest.get("source_code_sha256") or {}).get(
            "../base_orchestration/o1_compat.py"
        )
    else:
        raise ValueError("unknown_generation_stage")
    if not expected or digest(scenes_path) != expected:
        raise ValueError(f"{stage}_scene_manifest_hash_mismatch")
    check_renderer_compatibility(
        code_hash,
        ROOT / "environment/base_orchestration/o1_compat.py",
        stage,
    )
    max_attempts = cfg.get("max_attempts", cfg.get("max_rounds"))
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError(f"{stage}_invalid_max_attempts")
    sub_model, model_alias = cfg.get("sub_model"), cfg.get("model_alias")
    if not isinstance(sub_model, str) or not sub_model:
        raise ValueError(f"{stage}_invalid_sub_model")
    if not isinstance(model_alias, str) or not model_alias:
        raise ValueError(f"{stage}_invalid_model_alias")
    return {
        "max_attempts": max_attempts,
        "sub_model": sub_model,
        "model_alias": model_alias,
    }
