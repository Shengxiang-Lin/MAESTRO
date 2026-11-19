"""Frozen Scene-level splits shared by all MAESTRO training methods."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


VERSION = "maestro.scene_splits.v1"


def make_splits(scene_ids, *, seed=17, ratios=(0.8, 0.1, 0.1)):
    if (len(ratios) != 3 or
            any(type(x) not in (float, int) or x < 0 for x in ratios) or
            abs(sum(ratios) - 1) > 1e-9):
        raise ValueError("invalid_dataset_split_ratios")
    if type(seed) is not int:
        raise ValueError("invalid_dataset_split_seed")
    ids = list(scene_ids)
    if len(ids) != len(set(ids)) or any(
        not isinstance(x, str) or not x.strip() for x in ids
    ):
        raise ValueError("invalid_or_duplicate_scene_ids")
    ids.sort(key=lambda x: (sha256(f"{seed}:{x}".encode()).hexdigest(), x))
    n = len(ids)
    if not n:
        raise ValueError("no_scenes_to_split")
    if n < sum(x > 0 for x in ratios):
        raise ValueError("too_few_scenes_for_requested_nonempty_splits")
    # Largest remainder, with a guaranteed nonempty group for each
    # nonzero requested split. No episode, fault variant or decision
    # from one Scene may land in another split.
    counts = [1 if r else 0 for r in ratios]
    left = n - sum(counts)
    requested = [max(0, r * n - counts[i]) for i, r in enumerate(ratios)]
    total = sum(requested)
    raw = [left * x / total if total else 0 for x in requested]
    shares = [int(x) for x in raw]
    for i in sorted(range(3), key=lambda i: (-(raw[i]-shares[i]), i))[
        :left-sum(shares)
    ]:
        shares[i] += 1
    counts = [counts[i]+shares[i] for i in range(3)]
    mapping = {}
    offset = 0
    for name, count in zip(("train", "val", "test"), counts):
        mapping.update({scene: name for scene in ids[offset:offset+count]})
        offset += count
    return {
        "schema": VERSION, "seed": seed,
        "ratios": list(ratios), "scene_to_split": dict(sorted(mapping.items())),
        "counts": dict(zip(("train", "val", "test"), counts)),
    }


def verify_splits(data: dict, scene_ids) -> dict[str, str]:
    if not isinstance(data, dict) or data.get("schema") != VERSION:
        raise ValueError("wrong_dataset_split_schema")
    mapping = data.get("scene_to_split")
    if (not isinstance(mapping, dict)
            or set(mapping) != set(scene_ids)
            or any(v not in {"train", "val", "test"} for v in mapping.values())):
        raise ValueError("dataset_split_scene_coverage_mismatch")
    return mapping


def load_splits(path: Path, scene_ids) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_splits(data, scene_ids)
    return data
