"""Pinned upstream runtime helpers for the environment pipeline.

Stage 1 and Stage 2 intentionally execute the implementation pinned in the
``vendor/orchestra-o2`` git submodule.  This keeps the migrated MAESTRO pipeline
behavior-identical to the current Orchestra-O3 environment while exposing a
clean three-stage interface in this repository.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENV_DIR = REPO_ROOT / "environment"
LOCAL_DATA_DIR = LOCAL_ENV_DIR / "data"
LOCAL_WORLD_SOURCE_DIR = REPO_ROOT / "data" / "world-source"
UPSTREAM_REPO = REPO_ROOT / "vendor" / "orchestra-o2"
UPSTREAM_O3_DIR = UPSTREAM_REPO / "orchestra-o3"
UPSTREAM_ENV_DIR = UPSTREAM_O3_DIR / "environment"
UPSTREAM_DATA_DIR = UPSTREAM_ENV_DIR / "data"
UPSTREAM_WORLD_SOURCE_DIR = UPSTREAM_REPO / "data" / "world-source"
PINNED_UPSTREAM_COMMIT = "28a3b520cbfb71223e72aa90657b0893e45db683"


def require_upstream() -> None:
    marker = UPSTREAM_ENV_DIR / "build_all.py"
    if marker.is_file():
        return
    raise SystemExit(
        "Pinned environment implementation is not initialized. Run:\n"
        "  git submodule update --init --recursive vendor/orchestra-o2\n"
        f"Expected: {marker}"
    )


def _link_directory(link: Path, target: Path, *, label: str) -> None:
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        raise SystemExit(f"{label} symlink points elsewhere: {link} -> {link.resolve()}")
    if link.exists():
        if link.is_dir() and not any(link.iterdir()):
            link.rmdir()
        else:
            raise SystemExit(
                f"Cannot attach MAESTRO {label}: upstream path already contains data: {link}. "
                "Move it away or reuse that checkout explicitly."
            )
    link.symlink_to(target.resolve(), target_is_directory=True)


def prepare_runtime_links() -> None:
    """Keep generated artifacts in MAESTRO while running the untouched upstream code."""

    require_upstream()
    _link_directory(UPSTREAM_DATA_DIR, LOCAL_DATA_DIR, label="environment/data")

    # The original implementation expects data/world-source at the old repository
    # root.  If MAESTRO has the same source bank, expose it to the pinned runtime
    # without changing any Stage-1 path logic.
    if LOCAL_WORLD_SOURCE_DIR.exists() and not UPSTREAM_WORLD_SOURCE_DIR.exists():
        _link_directory(
            UPSTREAM_WORLD_SOURCE_DIR,
            LOCAL_WORLD_SOURCE_DIR,
            label="data/world-source",
        )


def run_upstream_module(module: str, args: Iterable[str]) -> int:
    """Execute one pinned upstream module with its original CLI semantics."""

    prepare_runtime_links()
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    command = [sys.executable, "-m", module, *list(args)]
    print("[MAESTRO environment] upstream commit:", PINNED_UPSTREAM_COMMIT)
    print("[MAESTRO environment] running:", " ".join(command))
    completed = subprocess.run(command, cwd=UPSTREAM_O3_DIR, env=env, check=False)
    return int(completed.returncode)
