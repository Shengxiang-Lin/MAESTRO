"""Compatibility launcher for the Orchestra-o1 local-Qwen benchmark.

Local/API model definitions now live in the shared ``config/model_config.yaml``.
This wrapper only selects the historical GRPO benchmark config and dispatches to
the normal Orchestra-o1 entrypoint.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE_ROOT = HERE.parent
REPO_ROOT = BASELINE_ROOT.parents[1]
for root in (REPO_ROOT, BASELINE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

DEFAULT_BENCH_CONFIG = HERE / "orchestra_o1_omnigaia_qwen_grpo.yaml"
BASELINE_ENTRYPOINT = BASELINE_ROOT / "bench_orchestra_o1_omnigaia.py"


def main() -> None:
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(DEFAULT_BENCH_CONFIG)])
    runpy.run_path(str(BASELINE_ENTRYPOINT), run_name="__main__")


if __name__ == "__main__":
    main()
