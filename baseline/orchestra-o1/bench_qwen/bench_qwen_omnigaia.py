"""Run Orchestra-o1 with a locally served Qwen Main-Agent.

The execution path is exactly the normal O1 baseline path. This launcher only
adds the local-Qwen model definitions before dispatching to the O1 entrypoint.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASELINE_ROOT = HERE.parent
REPO_ROOT = BASELINE_ROOT.parents[1]
for root in (REPO_ROOT, BASELINE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from base.engine.async_llm import LLMsConfig

MODEL_CONFIG = HERE / "model_config_qwen.yaml"
SHARED_MODEL_CONFIG = REPO_ROOT / "config/model_config.yaml"
DEFAULT_BENCH_CONFIG = HERE / "orchestra_o1_omnigaia_qwen_grpo.yaml"
BASELINE_ENTRYPOINT = BASELINE_ROOT / "bench_orchestra_o1_omnigaia.py"


def _read_models(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    models = raw.get("models", raw)
    return dict(models or {})


def _install_model_registry() -> None:
    models = _read_models(SHARED_MODEL_CONFIG)
    models.update(_read_models(MODEL_CONFIG))
    LLMsConfig._default_config = LLMsConfig(models)


def main() -> None:
    _install_model_registry()
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(DEFAULT_BENCH_CONFIG)])
    runpy.run_path(str(BASELINE_ENTRYPOINT), run_name="__main__")


if __name__ == "__main__":
    main()
