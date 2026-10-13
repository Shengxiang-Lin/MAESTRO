"""Run OmniGAIA with MAESTRO prospective orchestration.

Usage:
    python bench_orchestra_omnigaia.py \
      --config config/benchmarks/orchestra_omnigaia.yaml
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.engine.logs import logger
from benchmark.omnigaia.orchestra_benchmark import (
    OmniGAIAConfig,
    OmniGAIAOrchestraBenchmark,
)
from benchmark.omnigaia.tools import build_omnigaia_tools, resolve_tool_models
from orchestra.config import GAIAOrchestraConfig, ProspectiveConfig
from orchestra.runners.omnigaia_runner import OmniGAIARunner

DEFAULT_CONFIG_PATH = ROOT / "config/benchmarks/orchestra_omnigaia.yaml"


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OmniGAIA using MAESTRO prospective orchestration."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--max_concurrency", type=int, default=None)
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--skip_completed", type=str, default=None)
    parser.add_argument("--result_folder", type=str, default=None)
    parser.add_argument("--trajectory_folder", type=str, default=None)
    args = parser.parse_args()

    cfg = GAIAOrchestraConfig.load(args.config)
    prospective_cfg = ProspectiveConfig.load(args.config)
    if args.result_folder:
        cfg.result_folder = Path(args.result_folder).resolve()
    if args.trajectory_folder:
        cfg.trajectory_folder = Path(args.trajectory_folder).resolve()
    if not cfg.dataset_path.exists():
        logger.error(f"Dataset not found: {cfg.dataset_path}")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.timestamp = timestamp
    cfg.tool_models = resolve_tool_models(cfg.tool_models)
    runtime_models = cfg.prepare_model_backends()
    tools = build_omnigaia_tools(cfg.tool_models)

    benchmark = OmniGAIAOrchestraBenchmark(
        OmniGAIAConfig(
            dataset_path=cfg.dataset_path,
            attachments_dir=cfg.attachments_dir,
            level_filter=(
                [str(level) for level in cfg.level_filter]
                if cfg.level_filter
                else None
            ),
            max_tasks=cfg.max_tasks,
            result_folder=cfg.result_folder,
            trajectory_folder=cfg.trajectory_folder,
        ),
        tools=tools,
    )
    levels = benchmark.list_levels()
    if args.tasks:
        selected = {
            value.strip() for value in args.tasks.split(",") if value.strip()
        }
        levels = [
            level
            for level in levels
            if str(level.get("id") or level.get("task_id")) in selected
        ]
    elif cfg.max_tasks and len(levels) > cfg.max_tasks:
        levels = levels[: cfg.max_tasks]

    if args.skip_completed:
        path = Path(args.skip_completed)
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                completed = {
                    str(row.get("task_id") or row.get("id"))
                    for row in csv.DictReader(handle)
                    if row.get("task_id") or row.get("id")
                }
            levels = [
                level
                for level in levels
                if str(level.get("id") or level.get("task_id")) not in completed
            ]

    cfg.result_folder.mkdir(parents=True, exist_ok=True)
    cfg.trajectory_folder.mkdir(parents=True, exist_ok=True)
    model_manifest = {
        "main_model": cfg.main_model,
        "sub_models": cfg.sub_models,
        "tool_models": cfg.tool_models,
        "evaluation_model": cfg.evaluation_model,
        "local_runtime_models": runtime_models,
    }
    manifest_path = cfg.result_folder / f"model_roles_{timestamp}.json"
    manifest_path.write_text(
        json.dumps(model_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = cfg.result_folder / f"omnigaia_orchestra_{timestamp}.csv"
    runner = OmniGAIARunner(
        benchmark=benchmark,
        main_model=cfg.main_model,
        sub_models=cfg.sub_models,
        max_attempts=cfg.max_attempts,
        omnigaia_tools=tools,
        tool_models=cfg.tool_models,
        evaluation_model=cfg.evaluation_model,
        prospective_config=prospective_cfg,
    )
    logger.info(
        f"[MAESTRO] tasks={len(levels)}, main={cfg.main_model}, "
        f"sub={cfg.sub_models}, tools={cfg.tool_models}, "
        f"evaluation={cfg.evaluation_model}, local={runtime_models}, "
        f"prospective={prospective_cfg}"
    )
    results = await runner.run_levels(
        levels=levels,
        max_concurrency=args.max_concurrency or cfg.max_concurrency,
        csv_path=csv_path,
        trajectory_folder=cfg.trajectory_folder,
        timestamp=timestamp,
    )
    success = sum(1 for result in results.values() if result.get("success"))
    logger.info(
        f"[MAESTRO] completed={len(results)}, success={success}, "
        f"results={csv_path}, model_roles={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
