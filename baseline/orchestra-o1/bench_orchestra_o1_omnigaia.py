"""Run the Orchestra-o1 baseline on OmniGAIA.

From the MAESTRO repository root:
    python baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
      --config baseline/orchestra-o1/config/benchmarks/orchestra_o1_omnigaia.yaml
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

BASELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_ROOT.parents[1]
for root in (REPO_ROOT, BASELINE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from base.engine.logs import logger
from benchmark.omnigaia.orchestra_benchmark import (
    OmniGAIAConfig,
    OmniGAIAOrchestraBenchmark,
)
from benchmark.omnigaia.tools import (
    build_omnigaia_tools,
    load_web_tool_configs,
    resolve_tool_models,
)
from orchestra.config import GAIAOrchestraConfig
from orchestra.runners.base_omnigaia_runner import BaseOmniGAIARunner
from orchestra_o1.main_agent import MainAgent as O1MainAgent
from orchestra_o1.prompts.omnigaia import OmniGAIAMainAgentPrompt

DEFAULT_CONFIG_PATH = (
    BASELINE_ROOT / "config/benchmarks/orchestra_o1_omnigaia.yaml"
)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OmniGAIA benchmark using Orchestra-o1 baseline."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--max_concurrency", type=int, default=None)
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--skip_completed", type=str, default=None)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("OmniGAIA Benchmark with Orchestra-o1 baseline")
    logger.info("=" * 60)

    cfg = GAIAOrchestraConfig.load(args.config)
    search_cfg, browse_cfg = load_web_tool_configs(args.config)
    cfg.prepare_dataset()
    logger.info(f"Dataset: type={cfg.dataset_type}, source={cfg.dataset_source_path}, jsonl={cfg.dataset_path}")
    if not cfg.dataset_path.exists():
        logger.error(f"Dataset not found: {cfg.dataset_path}")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.timestamp = timestamp
    cfg.tool_models = resolve_tool_models(cfg.tool_models)
    runtime_models = cfg.prepare_model_backends()
    tools = build_omnigaia_tools(
        cfg.tool_models,
        search=search_cfg,
        browse=browse_cfg,
    )

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
    if not levels:
        logger.error("No tasks found in dataset!")
        return 1

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
        "dataset_type": cfg.dataset_type,
        "dataset_source_path": str(cfg.dataset_source_path),
        "dataset_path": str(cfg.dataset_path),
        "attachments_dir": str(cfg.attachments_dir),
        "main_model": cfg.main_model,
        "sub_models": cfg.sub_models,
        "search": search_cfg,
        "browse": browse_cfg,
        "tool_models": cfg.tool_models,
        "evaluation_model": cfg.evaluation_model,
        "trace_summary_model": cfg.trace_summary_model,
        "local_runtime_models": runtime_models,
    }
    manifest_path = cfg.result_folder / f"model_roles_{timestamp}.json"
    manifest_path.write_text(
        json.dumps(model_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = cfg.result_folder / f"{cfg.dataset_type}_orchestra_o1_{timestamp}.csv"
    runner = BaseOmniGAIARunner(
        benchmark=benchmark,
        main_model=cfg.main_model,
        sub_models=cfg.sub_models,
        max_attempts=cfg.max_attempts,
        omnigaia_tools=tools,
        tool_models=cfg.tool_models,
        evaluation_model=cfg.evaluation_model,
        trace_summary_model=cfg.trace_summary_model,
        main_agent_cls=O1MainAgent,
        prompt_builder=OmniGAIAMainAgentPrompt,
        log_prefix="Orchestra-o1",
        summarize_trace=True,
    )
    logger.info(
        f"[Orchestra-o1] main={cfg.main_model}, sub={cfg.sub_models}, "
        f"search={search_cfg}, browse={browse_cfg}, tools={cfg.tool_models}, "
        f"evaluation={cfg.evaluation_model}, "
        f"trace_summary={cfg.trace_summary_model}, local={runtime_models}"
    )
    results = await runner.run_levels(
        levels=levels,
        max_concurrency=args.max_concurrency or cfg.max_concurrency,
        csv_path=csv_path,
        trajectory_folder=cfg.trajectory_folder,
        timestamp=timestamp,
    )
    total = len(results)
    success = sum(1 for result in results.values() if result.get("success"))
    reward = sum(float(result.get("reward", 0) or 0) for result in results.values())
    logger.info("\n" + "=" * 60)
    logger.info(f"Total tasks: {total}")
    logger.info(f"Successful: {success}/{total}")
    logger.info(f"Total reward: {reward:.2f}")
    logger.info(f"Results: {csv_path}")
    logger.info(f"Model roles: {manifest_path}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
