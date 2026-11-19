"""Use the original O1 OmniGAIA runner with a reliability-review plugin."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from base.engine.logs import logger
from benchmark.omnigaia.orchestra_benchmark import OmniGAIAConfig, OmniGAIAOrchestraBenchmark
from benchmark.omnigaia.tools import build_omnigaia_tools, load_web_tool_configs, resolve_tool_models
from orchestra.config import GAIAOrchestraConfig
from orchestra.runners.base_omnigaia_runner import BaseOmniGAIARunner
from orchestra.prompts.omnigaia import OmniGAIAMainAgentPrompt
from reliability_core import METHODS, ReliabilityMainAgent


async def amain(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--config", default=str(
        HERE / "orchestra-o1/config/omnigaia/pretest/orchestra_o1_omnigaia.yaml"))
    parser.add_argument("--options", default="{}", help="JSON settings for reviewer")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--task-ids-file", default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    args = parser.parse_args(argv)
    options = json.loads(args.options)
    if not isinstance(options, dict):
        parser.error("--options must be JSON object")
    cfg = GAIAOrchestraConfig.load(args.config)
    if args.task_ids_file:
        path = Path(args.task_ids_file)
        cfg.task_ids_file = (path if path.is_absolute() else ROOT / path).resolve()
    search, browse = load_web_tool_configs(args.config)
    cfg.prepare_dataset()
    cfg.tool_models = resolve_tool_models(cfg.tool_models)
    runtime_models = cfg.prepare_model_backends()
    tools = build_omnigaia_tools(cfg.tool_models, search=search, browse=browse)
    benchmark = OmniGAIAOrchestraBenchmark(
        OmniGAIAConfig(
            dataset_path=cfg.dataset_path, attachments_dir=cfg.attachments_dir,
            level_filter=[str(x) for x in cfg.level_filter] if cfg.level_filter else None,
            max_tasks=cfg.max_tasks, task_ids_file=cfg.task_ids_file,
            result_folder=cfg.result_folder,
            trajectory_folder=cfg.trajectory_folder), tools=tools)
    levels = benchmark.list_levels()
    if args.tasks:
        selected = {x.strip() for x in args.tasks.split(",") if x.strip()}
        levels = [x for x in levels if str(x.get("id") or x.get("task_id")) in selected]
    if args.max_tasks is not None:
        levels = levels[:args.max_tasks]
    elif cfg.max_tasks:
        levels = levels[:cfg.max_tasks]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = ROOT / "logs/reliability_baselines" / args.method / stamp
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "manifest.json").write_text(json.dumps({
        "method": args.method, "options": options, "config": args.config,
        "main_model": cfg.main_model, "sub_models": cfg.sub_models,
        "runtime_models": runtime_models,
        "task_ids": [str(x.get("id") or x.get("task_id")) for x in levels],
    }, indent=2), encoding="utf-8")
    runner = BaseOmniGAIARunner(
        benchmark=benchmark, main_model=cfg.main_model,
        sub_models=cfg.sub_models, max_attempts=cfg.max_attempts,
        omnigaia_tools=tools, tool_models=cfg.tool_models,
        evaluation_model=cfg.evaluation_model,
        trace_summary_model=cfg.trace_summary_model,
        main_agent_cls=ReliabilityMainAgent,
        prompt_builder=OmniGAIAMainAgentPrompt,
        main_agent_kwargs={
            "baseline_method": args.method, "baseline_options": options,
            "review_audit_dir": str(dest / "review")},
        log_prefix=args.method, summarize_trace=True)
    results = await runner.run_levels(
        levels=levels, max_concurrency=args.max_concurrency or cfg.max_concurrency,
        csv_path=dest / "results.csv",
        trajectory_folder=dest / "trajectories", timestamp=stamp)
    success = sum(bool(item.get("success")) for item in results.values())
    logger.info(f"{args.method}: {success}/{len(results)} success; output {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
