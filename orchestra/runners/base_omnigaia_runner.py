"""Shared OmniGAIA runner used by MAESTRO and Orchestra-o1."""
from __future__ import annotations

import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

from base.agent.base_action import BaseAction
from base.engine.async_llm import LLMsConfig, create_llm_instance
from base.engine.cost_monitor import COST_CSV_FIELDS, CostMonitor, record_unmeasured_call
from base.engine.logs import logger
from benchmark.common.env import BasicInfo
from benchmark.common.runner import Runner
from benchmark.omnigaia.llm_scorer import llm_semantic_score
from orchestra.base_main_agent import BaseMainAgent
from orchestra.tools.complete import CompleteTool
from orchestra.tools.delegate import DelegateTaskTool


class BaseOmniGAIARunner:
    """Common execution path; methods only supply their Main-Agent policy."""

    def __init__(
        self,
        *,
        benchmark,
        main_model: str,
        sub_models: List[str],
        max_attempts: int,
        omnigaia_tools: List[BaseAction],
        tool_models: Dict[str, str] | None = None,
        evaluation_model: str = "qwen3.7-plus",
        trace_summary_model: str = "qwen3.7-plus",
        main_agent_cls: Type[BaseMainAgent] = BaseMainAgent,
        prompt_builder: Any = None,
        main_agent_kwargs: Dict[str, Any] | None = None,
        log_prefix: str = "Orchestra",
        summarize_trace: bool = True,
    ) -> None:
        self.benchmark = benchmark
        self.main_model = main_model
        self.sub_models = list(sub_models)
        self.max_attempts = max_attempts
        self.omnigaia_tools = omnigaia_tools
        self.tool_models = dict(tool_models or {})
        self.evaluation_model = evaluation_model
        self.trace_summary_model = trace_summary_model
        self.main_agent_cls = main_agent_cls
        self.prompt_builder = prompt_builder
        self.main_agent_kwargs = dict(main_agent_kwargs or {})
        self.log_prefix = log_prefix
        self.summarize_trace = summarize_trace

    def _prepare_csv(
        self, csv_path: Path | None
    ) -> Tuple[csv.DictWriter | None, asyncio.Lock | None, Any]:
        if not csv_path:
            return None, None, None
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "task_id", "level", "task_type", "category", "main_model",
                "sub_models", "tool_models", "evaluation_model", "trace_summary_model",
                "final_sub_model", "success", "reward", "attempts", "sub_cost",
                "main_cost", "total_cost", "timestamp", "start_time", "end_time",
                "error", *COST_CSV_FIELDS,
            ],
        )
        csv_writer.writeheader()
        csv_file.flush()
        return csv_writer, asyncio.Lock(), csv_file

    def _save_trajectory(
        self,
        *,
        trajectory_folder: Path,
        level_id: str,
        timestamp: str,
        level: str | None,
        question: str | None,
        expected_answer: str | None,
        task_type: str | None,
        category: str | None,
        omni_modal_input: list | None,
        instruction: str | None,
        meta: dict | None,
        attempts_detail: List[Dict[str, Any]],
        success: bool,
        total_reward: float,
        total_cost: float | None,
        main_cost: float,
        sub_cost: float,
        final_sub_model: str | None,
        error: str | None,
        start_time: str | None = None,
        end_time: str | None = None,
        cost_breakdown: Dict[str, Any] | None = None,
    ) -> None:
        trajectory_folder.mkdir(parents=True, exist_ok=True)
        filename = (
            f"omnigaia_{level_id}_{timestamp}.json"
            if timestamp else f"omnigaia_{level_id}.json"
        )
        data = {
            "task_id": level_id,
            "timestamp": timestamp,
            "start_time": start_time,
            "end_time": end_time,
            "level": level,
            "task_type": task_type,
            "category": category,
            "question": question,
            "expected_answer": expected_answer,
            "omni_modal_input": omni_modal_input,
            "main_model": self.main_model,
            "sub_models": self.sub_models,
            "tool_models": self.tool_models,
            "evaluation_model": self.evaluation_model,
            "trace_summary_model": self.trace_summary_model,
            "max_attempts": self.max_attempts,
            "success": success,
            "total_reward": total_reward,
            "total_cost": total_cost,
            "main_cost": main_cost,
            "sub_cost": sub_cost,
            "attempts": len(attempts_detail),
            "trajectory": attempts_detail,
            "final_sub_model": final_sub_model,
            "error": error,
            "instruction": instruction,
            "meta": meta or {},
            "cost_breakdown": cost_breakdown or {},
        }
        path = trajectory_folder / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        logger.info(f"[{self.log_prefix}] Trajectory saved to {path}")

    def _after_trajectory(
        self,
        *,
        trajectory_folder: Path,
        level_id: str,
        timestamp: str,
        main_agent: BaseMainAgent | None,
        attempts_detail: List[Dict[str, Any]],
    ) -> None:
        """Optional method-specific trajectory extension."""
        del trajectory_folder, level_id, timestamp, main_agent, attempts_detail

    def _build_main_agent(self, env, main_llm):
        model_to_alias = {
            model: f"model_{index + 1}"
            for index, model in enumerate(self.sub_models)
        }
        alias_to_model = {
            alias: model for model, alias in model_to_alias.items()
        }
        delegate_tool = DelegateTaskTool(
            env=env,
            runner=Runner(),
            models=self.sub_models,
            benchmark_type="omnigaia",
            alias_to_model=alias_to_model,
            summarize_trace=self.summarize_trace,
            trace_summary_model=self.trace_summary_model,
        )
        return self.main_agent_cls(
            llm=main_llm,
            sub_models=self.sub_models,
            tools=[delegate_tool, CompleteTool()],
            subagent_tools=self.omnigaia_tools,
            prompt_builder=self.prompt_builder,
            max_attempts=self.max_attempts,
            benchmark_type="omnigaia",
            mask_model_names=True,
            **self.main_agent_kwargs,
        )

    @staticmethod
    def _attempt_record(
        attempt_index: int,
        action: Dict[str, Any],
        response: str,
    ) -> Dict[str, Any]:
        """Keep a superset so the same runner records O1 and MAESTRO actions."""
        return {
            "attempt": attempt_index + 1,
            "action": action.get("action"),
            "reasoning": action.get("reasoning", ""),
            "params": action.get("params", {}),
            "execution_params": action.get("execution_params", {}),
            "execution_projection": action.get("execution_projection", []),
            "route_compilation": action.get("route_compilation", []),
            "result": action.get("result", {}),
            "epistemic_state": action.get("epistemic_state", {}),
            "prospective_state": action.get("prospective_state", {}),
            "validation_warnings": action.get("validation_warnings", []),
            "state_before": action.get("state_before", {}),
            "observed_state_after": action.get("observed_state_after", {}),
            "decision_record": action.get("decision_record", {}),
            "raw_response": response,
        }

    @staticmethod
    def _sub_cost(attempts_detail: List[Dict[str, Any]]) -> float:
        return sum(
            float(
                (attempt.get("result") or {}).get(
                    "total_cost",
                    (attempt.get("result") or {}).get("cost", 0.0),
                ) or 0.0
            )
            for attempt in attempts_detail
        )

    async def run_levels(
        self,
        levels,
        max_concurrency: int,
        csv_path: Path | None,
        trajectory_folder: Path,
        timestamp: str,
    ) -> Dict[str, Dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, max_concurrency))
        csv_writer, csv_lock, csv_file = self._prepare_csv(csv_path)
        results: Dict[str, Dict[str, Any]] = {}

        async def run_single(level_spec):
            level_id = str(level_spec.get("id") or level_spec.get("task_id") or str(level_spec))
            task_level = level_spec.get("Level", "")
            question = level_spec.get("question", "")
            expected_answer = level_spec.get("answer", "")
            task_type = level_spec.get("task_type", "")
            category = level_spec.get("category", "")
            omni_modal_input = level_spec.get("omni_modal_input", [])
            journal = (csv_path.parent if csv_path else trajectory_folder) / f"api_usage_{timestamp}.jsonl"

            async with semaphore:
                with CostMonitor(task_id=level_id, journal_path=journal) as monitor:
                    env = main_agent = basic_info = None
                    attempts_detail: List[Dict[str, Any]] = []
                    start_time = datetime.now().isoformat()
                    main_cost_before = 0.0
                    final_answer = None
                    total_reward = 0.0
                    success = False
                    error = None
                    try:
                        logger.info(f"[{self.log_prefix}] Starting task: {level_id}")
                        env = self.benchmark.make_env(level_spec, tools=self.omnigaia_tools)
                        basic_info = env.get_basic_info()
                        main_llm = create_llm_instance(LLMsConfig.default().get(self.main_model))
                        main_agent = self._build_main_agent(env, main_llm)
                        main_agent.reset(BasicInfo(
                            env_id=level_id, instruction=basic_info.instruction,
                            action_space="", max_steps=self.max_attempts,
                            meta_data=basic_info.meta_data,
                        ))
                        main_cost_before = main_agent.get_usage_cost()
                        for attempt_index in range(self.max_attempts):
                            action, response = await main_agent.step(None, [])
                            attempts_detail.append(self._attempt_record(attempt_index, action, response))
                            if action.get("action") == "complete":
                                final_answer = (action.get("params") or {}).get("answer")
                                break
                        if final_answer:
                            total_reward = await llm_semantic_score(
                                str(final_answer), expected_answer,
                                model=self.evaluation_model, question=question,
                            )
                        success = total_reward > 0.5
                        results[level_id] = {
                            "success": success, "reward": total_reward,
                            "answer": final_answer, "expected_answer": expected_answer,
                        }
                    except asyncio.CancelledError:
                        error = "Cancelled"
                        record_unmeasured_call(self.main_model, "agent", "cancelled_with_possible_in_flight_requests")
                        raise
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        logger.error(f"[{self.log_prefix}] Task {level_id} failed: {error}")
                        results[level_id] = {"success": False, "reward": 0.0, "error": error}
                    finally:
                        main_cost = max(0.0, main_agent.get_usage_cost() - main_cost_before) if main_agent else 0.0
                        agent_cost = sum(r.cost for r in monitor.records if r.category == "agent")
                        sub_cost = max(0.0, agent_cost - main_cost)
                        breakdown = monitor.cost_columns(main_cost, sub_cost)
                        end_time = datetime.now().isoformat()
                        self._save_trajectory(
                            trajectory_folder=trajectory_folder, level_id=level_id,
                            timestamp=timestamp, level=task_level, question=question,
                            expected_answer=expected_answer, task_type=task_type,
                            category=category, omni_modal_input=omni_modal_input,
                            instruction=basic_info.instruction if basic_info else None,
                            meta=basic_info.meta_data if basic_info else None,
                            attempts_detail=attempts_detail, success=success,
                            total_reward=total_reward, total_cost=breakdown["total_cost"],
                            main_cost=main_cost, sub_cost=sub_cost, final_sub_model=None,
                            error=error, start_time=start_time, end_time=end_time,
                            cost_breakdown={**breakdown, "usage": monitor.summary()},
                        )
                        self._after_trajectory(
                            trajectory_folder=trajectory_folder, level_id=level_id,
                            timestamp=timestamp, main_agent=main_agent,
                            attempts_detail=attempts_detail,
                        )
                        if csv_writer and csv_lock:
                            cost_fields = {
                                key: ("" if value is None else f"{value:.6f}")
                                if key.endswith(("_cost", "_cost_cny")) else value
                                for key, value in breakdown.items()
                            }
                            cost_fields["cost_complete"] = "yes" if breakdown["cost_complete"] else "no"
                            async with csv_lock:
                                csv_writer.writerow({
                                    "task_id": level_id, "level": task_level,
                                    "task_type": task_type, "category": category,
                                    "main_model": self.main_model,
                                    "sub_models": ",".join(self.sub_models),
                                    "tool_models": json.dumps(self.tool_models, ensure_ascii=False, sort_keys=True),
                                    "evaluation_model": self.evaluation_model,
                                    "trace_summary_model": self.trace_summary_model,
                                    "final_sub_model": None, "success": success,
                                    "reward": f"{total_reward:.4f}",
                                    "attempts": len(attempts_detail),
                                    "sub_cost": f"{sub_cost:.6f}", "main_cost": f"{main_cost:.6f}",
                                    "timestamp": timestamp, "start_time": start_time,
                                    "end_time": end_time, "error": error, **cost_fields,
                                })
                                csv_file.flush()
                        if env and hasattr(env, "close"):
                            try:
                                await env.close()
                            except Exception as exc:
                                logger.debug(f"[{self.log_prefix}] Error closing env: {exc}")

        tasks = [asyncio.create_task(run_single(level)) for level in levels]
        try:
            await asyncio.gather(*tasks)
        finally:
            if csv_file:
                csv_file.close()
        return results
