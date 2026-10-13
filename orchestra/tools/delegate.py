"""DelegateTaskTool - shared parallel Sub-Agent execution."""
from __future__ import annotations

import asyncio
import hashlib
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

from pydantic import Field, PrivateAttr

from base.agent.base_action import BaseAction
from base.agent.memory import Memory
from base.engine.async_llm import LLMsConfig, create_llm_instance
from base.engine.logs import logger
from orchestra.subagents import ReActAgent
from orchestra.tools.trace_formatter import create_gaia_formatter


def _make_serializable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _make_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    return str(obj)


class DelegateTaskTool(BaseAction):
    name: str = "delegate_task"
    description: str = (
        "Delegate one or more subtasks to SubAgents for parallel execution. "
        "Each subtask runs independently with its own SubAgent."
    )
    parameters: Dict[str, Any] = Field(default_factory=dict)
    env: Any = Field(default=None, exclude=True)
    runner: Any = Field(default=None, exclude=True)
    models: list = Field(default_factory=list)
    benchmark_type: str = Field(default="omnigaia")
    alias_to_model: Dict[str, str] = Field(default_factory=dict)
    summarize_trace: bool = Field(default=True)
    _trace_formatter: Any = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        env,
        runner,
        models: list,
        benchmark_type: str = "omnigaia",
        alias_to_model: Dict[str, str] | None = None,
        summarize_trace: bool = True,
    ):
        super().__init__()
        self.env = env
        self.runner = runner
        self.models = list(models)
        self.benchmark_type = benchmark_type
        self.alias_to_model = alias_to_model or {}
        self.summarize_trace = bool(summarize_trace)
        self._trace_formatter = create_gaia_formatter()
        display_models = (
            list(self.alias_to_model.keys()) if self.alias_to_model else self.models
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "List of parallel subtasks.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_instruction": {"type": "string"},
                            "context": {"type": "string"},
                            "model": {"type": "string", "enum": display_models},
                            "tools": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["task_instruction", "model"],
                    },
                }
            },
            "required": ["tasks"],
        }

    def _resolve_runtime_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        runtime_task = deepcopy(task)
        requested = str(runtime_task.get("model") or "")
        runtime_task["model"] = self.alias_to_model.get(requested, requested)
        return runtime_task

    def _environment_instruction(self) -> str:
        getter = getattr(self.env, "get_basic_info", None)
        if callable(getter):
            try:
                return str(getter().instruction or "")
            except Exception as exc:
                logger.warning(
                    f"[DelegateTool] Could not read environment instruction: {exc}"
                )
        return str(getattr(self.env, "instruction", "") or "")

    async def __call__(
        self,
        tasks: List[Dict[str, Any]] | None = None,
        task_instruction: str | None = None,
        model: str | None = None,
        context: str = "",
        tools: List[str] | None = None,
    ) -> Dict[str, Any]:
        if tasks is None:
            if task_instruction and model:
                tasks = [
                    {
                        "task_instruction": task_instruction,
                        "model": model,
                        "context": context,
                        "tools": tools,
                    }
                ]
            else:
                return {
                    "error": "Must provide either 'tasks' or task_instruction+model",
                    "subtask_results": [],
                }
        if not tasks:
            return {"error": "Empty tasks list", "subtask_results": []}

        intended_instructions = [
            str(task.get("task_instruction") or "") for task in tasks
        ]
        runtime_tasks = [self._resolve_runtime_task(task) for task in tasks]
        logger.info(
            f"[DelegateTool] Launching {len(runtime_tasks)} parallel subtask(s)"
        )
        subtask_results = await asyncio.gather(
            *[
                self._execute_single_task(task, index)
                for index, task in enumerate(runtime_tasks)
            ],
            return_exceptions=True,
        )

        processed_results = []
        for index, result in enumerate(subtask_results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "subtask_index": index,
                        "task_instruction": intended_instructions[index],
                        "model": runtime_tasks[index].get("model", ""),
                        "error": str(result),
                        "steps_taken": 0,
                        "done": False,
                        "cost": 0.0,
                    }
                )
            else:
                result["subtask_index"] = index
                result["task_instruction"] = intended_instructions[index]
                processed_results.append(result)

        total_cost = sum(
            float(item.get("cost", 0.0) or 0.0) for item in processed_results
        )
        total_steps = sum(
            int(item.get("steps_taken", 0) or 0) for item in processed_results
        )
        return {
            "subtask_results": processed_results,
            "total_subtasks": len(tasks),
            "total_cost": total_cost,
            "total_steps": total_steps,
            "all_done": all(item.get("done", False) for item in processed_results),
            "usage": {
                "total_input_tokens": sum(
                    int((item.get("usage") or {}).get("total_input_tokens", 0) or 0)
                    for item in processed_results
                ),
                "total_output_tokens": sum(
                    int((item.get("usage") or {}).get("total_output_tokens", 0) or 0)
                    for item in processed_results
                ),
                "total_cost": total_cost,
                "call_count": sum(
                    int((item.get("usage") or {}).get("call_count", 0) or 0)
                    for item in processed_results
                ),
            },
        }

    async def _execute_single_task(
        self, task: Dict[str, Any], idx: int
    ) -> Dict[str, Any]:
        task_instruction = str(task.get("task_instruction") or "")
        intended_task_instruction = str(
            task.get("intended_task_instruction") or task_instruction
        )
        model = str(task.get("model") or "")
        context = str(task.get("context") or "")
        tools = task.get("tools")
        preferred_tools = task.get("preferred_tools") or []
        route_compilation = deepcopy(task.get("route_compilation") or {})
        agent_spec = task.get("agent_spec") or {}
        normalized_context = " ".join(context.lower().split())
        context_fingerprint = (
            hashlib.sha256(normalized_context.encode("utf-8")).hexdigest()[:12]
            if normalized_context
            else "empty"
        )
        real_model = self.alias_to_model.get(model, model)
        if real_model not in self.models:
            return {
                "error": f"Invalid model selection: {model}",
                "steps_taken": 0,
                "done": False,
                "cost": 0.0,
                "task_instruction": task_instruction,
                "model": real_model,
            }

        env_clone = self.env.clone()
        execution_max_steps = int(
            getattr(getattr(env_clone, "config", None), "max_steps", 30)
        )
        if agent_spec:
            try:
                execution_max_steps = min(
                    execution_max_steps,
                    max(1, int(agent_spec.get("max_calls", 12))) + 1,
                )
                env_clone.config = deepcopy(env_clone.config)
                env_clone.config.max_steps = execution_max_steps
            except (AttributeError, TypeError, ValueError) as exc:
                logger.warning(
                    f"[DelegateTool] Could not enforce agent max_calls: {exc}"
                )

        original_question = str(
            task.get("original_question")
            if "original_question" in task
            else self._environment_instruction()
        )
        set_delegation_view = getattr(env_clone, "set_delegation_view", None)
        if callable(set_delegation_view):
            set_delegation_view(
                instruction=original_question,
                question=original_question,
            )

        llm = create_llm_instance(LLMsConfig.default().get(real_model))
        sub_agent = ReActAgent(
            llm=llm,
            benchmark_type=self.benchmark_type,
            task_instruction=task_instruction,
            context=context,
            original_question=original_question,
            allowed_tools=tools,
            preferred_tools=preferred_tools,
            memory=Memory(llm=llm, max_memory=10),
        )
        try:
            result = await self.runner.run(sub_agent, env_clone)
            finish_result = None
            if result.trace:
                last = result.trace[-1]
                if last.info.get("finished") and last.info.get("finish_result"):
                    finish_result = last.info["finish_result"]
            trace_summary = await self._trace_summary_if_enabled(
                result.trace,
                intended_task_instruction,
                real_model,
            )
            return {
                "task_instruction": task_instruction,
                "model": real_model,
                "tools_assigned": tools,
                "preferred_tools": preferred_tools,
                "route_compilation": route_compilation,
                "steps_taken": result.steps,
                "context_fingerprint": context_fingerprint,
                "agent_spec": agent_spec,
                "done": result.done,
                "cost": result.cost,
                "finish_result": finish_result,
                "usage": llm.get_usage_summary(),
                "trace": [
                    _make_serializable(step) for step in result.trace
                ] if result.trace else [],
                "trace_summary": trace_summary,
                "statistics": {
                    "total_steps": result.steps,
                    "max_steps": execution_max_steps,
                    "completed": result.done,
                },
            }
        except Exception as exc:
            logger.error(f"[DelegateTool] Subtask {idx} error: {exc}")
            return {
                "error": str(exc),
                "task_instruction": task_instruction,
                "model": real_model,
                "steps_taken": 0,
                "tools_assigned": tools,
                "preferred_tools": preferred_tools,
                "route_compilation": route_compilation,
                "context_fingerprint": context_fingerprint,
                "agent_spec": agent_spec,
                "done": False,
                "cost": 0.0,
                "usage": llm.get_usage_summary(),
            }
        finally:
            if hasattr(env_clone, "close"):
                try:
                    await env_clone.close()
                except Exception:
                    pass

    async def _summarize_trace(
        self,
        trace,
        intended_task_instruction: str,
        summary_model: str,
    ) -> str:
        if not trace:
            return "No steps executed"
        del intended_task_instruction
        trace_text = self._trace_formatter.format_trace(trace)
        prompt = (
            "Review the SubAgent execution trace and faithfully summarize it in "
            "5-10 bullets: tool names, targets, evidence/provenance, final result, "
            "problems, remaining issues. Output ONLY bullets.\n\n"
            + trace_text
        )
        try:
            review_llm = create_llm_instance(LLMsConfig.default().get(summary_model))
            return (await review_llm(prompt)).strip()
        except Exception as exc:
            logger.warning(f"[DelegateTool] Trace summarization failed: {exc}")
            return f"Steps: {len(trace)}"

    async def _trace_summary_if_enabled(
        self,
        trace,
        intended_task_instruction: str,
        summary_model: str,
    ) -> str:
        if not self.summarize_trace:
            return ""
        return await self._summarize_trace(
            trace,
            intended_task_instruction,
            summary_model,
        )
