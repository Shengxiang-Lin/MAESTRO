"""Shared Main-Agent mechanics used by MAESTRO and Orchestra-o1."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import Field

from base.agent.base_agent import BaseAgent
from base.agent.memory import Memory
from base.engine.async_llm import ModelPricing
from base.engine.logs import LogLevel, logger
from benchmark.common.env import BasicInfo
from orchestra.common.provenance import trace_control_provenance
from orchestra.common.utils import indent_text, parse_json_response


def build_model_pricing_table(
    sub_models: List[str],
    model_to_alias: Dict[str, str] | None = None,
) -> str:
    lines = [
        "| Model | Input $/1K | Output $/1K |",
        "|-------|-----------|------------|",
    ]
    alias_to_model = {v: k for k, v in (model_to_alias or {}).items()}
    for model_display in sub_models:
        real_model = alias_to_model.get(model_display, model_display)
        input_price = ModelPricing.get_price(real_model, "input")
        output_price = ModelPricing.get_price(real_model, "output")
        lines.append(
            f"| {model_display} | ${input_price:.5f} | ${output_price:.5f} |"
        )
    return "\n".join(lines)


class BaseMainAgent(BaseAgent):
    """Common orchestration shell; concrete methods only customize decision logic."""

    name: str = Field(default="BaseMainAgent")
    description: str = Field(default="Multi-agent orchestrator")
    sub_models: List[str] = Field(default_factory=list)
    tools: List[Any] = Field(default_factory=list)
    subagent_tools: List[Any] = Field(default_factory=list)
    prompt_builder: Optional[Any] = Field(default=None)
    max_attempts: int = Field(default=10)
    benchmark_type: str = Field(default="omnigaia")

    mask_model_names: bool = Field(default=False)
    model_to_alias: Dict[str, str] = Field(default_factory=dict)
    alias_to_model: Dict[str, str] = Field(default_factory=dict)
    masked_sub_models: List[str] = Field(default_factory=list)

    memory: Memory = Field(default=None)
    instruction: str = Field(default="")
    meta: Dict[str, Any] = Field(default_factory=dict)
    attempt: int = Field(default=0)
    context: str = Field(default="")
    history: List[Dict[str, Any]] = Field(default_factory=list)
    task_entries: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.mask_model_names and self.sub_models:
            self.model_to_alias = {
                model: f"model_{index + 1}"
                for index, model in enumerate(self.sub_models)
            }
            self.alias_to_model = {
                alias: model for model, alias in self.model_to_alias.items()
            }
            self.masked_sub_models = list(self.model_to_alias.values())
        else:
            self.model_to_alias = {model: model for model in self.sub_models}
            self.alias_to_model = dict(self.model_to_alias)
            self.masked_sub_models = list(self.sub_models)

    def reset(self, env_info: BasicInfo) -> None:
        self.memory = Memory(llm=self.llm, max_memory=20)
        self.instruction = env_info.instruction
        self.meta = env_info.meta_data or {}
        self.attempt = 0
        self.context = ""
        self.history = []
        self.task_entries = []

    def get_usage_summary(self) -> Dict[str, Any]:
        if self.llm is None or not hasattr(self.llm, "get_usage_summary"):
            return {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "call_count": 0,
            }
        return dict(self.llm.get_usage_summary() or {})

    def get_usage_cost(self) -> float:
        return float(self.get_usage_summary().get("total_cost", 0.0) or 0.0)

    @staticmethod
    def _runtime_telemetry(subtask: Dict[str, Any]) -> Dict[str, Any]:
        provenance = trace_control_provenance(subtask.get("trace") or [])
        assigned = subtask.get("tools_assigned") or []
        if isinstance(assigned, str):
            assigned = [assigned]
        return {
            "assigned_tool_ids": sorted(
                {str(value) for value in assigned if value}
            ),
            "executed_tool_ids": list(provenance.get("executed_tool_ids") or []),
            "attested_source_ids": list(
                provenance.get("attested_source_ids") or []
            )[:12],
            "context_fingerprint": str(
                subtask.get("context_fingerprint") or "empty"
            ),
        }

    @staticmethod
    def _format_runtime_telemetry(entry: Dict[str, Any]) -> str:
        def _fmt(values: Any) -> str:
            values = [str(value) for value in (values or []) if str(value)]
            return ", ".join(values) if values else "none"

        return (
            f"model={entry.get('model') or 'unknown'}; "
            f"assigned_tools=[{_fmt(entry.get('assigned_tool_ids'))}]; "
            f"executed_tools=[{_fmt(entry.get('executed_tool_ids'))}]; "
            f"attested_sources=[{_fmt(entry.get('attested_source_ids'))}]; "
            f"context_fp={entry.get('context_fingerprint') or 'empty'}"
        )

    def _format_subtask_history(self) -> str:
        if not self.task_entries:
            return "No subtasks completed yet."

        lines = [
            "Runtime telemetry below records the execution path only; it does not "
            "certify the semantic correctness of a Sub-Agent result."
        ]
        done_count = 0
        total_subtasks = 0
        for entry in self.task_entries:
            if entry.get("is_parallel_batch"):
                batch_results = entry.get("subtask_results", [])
                total_subtasks += len(batch_results)
                block = [
                    f'[Attempt {entry["attempt"]}] Parallel Batch '
                    f'({len(batch_results)} subtasks)'
                ]
                for index, sub in enumerate(batch_results):
                    status = sub.get("status", "?")
                    model_display = sub.get("model", "?")
                    if self.mask_model_names and model_display in self.model_to_alias:
                        model_display = self.model_to_alias[model_display]
                    block.extend(
                        [
                            f"  [{index + 1}] {status} | Model: {model_display} | "
                            f"Steps: {sub.get('steps_taken', '?')}/{sub.get('max_steps', 30)}",
                            f"  |- Task: {sub.get('instruction', 'N/A')}",
                            f"  |- Result: {sub.get('result', '(no result)')}",
                            f"  |- Runtime telemetry: {self._format_runtime_telemetry(sub)}",
                        ]
                    )
                    if sub.get("summary"):
                        block.append(f"  |- Summary: {sub['summary']}")
                    if sub.get("trace_summary"):
                        block.append(
                            "  `- Trace summary:\n"
                            + indent_text(str(sub["trace_summary"]), "     ")
                        )
                    if status == "done":
                        done_count += 1
                lines.append("\n".join(block))
                continue

            total_subtasks += 1
            status = entry.get("status", "?")
            model_display = entry.get("model", "?")
            if self.mask_model_names and model_display in self.model_to_alias:
                model_display = self.model_to_alias[model_display]
            block = [
                f'[Attempt {entry.get("attempt", "?")}] {status} | '
                f'Model: {model_display} | Steps: '
                f'{entry.get("steps_taken", "?")}/{entry.get("max_steps", 30)}',
                f"|- Task: {entry.get('instruction', 'N/A')}",
                f"|- Result: {entry.get('result', '(no result)')}",
                f"|- Runtime telemetry: {self._format_runtime_telemetry(entry)}",
            ]
            if entry.get("summary"):
                block.append(f"|- Summary: {entry['summary']}")
            if entry.get("trace_summary"):
                block.append(
                    "`- Trace summary:\n"
                    + indent_text(str(entry["trace_summary"]), "   ")
                )
            lines.append("\n".join(block))
            if status == "done":
                done_count += 1

        lines.append(f"---\nSummary: {done_count}/{total_subtasks} subtasks done")
        return "\n\n".join(lines)

    def _build_prompt(
        self,
        *,
        context: str,
        subtask_history: str,
        attempt_index: int,
    ) -> str:
        if not self.prompt_builder:
            return f"Task: {self.instruction}\n\nContext:\n{context or 'First attempt'}"
        return self.prompt_builder.build_prompt(
            instruction=self.instruction,
            meta=self.meta,
            prior_context=context,
            attempt_index=attempt_index,
            max_attempts=self.max_attempts,
            sub_models=self.masked_sub_models,
            subtask_history=subtask_history,
            model_to_alias=self.model_to_alias if self.mask_model_names else None,
            tools=self.subagent_tools,
        )

    async def generate_next_action(
        self,
        *,
        context: str,
        subtask_history: str,
        dry_run: bool = False,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        attempt_index = min(self.attempt + 1, self.max_attempts)
        prompt = self._build_prompt(
            context=context,
            subtask_history=subtask_history,
            attempt_index=attempt_index,
        )
        response = await self.llm(prompt, temperature=temperature, seed=seed)
        decision = parse_json_response(response)
        if not isinstance(decision, dict):
            raise ValueError("Main-Agent did not return a JSON object")
        if dry_run:
            logger.info("[MainAgent] Generated dry-run action without tool execution")
        return decision

    async def step(self, observation, history, **kwargs) -> tuple:
        del observation, history, kwargs
        self.attempt += 1
        logger.info(f"[MainAgent] Step {self.attempt}/{self.max_attempts}")
        subtask_history = self._format_subtask_history()
        prompt = self._build_prompt(
            context=self.context,
            subtask_history=subtask_history,
            attempt_index=self.attempt,
        )
        logger.log_to_file(
            LogLevel.INFO,
            f"\n{'=' * 80}\n[MainAgent Attempt {self.attempt}] PROMPT:\n"
            f"{prompt}\n{'=' * 80}\n",
        )
        response = await self.llm(prompt)
        decision = parse_json_response(response)
        if not isinstance(decision, dict):
            return {"action": "error", "error": "Invalid Main-Agent JSON"}, response

        logger.log_to_file(
            LogLevel.INFO,
            f"[MainAgent Attempt {self.attempt}] PARSED DECISION:\n"
            f"{json.dumps(decision, indent=2, ensure_ascii=False)}",
        )
        action_name = decision.get("action")
        params = decision.get("params", {})
        tool = next(
            (candidate for candidate in self.tools if candidate.name == action_name),
            None,
        )
        if tool is None:
            return {
                "action": "error",
                "error": f"Unknown action: {action_name}",
            }, response

        result = await tool(**params)
        self._update_context(str(action_name), params, result)
        return {
            "action": action_name,
            "params": params,
            "result": result,
            "subtask_history": subtask_history,
        }, response

    def _update_context(self, action: str, params: Dict, result: Dict) -> None:
        summary = f"[{self.attempt}] {action}\n"
        if action == "delegate_task":
            subtask_results = result.get("subtask_results", [])
            if subtask_results:
                summary += (
                    f"  Parallel batch: {len(subtask_results)} subtask(s)\n"
                    f"  Total cost: {result.get('total_cost', 0.0):.6f}\n"
                    f"  Total steps: {result.get('total_steps', 0)}\n"
                )
                batch_entries = []
                for sub_result in subtask_results:
                    finish = sub_result.get("finish_result", {}) or {}
                    status = finish.get("status", "partial") if finish else "partial"
                    result_text = finish.get("result", "-") if finish else "-"
                    entry = {
                        "status": status,
                        "instruction": sub_result.get("task_instruction", "N/A"),
                        "model": sub_result.get("model", "unknown"),
                        "steps_taken": sub_result.get("steps_taken", 0),
                        "max_steps": sub_result.get("statistics", {}).get(
                            "max_steps", 30
                        ),
                        "cost": sub_result.get("cost", 0),
                        "result": result_text,
                        "summary": finish.get("summary", "") if finish else "",
                        "trace_summary": sub_result.get("trace_summary", ""),
                        **self._runtime_telemetry(sub_result),
                    }
                    batch_entries.append(entry)
                    summary += (
                        f"  - Subtask: {entry['instruction'][:60]}...\n"
                        f"    Status: {status}, Result: {result_text}\n"
                    )
                self.task_entries.append(
                    {
                        "attempt": self.attempt,
                        "is_parallel_batch": True,
                        "subtask_results": batch_entries,
                        "total_cost": result.get("total_cost", 0.0),
                        "total_steps": result.get("total_steps", 0),
                    }
                )
            else:
                finish = result.get("finish_result", {}) or {}
                self.task_entries.append(
                    {
                        "attempt": self.attempt,
                        "status": finish.get("status", "partial") if finish else "partial",
                        "instruction": result.get(
                            "task_instruction",
                            params.get("task_instruction", "N/A"),
                        ),
                        "model": result.get("model", params.get("model", "unknown")),
                        "steps_taken": result.get("steps_taken", 0),
                        "max_steps": result.get("statistics", {}).get(
                            "max_steps", 30
                        ),
                        "cost": result.get("cost", 0),
                        "result": finish.get("result", "-") if finish else "-",
                        "summary": finish.get("summary", "") if finish else "",
                        "trace_summary": result.get("trace_summary", ""),
                        **self._runtime_telemetry(result),
                    }
                )
        elif action == "complete":
            summary += f"  Answer: {params.get('answer', 'N/A')}\n"

        self.context = summary + "\n" + self.context
        self.history.append(
            {"attempt": self.attempt, "action": action, "result": result}
        )

    async def run(self, request: Optional[str] = None) -> str:
        del request
        return "Orchestration via Runner"


# Backward-compatible name for the O1 baseline.
MainAgent = BaseMainAgent
