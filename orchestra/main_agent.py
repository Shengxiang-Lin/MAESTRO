"""Prospective epistemic MAESTRO Main-Agent."""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Dict, List, Optional

from pydantic import Field

from base.engine.logs import LogLevel
from orchestra.base_main_agent import BaseMainAgent
from orchestra.common.utils import parse_json_response

from .config import ProspectiveConfig
from .prospective import normalize_prospective_decision
from .routing import project_evidence_requests
from .state import ObservedStateLedger


class MainAgent(BaseMainAgent):
    """Persistent epistemic state + prospective deliberation + one macro-action."""

    name: str = Field(default="MAESTROMainAgent")
    prospective_config: Any = Field(default=None, exclude=True)
    observed_state: Any = Field(default=None, exclude=True)

    def __init__(self, **data: Any) -> None:
        cfg = data.pop("prospective_config", None) or ProspectiveConfig()
        super().__init__(**data)
        self.prospective_config = cfg
        self.observed_state = self._new_state()

    def _new_state(self) -> ObservedStateLedger:
        cfg = self.prospective_config
        return ObservedStateLedger(
            max_attempts=self.max_attempts,
            max_source_ids_per_result=cfg.max_source_ids_per_result,
            max_belief_history=cfg.max_belief_history,
            max_evidence_attempts_per_target=cfg.max_evidence_attempts_per_target,
        )

    def reset(self, env_info) -> None:
        super().reset(env_info)
        self.observed_state = self._new_state()

    def _visible_state(self) -> Dict[str, Any]:
        cfg = self.prospective_config
        return self.observed_state.prompt_snapshot(
            cfg.max_prompt_state_chars,
            max_results=cfg.max_prompt_results,
        )

    def _tool_names(self) -> List[str]:
        return [
            str(tool.name)
            for tool in self.subagent_tools
            if getattr(tool, "name", None)
            and str(tool.name) not in {"finish", "complete", "delegate_task"}
        ]

    def _provider_tool_names(self) -> List[str]:
        return [name for name in self._tool_names() if name != "ExecuteCodeAction"]

    def _build_prospective_prompt(
        self,
        *,
        context: str,
        subtask_history: str,
        attempt_index: int,
    ) -> str:
        cfg = self.prospective_config
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
            observed_state=self._visible_state(),
            max_future_requirements=cfg.max_future_requirements,
            max_critical_dependencies=cfg.max_critical_dependencies,
            max_tasks_per_action=cfg.max_tasks_per_action,
        )

    def _normalize_decision(self, raw: Any) -> Dict[str, Any]:
        return normalize_prospective_decision(
            raw,
            config=self.prospective_config,
            selection_field="model",
            choices=self.masked_sub_models,
            tool_names=self._tool_names(),
        )

    def _project_execution_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.prospective_config
        if not cfg.execution_projection_enabled:
            return {
                "execution_params": deepcopy(params),
                "projection_reports": [],
                "projection_failures": [],
            }
        return project_evidence_requests(
            params,
            ledger=self.observed_state,
            available_tools=self._tool_names(),
            provider_tools=self._provider_tool_names(),
            strict_diversity=cfg.strict_provider_diversity,
            blind_requery=cfg.blind_requery,
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
        del dry_run
        attempt_index = min(self.attempt + 1, self.max_attempts)
        self.observed_state.set_attempt(attempt_index)
        prompt = self._build_prospective_prompt(
            context=context,
            subtask_history=subtask_history,
            attempt_index=attempt_index,
        )
        response = await self.llm(prompt, temperature=temperature, seed=seed)
        decision = self._normalize_decision(parse_json_response(response))
        decision["called"] = True
        return decision

    @staticmethod
    def _merge_projection_failures(
        result: Dict[str, Any], failures: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not failures:
            return result
        merged = deepcopy(result or {})
        batch = merged.get("subtask_results")
        if not isinstance(batch, list):
            batch = []
        batch.extend(deepcopy(failures))
        merged["subtask_results"] = batch
        merged["total_subtasks"] = len(batch)
        merged["all_done"] = bool(batch) and all(
            item.get("done", False) for item in batch
        )
        return merged

    async def step(self, observation, history, **kwargs) -> tuple:
        del observation, history, kwargs
        self.attempt += 1
        self.observed_state.set_attempt(self.attempt)
        subtask_history = self._format_subtask_history()
        state_before = {
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "instruction": deepcopy(self.instruction),
            "meta": deepcopy(self.meta),
            "context": deepcopy(self.context),
            "subtask_history": deepcopy(subtask_history),
            "observed_state": deepcopy(self._visible_state()),
            "sub_models": deepcopy(self.masked_sub_models),
            "tools": self._tool_names(),
        }
        prompt = self._build_prospective_prompt(
            context=self.context,
            subtask_history=subtask_history,
            attempt_index=self.attempt,
        )
        state_before["main_prompt"] = prompt
        from base.engine.logs import logger

        logger.log_to_file(
            LogLevel.INFO,
            f"\n{'='*80}\n[MAESTRO Attempt {self.attempt}] "
            f"EPISTEMIC/PROSPECTIVE PROMPT:\n{prompt}\n{'='*80}\n",
        )

        response = await self.llm(prompt)
        decision = self._normalize_decision(parse_json_response(response))
        action_name = str(decision["action"])
        params = deepcopy(decision.get("params") or {})
        epistemic_state = deepcopy(
            decision.get("epistemic_state") or {"bindings": []}
        )
        prospective_state = deepcopy(decision.get("prospective_state") or {})
        decision_record = self.observed_state.record_decision(
            attempt=self.attempt,
            epistemic_state=epistemic_state,
            prospective_state=prospective_state,
            action=action_name,
            params=params,
            reasoning=str(decision.get("reasoning") or ""),
            validation_warnings=decision.get("validation_warnings") or [],
        )

        projection_reports: List[Dict[str, Any]] = []
        projection_failures: List[Dict[str, Any]] = []
        execution_params = deepcopy(params)
        if action_name == "delegate_task":
            projected = self._project_execution_params(params)
            execution_params = projected["execution_params"]
            projection_reports = projected.get("projection_reports") or []
            projection_failures = projected.get("projection_failures") or []
        logger.log_to_file(
            LogLevel.INFO,
            f"\n{'='*80}\n[MAESTRO Attempt {self.attempt}] DECISION:\n"
            f"{json.dumps(decision_record, indent=2, ensure_ascii=False)}\n"
            f"EXECUTION PROJECTION:\n"
            f"{json.dumps(projection_reports, indent=2, ensure_ascii=False)}\n"
            f"{'='*80}\n",
        )

        tool = next(
            (candidate for candidate in self.tools if candidate.name == action_name),
            None,
        )
        if tool is None:
            result = {
                "success": False,
                "done": False,
                "error": f"Unknown action: {action_name}",
            }
            action_name = "orchestration_error"
        elif action_name == "delegate_task" and not (
            execution_params.get("tasks") or []
        ):
            result = self._merge_projection_failures({}, projection_failures)
        else:
            result = await tool(**execution_params)
            if action_name == "delegate_task":
                result = self._merge_projection_failures(
                    result,
                    projection_failures,
                )

        if action_name == "delegate_task":
            result_ids = self.observed_state.observe_feedback(result, self.attempt)
            self.observed_state.record_evidence_attempts(
                attempt=self.attempt,
                execution_params=execution_params,
                projection_reports=projection_reports,
                result_ids=result_ids,
            )

        context_params = (
            execution_params if action_name == "delegate_task" else params
        )
        self._update_context(action_name, context_params, result)
        return {
            "action": action_name,
            "reasoning": decision.get("reasoning", ""),
            "params": params,
            "execution_params": execution_params,
            "execution_projection": projection_reports,
            "route_compilation": projection_reports,
            "result": result,
            "epistemic_state": epistemic_state,
            "prospective_state": prospective_state,
            "validation_warnings": deepcopy(
                decision.get("validation_warnings") or []
            ),
            "state_before": state_before,
            "observed_state_after": self._visible_state(),
            "decision_record": decision_record,
            "subtask_history": subtask_history,
        }, response
