"""MAESTRO OmniGAIA runner built on the shared Orchestra runtime."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from orchestra.config import ProspectiveConfig
from orchestra.main_agent import MainAgent
from orchestra.prompts.prospective import ProspectiveOmniGAIAPrompt
from orchestra.runners.base_omnigaia_runner import BaseOmniGAIARunner


class OmniGAIARunner(BaseOmniGAIARunner):
    """MAESTRO specialization: only the Main-Agent policy differs from O1."""

    def __init__(
        self,
        *,
        benchmark,
        main_model: str,
        sub_models: List[str],
        max_attempts: int,
        omnigaia_tools: List[Any],
        prospective_config: ProspectiveConfig | None = None,
    ) -> None:
        self.prospective_config = prospective_config or ProspectiveConfig()
        if not self.prospective_config.enabled:
            raise ValueError("MAESTRO requires prospective.enabled=true")
        super().__init__(
            benchmark=benchmark,
            main_model=main_model,
            sub_models=sub_models,
            max_attempts=max_attempts,
            omnigaia_tools=omnigaia_tools,
            main_agent_cls=MainAgent,
            prompt_builder=ProspectiveOmniGAIAPrompt,
            main_agent_kwargs={"prospective_config": self.prospective_config},
            log_prefix="MAESTRO",
            summarize_trace=True,
        )

    def _after_trajectory(
        self,
        *,
        trajectory_folder: Path,
        level_id: str,
        timestamp: str,
        main_agent: MainAgent | None,
        attempts_detail: List[Dict[str, Any]],
    ) -> None:
        if main_agent is None or not hasattr(main_agent, "observed_state"):
            return
        filename = (
            f"omnigaia_{level_id}_{timestamp}.json"
            if timestamp
            else f"omnigaia_{level_id}.json"
        )
        path = trajectory_folder / filename
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        data["orchestrator_version"] = "maestro_epistemic_projection_v4"
        data["maestro_trace"] = main_agent.observed_state.audit_trace()
        data["usage_summary"] = self._aggregate_usage(
            main_agent,
            attempts_detail,
        )
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _aggregate_usage(
        main_agent: MainAgent,
        attempts_detail: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        main_usage = main_agent.get_usage_summary()
        sub_input = sub_output = sub_calls = 0
        sub_cost = 0.0
        for attempt in attempts_detail:
            usage = (attempt.get("result") or {}).get("usage") or {}
            sub_input += int(usage.get("total_input_tokens", 0) or 0)
            sub_output += int(usage.get("total_output_tokens", 0) or 0)
            sub_cost += float(usage.get("total_cost", 0.0) or 0.0)
            sub_calls += int(usage.get("call_count", 0) or 0)
        return {
            "main_agent": main_usage,
            "subagents": {
                "total_input_tokens": sub_input,
                "total_output_tokens": sub_output,
                "total_tokens": sub_input + sub_output,
                "total_cost": sub_cost,
                "call_count": sub_calls,
            },
            "total_input_tokens": int(
                main_usage.get("total_input_tokens", 0) or 0
            ) + sub_input,
            "total_output_tokens": int(
                main_usage.get("total_output_tokens", 0) or 0
            ) + sub_output,
            "total_tokens": int(main_usage.get("total_tokens", 0) or 0)
            + sub_input
            + sub_output,
            "total_cost": float(main_usage.get("total_cost", 0.0) or 0.0)
            + sub_cost,
            "currency": "USD",
        }
