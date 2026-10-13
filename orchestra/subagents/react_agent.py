"""ReAct-style SubAgent shared by MAESTRO and Orchestra-o1."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from pydantic import Field

from base.agent.base_agent import BaseAgent
from base.agent.memory import Memory
from base.engine.logs import LogLevel, logger
from base.engine.utils import parse_llm_action_response, parse_llm_output
from benchmark.common.env import Action, BasicInfo, Observation


OMNIGAIA_PROMPT = """You are a specialized SubAgent. Complete the assigned task efficiently.

==== Progress ====
[Step {current_step}/{max_steps}] Remaining {remaining_steps} steps
{budget_warning}

==== Your Task (from MainAgent) ====
{task_instruction}

==== Context ====
{context}

==== Original Question (for reference) ====
{original_question}

==== Available Tools ====
{action_space}

==== Guidelines ====
1. Focus on completing YOUR TASK above.
2. Think step by step before outputting an action.
3. Write key observations to the "memory" field.
4. Use print() in ExecuteCodeAction to see computation results. Every generated, downloaded, extracted, or intermediate file MUST be written under ORCHESTRA_TMP_DIR (normally the repository's tmp/ directory).
5. Once done, use 'finish' IMMEDIATELY.
6. IMAGE ANALYSIS RULE: You may ONLY use ImageAnalysisAction on image URLs explicitly provided in your TASK or CONTEXT from the MainAgent.
7. EFFICIENCY RULE: after repeated unsuccessful attempts with the same pattern, stop and finish with partial results.
8. COMPLETENESS vs PERFECTION: return useful partial results rather than exhausting all steps.
9. FORBIDDEN IMAGE SOURCES: never analyze web-discovered images; only original task images.

==== Output Format ====
Reply with ONLY one valid JSON object. The "action" must be an exact available tool name or "finish". The "params" field must use the exact parameter names.

{{
    "action": "<EXACT_TOOL_NAME>",
    "params": {{ <tool-specific parameters> }},
    "memory": "<your key observations>"
}}

==== Memory ====
{memory}

==== Current Observation ====
{obs}
"""


class ReActAgent(BaseAgent):
    name: str = Field(default="ReActAgent")
    description: str = Field(default="ReAct-style SubAgent for Orchestra framework")
    benchmark_type: str = Field(default="omnigaia")
    task_instruction: str = Field(default="")
    context: str = Field(default="")
    original_question: str | None = Field(default=None)
    allowed_tools: List[str] | None = Field(default=None)
    preferred_tools: List[str] | None = Field(default=None)
    current_env_instruction: str = Field(default="")
    current_action_space: str = Field(default="")
    memory: Memory = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def reset(self, env_info: BasicInfo) -> None:
        if self.memory is None:
            self.memory = Memory(llm=self.llm, max_memory=10)
        else:
            self.memory.clear()
        if self.original_question is None:
            self.original_question = env_info.instruction
        self.current_env_instruction = env_info.instruction
        if self.allowed_tools:
            self.current_action_space = self._filter_action_space(
                env_info.action_space,
                self.allowed_tools,
                self.preferred_tools,
            )
            logger.info(
                f"[ReActAgent] Filtered to tools: {self.allowed_tools}; "
                f"preferred first: {self.preferred_tools or []}"
            )
        else:
            self.current_action_space = env_info.action_space

    @staticmethod
    def _normalize_tool_name(name: str) -> str:
        normalized = name.lower().replace("_", "")
        return normalized[:-6] if normalized.endswith("action") else normalized

    def _tool_matches(self, tool_name: str, allowed_tools: List[str]) -> bool:
        return tool_name in allowed_tools or any(
            self._normalize_tool_name(allowed)
            == self._normalize_tool_name(tool_name)
            for allowed in allowed_tools
        )

    def _filter_action_space(
        self,
        action_space: str,
        allowed_tools: List[str],
        preferred_tools: List[str] | None = None,
    ) -> str:
        blocks = re.split(r"\n(?=### )", action_space)
        header_blocks, tool_blocks = [], []
        for block in blocks:
            if block.startswith("Available actions"):
                header_blocks.append(block.rstrip())
                continue
            match = re.match(r"### (\w+)", block)
            if match and self._tool_matches(match.group(1), allowed_tools):
                tool_blocks.append((match.group(1), block.rstrip()))
        preferred_tools = preferred_tools or []
        preferred = [
            block
            for name, block in tool_blocks
            if self._tool_matches(name, preferred_tools)
        ]
        remaining = [
            block
            for name, block in tool_blocks
            if not self._tool_matches(name, preferred_tools)
        ]
        return "\n\n".join(header_blocks + preferred + remaining)

    def parse_action(self, resp: str) -> Dict[str, Any]:
        return parse_llm_action_response(resp)

    def _get_memory(self) -> str:
        return self.memory.as_text()

    @staticmethod
    def _get_budget_warning(remaining_steps: int) -> str:
        if remaining_steps <= 3:
            return (
                f"CRITICAL: Only {remaining_steps} steps left! Use 'finish' NOW "
                "with whatever results you have!"
            )
        if remaining_steps <= 5:
            return (
                f"Warning: {remaining_steps} steps remaining. Wrap up and prepare "
                "to 'finish' soon."
            )
        return ""

    async def step(
        self,
        observation: Observation,
        history: Any,
        current_step: int = 1,
        max_steps: int = 30,
    ) -> tuple[Action, str, str]:
        remaining_steps = max_steps - current_step
        prompt = OMNIGAIA_PROMPT.format(
            task_instruction=self.task_instruction,
            context=self.context or "None",
            original_question=self.original_question or "Not provided",
            action_space=self.current_action_space,
            memory=self._get_memory(),
            obs=observation,
            current_step=current_step,
            max_steps=max_steps,
            remaining_steps=remaining_steps,
            budget_warning=self._get_budget_warning(remaining_steps),
        )
        logger.log_to_file(LogLevel.INFO, f"ReActAgent Input:\n{prompt}\n")
        try:
            resp = await self.llm(prompt)
        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            resp = ""
        memory_content = parse_llm_output(resp, "memory")
        thinking = (
            memory_content.get("memory")
            if isinstance(memory_content, dict)
            else None
        )
        action = self.parse_action(resp)
        logger.agent_action(f"ReActAgent Action: {action}")
        agent_obs = history[-1].info.get("last_action_result") if history else None
        await self.memory.add_memory(
            obs=agent_obs,
            action=action,
            thinking=thinking,
            raw_response=resp,
        )
        return action, resp, prompt

    async def run(self, request: str = None) -> str:
        del request
        return ""
