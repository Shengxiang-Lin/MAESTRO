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


# Keep the official Orchestra-o1 SubAgent-visible prompt unchanged. Backend and
# tool implementations may differ, but the worker should receive the same
# instructions as the official baseline.
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
1. Focus on completing YOUR TASK above
2. Think step by step before outputting an action
3. Write key observations to the "memory" field
4. Use print() in ExecuteCodeAction to see computation results
5. Once done, use 'finish' IMMEDIATELY
6. **IMAGE ANALYSIS RULE:** You may ONLY use ImageAnalysisAction on image URLs that are explicitly provided in your TASK or CONTEXT from the MainAgent. Do NOT use ImageAnalysisAction on any image URLs you encounter during web search or browsing (e.g., thumbnails, page images, search result images). These external image URLs are often inaccessible and will waste your steps.
7. **EFFICIENCY RULE — Avoid Repetitive Attempts:**
   - Count your attempts by **behavior pattern**, not just individual tool names. A "search-then-extract" cycle (e.g., GoogleSearchAction → ExtractUrlContentAction) counts as ONE search attempt, not two separate tool uses.
   - If you have performed the same **behavior pattern** 5 times without finding the target information, STOP immediately. Use 'finish' with whatever partial results you have gathered so far.
   - Examples of behavior patterns that count as the SAME attempt:
     • GoogleSearchAction alone (one search attempt)
     • GoogleSearchAction → ExtractUrlContentAction (one search-and-read attempt)
     • ExtractUrlContentAction alone on different URLs (one URL extraction attempt each)
   - Do NOT keep trying different keyword variants or URLs endlessly. After 5 rounds of the same behavior pattern, you have likely exhausted what can be found.
   - When finishing with partial results, set status to "partial" and clearly describe what you DID find and what you could NOT find. The MainAgent can decide how to proceed.
8. **COMPLETENESS vs PERFECTION:** It is better to return partial results quickly than to waste all your steps searching for information that may not exist. The MainAgent can assign follow-up tasks if needed.
9. **FORBIDDEN IMAGE SOURCES:** Never attempt ImageAnalysisAction on URLs you discovered through GoogleSearchAction or ExtractUrlContentAction. Only analyze images that were part of the ORIGINAL task assignment.

⚠️ BUDGET: When remaining_steps <= 5, use 'finish' NOW with your best available results!
⚠️ EFFICIENCY: After 5 rounds of the same behavior pattern (e.g., repeated search→extract cycles), use 'finish' NOW with partial results!

==== Output Format ====
⚠️ CRITICAL: You MUST reply with ONLY a valid JSON object. No markdown, no extra text.
⚠️ The "action" field MUST be one of the exact tool names listed in Available Tools (e.g., "ImageAnalysisAction"), or "finish".
⚠️ Do NOT use "execute" as the action. Do NOT pass tool names via a "command" field.
⚠️ The "params" field MUST be a JSON object with the exact parameter names defined for that tool.

```json
{{
    "action": "<EXACT_TOOL_NAME>",
    "params": {{ <tool-specific parameters as key-value pairs> }},
    "memory": "<your key observations>"
}}
```

==== Tool Call Examples (Few-Shot) ====

1. GoogleSearchAction — web search:
{{"action": "GoogleSearchAction", "params": {{"query": "population of Tokyo 2024"}}, "memory": "Searching for Tokyo population data."}}

2. ImageAnalysisAction — analyze an image (ONLY for images provided in the original task, NOT for images found via web search):
{{"action": "ImageAnalysisAction", "params": {{"query": "Identify the brand logo visible on the product.", "image_path": "https://example.com/image.jpg"}}, "memory": "Analyzing task-provided image for brand identification."}}

3. ParseAudioAction — transcribe/analyze audio:
{{"action": "ParseAudioAction", "params": {{"query": "Transcribe the audio and extract the speaker's main argument.", "audio_path": "/path/to/audio.wav"}}, "memory": "Transcribing audio to extract key information."}}

4. VideoAnalysisAction — analyze video (frames + audio):
{{"action": "VideoAnalysisAction", "params": {{"query": "Identify the product shown at the 30-second mark.", "video_path": "/path/to/video.mp4", "max_frames": 12, "start_time": 25, "end_time": 35, "analyze_audio": true}}, "memory": "Analyzing video segment 25-35s for product identification."}}

5. ExecuteCodeAction — run Python/Bash code:
{{"action": "ExecuteCodeAction", "params": {{"code": "from math import radians, sin, cos, sqrt, atan2\\nlat1, lon1 = radians(48.8566), radians(2.3522)\\nlat2, lon2 = radians(40.7128), radians(-74.0060)\\ndlat = lat2-lat1\\ndlon = lon2-lon1\\na = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2\\nprint(round(6371*2*atan2(sqrt(a),sqrt(1-a))))", "code_type": "python"}}, "memory": "Calculating distance between Paris and New York."}}

6. ExtractUrlContentAction — fetch web page content:
{{"action": "ExtractUrlContentAction", "params": {{"url": "https://en.wikipedia.org/wiki/Example", "browse_query": "key facts about the topic"}}, "memory": "Extracting content from Wikipedia page."}}

7. finish — report result to MainAgent (task fully completed):
{{"action": "finish", "params": {{"result": "42", "status": "done", "summary": "Computed the answer by analyzing the video and performing calculation."}}, "memory": "Task complete."}}

8. finish — report partial result (when target info cannot be fully found after multiple attempts):
{{"action": "finish", "params": {{"result": "Found that the statistic is cited by multiple sources but the original study/report is not publicly available online.", "status": "partial", "summary": "Performed 5 search-and-extract rounds with different queries. Found secondary citations but could not locate the primary source with methodology details."}}, "memory": "Exhausted 5 search rounds, returning partial findings."}}

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
    # MAESTRO may attach this internally; O1 never emits it and it is not part
    # of the public delegation contract.
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
            logger.info(f"[ReActAgent] Filtered to tools: {self.allowed_tools}")
        else:
            self.current_action_space = env_info.action_space

    @staticmethod
    def _normalize_tool_name(name: str) -> str:
        normalized = name.lower().replace("_", "")
        return normalized[:-6] if normalized.endswith("action") else normalized

    def _tool_matches(self, tool_name: str, allowed_tools: List[str]) -> bool:
        if tool_name in allowed_tools:
            return True
        normalized_tool = self._normalize_tool_name(tool_name)
        return any(
            self._normalize_tool_name(allowed) == normalized_tool
            for allowed in allowed_tools
        )

    def _filter_action_space(
        self,
        action_space: str,
        allowed_tools: List[str],
        preferred_tools: List[str] | None = None,
    ) -> str:
        blocks = re.split(r"\n(?=### )", action_space)
        filtered_blocks: list[tuple[str | None, str]] = []
        for block in blocks:
            if block.startswith("Available actions"):
                filtered_blocks.append((None, block.rstrip()))
                continue
            match = re.match(r"### (\w+)", block)
            if match and self._tool_matches(match.group(1), allowed_tools):
                filtered_blocks.append((match.group(1), block.rstrip()))

        if not preferred_tools:
            return "\n\n".join(block for _, block in filtered_blocks)

        header = [block for name, block in filtered_blocks if name is None]
        preferred = [
            block for name, block in filtered_blocks
            if name is not None and self._tool_matches(name, preferred_tools)
        ]
        remaining = [
            block for name, block in filtered_blocks
            if name is not None and not self._tool_matches(name, preferred_tools)
        ]
        return "\n\n".join(header + preferred + remaining)

    def parse_action(self, resp: str) -> Dict[str, Any]:
        return parse_llm_action_response(resp)

    def _get_memory(self) -> str:
        return self.memory.as_text()

    @staticmethod
    def _get_budget_warning(remaining_steps: int) -> str:
        if remaining_steps <= 3:
            return f"🚨 CRITICAL: Only {remaining_steps} steps left! Use 'finish' NOW with whatever results you have!"
        if remaining_steps <= 5:
            return f"⚠️ Warning: {remaining_steps} steps remaining. Wrap up and prepare to 'finish' soon."
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
        thinking = memory_content.get("memory") if isinstance(memory_content, dict) else None
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
