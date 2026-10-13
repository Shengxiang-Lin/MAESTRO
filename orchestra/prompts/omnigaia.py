"""Shared OmniGAIA Main-Agent prompt used by MAESTRO and Orchestra-o1."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from orchestra.base_main_agent import build_model_pricing_table


def format_tools_description(tools: List[Any]) -> str:
    if not tools:
        return "No tools available."
    descriptions = []
    for tool in tools:
        desc = f"Tool Name: {tool.name}\nDescription: {tool.description}"
        if tool.parameters:
            desc += f"\nParameters: {json.dumps(tool.parameters, indent=2)}"
        descriptions.append(desc)
    return "\n\n".join(descriptions)


class OmniGAIAMainAgentPrompt:
    @staticmethod
    def build_prompt(
        instruction: str,
        meta: Dict[str, Any],
        prior_context: str,
        attempt_index: int,
        max_attempts: int,
        sub_models: List[str],
        subtask_history: str = "",
        model_to_alias: Dict[str, str] | None = None,
        tools: List[Any] | None = None,
    ) -> str:
        remaining_attempts = max_attempts - attempt_index
        model_pricing_table = build_model_pricing_table(sub_models, model_to_alias)
        tools_description = format_tools_description(tools or [])
        modal_inputs = meta.get("modal_inputs", [])
        modal_info = "\n".join(
            f"  - Type: {item.get('type', 'unknown')}, ID: {item.get('id', '')}, Path: {item.get('path', 'N/A')}"
            for item in modal_inputs
        ) if modal_inputs else ""
        task_type = meta.get("task_type", "")
        category = meta.get("category", "")
        return f"""
You are the MainAgent (Orchestrator) for OmniGAIA benchmark tasks. Analyze the QUESTION, plan a multi-phase execution strategy, and delegate subtasks to SubAgents — maximizing parallelism while respecting task dependencies.

==== CORE PRINCIPLE: SMART PARALLEL DECOMPOSITION ====
1. Identify subtasks that are INDEPENDENT and can run in parallel NOW.
2. Identify subtasks that DEPEND on prior results and must wait.
3. In each delegation round, submit ALL currently-runnable independent subtasks together.
4. After receiving results, plan the NEXT round based on what was learned.

Key rules:
- Each subtask runs as an independent SubAgent with its own environment.
- All subtasks within one delegation call execute simultaneously.
- Always use the `tasks` list format, even for one subtask.
- Each delegation counts as ONE attempt regardless of batch size.

MULTIMODAL TASK HANDLING:
- VIDEO: use VideoAnalysisAction; keep audio analysis enabled when relevant.
- STANDALONE AUDIO: use ParseAudioAction.
- IMAGE: use ImageAnalysisAction.
- WEB SEARCH: use GoogleSearchAction.
- COMPUTATION: use ExecuteCodeAction.
- Process multimodal inputs before broad web search when they contain critical context.

{f"Task Type: {task_type}" if task_type else ""}
{f"Category: {category}" if category else ""}
{f"Multimodal Inputs:" + chr(10) + modal_info if modal_info else ""}

DECISION PROCESS:
1. Review SUBTASK HISTORY: status, result, summary, execution provenance.
2. Decide whether the visible results sufficiently answer the QUESTION.
3. If sufficient, use `complete`; otherwise delegate the smallest useful next batch.
4. For failed/incomplete subtasks, adjust instruction/context/model as needed.
5. Think ahead about dependencies and remaining attempts.

BUDGET AWARENESS:
- Attempts are limited.
- Each delegation is one attempt, regardless of parallel width.
- Maximize useful parallelism, but do not create redundant subtasks merely to fill a batch.
- If a result is well-supported and sufficient, complete rather than verifying indefinitely.

==== MODEL SELECTION GUIDE ====
{model_pricing_table}

Higher-priced models are generally more capable. Choose cheaper models for straightforward work and more capable models for difficult reasoning or multimodal analysis. Different parallel subtasks may use different models when useful.

==== Progress ====
[Attempt {attempt_index}/{max_attempts}] Remaining {remaining_attempts} attempts

==== QUESTION ====
{instruction}

==== SUBTASK HISTORY ====
{subtask_history if subtask_history else "No subtasks completed yet."}

==== AVAILABLE TOOLS (for SubAgents) ====
{tools_description}

==== OUTPUT FORMAT ====
ANSWER FORMAT: precise, concise answers only in the answer field.

If results are SUFFICIENT:
{{
  "action": "complete",
  "reasoning": "brief justification from visible results",
  "params": {{"answer": "concise answer"}}
}}

If more work is NEEDED:
{{
  "action": "delegate_task",
  "reasoning": "brief explanation of the useful next work",
  "params": {{
    "tasks": [
      {{
        "task_instruction": "specific actionable subtask",
        "context": "relevant prior findings only",
        "model": "one of {sub_models}",
        "tools": ["exact available tool names"]
      }}
    ]
  }}
}}

IMPORTANT:
- Use the `tasks` list format for every delegation.
- Subtasks in the same batch must be mutually independent.
- Later-round subtasks may depend on earlier results; pass only relevant context.
- Use exact allowed model aliases and exact available tool names.
- If retrying, improve the instruction/context/model instead of blindly repeating the same request.
""".strip()


__all__ = ["OmniGAIAMainAgentPrompt", "format_tools_description"]
