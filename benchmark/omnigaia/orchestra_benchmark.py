"""OmniGAIA benchmark environment shared by MAESTRO and Orchestra-o1."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from base.agent.base_action import BaseAction
from base.engine.logs import logger
from benchmark.benchmark import Benchmark, LevelSpec
from benchmark.common.env import Action, BasicInfo, Environment, Observation
from benchmark.omnigaia.llm_scorer import llm_semantic_score

PROJECT_ROOT = Path(__file__).parent.parent.parent

OMNI_MODAL_TOOL_HINTS = {
    "video": (
        "Use VideoAnalysisAction to analyze video content. Enable audio analysis "
        "when speech or sound is relevant."
    ),
    "audio": "Use ParseAudioAction to transcribe and analyze audio content.",
    "image": "Use ImageAnalysisAction to analyze image content.",
}

ACTION_SPACE_TEMPLATE = """
### finish
Description: Report your result to MainAgent. Use when you have found the answer or cannot proceed.
Parameters: {"result": "<answer>", "status": "done|partial|blocked", "summary": "<brief summary>"}

[IMPORTANT RULES]
- Use print() in ExecuteCodeAction to see computation results.
- Write generated/downloaded/intermediate files under ORCHESTRA_TMP_DIR.
- Use 'finish' to report your result when done.
- When remaining steps < 5, finish with your best result.
- Use VideoAnalysisAction for video, ParseAudioAction for audio, and ImageAnalysisAction for task-provided images.

[STRICT OUTPUT FORMAT]
Return one JSON object whose action is an exact available tool name or "finish".
""".strip()


@dataclass
class OmniGAIAConfig:
    dataset_path: Path
    attachments_dir: Path
    max_steps: int = 30
    max_tasks: Optional[int] = None
    level_filter: Optional[List[str]] = None
    result_folder: Path = PROJECT_ROOT / "workspace/logs/omnigaia_results"
    trajectory_folder: Path = PROJECT_ROOT / "workspace/logs/omnigaia_trajectories"
    timestamp: Optional[str] = None

    @classmethod
    def load(cls, config_path: Path | str) -> "OmniGAIAConfig":
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        def resolve_required(key: str) -> Path:
            value = raw.get(key)
            if not value:
                raise ValueError(f"{key} is required for OmniGAIA")
            path = Path(value)
            return path if path.is_absolute() else PROJECT_ROOT / path

        def resolve_optional(key: str, default: str) -> Path:
            path = Path(raw.get(key, default))
            return path if path.is_absolute() else PROJECT_ROOT / path

        level_filter = raw.get("level_filter")
        if level_filter is not None:
            if isinstance(level_filter, str):
                level_filter = [level_filter]
            else:
                level_filter = [str(level) for level in level_filter]

        return cls(
            dataset_path=resolve_required("dataset_path"),
            attachments_dir=resolve_required("attachments_dir"),
            max_steps=int(raw.get("max_steps", 30)),
            max_tasks=int(raw["max_tasks"]) if raw.get("max_tasks") else None,
            level_filter=level_filter,
            result_folder=resolve_optional(
                "result_folder", "workspace/logs/omnigaia_results"
            ),
            trajectory_folder=resolve_optional(
                "trajectory_folder", "workspace/logs/omnigaia_trajectories"
            ),
        )


class OmniGAIAOrchestraEnvironment(Environment):
    """Environment executed independently by each shared SubAgent."""

    def __init__(
        self,
        level: LevelSpec,
        config: OmniGAIAConfig,
        tools: List[BaseAction],
    ) -> None:
        self.task_id = str(level.get("id") or level.get("task_id") or "unknown")
        self.config = config
        self.tools: Dict[str, BaseAction] = {tool.name: tool for tool in tools}
        self.question = level.get("question") or level.get("Question") or str(level)
        self.expected_answer = level.get("answer") or level.get("Final answer")
        self.task_level = level.get("Level") or level.get("level")
        self.omni_modal_input = level.get("omni_modal_input", [])
        self.resolved_modal_inputs = self._resolve_modal_inputs(config.attachments_dir)
        self.task_type = level.get("task_type", "")
        self.category = level.get("category", "")
        self.required_external_tools = level.get("required_external_tools", [])
        self.total_steps = level.get("total_steps")
        self.level_data = level
        self.meta_data = {
            "task_id": self.task_id,
            "level": self.task_level,
            "task_type": self.task_type,
            "category": self.category,
            "required_external_tools": self.required_external_tools,
            "modal_inputs": [
                {
                    "type": item["type"],
                    "id": item.get("id", ""),
                    "path": str(item.get("resolved_path", "")),
                }
                for item in self.resolved_modal_inputs
            ],
            "expected_steps": self.total_steps,
        }
        self._steps = 0
        self._done = False
        self._delegation_instruction_override: str | None = None
        self._delegation_question_override: str | None = None

    def clone(self) -> "OmniGAIAOrchestraEnvironment":
        return OmniGAIAOrchestraEnvironment(
            level=self.level_data,
            config=self.config,
            tools=list(self.tools.values()),
        )

    def set_delegation_view(self, *, instruction: str, question: str) -> None:
        self._delegation_instruction_override = str(instruction)
        self._delegation_question_override = str(question)

    def _resolve_modal_inputs(self, attachments_dir: Path) -> list:
        resolved = []
        for modal_item in self.omni_modal_input:
            item = dict(modal_item)
            rel_path = item.get("path", "")
            if rel_path:
                full_path = attachments_dir / rel_path
                if full_path.exists():
                    item["resolved_path"] = full_path
                else:
                    logger.warning(
                        f"[OmniGAIA] Modal file not found: {full_path} "
                        f"(task {self.task_id}, type={item.get('type')})"
                    )
                    item["resolved_path"] = None
            else:
                item["resolved_path"] = None
            resolved.append(item)
        return resolved

    def _build_action_space(self) -> str:
        descriptions = []
        for name, tool in self.tools.items():
            desc = f"### {name}\nDescription: {tool.description}"
            if tool.parameters:
                desc += f"\nParameters: {json.dumps(tool.parameters, indent=2)}"
            descriptions.append(desc)
        return (
            "Available actions:\n\n"
            + "\n\n".join(descriptions)
            + "\n\n"
            + ACTION_SPACE_TEMPLATE
        )

    def _build_instruction(self) -> str:
        if self._delegation_instruction_override is not None:
            return self._delegation_instruction_override
        instruction = f"Question: {self.question}"
        if self.resolved_modal_inputs:
            instruction += "\n\n[MULTIMODAL INPUTS]"
            for index, modal in enumerate(self.resolved_modal_inputs, 1):
                modal_type = modal.get("type", "unknown")
                modal_id = modal.get("id", "")
                resolved_path = modal.get("resolved_path")
                if resolved_path:
                    hint = OMNI_MODAL_TOOL_HINTS.get(
                        modal_type, "Use the appropriate tool to process this file."
                    )
                    instruction += (
                        f"\n{index}. [{modal_type.upper()}] ID: {modal_id}"
                        f"\n   Path: {resolved_path}"
                        f"\n   Hint: {hint}"
                    )
                else:
                    instruction += (
                        f"\n{index}. [{modal_type.upper()}] ID: {modal_id} "
                        "(file not available)"
                    )
        return instruction

    def get_basic_info(self) -> BasicInfo:
        return BasicInfo(
            env_id=self.task_id,
            instruction=self._build_instruction(),
            action_space=self._build_action_space(),
            max_steps=self.config.max_steps,
            meta_data=self.meta_data,
        )

    async def reset(self, seed: int | None = None) -> Observation:
        del seed
        self._done = False
        self._steps = 0
        visible_question = (
            self._delegation_question_override
            if self._delegation_question_override is not None
            else self.question
        )
        return {
            "message": "Environment ready. Use the available tools to answer the question.",
            "question": visible_question,
            "current_step": 0,
            "max_steps": self.config.max_steps,
        }

    async def step(
        self, action: Action
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Environment already finished. Call reset() first.")
        self._steps += 1
        action_type = action.get("action", "")
        params = action.get("params", {})
        if action_type == "finish":
            return self._handle_finish(params)
        if action_type == "SubmitAnswer":
            return await self._handle_submit(params)
        return await self._handle_tool(action_type, params)

    def _handle_finish(
        self, params: Dict
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        result = params.get("result", "")
        status = params.get("status", "done")
        summary = params.get("summary", "")
        self._done = True
        finish_result = {
            "result": result,
            "status": status,
            "summary": summary,
        }
        logger.info(
            f"[OmniGAIA] Task {self.task_id} finish: result='{result}', "
            f"status='{status}'"
        )
        return (
            {
                "message": "Result reported to MainAgent.",
                "current_step": self._steps,
                "finish_result": finish_result,
            },
            0.0,
            True,
            {"finished": True, "finish_result": finish_result},
        )

    async def _handle_submit(
        self, params: Dict
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        answer = params.get("answer", "")
        try:
            reward = await llm_semantic_score(
                answer,
                self.expected_answer,
                question=self.question,
            )
        except Exception:
            reward = 0.0
        self._done = True
        return (
            {
                "message": "Answer submitted",
                "submitted_answer": answer,
                "expected_answer": self.expected_answer,
                "reward": reward,
                "correct": reward == 1.0,
                "current_step": self._steps,
            },
            reward,
            True,
            {"submitted": True, "correct": reward == 1.0},
        )

    async def _handle_tool(
        self, action_type: str, params: Dict
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        tool = self.tools.get(action_type)
        if tool is None:
            return self._handle_unknown_action(action_type)
        try:
            result = await tool(**params)
            observation = {
                "action": action_type,
                "success": result.get("success", False),
                "output": result.get("output") if result.get("success") else None,
                "error": result.get("error") if not result.get("success") else None,
                "current_step": self._steps,
                "max_steps": self.config.max_steps,
            }
            logger.info(
                f"[OmniGAIA] Task {self.task_id} step {self._steps}: "
                f"{action_type} -> success={result.get('success')}"
            )
        except Exception as exc:
            observation = {
                "action": action_type,
                "success": False,
                "error": str(exc),
                "current_step": self._steps,
                "max_steps": self.config.max_steps,
            }
            logger.error(f"[OmniGAIA] Tool execution error: {exc}")
        return self._check_max_steps(observation)

    def _handle_unknown_action(
        self, action_type: str
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        observation = {
            "error": (
                f"Unknown action: {action_type}. Available actions: "
                f"{list(self.tools.keys()) + ['finish']}"
            ),
            "current_step": self._steps,
            "max_steps": self.config.max_steps,
        }
        if self._steps >= self.config.max_steps:
            return self._timeout_response(observation, {"error": "unknown_action"})
        return observation, 0.0, False, {"error": "unknown_action"}

    def _check_max_steps(
        self, observation: Dict
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._steps >= self.config.max_steps:
            return self._timeout_response(observation, {})
        return observation, 0.0, False, {}

    def _timeout_response(
        self, observation: Dict, extra_info: Dict
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        self._done = True
        finish_result = {
            "result": "",
            "status": "timeout",
            "summary": f"Used all {self.config.max_steps} steps without finish",
        }
        observation["message"] = "Max steps reached"
        observation["finish_result"] = finish_result
        return (
            observation,
            0.0,
            True,
            {
                **extra_info,
                "max_steps_reached": True,
                "finished": True,
                "finish_result": finish_result,
            },
        )

    async def close(self) -> None:
        pass


class OmniGAIAOrchestraBenchmark(Benchmark):
    def __init__(
        self,
        config: OmniGAIAConfig,
        tools: List[BaseAction] | None = None,
    ) -> None:
        self.config = config
        self.tools = tools or []
        self._levels: List[LevelSpec] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        if not self.config.dataset_path.exists():
            logger.warning(
                f"[OmniGAIA] Dataset not found: {self.config.dataset_path}"
            )
            return
        self._levels = []
        with self.config.dataset_path.open("r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        f"[OmniGAIA] Failed to parse line {line_num}: {exc}"
                    )
                    continue
                if "task_id" not in data and "id" in data:
                    data["task_id"] = str(data["id"])
                elif "task_id" not in data:
                    data["task_id"] = f"omnigaia_task_{line_num}"
                if self.config.level_filter is not None:
                    if data.get("Level", "") not in self.config.level_filter:
                        continue
                self._levels.append(data)

        level_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        for level in self._levels:
            level_name = level.get("Level", "unknown")
            task_type = level.get("task_type", "unknown")
            level_counts[level_name] = level_counts.get(level_name, 0) + 1
            type_counts[task_type] = type_counts.get(task_type, 0) + 1
        with_modal = sum(
            1 for level in self._levels if level.get("omni_modal_input")
        )
        logger.info(f"[OmniGAIA] Loaded {len(self._levels)} tasks")
        logger.info(f"[OmniGAIA] Level distribution: {level_counts}")
        logger.info(f"[OmniGAIA] Task type distribution: {type_counts}")
        logger.info(f"[OmniGAIA] With multimodal inputs: {with_modal}")

    def list_levels(self) -> List[LevelSpec]:
        levels = self._levels
        if self.config.max_tasks and len(levels) > self.config.max_tasks:
            levels = levels[: self.config.max_tasks]
        return levels

    def make_env(
        self,
        level: LevelSpec,
        tools: List[BaseAction] | None = None,
    ) -> OmniGAIAOrchestraEnvironment:
        return OmniGAIAOrchestraEnvironment(
            level,
            self.config,
            tools if tools is not None else self.tools,
        )

    def get_level_by_id(self, task_id: str) -> Optional[LevelSpec]:
        return next(
            (
                level
                for level in self._levels
                if str(level.get("task_id")) == task_id
                or str(level.get("id")) == task_id
            ),
            None,
        )
