"""Main decisions and per-task simulated feedback, plus legacy reference roles."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Mapping

from environment.question_generation.backend import (
    BackendError, OpenAICompatibleBackend, _extract_content, _json_object,
)


class _GroundingReviewBackend(OpenAICompatibleBackend):
    """Bound verdicts so reviewer reasoning cannot truncate the verdict JSON."""

    def complete_json(self,system_prompt,user_prompt):
        schema={
            "type":"object","properties":{
                "accept":{"type":"boolean"},
                "issues":{"type":"array","maxItems":3,"items":{
                    "type":"object","properties":{
                        "quote":{"type":"string","maxLength":180},
                        "reason":{"type":"string","maxLength":240}},
                    "required":["quote","reason"],"additionalProperties":False}}},
            "required":["accept","issues"],"additionalProperties":False,
        }
        body={
            "model":self.model,"messages":[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt}],
            "temperature":0.0,"max_tokens":self.max_tokens,
        }
        if self.json_mode:
            body["response_format"]={"type":"json_schema","json_schema":{
                "name":"grounding_review","strict":True,"schema":schema}}
        headers={"Content-Type":"application/json"}
        if self.api_key:headers["Authorization"]="Bearer "+self.api_key
        url=self.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):url+="/chat/completions"
        request=urllib.request.Request(
            url,data=json.dumps(body,ensure_ascii=False).encode("utf-8"),
            headers=headers,method="POST")
        try:
            with urllib.request.urlopen(request,timeout=self.timeout) as response:
                payload=json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError,TimeoutError,json.JSONDecodeError) as exc:
            raise BackendError(f"grounding review request failed: {exc}") from exc
        if any(choice.get("finish_reason")=="length" for choice in payload.get("choices",())):
            raise BackendError("grounding review output truncated")
        return _json_object(_extract_content(payload))


class _O1GenerationBackend(OpenAICompatibleBackend):
    """Constrain native wire structure without constraining Main's decisions."""

    def complete_structured(self, system_prompt, user_prompt, schema):
        body = {
            "model": self.model, "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}],
            "temperature": self.temperature, "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "o1_generation", "strict": True, "schema": schema}}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        url = self.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        request = urllib.request.Request(
            url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise BackendError(f"O1 generation request failed: {exc}") from exc
        if any(choice.get("finish_reason") == "length" for choice in payload.get("choices", ())):
            raise BackendError("O1 generation output truncated; increase max_tokens")
        return _json_object(_extract_content(payload))


class PassthroughPlanner:
    offline=True
    """Deterministic offline teacher for regression tests, not benchmark data."""

    def decide(self, payload, prompt):
        from .o1_contract import answer_string
        if payload["evidence_sufficient"]:
            return {"action":"complete","reasoning":"",
                    "params":{"answer":answer_string(payload["offline_answer"])}}
        work=payload["feasible_work"]
        return {
            "action":"delegate_task","reasoning":"",
            "params":{"tasks":[{
                "task_instruction":f"{item['kind'].capitalize()} {item['resource_id']} and report its evidence.",
                "context":payload["public_state"]["question"],
                "model":"model_1","tools":item["suggested_tools"],
            } for item in work]},
            "task_bindings":[{"task_index":index,"subtask_ids":[item["simulation_subtask_id"]]}
                             for index,item in enumerate(work)],
        }

    def plan(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        return {
            "parallel_subtasks": [
                {
                    "subtask_id": task["subtask_id"],
                    "agent": task["agent"],
                    "resource_id": task["resource_id"],
                    "instruction": f"Inspect {task['resource_id']} for evidence relevant to the question.",
                }
                for task in payload["ready_tasks"]
            ]
        }

    def summarize(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        ids = [result["subtask_id"] for result in payload["current_results"]]
        return {"main_summary": "Completed and incorporated evidence from " + ", ".join(ids) + "."}

    def review(self,prompt):
        return {"accept":True,"issues":[]}

    def finalize(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        return {
            "action": "complete",
            "answer": payload["expected_answer"],
            "summary": "All necessary evidence has been processed; the official answer is confirmed.",
        }


class LLMPlanner:
    offline=False
    def __init__(self, backend: OpenAICompatibleBackend):
        self.backend = backend
        self.native_backend = _O1GenerationBackend(**asdict(backend))
        self.review_backend = _GroundingReviewBackend(**{**asdict(backend),"temperature":0.0})

    def review(self,prompt):
        return self.review_backend.complete_json("Verify summary grounding against supplied evidence. Return JSON only.",prompt)

    def decide(self,payload,prompt):
        from .o1_contract import action_schema
        return self.native_backend.complete_structured(
            "You are the Orchestra-o1 Main Agent. Decide your next native action from "
            "the user's question and previous Sub-Agent replies. You choose the task "
            "grouping and parallel width. Return the requested JSON only.",prompt,
            action_schema(payload))

    def plan(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        return self.backend.complete_json(
            "You are a gold-trajectory Main Orchestrator. "
            "Do not modify the frozen solution plan or access future evidence. "
            "Return only the requested JSON.", prompt
        )

    def summarize(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        return self.backend.complete_json(
            "You are a gold-trajectory Main Orchestrator. Summarize only "
            "already-observed subagent results, not unseen evidence. Return JSON.", prompt
        )

    def finalize(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        return self.backend.complete_json(
            "You are a gold-trajectory Main Orchestrator. Conclude the task "
            "from completed subagent evidence; return only JSON.", prompt
        )


class PassthroughExecutor:
    offline=True
    """Offline deterministic gold-copying reference (no LLM)."""

    def respond(self,payload,prompt):
        claims={task["subtask_id"]:deepcopy(task["gold_answer_claims"]) for task in payload["scopes"]}
        report="\n".join(
            f"{task['resource_id']} ({task['observation']['title']}): "
            +json.dumps(task["gold_answer_claims"],ensure_ascii=False)
            for task in payload["scopes"])
        return {
            "finish_result":{"result":report,"status":"done",
                             "summary":"Returned the assigned source-scoped evidence."},
            "trace_summary":"Processed the assigned observations and returned their verified evidence; discovery descriptors do not establish read contents.",
            "grounded_claims":claims,
        }

    def review(self,prompt):
        return {"accept":True,"issues":[]}

    def execute(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        task = payload["task"]
        return {
            "subtask_id": task["subtask_id"],
            "status": "success",
            "answer": task["gold_answer_claims"],
            "answer_summary": "Grounded facts have been obtained from the assigned resource.",
            "trajectory_summary": "Inspected the supplied evidence and extracted supported facts.",
            "evidence_source_ids": [task["resource_id"]],
        }


class LLMExecutor:
    offline=False
    def __init__(self, backend: OpenAICompatibleBackend):
        self.backend = backend
        self.native_backend = _O1GenerationBackend(**asdict(backend))
        self.review_backend = _GroundingReviewBackend(**{**asdict(backend),"temperature":0.0})

    def review(self,prompt):
        return self.review_backend.complete_json("Verify summary grounding against supplied evidence. Return JSON only.",prompt)

    def respond(self,payload,prompt):
        from .o1_contract import feedback_schema
        return self.native_backend.complete_structured(
            "You simulate one Orchestra-o1 Sub-Agent task's answer and trace_summary. "
            "Use only its assigned evidence and known context. Do not simulate other tasks "
            "or fabricate raw tool traces. Return the requested JSON only.",prompt,
            feedback_schema(payload))

    def execute(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        return self.backend.complete_json(
            "You are a clean Sub-Agent teacher. Use only the assigned source. "
            "Copy the authoritative structured facts exactly; do not invent "
            "additional evidence, conclusions, or unseen results. Return JSON.", prompt
        )


__all__ = [
    "LLMPlanner", "LLMExecutor", "PassthroughPlanner", "PassthroughExecutor",
]
