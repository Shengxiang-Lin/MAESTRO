"""Teacher backends for round-by-round clean trajectory simulation."""
from __future__ import annotations
from typing import Any,Mapping
from environment.question_generation.backend import OpenAICompatibleBackend
class PassthroughTeacher:
    def generate(self,payload:Mapping[str,Any],prompt:str)->dict[str,Any]:
        frontier=payload.get("frontier",{}) if isinstance(payload.get("frontier"),Mapping) else {};resources=list(frontier.get("available_resource_ids",()))
        return {"action":"delegate_task" if resources else "complete","resource_ids":resources,"subtask":"Inspect the currently available evidence and report only grounded observations." if resources else "","summary":"Use the frozen evidence frontier.","completion_claim":not resources}
class LLMTeacher:
    def __init__(self,backend:OpenAICompatibleBackend):self.backend=backend
    def generate(self,payload,prompt):return self.backend.complete_json("You are a teacher generating one clean orchestration step. Obey the environment-authoritative frontier. Return JSON only.",prompt)
__all__=["LLMTeacher","PassthroughTeacher"]
