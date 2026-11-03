"""Teacher prompts for clean orchestration trajectory synthesis."""
from __future__ import annotations
import json
from typing import Any,Mapping

def teacher_prompt(payload:Mapping[str,Any])->str:
    public={"question":payload.get("question"),"frontier":payload.get("frontier"),"resources":payload.get("resources",()),"history":payload.get("history",())};return """Generate exactly one clean orchestration step. You may use only resource ids in frontier.available_resource_ids in this round. Do not invent evidence, hidden facts, or future resources. Prefer useful parallel work when multiple resources are simultaneously available. Return JSON with: action (delegate_task|complete), resource_ids, subtask, summary, completion_claim. `complete` is allowed only when no further frozen evidence is needed and the history supports answering the question.\n\n"""+json.dumps(public,ensure_ascii=False,indent=2)
__all__=["teacher_prompt"]
