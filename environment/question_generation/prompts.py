"""Public-only author and critic prompts for frozen task realization."""
from __future__ import annotations
import json
from typing import Any,Mapping
_AUTHOR_SYSTEM="""Rewrite the deterministic draft into one natural benchmark question while preserving the frozen public task contract exactly. You may improve wording only. Do not add facts, resources, proof requirements, hidden criteria, or answer hints. Do not reveal internal resource ids, capability ids, orchestration labels, or hidden acquisition structure. Return JSON: {\"question\": \"...\", \"notes\": \"brief wording note\"}."""
_CRITIC_SYSTEM="""Check whether the candidate question has exactly the same public semantics as the frozen contract. Reject changed operators, missing operands, changed quantifiers, changed comparison direction, invented constraints/resources, answer leakage, or wording that exposes hidden acquisition semantics. Cosmetic changes are allowed. Return JSON: {\"accept\": true|false, \"semantic_fidelity\": \"pass|fail\", \"issues\": [\"...\"], \"repair_question\": \"...\"}. If repair is possible, repair wording only; never change the contract."""
def author_prompt(payload:Mapping[str,Any],*,retry_feedback:str|None=None)->str:
    public={"draft_question":payload.get("draft_question"),"task_design":payload.get("task_design",{}),"resources":payload.get("resources",()),"available_agents":payload.get("available_agents",())};parts=[_AUTHOR_SYSTEM,"\nFrozen public task:\n"+json.dumps(public,ensure_ascii=False,indent=2)]
    if retry_feedback:parts.append("\nPrevious deterministic/critic feedback:\n"+retry_feedback+"\nRepair only those wording defects.")
    return "\n".join(parts)
def critic_prompt(payload:Mapping[str,Any],question:str)->str:
    public={"draft_question":payload.get("draft_question"),"task_design":payload.get("task_design",{}),"resources":payload.get("resources",())};return _CRITIC_SYSTEM+"\n\nFrozen public task:\n"+json.dumps(public,ensure_ascii=False,indent=2)+"\n\nCandidate question:\n"+str(question)
__all__=["author_prompt","critic_prompt"]
