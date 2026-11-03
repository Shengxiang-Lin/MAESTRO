"""Environment-authoritative validation for synthesized trajectory steps."""
from __future__ import annotations
from typing import Any,Mapping

def validate_step(step:Mapping[str,Any],frontier:Mapping[str,Any],*,is_final_frontier:bool)->tuple[bool,list[str]]:
    issues=[];action=str(step.get("action") or "");available=set(map(str,frontier.get("available_resource_ids",()))) ;selected=set(map(str,step.get("resource_ids",()) or ()))
    if action not in {"delegate_task","complete"}:issues.append("invalid_action")
    if not selected.issubset(available):issues.append("resource_outside_frontier")
    if action=="delegate_task" and not selected:issues.append("empty_delegation")
    if action=="complete" and not is_final_frontier:issues.append("premature_complete")
    if action=="complete" and selected:issues.append("complete_with_resource_selection")
    return not issues,issues

def validate_trajectory(plan:Mapping[str,Any],steps:list[Mapping[str,Any]])->dict[str,Any]:
    rounds=plan.get("rounds",());issues=[]
    if len(steps)<len(rounds):issues.append("missing_rounds")
    for index,frontier in enumerate(rounds):
        if index>=len(steps):break
        ok,current=validate_step(steps[index],frontier,is_final_frontier=index==len(rounds)-1)
        issues.extend(f"round_{index+1}:{item}" for item in current)
    used={str(resource_id) for step in steps for resource_id in step.get("resource_ids",()) or ()};reachable=set(map(str,plan.get("reachable_resource_ids",())));missing=sorted(reachable-used)
    if missing:issues.append("unobserved_reachable_resources:"+",".join(missing))
    return {"accepted":not issues,"issues":issues,"observed_resource_ids":sorted(used),"reachable_resource_ids":sorted(reachable)}
__all__=["validate_step","validate_trajectory"]
