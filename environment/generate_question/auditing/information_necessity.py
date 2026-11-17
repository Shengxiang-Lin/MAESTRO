"""Current Task Plan information-necessity audit only."""
from __future__ import annotations


def audit_candidate_information(scene: dict) -> dict:
    policy = scene["problem_spec"]["macro_plan"].get("information_necessity_policy")
    if policy != "single_task_plan_counterfactual_v1":
        raise ValueError("unsupported_stage1_information_necessity_policy")
    from environment.generate_question.execution.task_plan_runtime import audit_information
    return audit_information(scene)
