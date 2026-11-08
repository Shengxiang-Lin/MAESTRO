"""Task-derived Question brief and deterministic reference for V4.5.

The MMQA question is embedded in the terminal typed WorkProduct goal. No
unrelated domain-specific incident template or separate Intent is sampled.
"""
from __future__ import annotations

SOURCE_LABELS = {
    "material": "source media",
    "records": "structured records",
    "context": "supporting documents",
    "rule_source": "applicable guidance",
    "capacity": "capacity records",
    "stakeholders": "stakeholder material",
    "independent_reports": "independent reports",
}


def task_level_brief(plan: dict) -> dict:
    graph = plan["macro_graph"]
    tasks = graph["tasks"]
    terminal = graph["terminal_handoffs"][0]
    final = next(t for t in tasks if t["task_id"] == terminal["from_task"])
    roots = [t for t in tasks if not t["depends_on"]]
    sources = []
    for task in tasks:
        for requirement in task.get("source_requirements", ()):
            role = requirement["role"]
            label = SOURCE_LABELS.get(role, "supporting documentation")
            if label not in sources:
                sources.append(label)
    return {
        "domain": plan["scenario"]["domain"],
        "setting": plan["scenario"]["setting"],
        "issue": plan["scenario"]["issue"],
        "user_role": plan["scenario"]["user_role"],
        "source_kind": plan["scenario"]["source_kind"],
        "intent": plan["goal"]["intent"],
        "final_work_goal": final["goal"],
        "final_deliverable": final["deliverable"],
        "root_work_goals": [t["goal"] for t in roots],
        "supporting_work_types": sorted({t["work_type"] for t in tasks}),
        "public_resource_categories": sources,
    }


def author_question(brief: dict, *, ordinal: int, seed: int) -> str:
    """Reference wording is only the source question carried by terminal goal."""
    if brief["source_kind"] != "MultiModalQA":
        raise ValueError("V4.5 Task-First expects an MMQA-backed information need")
    from .data_taskfirst import FINAL_PREFIX
    goal = brief["final_work_goal"]
    if not goal.startswith(FINAL_PREFIX):
        raise ValueError("missing immutable terminal MMQA information need")
    return goal[len(FINAL_PREFIX):].strip()
