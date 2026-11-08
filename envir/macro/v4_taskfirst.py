"""Derive initial questions AFTER composition of the complete Macro TaskGraph.

The original MultiModalQA question is a source provenance item, not a writing
prompt, a work-product goal or a terminal Question. Scene Bank and Goal Bank
are combined before graph construction; the terminal task carries the
structured blueprint required to render a fresh user-facing information need.
"""
from __future__ import annotations

from .goal_grammar import GOALS, source_phrase

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
    blueprint = final.get("question_blueprint")
    if not isinstance(blueprint, dict):
        raise ValueError("terminal Macro Task is missing its question blueprint")
    if final["outputs"][terminal["output_slot"]] != blueprint["goal_artifact"]:
        raise ValueError("terminal WorkProduct and question blueprint disagree")
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
        "question_blueprint": dict(blueprint),
    }


def author_question(brief: dict, *, ordinal: int, seed: int) -> str:
    """Create a new Question from the TERMINAL task, not any MMQA question."""
    if brief["source_kind"] != "MultiModalQA":
        raise ValueError("Task-First generation expects an MMQA-backed scene")
    blueprint = brief["question_blueprint"]
    pattern_id = blueprint["pattern_id"]
    spec = GOALS.get(pattern_id)
    if not spec:
        raise ValueError("unknown goal-pattern bank identifier")
    if blueprint["goal_artifact"] != spec["artifact"]:
        raise ValueError("goal-pattern artifact does not match terminal work")
    if blueprint["source_phrase"] != source_phrase(blueprint["modalities"]):
        raise ValueError("source phrase does not match visible graph modalities")
    if blueprint["origin"] != "taskgraph_terminal_workproduct":
        raise ValueError("question blueprint lacks terminal-task provenance")
    return spec["question"].format(
        topic=blueprint["topic"], sources=blueprint["source_phrase"],
    )
