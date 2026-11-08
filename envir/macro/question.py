"""New natural user Question derived from the terminal Macro WorkProduct.

The TaskGraph is compiled BEFORE this module runs. MMQA original questions
are private provenance, never authoring hints or target operations. The
deterministic variants are fallbacks; an optional independent LLM Author and
Critic may refine the user-facing wording without mutating the TaskGraph.
"""
from __future__ import annotations

from envir.schema import digest
from .goal_grammar import GOALS, source_phrase

SOURCE_LABELS = {
    "material": "source media",
    "records": "structured records",
    "context": "supporting documents",
}

# Three semantically equivalent constructions for each goal. No template
# adds entities, observations or capabilities missing from the source world.
QUESTION_VARIANTS = {
    "lookup": (
        "Which candidates in the {sources} observation records concerning {topic} "
        "are marked as having the requested attribute? Identify the supporting source.",
        "From the {sources} records about {topic}, identify the candidates with a "
        "positive attribute observation and indicate where each result is recorded.",
    ),
    "composition": (
        "How can the candidate references in the {sources} records about {topic} "
        "be connected, and which sources establish those links?",
        "What source-to-candidate relationships can be traced across the {sources} "
        "records for {topic}? Give the references supporting each connection.",
    ),
    "comparison": (
        "Which candidate ranks first by the recorded attribute scores in the "
        "{sources} observations for {topic}, and what evidence supports the ranking?",
        "Compare the recorded candidate scores across the {sources} references "
        "concerning {topic}. Who or what ranks highest, and why?",
    ),
    "intersection": (
        "Which candidate groups appear in more than one {sources} source for "
        "{topic}? Identify the records that establish each match.",
        "Across the {sources} sources related to {topic}, find the recurring "
        "candidate groups and cite the references supporting the overlaps.",
    ),
    "aggregation": (
        "For {topic}, how many observed candidates have a positive attribute "
        "flag across the {sources} records, and what evidence contributes to the total?",
        "Count the positive candidate observations documented in the {sources} "
        "sources concerning {topic}, identifying the contributing records.",
    ),
    "temporal": (
        "Using the {sources} observations about {topic}, establish the relative "
        "sequence of candidate events and identify the earliest one.",
        "What chronology is supported by the recorded time indices in the "
        "{sources} sources for {topic}, and which observation comes first?",
    ),
    "prioritization": (
        "Which candidates should be listed first by the recorded scores in "
        "the {sources} records for {topic}? Explain their ordering using the source references.",
        "Produce a source-grounded priority ordering for candidates associated "
        "with {topic}, using the scores in the {sources} observations.",
    ),
    "feasibility": (
        "Which candidates related to {topic} pass both the positive-attribute "
        "and score-at-least-50 checks in the available {sources} records?",
        "Based on the {sources} observations for {topic}, identify records that "
        "meet both requirements: attribute present and recorded score at least 50.",
    ),
}


def task_level_brief(plan: dict) -> dict:
    graph = plan["macro_graph"]
    tasks = graph["tasks"]
    terminal = graph["terminal_handoffs"][0]
    final = next(t for t in tasks if t["task_id"] == terminal["from_task"])
    roots = [t for t in tasks if not t["depends_on"]]
    source_labels = []
    for task in tasks:
        for required in task.get("source_requirements", []):
            label = SOURCE_LABELS.get(required["role"], "supporting material")
            if label not in source_labels:
                source_labels.append(label)
    blueprint = final.get("question_blueprint")
    if not isinstance(blueprint, dict):
        raise ValueError("terminal WorkProduct lacks a question blueprint")
    if final["outputs"].get(terminal["output_slot"]) != blueprint["goal_artifact"]:
        raise ValueError("terminal output does not match question blueprint")
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
        "public_resource_categories": source_labels,
        "question_blueprint": dict(blueprint),
    }


def author_question(brief: dict, *, ordinal: int, seed: int) -> str:
    if brief["source_kind"] != "MultiModalQA":
        raise ValueError("Task-First expects a dataset-grounded scene")
    blueprint = brief["question_blueprint"]
    pattern_id = blueprint["pattern_id"]
    spec = GOALS.get(pattern_id)
    if not spec or blueprint["goal_artifact"] != spec["artifact"]:
        raise ValueError("selected goal and terminal question blueprint mismatch")
    if blueprint["source_phrase"] != source_phrase(blueprint["modalities"]):
        raise ValueError("source modalities and question blueprint mismatch")
    if blueprint["origin"] != "taskgraph_terminal_workproduct":
        raise ValueError("question must originate from a terminal WorkProduct")
    options = (spec["question"], *QUESTION_VARIANTS[pattern_id])
    index = int(digest(
        seed, ordinal, pattern_id, blueprint.get("recipe_route"),
        "question_variant"
    )[:10], 16) % len(options)
    return options[index].format(
        topic=blueprint["topic"], sources=blueprint["source_phrase"]
    )
