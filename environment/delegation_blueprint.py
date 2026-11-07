"""Shared, PUBLIC delegation blueprint for Stage 2 question writing and Stage 3.

Immutable topology is compiled from compact Gold fragments. The LLM can
ONLY write short task goals; it cannot modify cuts, waves, tools or outputs.
Nothing from intermediate results, oracle answers, witnesses or source
content is sent to the writer. A blueprint is teacher-only scaffolding,
NEVER part of Main's public state.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping

from environment.trajectory_simulation.eligibility import digest
from environment.question_generation.public_semantics import public_semantic_scaffold
from environment.question_generation.validators import postflight


VERSION = "maestro.delegation_blueprint.v1"
_GOAL_SYSTEM = """Write concise, natural Main-to-Sub-Agent GOALS for a benchmark task.
A verified task graph fixes the independent work, dependency order and
final output. Describe what EACH assigned worker should discover, inspect,
compare or return in a clear 1-2 sentence goal (not a numbered query plan).
These are reusable INTENT descriptions, not actual runtime assignments.

DO NOT change task boundaries or invent another task. Preserve each directed
relation, attribute owner, AND/OR join, comparison, candidate pool and output.
A task that requires earlier results must say 'the preceding task result',
NOT reveal those results. Do not claim future sources were already provided:
'discoverable' means the agent must locate them. Do not invent source titles,
media contents, entity names, dates, candidate lists, search results or
answers. Refer to a video/image/audio only if the PUBLIC resource modalities
actually include that medium; for discoverable media, say it must be found.
Do not include internal node IDs or fragment IDs inside the goal text.

Return exactly {"goals":[{"fragment_id":"EXACT ID","goal":"..."}]} with one
entry per input fragment, in the same order. Only the goal strings are free.
"""

_GOAL_SCHEMA = {
    "type": "object",
    "properties": {
        "goals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fragment_id": {"type": "string"},
                    "goal": {"type": "string"},
                },
                "required": ["fragment_id", "goal"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["goals"],
    "additionalProperties": False,
}

_ALLOWED_STEP_FIELDS = (
    "id", "operation", "inputs_from", "entity_type", "attribute",
    "comparison", "public_value", "relation", "direction",
    "rank_attribute", "ranking", "output_attribute", "goal",
)


def _public_modalities(resources):
    modes = set()
    for resource in resources:
        if not isinstance(resource, Mapping):
            continue
        medium = str(resource.get("source_type") or "").lower()
        if not medium:
            continue
        availability = (
            "provided" if resource.get("visibility") in {
                "provided", "attached", "initial", "visible", "provided_initial",
            } else "discoverable"
        )
        modes.add((medium, availability))
    return [
        {"medium": medium, "availability": availability}
        for medium, availability in sorted(modes)
    ]


def compile_delegation_outline(gold, world, canonical, plan, resources=()):
    """Project the SAME compact Gold DAG into an answer-free public brief.

    This deliberately uses public_semantic_scaffold's exact public-match
    filter. NEVER copy Gold operations' raw inputs, source/observation arrays,
    expected_answer, operation_results, or symbolic_trace into the prompt.
    """
    scaffold = public_semantic_scaffold(world, canonical)
    public_steps = {s["id"]: s for s in scaffold["steps"]}
    waves = gold["waves"]
    wave_by_id = {
        fid: i + 1 for i, wave in enumerate(waves) for fid in wave
    }
    fragments = gold["fragments"]
    identifiers = [f["fragment_id"] for f in fragments]
    if set(identifiers) != set(wave_by_id) or len(set(identifiers)) != len(identifiers):
        raise ValueError("blueprint_fragment_wave_mismatch")
    tasks = []
    for fragment in fragments:
        fid = fragment["fragment_id"]
        ops = []
        for raw in fragment["operations"]:
            step = public_steps.get("S" + str(raw["node_id"]))
            if step is None:
                # A compiler-certified concealed identity need not be
                # surfaced as a public constraint to the Author.
                continue
            ops.append({k: step[k] for k in _ALLOWED_STEP_FIELDS if k in step})
        if not ops:
            raise ValueError("blueprint_empty_public_fragment")
        dependencies = list(fragment["depends_on"])
        if any(d not in wave_by_id or wave_by_id[d] >= wave_by_id[fid]
               for d in dependencies):
            raise ValueError("blueprint_invalid_dependency")
        tasks.append({
            "fragment_id": fid,
            "wave": wave_by_id[fid],
            "depends_on": dependencies,
            "input_slots": [
                {"from_fragment": d, "kind": "completed_result"}
                for d in dependencies
            ],
            "output_kind": fragment["answer_kind"],
            "public_steps": ops,
            "capabilities": list(fragment["suggested_tools"]),
        })
    return {
        "schema": "public_delegation_outline.v1",
        "schedule": gold["schedule"],
        "waves": [list(wave) for wave in waves],
        "task_semantics": plan.query,
        "public_modalities": _public_modalities(resources),
        "tasks": tasks,
    }


def _validate_goal(goal, task_id, payload=None, oracle_answer=None):
    if not isinstance(goal, str):
        raise ValueError("blueprint_goal_not_text")
    goal = " ".join(goal.split())
    if len(goal) < 18 or len(goal) > 480:
        raise ValueError("blueprint_goal_length")
    # Do not let the LLM serialize teacher-private identifiers in the goal.
    if re.search(r"\b(?:fragment|node|step)[:_ -]?\d+\b", goal, re.I):
        raise ValueError("blueprint_internal_graph_id_in_goal")
    public_payload = {
        "task_design": {},
        "draft_question": "",
        "resources": (payload or {}).get("resources", []),
    }
    checked = postflight(
        public_payload,
        goal if goal.endswith((".", "?")) else goal + ".",
        oracle_answer=oracle_answer, check_contract=False,
    )
    if not checked.accepted:
        raise ValueError("blueprint_goal_boundary:" + ",".join(checked.issues))
    return goal.rstrip(". ") + "."


def naturalize_delegation_blueprint(backend, outline, reference, *,
                                     payload=None, oracle_answer=None):
    """One LLM request creates reusable GOALS for the whole immutable DAG."""
    if not callable(getattr(backend, "complete_json", None)):
        raise ValueError("blueprint_author_backend_missing")
    response = backend.complete_json(
        _GOAL_SYSTEM,
        json.dumps({
            "verified_public_outline": outline,
            "verified_reference_for_literal_spelling_only": reference,
        }, ensure_ascii=False),
        response_schema=_GOAL_SCHEMA,
    )
    if not isinstance(response, dict) or set(response) != {"goals"}:
        raise ValueError("blueprint_response_schema")
    goals = response["goals"]
    if not isinstance(goals, list) or len(goals) != len(outline["tasks"]):
        raise ValueError("blueprint_goal_count")
    checked = []
    for item, fragment in zip(goals, outline["tasks"]):
        if not isinstance(item, dict) or set(item) != {"fragment_id", "goal"}:
            raise ValueError("blueprint_goal_item_schema")
        if item["fragment_id"] != fragment["fragment_id"]:
            raise ValueError("blueprint_goal_id_mismatch")
        checked.append({
            "fragment_id": item["fragment_id"],
            "goal": _validate_goal(
                item["goal"], item["fragment_id"],
                payload=payload, oracle_answer=oracle_answer,
            ),
        })
    return {"version": VERSION, "outline": outline, "goals": checked}


def validate_delegation_blueprint(blueprint, outline, *,
                                  payload=None, oracle_answer=None):
    """Pure replay validation: no new LLM calls or future data disclosure."""
    if not isinstance(blueprint, dict) or set(blueprint) != {
        "version", "outline", "goals",
    } or blueprint["version"] != VERSION or blueprint["outline"] != outline:
        raise ValueError("delegation_blueprint_outline_mismatch")
    goals = blueprint["goals"]
    if not isinstance(goals, list) or len(goals) != len(outline["tasks"]):
        raise ValueError("delegation_blueprint_goal_count")
    for item, fragment in zip(goals, outline["tasks"]):
        if (not isinstance(item, dict) or set(item) != {"fragment_id", "goal"}
                or item["fragment_id"] != fragment["fragment_id"]):
            raise ValueError("delegation_blueprint_goal_binding")
        checked = _validate_goal(
            item["goal"], item["fragment_id"],
            payload=payload, oracle_answer=oracle_answer,
        )
        if item["goal"] != checked:
            raise ValueError("delegation_blueprint_goal_not_normalized")
    return digest(blueprint)


def goal_for_fragment(blueprint, fragment_id):
    if not isinstance(blueprint, Mapping):
        return None
    for item in blueprint.get("goals", ()):
        if item["fragment_id"] == fragment_id:
            return item["goal"]
    return None


__all__ = [
    "VERSION", "compile_delegation_outline",
    "naturalize_delegation_blueprint", "validate_delegation_blueprint",
    "goal_for_fragment",
]
