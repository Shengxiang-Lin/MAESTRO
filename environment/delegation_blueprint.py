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
from concurrent.futures import ThreadPoolExecutor

from environment.trajectory_simulation.eligibility import digest
from environment.question_generation.public_semantics import public_semantic_scaffold
from environment.question_generation.validators import postflight


VERSION = "maestro.delegation_blueprint.v2"
_GOAL_SYSTEM = """Write ONE concise, natural-language SUBTASK INTENT for the
CURRENT fragment. This is a high-level planning hint, NOT an executable
replacement for the precise public semantic contract.

Use current_fragment.public_steps, output_kind, and direct
dependency_contracts to identify:
(1) the main action or investigation this fragment contributes,
(2) its principal entity/information target, and
(3) the kind of result it should return.

Summarize the intent in ONE short sentence, usually 15-45 words. It can
omit numeric thresholds, dates, most intermediate filters, exact relation
names, mechanical operators, and execution details. Prefer a meaningful
description such as 'Find the events involving people who satisfy the
specified biographical and physical criteria' to a full list of thresholds.
The full contract remains available separately to Question and Sub-Agent
writers; DO NOT try to encode every step in this short Goal.

Nevertheless, a summary must NOT contradict the fragment. Do not change
which entity is returned, turn a count into a list, invent a requirement,
reverse an explicitly mentioned relation, swap AND for OR, or borrow filters
from another branch. For a multi-input operation, describe its role
(e.g. reconcile or combine the upstream candidate groups) without falsely
imposing a dependency on a parallel sibling. Describe prerequisite outputs
by their public meaning; do not call them 'the previous task'.

Do not generate fragment IDs, waves, tools, evidence facts, candidate rosters,
source titles, private labels, Gold results or output answers. Public null
labels are unspecified: do not guess their precise values. Discoverable
sources must not be described as already attached or inspected.

Your output is only the intent; the immutable DAG retains ALL exact
numbers, dates, filters and conditions for downstream semantic checks.
Return exactly one JSON object: {"goal":"..."}.
"""

_GOAL_SCHEMA = {
    "type": "object",
    "properties": {"goal": {"type": "string"}},
    "required": ["goal"],
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
    if re.search(
        r"\b(?:previous|preceding|last)\s+(?:task|subtask)(?:'s)?\b",
        goal, re.I,
    ):
        raise ValueError("blueprint_ambiguous_positional_dependency")
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


def _fragment_goal_packet(outline, fragment, reference):
    """Only this fragment, immutable direct prerequisites and final goal.

    Deliberately exclude siblings' authored prose, operation_results,
    future answers and evidence. Internal IDs are routing labels in the
    PRIVATE author prompt, never user-visible text.
    """
    by_id = {task["fragment_id"]: task for task in outline["tasks"]}
    dependencies = [{
        "input_slot": {"from_fragment": dep, "kind": "completed_result"},
        "output_kind": by_id[dep]["output_kind"],
        "public_steps": by_id[dep]["public_steps"],
    } for dep in fragment["depends_on"]]
    return {
        "current_fragment": fragment,
        "dependency_contracts": dependencies,
        "global_output": {
            "kind": outline["tasks"][-1]["output_kind"],
        },
        "public_modalities": outline["public_modalities"],
    }


def naturalize_fragment_goal(backend, outline, fragment, reference, *,
                             payload=None, oracle_answer=None,
                             max_attempts=2, call_log=None, feedback=None):
    """Naturalize ONE fragment; caller binds ID and DAG properties itself."""
    if not callable(getattr(backend, "complete_json", None)):
        raise ValueError("blueprint_author_backend_missing")
    if not 1 <= max_attempts <= 2:
        raise ValueError("invalid_blueprint_goal_attempts")
    packet = _fragment_goal_packet(outline, fragment, reference)
    if feedback:
        packet["revision_feedback"] = {
            "previous_goal": feedback.get("previous_goal"),
            "issues": feedback.get("issues", []),
        }
    last_error = None
    for attempt in range(max_attempts):
        if call_log is not None:
            call_log["calls"] += 1
        try:
            # The caller may generate multiple fragments concurrently.
            # Each request has its own immutable local packet.
            answer = backend.complete_json(
                _GOAL_SYSTEM + (
                    "\nYour prior output was invalid. Correct the JSON "
                    "format and respect the exact local constraints."
                    if attempt else ""
                ),
                json.dumps(packet, ensure_ascii=False),
                response_schema=_GOAL_SCHEMA,
            )
            if not isinstance(answer, dict) or set(answer) != {"goal"}:
                raise ValueError("blueprint_fragment_goal_schema")
            goal = _validate_goal(
                answer["goal"], fragment["fragment_id"],
                payload=payload, oracle_answer=oracle_answer,
            )
            return {"fragment_id": fragment["fragment_id"], "goal": goal}
        except (ValueError, KeyError, TypeError) as exc:
            last_error = exc
            if call_log is not None:
                call_log.setdefault("issues", []).append(str(exc)[:160])
    raise ValueError(
        "blueprint_fragment_failed:" + fragment["fragment_id"] +
        ":" + str(last_error)
    )


def naturalize_delegation_blueprint(backend, outline, reference, *,
                                     payload=None, oracle_answer=None,
                                     max_workers=2, audit=None):
    """Independently write one goal PER fragment, then assemble by Gold DAG.

    Calls can run concurrently, but assembly is deterministic and cannot
    change fragment order, wave/parallelism, dependencies or tool choices.
    One malformed fragment is locally retried; another's successful goal
    is not sent back into the failing fragment's prompt.
    """
    if max_workers < 1:
        raise ValueError("blueprint_workers_must_be_positive")
    fragments = outline["tasks"]
    logs = [{
        "fragment_id": f["fragment_id"], "calls": 0, "issues": [],
    } for f in fragments]
    def write_one(pair):
        index, fragment = pair
        return naturalize_fragment_goal(
            backend, outline, fragment, reference,
            payload=payload, oracle_answer=oracle_answer,
            call_log=logs[index],
        )
    try:
        with ThreadPoolExecutor(
            max_workers=min(max_workers, max(1, len(fragments))),
            thread_name_prefix="blueprint-goal",
        ) as pool:
            # map() returns in canonical Fragment order regardless of which
            # request completes first. No model-generated Fragment IDs.
            goals = list(pool.map(write_one, enumerate(fragments)))
    finally:
        if audit is not None:
            audit["author_calls"] = sum(x["calls"] for x in logs)
            audit["fragments"] = logs
    return {"version": VERSION, "outline": outline, "goals": goals}


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
    "naturalize_fragment_goal", "naturalize_delegation_blueprint",
    "validate_delegation_blueprint",
    "goal_for_fragment",
]
