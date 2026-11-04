"""Compile an official solution graph from the already-frozen MAESTRO world.

No LLM invents the solution or its dependencies. ProgramIR supplies symbolic
ground truth; the frozen capability graph supplies acquisition dependencies;
rendered sources supply the only agent-visible observations.
"""
from __future__ import annotations

from typing import Any, Mapping

from environment.task_construction.executor import SymbolicExecutor
from environment.task_construction.rendering.pipeline import program_from_dict, world_from_dict
from environment.task_construction.validation import minimal_source_proof


class GoldCompilationError(ValueError):
    pass


def json_value(value: Any) -> Any:
    if isinstance(value, set):
        return [json_value(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    return value


def _source_closure(ids: set[str], dependencies: Mapping[str, set[str]]) -> set[str]:
    result = set(ids)
    while True:
        parents = {p for sid in result for p in dependencies[sid]}
        new = result | parents
        if new == result:
            return result
        result = new


def ready_subtasks(plan: Mapping[str, Any], completed: set[str]) -> list[dict[str, Any]]:
    """All independent tasks currently executable in the official solution."""
    return [
        task for task in plan["tasks"]
        if task["subtask_id"] not in completed
        and set(task["depends_on"]).issubset(completed)
    ]


def compile_gold_solution(
    realized: Mapping[str, Any], world_record: Mapping[str, Any],
    rendered: Any, *, max_proof_sources: int = 8,
) -> dict[str, Any]:
    task_id = str(realized.get("task_id", ""))
    if not task_id or task_id != rendered.task_id:
        raise GoldCompilationError("realized task ID does not match frozen world")
    program = program_from_dict(world_record["program"])
    world = world_from_dict(world_record["world"])
    trace = SymbolicExecutor().execute_trace(program, world)
    order = program.topological_order()
    raw_answer = json_value(trace[order[-1]])
    proof = minimal_source_proof(program, world, max_sources=max_proof_sources)
    witnesses = proof.get("witnesses") or []
    if not witnesses:
        raise GoldCompilationError("no executable minimal source proof for this task")

    capability = realized.get("capability_world", {})
    resources = capability.get("resources", ()) if isinstance(capability, Mapping) else ()
    by_id = {str(r.get("resource_id")): r for r in resources
             if isinstance(r, Mapping) and r.get("resource_id")}
    if len(by_id) != len(resources):
        raise GoldCompilationError("invalid or duplicate frozen capability resources")
    missing = set(world.sources) - set(by_id)
    if missing:
        raise GoldCompilationError(f"capability graph missing world resources: {sorted(missing)}")
    dep = {
        sid: set(map(str, by_id[sid].get("anchor_dependency_resource_ids", ()) or ()))
        if str(by_id[sid].get("visibility", "")) not in {"attached", "provided", "initial"}
        else set()
        for sid in world.sources
    }
    for sid, parents in dep.items():
        if not parents.issubset(by_id) or sid in parents:
            raise GoldCompilationError(f"invalid acquisition parents for {sid}")
    options = []
    for witness in witnesses:
        proof_ids = set(map(str, witness))
        if not proof_ids or not proof_ids.issubset(world.sources):
            continue
        closure = _source_closure(proof_ids, dep)
        options.append((len(closure), sorted(closure), sorted(proof_ids)))
    if not options:
        raise GoldCompilationError("minimal source proof references unavailable sources")
    _, selected, proof_ids = min(options)
    selected_ids = set(selected)

    # Check an acyclic acquisition graph and deterministic parallel wave order.
    done: set[str] = set()
    waves: list[list[str]] = []
    while len(done) < len(selected):
        ready = sorted(sid for sid in selected if sid not in done
                       and (dep[sid] & selected_ids).issubset(done))
        if not ready:
            raise GoldCompilationError("gold resource dependency graph is cyclic")
        waves.append(["subtask:" + sid for sid in ready])
        done.update(ready)

    rendered_sources = {source.source_id: source for source in rendered.sources}
    if not selected_ids.issubset(rendered_sources):
        raise GoldCompilationError("gold resources missing rendered observations")
    fact_map = {fact.fact_id: fact for fact in world.facts}
    cap_by_id = {str(cap.get("capability_id")): cap
                 for cap in capability.get("capabilities", ())
                 if isinstance(cap, Mapping)}
    tasks: list[dict[str, Any]] = []
    for sid in selected:
        src = world.sources[sid]
        resource = by_id[sid]
        parents = sorted(dep[sid] & selected_ids)
        cap_ids = list(resource.get("required_capability_ids") or [])
        specialist = (str(cap_by_id[cap_ids[0]].get("specialist", "")) if cap_ids
                      and cap_ids[0] in cap_by_id else "")
        if not specialist:
            specialist = str(resource.get("source_type", "evidence")) + "_specialist"
        facts = []
        for fid in src.fact_ids:
            if fid not in fact_map:
                raise GoldCompilationError(f"unknown frozen fact {fid} in {sid}")
            fact = fact_map[fid]
            facts.append({
                "fact_id": fact.fact_id,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": json_value(fact.object),
                "object_kind": fact.object_kind,
            })
        tasks.append({
            "subtask_id": "subtask:" + sid,
            "resource_id": sid,
            "source_type": str(resource.get("source_type", src.source_type)),
            "visibility": str(resource.get("visibility", "")),
            "role": str(resource.get("role", "")),
            "agent": specialist,
            "depends_on": ["subtask:" + p for p in parents],
            "required_for_proof": sid in proof_ids,
            "gold_answer_claims": facts,
            "observation": {
                "resource_id": sid,
                "source_type": rendered_sources[sid].source_type,
                "title": rendered_sources[sid].title,
                "content": json_value(rendered_sources[sid].content),
            },
        })

    return {
        "task_id": task_id,
        "question": str(realized.get("question", "")),
        "final_answer": json_value(rendered.answer),
        "symbolic_answer": raw_answer,
        "symbolic_trace": {str(k): json_value(v) for k, v in trace.items()},
        "reasoning_graph": [
            {"node_id": node_id, "op": program.node_map()[node_id].op,
             "dependencies": list(program.node_map()[node_id].dependencies)}
            for node_id in order
        ],
        "proof_source_ids": proof_ids,
        "required_resource_ids": selected,
        "tasks": tasks,
        "gold_parallel_waves": waves,
        "policy": "frozen_symbolic_proof_plus_dependency_closed_acquisition",
    }


def public_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Teacher Main sees architecture, not future observations or gold answers."""
    return {
        "task_id": plan["task_id"],
        "question": plan["question"],
        "reasoning_graph": plan["reasoning_graph"],
        "tasks": [
            {key: task[key] for key in (
                "subtask_id", "resource_id", "source_type", "visibility", "role",
                "agent", "depends_on",
            )}
            for task in plan["tasks"]
        ],
    }


__all__ = [
    "GoldCompilationError", "compile_gold_solution", "public_plan",
    "ready_subtasks", "json_value",
]
