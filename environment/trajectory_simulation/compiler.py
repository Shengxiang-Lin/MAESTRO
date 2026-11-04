"""Compile an official solution graph from the already-frozen MAESTRO world.

No LLM invents the solution or its dependencies. ProgramIR supplies symbolic
ground truth; the frozen capability graph supplies acquisition dependencies;
rendered sources supply the only agent-visible observations.
"""
from __future__ import annotations

from typing import Any, Mapping

from environment.task_construction.executor import SymbolicExecutor
from environment.task_construction.rendering.pipeline import program_from_dict, world_from_dict
from environment.task_construction.rendering.naming import predicate_label
from .acquisition import INITIAL,resolve_dependencies,choose_evidence


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


def ready_subtasks(plan: Mapping[str, Any], completed: set[str]) -> list[dict[str, Any]]:
    """All independent tasks currently executable in the official solution."""
    return [
        task for task in plan["tasks"]
        if task["subtask_id"] not in completed
        and set(task["depends_on"]).issubset(completed)
        and all(set(group) & completed for group in task.get("depends_on_any", ()))
    ]


def compile_gold_solution(
    realized: Mapping[str, Any], world_record: Mapping[str, Any],
    rendered: Any, *, max_proof_sources: int = 8, full_environment: bool = False,
) -> dict[str, Any]:
    task_id = str(realized.get("task_id", ""))
    if not task_id or task_id != rendered.task_id:
        raise GoldCompilationError("realized task ID does not match frozen world")
    program = program_from_dict(world_record["program"])
    world = world_from_dict(world_record["world"])
    trace = SymbolicExecutor().execute_trace(program, world)
    order = program.topological_order()
    raw_answer = json_value(trace[order[-1]])
    capability = realized.get("capability_world", {})
    resources = capability.get("resources", ()) if isinstance(capability, Mapping) else ()
    by_id = {str(r.get("resource_id")): r for r in resources
             if isinstance(r, Mapping) and r.get("resource_id")}
    if len(by_id) != len(resources):
        raise GoldCompilationError("invalid or duplicate frozen capability resources")
    missing = set(world.sources) - set(by_id)
    if missing:
        raise GoldCompilationError(f"capability graph missing world resources: {sorted(missing)}")
    audit=capability.get("binding_audit",{})
    discovery=audit.get("progressive_discovery",{}) or audit.get("p1_orchestration_profile",{}).get("progressive_discovery",{})
    registry=discovery.get("anchor_registry",{})
    try:
        dep,levels=resolve_dependencies(list(by_id.values()),registry)
        selected,required_facts=choose_evidence(program,world,dep,max_proof_sources)
    except ValueError as exc:
        raise GoldCompilationError(str(exc)) from exc
    proof_ids=list(selected)
    if full_environment:selected=sorted(world.sources)
    selected_ids=set(selected)
    rendered_sources = {source.source_id: source for source in rendered.sources}
    if not selected_ids.issubset(rendered_sources):
        raise GoldCompilationError("gold resources missing rendered observations")
    fact_map = {fact.fact_id: fact for fact in world.facts}
    # The Sub-Agent's answers use the same deterministic public surfaces as
    # the rendered evidence, never private synthetic entity/fact IDs.
    surface = getattr(rendered, "hidden_metadata", {}) or {}
    aliases = surface.get("entity_aliases", {})
    literal_scoped = surface.get("literal_surfaces_by_predicate", {})
    literal_fallback = surface.get("literal_surfaces", {})
    cap_by_id = {str(cap.get("capability_id")): cap
                 for cap in capability.get("capabilities", ())
                 if isinstance(cap, Mapping)}
    tasks: list[dict[str, Any]] = []
    for sid in selected:
        src = world.sources[sid]
        resource = by_id[sid]
        groups = [sorted(group & selected_ids) for group in dep[sid]]
        parents = sorted({group[0] for group in groups if len(group) == 1})
        alternatives = [["subtask:"+p for p in group] for group in groups if len(group)>1]
        cap_ids = list(resource.get("required_capability_ids") or [])
        specialist = (str(cap_by_id[cap_ids[0]].get("specialist", "")) if cap_ids
                      and cap_ids[0] in cap_by_id else "")
        if not specialist:
            raise GoldCompilationError(f"resource {sid} has no registered specialist")
        facts = []
        for fid in src.fact_ids:
            if fid not in fact_map:
                raise GoldCompilationError(f"unknown frozen fact {fid} in {sid}")
            fact = fact_map[fid]
            if fact.object_kind == "entity":
                display_object = aliases.get(str(fact.object), str(fact.object))
            else:
                display_object = literal_scoped.get(fact.predicate, {}).get(
                    str(fact.object),
                    literal_fallback.get(str(fact.object), fact.object),
                )
            facts.append({
                "subject": aliases.get(fact.subject, fact.subject),
                "predicate": predicate_label(fact.predicate),
                "object": json_value(display_object),
                "object_kind": fact.object_kind,
            })
        inspection_parents=["subtask:"+p for p in parents]
        if resource.get("visibility") not in INITIAL:
            descriptor={"resource_id":sid,"source_type":str(src.source_type),
                        "title":rendered_sources[sid].title}
            terms=[]
            for aid in resource.get("requires_anchors",()):
                raw=str(registry.get("anchors",{}).get(aid,{}).get("surface",aid))
                terms.append(aliases.get(raw,literal_fallback.get(raw,
                    predicate_label(raw) if raw.startswith(("schema:","yago:")) else raw)))
            tasks.append({"subtask_id":"discover:"+sid,"kind":"discover",
                "resource_id":sid,"source_type":str(src.source_type),
                "visibility":"discoverable","role":"resource_discovery","agent":"main",
                "available_tool_families":["search"],"depends_on":inspection_parents,
                "depends_on_any":alternatives,
                "discovery_terms":terms,"required_for_proof":False,
                "gold_answer_claims":[descriptor],"observation":descriptor})
            inspection_parents=["discover:"+sid]
        tasks.append({
            "subtask_id": "subtask:" + sid,
            "kind":"inspect",
            "resource_id": sid,
            "source_type": str(resource.get("source_type", src.source_type)),
            "visibility": str(resource.get("visibility", "")),
            "role": str(resource.get("role", "")),
            "agent": specialist,
            "available_tool_families": list(resource.get("available_tool_families") or []),
            "depends_on": inspection_parents,
            "depends_on_any": [],
            "required_for_proof": sid in proof_ids,
            "gold_answer_claims": facts,
            "observation": {
                "resource_id": sid,
                "source_type": rendered_sources[sid].source_type,
                "title": rendered_sources[sid].title,
                "content": json_value(rendered_sources[sid].content),
            },
        })

    done=set();waves=[]
    while len(done)<len(tasks):
        ready=sorted(t["subtask_id"] for t in ready_subtasks({"tasks":tasks},done))
        if not ready:raise GoldCompilationError("compiled acquisition graph is cyclic")
        waves.append(ready);done.update(ready)
    final_op=program.node_map()[order[-1]].op
    answer_contract={"operation":final_op,"collection_mode":
        "set" if final_op in {"QUERY_NAME","QUERY_RELATION","SELECT_BETWEEN","SELECT_AMONG"}
        else "multiset" if isinstance(raw_answer,list) else "scalar"}
    return {
        "task_id": task_id,
        "answer_contract":answer_contract,
        "required_fact_ids":required_facts,
        "acquisition_validation":{"anchor_semantics":"AND_over_anchors_OR_over_providers",
                                  "decision_information_complete":True},
        "acquisition_alternatives":{
            sid:[sorted(group) for group in groups] for sid,groups in sorted(dep.items())
        },
        "question": str(realized.get("question", "")),
        "public_task_design": dict(realized.get("task_design", {}) or {}),
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
        "policy": "decision_complete_evidence_and_anchor_aware_discovery",
    }


def public_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Teacher Main sees architecture, not future observations or gold answers."""
    return {
        "task_id": plan["task_id"],
        "question": plan["question"],
        "task_design": plan["public_task_design"],
        "answer_contract":plan["answer_contract"],
        "reasoning_graph": plan["reasoning_graph"],
        "tasks": [
            {key: task[key] for key in (
                "subtask_id", "kind", "resource_id", "source_type", "visibility", "role",
                "agent", "available_tool_families", "depends_on",
                "depends_on_any",
            )}
            for task in plan["tasks"]
        ],
    }


__all__ = [
    "GoldCompilationError", "compile_gold_solution", "public_plan",
    "ready_subtasks", "json_value",
]
