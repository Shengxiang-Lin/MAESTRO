"""Compile first-stage composed ProgramIR fragments into executable gold subtasks.

Gold subtask boundaries are composition_links output nodes, NOT source IDs.
The executor supplies exact intermediate answers; sources are selected only
to ground those answers. Neither model is permitted to invent new subgoals.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Any, Mapping

from environment.task_construction.executor import SymbolicExecutor
from environment.task_construction.scenario.information import assess_goal_information_completeness
from environment.task_construction.ir import ProgramIR
from environment.task_construction.rendering.naming import predicate_label, type_label
from environment.task_construction.rendering.pipeline import program_from_dict, world_from_dict
from environment.task_construction.validation import normalize_answer

from .acquisition import reachable_subset, resolve_dependencies


class GoldCompilationError(ValueError):
    pass


def json_value(value: Any) -> Any:
    if isinstance(value, set):
        return [json_value(x) for x in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [json_value(x) for x in value]
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    return value


def _public_inputs(inputs: tuple[str, ...], rendered: Any = None) -> list[str]:
    """Never reveal synthetic internal entity IDs as future Main instructions."""
    visible = []
    aliases = (getattr(rendered, "hidden_metadata", {}) or {}).get("entity_aliases", {})
    comparators = {
        "<CMP:=>": "=", "<CMP:==>": "==", "<CMP:!=>": "!=",
        "<CMP:>>": ">", "<CMP:greater>": ">",
        "<CMP:<>": "<", "<CMP:less>": "<",
        "<CMP:>=>": ">=", "<CMP:<=>": "<=",
    }
    for token in inputs:
        token = str(token)
        if token in comparators:
            visible.append(comparators[token])
        elif token.startswith("entity:"):
            visible.append(aliases.get(
                token.split(":", 1)[1],
                "entity identified by the question or predecessor clues"))
        elif token.startswith("type:"):
            visible.append(type_label(token.split(":", 1)[1]))
        elif token.startswith(("schema:", "yago:", "r:")):
            visible.append(predicate_label(token))
        elif token.startswith(("value_", "literal_")):
            visible.append("value specified by the public question")
        else:
            visible.append(token)
    return visible


def _ancestors(program: ProgramIR, output: int) -> set[int]:
    nodes = program.node_map()
    result: set[int] = set()
    def visit(node: int) -> None:
        if node in result:
            return
        result.add(node)
        for parent in nodes[node].dependencies:
            visit(parent)
    visit(output)
    return result


def _subprogram(program: ProgramIR, output: int) -> ProgramIR:
    needed = _ancestors(program, output)
    return ProgramIR(
        [n for n in program.nodes if n.node_id in needed],
        metadata={"gold_output_node_id": output},
    )


def _evidence_world(world: Any, source_ids: set[str]) -> Any:
    """Keep a fact if any retained source provides it (shared facts included)."""
    subset = world.copy()
    subset.sources = {sid: s for sid, s in world.sources.items() if sid in source_ids}
    fact_ids = {fid for source in subset.sources.values() for fid in source.fact_ids}
    subset.facts = [
        fact for fact in world.facts
        if fact.fact_id in fact_ids or fact.source_id in source_ids
    ]
    return subset


def _same(left: Any, right: Any) -> bool:
    return normalize_answer(json_value(left)) == normalize_answer(json_value(right))


def _read_fact_ids(program: ProgramIR, world: Any) -> set[str]:
    """Retain evaluated evidence, including values that reject a candidate."""
    trace = SymbolicExecutor().execute_trace(program, world)
    required = set(assess_goal_information_completeness(program, world).required_fact_ids)
    for node in program.nodes:
        if not node.dependencies:
            continue
        candidates = set()
        dependencies = node.dependencies[:2] if node.op in {
            "SELECT_BETWEEN", "QUERY_RELATION",
        } else node.dependencies[:1]
        for parent in dependencies:
            value = trace[parent]
            if isinstance(value, (set, list, tuple)):
                candidates.update(str(item) for item in value)
        for fact in world.facts:
            if node.op == "RELATE" and fact.predicate == node.inputs[0]:
                backward = "backward" in node.inputs
                endpoint = str(fact.object) if backward else fact.subject
                if fact.object_kind == "entity" and endpoint in candidates:
                    required.add(fact.fact_id)
            elif node.op == "QUERY_RELATION":
                left = {str(item) for item in trace[node.dependencies[0]]}
                right = {str(item) for item in trace[node.dependencies[1]]}
                if fact.object_kind == "entity" and fact.subject in left and str(fact.object) in right:
                    required.add(fact.fact_id)
            elif (node.op in {
                "FILTER_STRING", "FILTER_NUMBER", "FILTER_YEAR", "FILTER_DATE",
                "QUERY_ATTR", "SELECT_AMONG", "SELECT_BETWEEN",
            } and fact.subject in candidates and fact.predicate == node.inputs[0]):
                required.add(fact.fact_id)
    return required


def _minimal_witness(
    subprogram: ProgramIR, world: Any, expected: Any, dependencies: Mapping[str, Any],
    *, max_sources: int, required_fact_ids: tuple[str, ...] = (), required_entity_ids=(),
) -> list[str]:
    """Brute-force the small frozen source set; never trust a model's evidence."""
    all_ids = sorted(world.sources)
    required_fact_ids = set(required_fact_ids) | _read_fact_ids(subprogram, world)
    if len(all_ids) > max_sources:
        raise GoldCompilationError("world exceeds exact fragment witness budget")
    for size in range(len(all_ids) + 1):
        for subset in combinations(all_ids, size):
            if not reachable_subset(subset, dependencies):
                continue
            scoped_world = _evidence_world(world, set(subset))
            observed_entities={f.subject for f in scoped_world.facts}|{str(f.object) for f in scoped_world.facts if f.object_kind=="entity"}
            if not set(required_entity_ids).issubset(observed_entities):continue
            if not set(required_fact_ids).issubset({f.fact_id for f in scoped_world.facts}):
                continue
            try:
                answer = SymbolicExecutor().execute(subprogram, scoped_world)
                info = assess_goal_information_completeness(subprogram, scoped_world)
            except (ValueError, TypeError, RuntimeError):
                continue
            if info.accepted and _same(answer, expected):
                return list(subset)
    raise GoldCompilationError(
        f"no source-complete witness for fragment output node "
        f"{subprogram.topological_order()[-1]}"
    )


def _public_answer(value: Any, rendered: Any, node: Any) -> Any:
    hidden = getattr(rendered, "hidden_metadata", {}) or {}
    aliases = hidden.get("entity_aliases", {})
    literal = hidden.get("literal_surfaces", {})
    scoped = hidden.get("literal_surfaces_by_predicate", {})
    predicate = node.inputs[0] if node.op == "QUERY_ATTR" and node.inputs else None
    def visit(item: Any) -> Any:
        if isinstance(item, set):
            return sorted((visit(x) for x in item), key=str)
        if isinstance(item, (list, tuple)):
            return [visit(x) for x in item]
        if isinstance(item, Mapping):
            return {str(k): visit(v) for k, v in item.items()}
        if isinstance(item, str):
            if item in aliases:
                return aliases[item]
            if node.op == "QUERY_RELATION":
                return predicate_label(item)
        key = str(item)
        if predicate and key in scoped.get(predicate, {}):
            return scoped[predicate][key]
        return literal.get(key, item)
    return visit(value)


def _maximal_dependencies(program: ProgramIR, output: int, previous: Mapping[int, str]) -> list[str]:
    ancestor = _ancestors(program, output)
    candidates = {node for node in previous if node in ancestor and node != output}
    maximal = {node for node in candidates
               if not any(node in _ancestors(program, other) for other in candidates if other != node)}
    return [previous[node] for node in sorted(maximal)]


def _source_claims(world: Any, rendered: Any, source_ids: list[str]) -> list[dict[str, Any]]:
    hidden = getattr(rendered, "hidden_metadata", {}) or {}
    aliases = hidden.get("entity_aliases", {})
    literals = hidden.get("literal_surfaces", {})
    scoped = hidden.get("literal_surfaces_by_predicate", {})
    facts = {f.fact_id: f for f in world.facts}
    visible = {src.source_id: src for src in rendered.sources}
    rows = []
    for sid in source_ids:
        if sid not in visible:
            raise GoldCompilationError(f"rendered observation missing source {sid}")
        claims = []
        for fid in world.sources[sid].fact_ids:
            f = facts[fid]
            obj = (aliases.get(str(f.object), str(f.object)) if f.object_kind == "entity"
                   else scoped.get(f.predicate, {}).get(
                       str(f.object), literals.get(str(f.object), f.object)))
            claims.append({
                "observation_key": "claim:"+hashlib.sha256((rendered.task_id+":"+fid).encode()).hexdigest()[:24],
                "subject": aliases.get(f.subject, f.subject),
                "predicate": predicate_label(f.predicate),
                "object": json_value(obj),
                "object_kind": f.object_kind,
            })
        source = visible[sid]
        rows.append({
            "resource_id": sid, "source_type": source.source_type,
            "title": source.title, "content": json_value(source.content),
            "claims": claims,
            "entity_types": {aliases.get(e,e):type_label(world.entities[e]) for e in
                {f.subject for fid in world.sources[sid].fact_ids for f in [facts[fid]]} |
                {str(f.object) for fid in world.sources[sid].fact_ids for f in [facts[fid]] if f.object_kind=="entity"}
                if e in world.entities},
        })
    return rows


def ready_fragments(plan: Mapping[str, Any], completed: set[str]) -> list[dict[str, Any]]:
    return [
        task for task in plan["fragments"]
        if task["fragment_id"] not in completed
        and set(task["depends_on"]).issubset(completed)
    ]


def compile_gold_solution(realized, world_record, rendered, *, max_sources=8, scene="omnigaia", schedule="baseline"):
    from .skeleton import compile_fragments
    plan=compile_fragments(realized, world_record, rendered, max_sources=max_sources, scene=scene)
    if schedule=="compact":
        from .scheduling import compact_plan
        return compact_plan(plan)
    if schedule!="baseline":raise ValueError("unknown scheduling policy")
    plan["schedule"]="baseline"
    return plan


def private_guidance(plan: Mapping[str, Any], completed: set[str]) -> dict[str, Any]:
    """For teacher-only supervision. Never include in public O1 state."""
    ready = ready_fragments(plan, completed)
    return {
        "goal": "Follow the frozen composed solution, one parallel wave at a time",
        "fragment_graph": [{
            "fragment_id": f["fragment_id"], "depends_on": f["depends_on"],
            "stage": f["stage"],
            "operations": [{"node_id": op["node_id"], "op": op["op"],
                            "dependencies": op["dependencies"], "inputs": op["inputs"],
                             "input_bindings": op.get("input_bindings", []), "find_kind":op.get("find_kind","entity")}
                           for op in f["operations"]],
        } for f in plan["fragments"]],
        "completed_fragments": sorted(completed),
        "ready": [{
            "fragment_id": f["fragment_id"], "depends_on": f["depends_on"],
            "operations": [{"node_id": op["node_id"], "op": op["op"],
                            "dependencies": op["dependencies"], "inputs": op["inputs"],
                             "input_bindings": op.get("input_bindings", []), "find_kind":op.get("find_kind","entity")}
                           for op in f["operations"]],
            "required_resources": [{
                "resource_id": obs["resource_id"],
                "source_type": obs["source_type"],
                # An undiscovered source's title is private evidence, not
                # an O1-visible clue for a future Main instruction.
            } for obs in f["observations"]],
            "scene":plan["scene"],
            "suggested_tools": f["suggested_tools"],
            "known_resource_ids": [o["resource_id"] for o in f.get("known_observations",[])],
            "available_tools": plan["available_tools"],
            "input_bindings": [op.get("input_bindings", []) for op in f["operations"]],
        } for f in ready],
        "remaining_fragment_ids": [
            f["fragment_id"] for f in plan["fragments"] if f["fragment_id"] not in completed
        ],
        "terminal_operation": plan["terminal"]["op"],
        "may_complete": len(completed) == len(plan["fragments"]),
        "available_tools": plan["available_tools"],
    }


__all__ = [
    "GoldCompilationError", "compile_gold_solution", "private_guidance",
    "ready_fragments", "json_value",
]
