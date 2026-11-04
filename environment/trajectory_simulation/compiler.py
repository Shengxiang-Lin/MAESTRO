"""Compile first-stage composed ProgramIR fragments into executable gold subtasks.

Gold subtask boundaries are composition_links output nodes, NOT source IDs.
The executor supplies exact intermediate answers; sources are selected only
to ground those answers. Neither model is permitted to invent new subgoals.
"""
from __future__ import annotations

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


def _public_inputs(inputs: tuple[str, ...]) -> list[str]:
    """Never reveal synthetic internal entity IDs as future Main instructions."""
    visible = []
    for token in inputs:
        token = str(token)
        if token.startswith("entity:"):
            visible.append("entity identified by the question or predecessor clues")
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


def _minimal_witness(
    subprogram: ProgramIR, world: Any, expected: Any, dependencies: Mapping[str, Any],
    *, max_sources: int, required_fact_ids: tuple[str, ...] = (),
) -> list[str]:
    """Brute-force the small frozen source set; never trust a model's evidence."""
    all_ids = sorted(world.sources)
    if len(all_ids) > max_sources:
        raise GoldCompilationError("world exceeds exact fragment witness budget")
    for size in range(len(all_ids) + 1):
        for subset in combinations(all_ids, size):
            if not reachable_subset(subset, dependencies):
                continue
            scoped_world = _evidence_world(world, set(subset))
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
        })
    return rows


def ready_fragments(plan: Mapping[str, Any], completed: set[str]) -> list[dict[str, Any]]:
    return [
        task for task in plan["fragments"]
        if task["fragment_id"] not in completed
        and set(task["depends_on"]).issubset(completed)
    ]


def compile_gold_solution(
    realized: Mapping[str, Any], world_record: Mapping[str, Any],
    rendered: Any, *, max_sources: int = 8,
) -> dict[str, Any]:
    task_id = str(realized.get("task_id", ""))
    if not task_id or task_id != rendered.task_id:
        raise GoldCompilationError("question/world task_id mismatch")
    program = program_from_dict(world_record["program"])
    world = world_from_dict(world_record["world"])
    node_map = program.node_map()
    trace = SymbolicExecutor().execute_trace(program, world)
    order = program.topological_order()
    links = (program.metadata.get("deep_dependency", {}) or {}).get("composition_links")
    if not isinstance(links, list) or not links:
        raise GoldCompilationError("missing first-stage composition_links; no inferred fallback")
    output_ids = [int(link["output_node_id"]) for link in links]
    if len(set(output_ids)) != len(output_ids) or any(n not in node_map for n in output_ids):
        raise GoldCompilationError("invalid/duplicate composition output nodes")
    resources = (realized.get("capability_world") or {}).get("resources", [])
    if not resources or {str(x["resource_id"]) for x in resources} != set(world.sources):
        raise GoldCompilationError("capability resources do not match frozen world")
    audit = (realized.get("capability_world") or {}).get("binding_audit", {})
    discovery = (audit.get("progressive_discovery", {}) or
                 audit.get("p1_orchestration_profile", {}).get("progressive_discovery", {}))
    dependencies, _ = resolve_dependencies(resources, discovery.get("anchor_registry", {}))

    fragments = []
    output_to_fragment: dict[int, str] = {}
    support_by_fragment: dict[str, set[str]] = {}
    support_sources: set[str] = set()
    for index, (link, output) in enumerate(zip(links, output_ids)):
        identifier = f"fragment:{index}"
        subtree = _subprogram(program, output)
        witness = _minimal_witness(subtree, world, trace[output], dependencies,
                                   max_sources=max_sources)
        support_sources.update(witness)
        prev = _maximal_dependencies(program, output, output_to_fragment)
        known_sources = set().union(*(support_by_fragment[p] for p in prev)) if prev else set()
        new_sources = sorted(set(witness) - known_sources)
        owned = _ancestors(program, output)
        for predecessor in output_to_fragment:
            owned -= _ancestors(program, predecessor)
        operations = [
            {"node_id": nid, "op": node_map[nid].op,
             "inputs": _public_inputs(node_map[nid].inputs)}
            for nid in order if nid in owned
        ]
        fragment = {
            "fragment_id": identifier, "stage": int(link.get("stage", index)),
            "output_node_id": output, "depends_on": prev,
            "operations": operations,
            "resource_ids": new_sources,
            "support_resource_ids": witness,
            "observations": _source_claims(world, rendered, new_sources),
            "expected_answer": _public_answer(trace[output], rendered, node_map[output]),
            "answer_kind": "entity_set" if isinstance(trace[output], set) else "value",
        }
        fragments.append(fragment)
        output_to_fragment[output] = identifier
        support_by_fragment[identifier] = set(witness)

    final_id = order[-1]
    final_subprogram = _subprogram(program, final_id)
    final_info = assess_goal_information_completeness(program, world)
    if not final_info.accepted:
        raise GoldCompilationError("frozen world lacks decision-complete information")
    final_proof_sources = _minimal_witness(
        final_subprogram, world, trace[final_id], dependencies,
        max_sources=max_sources, required_fact_ids=final_info.required_fact_ids,
    )
    current_world = _evidence_world(world, support_sources)
    current_info = assess_goal_information_completeness(program, current_world)
    current_answer = None
    try:
        current_answer = SymbolicExecutor().execute(program, current_world)
    except (ValueError, TypeError, RuntimeError):
        pass
    sufficient = (
        current_info.accepted
        and set(final_info.required_fact_ids).issubset(
            {f.fact_id for f in current_world.facts})
        and current_answer is not None
        and _same(current_answer, trace[final_id])
    )
    if not sufficient:
        # Some terminals need additional source observations beyond all
        # producer fragments (e.g. QUERY_ATTR or SELECT_BETWEEN).
        fragments.append({
            "fragment_id": "fragment:terminal",
            "stage": max(x["stage"] for x in fragments) + 1,
            "output_node_id": final_id,
            "depends_on": _maximal_dependencies(program, final_id, output_to_fragment),
            "operations": [{"node_id": final_id, "op": node_map[final_id].op,
                            "inputs": _public_inputs(node_map[final_id].inputs)}],
            "resource_ids": sorted(set(final_proof_sources) - support_sources),
            "support_resource_ids": final_proof_sources,
            "observations": _source_claims(
                world, rendered, sorted(set(final_proof_sources) - support_sources)
            ),
            "expected_answer": json_value(rendered.answer),
            "answer_kind": "terminal_evidence",
        })
        support_sources.update(final_proof_sources)

    if not fragments:
        raise GoldCompilationError("no composed gold fragments")
    waves = []
    completed: set[str] = set()
    while len(completed) < len(fragments):
        ready = ready_fragments({"fragments": fragments}, completed)
        if not ready:
            raise GoldCompilationError("cyclic or unresolved fragment dependencies")
        wave = [item["fragment_id"] for item in ready]
        waves.append(wave)
        completed.update(wave)

    final_op = node_map[final_id].op
    return {
        "task_id": task_id,
        "question": str(realized.get("question", "")),
        "fragments": fragments,
        "waves": waves,
        "final_answer": json_value(rendered.answer),
        "answer_contract": {
            "operation": final_op,
            "collection_mode": (
                "set" if final_op in {"QUERY_NAME", "QUERY_RELATION", "SELECT_BETWEEN", "SELECT_AMONG"}
                else "multiset" if isinstance(trace[final_id], (list, tuple, set))
                else "scalar"
            ),
        },
        "terminal": {"op": final_op, "node_id": final_id,
                     "depends_on": _maximal_dependencies(program, final_id, output_to_fragment)},
        "symbolic_trace": {str(k): json_value(v) for k, v in trace.items()},
        "final_proof_source_ids": final_proof_sources,
        "policy": "first_stage_composition_links_fragment_gold",
    }


def private_guidance(plan: Mapping[str, Any], completed: set[str]) -> dict[str, Any]:
    """For teacher-only supervision. Never include in public O1 state."""
    ready = ready_fragments(plan, completed)
    return {
        "goal": "Follow the frozen composed solution, one parallel wave at a time",
        "fragment_graph": [{
            "fragment_id": f["fragment_id"], "depends_on": f["depends_on"],
            "stage": f["stage"],
            "operations": [{"op": op["op"], "inputs": op["inputs"]}
                           for op in f["operations"]],
        } for f in plan["fragments"]],
        "completed_fragments": sorted(completed),
        "ready": [{
            "fragment_id": f["fragment_id"], "depends_on": f["depends_on"],
            "operations": [{"op": op["op"], "inputs": op["inputs"]}
                           for op in f["operations"]],
            "required_resources": [{
                "resource_id": obs["resource_id"],
                "source_type": obs["source_type"],
                # An undiscovered source's title is private evidence, not
                # an O1-visible clue for a future Main instruction.
            } for obs in f["observations"]],
            "suggested_tools": sorted({
                "GoogleSearchAction" if obs["source_type"] == "search" else
                "ImageAnalysisAction" if obs["source_type"].startswith("image") else
                "AudioAnalysisAction" if obs["source_type"].startswith("audio") else
                "VideoAnalysisAction" if obs["source_type"].startswith("video") else
                "ExecuteCodeAction" if obs["source_type"].startswith(("table", "database")) else
                "ExtractUrlContentAction"
                for obs in f["observations"]
            }) or ["ExecuteCodeAction"],
        } for f in ready],
        "remaining_fragment_ids": [
            f["fragment_id"] for f in plan["fragments"] if f["fragment_id"] not in completed
        ],
        "terminal_operation": plan["terminal"]["op"],
        "may_complete": len(completed) == len(plan["fragments"]),
    }


__all__ = [
    "GoldCompilationError", "compile_gold_solution", "private_guidance",
    "ready_fragments", "json_value",
]
