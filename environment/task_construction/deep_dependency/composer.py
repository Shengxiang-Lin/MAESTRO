"""Compose dependency-closed, monotone KQA ProgramIR fragments.

Only genuine FIND/FIND_ALL roots are replaceable. Every selected fragment keeps
the COMPLETE ancestor closure of its output; no dependency is silently dropped
to satisfy a depth limit. A chained fragment replaces its own root with the
producer's output instead of introducing an unrelated FIND root.

The production orchestra-o3 anchor-fragment composer is still richer (motif
quotas and fork/join topology). This module safeguards the current MAESTRO
implementation; it is not a claim of full upstream algorithm equivalence.
"""
from __future__ import annotations

import random
from typing import Iterable, Sequence

from ..ir import ProgramIR, ProgramNode

_TERMINALS = {
    "COUNT", "QUERY_NAME", "QUERY_ATTR", "QUERY_RELATION",
    "SELECT_BETWEEN", "SELECT_AMONG",
    "VERIFY_STRING", "VERIFY_NUMBER", "VERIFY_YEAR", "VERIFY_DATE",
}
_REPLACEABLE_ROOTS = {"FIND", "FIND_ALL"}
# SELECT_* can change its chosen winner when a branch is extended. It must
# never be a reused producer/latent anchor.
_MONOTONE_ENTITY_OPS = {
    "FIND", "FIND_ALL", "RELATE",
    "FILTER_TYPE", "FILTER_STRING", "FILTER_NUMBER",
    "FILTER_YEAR", "FILTER_DATE", "INTERSECT", "UNION",
}
_ENTITY_OPS = _MONOTONE_ENTITY_OPS | {"SELECT_BETWEEN", "SELECT_AMONG"}


def _renumber(nodes: Sequence[ProgramNode]) -> list[ProgramNode]:
    """Renumber topologically ordered nodes while preserving dependency edges."""
    remap = {node.node_id: index for index, node in enumerate(nodes)}
    return [
        ProgramNode(
            remap[node.node_id], node.op,
            tuple(remap[dep] for dep in node.dependencies),
            node.inputs, node.raw_function,
        )
        for node in nodes
    ]


def _ancestor_closure(nodes: dict[int, ProgramNode], endpoint: int) -> set[int]:
    closure: set[int] = set()
    pending = [endpoint]
    while pending:
        current = pending.pop()
        if current in closure:
            continue
        closure.add(current)
        pending.extend(nodes[current].dependencies)
    return closure


def _fragment(
    program: ProgramIR, *, min_depth: int, max_depth: int, rng: random.Random
) -> list[ProgramNode]:
    """Extract a complete single-root producer fragment.

    Depth bounds apply to longest dependency path, not a prefix/truncated
    number of ancestors. A multi-root or non-monotone candidate is rejected
    rather than being rewritten into a syntactically valid but wrong graph.
    """
    if max_depth < min_depth or min_depth < 2:
        raise ValueError("invalid fragment depth bounds")
    order = program.topological_order()
    nodes = program.node_map()
    depths: dict[int, int] = {}
    eligible: list[tuple[int, set[int]]] = []
    for node_id in order:
        node = nodes[node_id]
        depths[node_id] = (
            1 if not node.dependencies
            else 1 + max(depths[dep] for dep in node.dependencies)
        )
        if node.op not in _MONOTONE_ENTITY_OPS or node.op in _REPLACEABLE_ROOTS:
            continue
        if not (min_depth <= depths[node_id] <= max_depth):
            continue
        ancestors = _ancestor_closure(nodes, node_id)
        if any(nodes[ancestor].op not in _MONOTONE_ENTITY_OPS for ancestor in ancestors):
            continue
        roots = [
            ancestor for ancestor in ancestors
            if not nodes[ancestor].dependencies
        ]
        if len(roots) != 1 or nodes[roots[0]].op not in _REPLACEABLE_ROOTS:
            continue
        eligible.append((node_id, ancestors))
    if not eligible:
        return []

    _, selected = rng.choice(eligible)
    chosen = [nodes[node_id] for node_id in order if node_id in selected]
    # The closure must be internally valid before IDs are remapped.
    if any(dep not in selected for node in chosen for dep in node.dependencies):
        raise ValueError("fragment dependency closure failed")
    return _renumber(chosen)


def _tail(nodes: Sequence[ProgramNode]) -> int | None:
    return nodes[-1].node_id if nodes else None


def _attach_chain(
    base: Sequence[ProgramNode], fragment: Sequence[ProgramNode]
) -> list[ProgramNode]:
    """Substitute a fragment's FIND root with the previous producer output."""
    if not fragment:
        raise ValueError("no dependency-closed producer fragment available")
    roots = [node for node in fragment if not node.dependencies]
    if len(roots) != 1 or roots[0].op not in _REPLACEABLE_ROOTS:
        raise ValueError("producer fragment must have one FIND/FIND_ALL root")
    if any(node.op not in _MONOTONE_ENTITY_OPS for node in fragment):
        raise ValueError("producer fragment contains non-monotone operator")
    if not base:
        return list(fragment)

    producer_output = _tail(base)
    if producer_output is None or base[-1].op not in _MONOTONE_ENTITY_OPS:
        raise ValueError("chain producer output must be a monotone entity node")
    root_id = roots[0].node_id
    # The fragment's public root becomes an *existing* latent producer output.
    appended_ids = [n.node_id for n in fragment if n.node_id != root_id]
    if not appended_ids:
        raise ValueError("fragment has no producer operations after its root")
    remap = {root_id: producer_output}
    remap.update({old: len(base) + index for index, old in enumerate(appended_ids)})
    appended = []
    for node in fragment:
        if node.node_id == root_id:
            continue
        appended.append(ProgramNode(
            node_id=remap[node.node_id],
            op=node.op,
            dependencies=tuple(remap[dep] for dep in node.dependencies),
            inputs=node.inputs,
            raw_function=node.raw_function,
        ))
    return [*base, *appended]


def _add_branch_join(nodes: list[ProgramNode], rng: random.Random) -> list[ProgramNode]:
    entity_ids = [node.node_id for node in nodes if node.op in _MONOTONE_ENTITY_OPS]
    if len(entity_ids) < 2:
        return nodes
    left = entity_ids[-1]
    right = rng.choice(entity_ids[:-1])
    op = rng.choice(("INTERSECT", "UNION"))
    return [*nodes, ProgramNode(len(nodes), op, (left, right), (), op)]


def _terminal(nodes: list[ProgramNode], template: ProgramIR, rng: random.Random) -> list[ProgramNode]:
    terminals = [
        template.node_map()[node_id]
        for node_id in template.topological_order()
        if template.node_map()[node_id].op in _TERMINALS
    ]
    terminal = rng.choice(terminals) if terminals else ProgramNode(-1, "QUERY_NAME", (), (), "QueryName")
    entity_ids = [node.node_id for node in nodes if node.op in _ENTITY_OPS]
    if not entity_ids:
        return nodes
    if terminal.op in {"QUERY_RELATION", "SELECT_BETWEEN"} and len(entity_ids) >= 2:
        deps = (entity_ids[-2], entity_ids[-1])
    else:
        deps = (entity_ids[-1],)
    return [*nodes, ProgramNode(len(nodes), terminal.op, deps, terminal.inputs, terminal.raw_function)]


def compose_deep_program(
    fragments: Sequence[ProgramIR], *,
    min_fragment_depth: int = 2, max_fragment_depth: int = 4,
    min_orchestration_depth: int = 3, max_orchestration_depth: int = 5,
    min_parallel_width: int = 2, seed: int = 17
) -> ProgramIR:
    if not fragments:
        raise ValueError("at least one ProgramIR fragment is required")
    rng = random.Random(seed)
    fragment_count = max(
        2, min(len(fragments), rng.randint(min_orchestration_depth, max_orchestration_depth))
    )
    chosen = (
        rng.sample(list(fragments), fragment_count)
        if len(fragments) >= fragment_count
        else [rng.choice(fragments) for _ in range(fragment_count)]
    )
    nodes: list[ProgramNode] = []
    links = []
    for index, program in enumerate(chosen):
        fragment = _fragment(
            program, min_depth=min_fragment_depth, max_depth=max_fragment_depth, rng=rng
        )
        if not fragment:
            raise ValueError("source has no eligible single-root monotone producer fragment")
        before = len(nodes)
        producer = _tail(nodes)
        nodes = _attach_chain(nodes, fragment)
        links.append({
            "fragment": index, "start": before, "end": len(nodes)-1,
            "producer_node_id": producer,
            "replaced_root": index > 0,
        })
    if min_parallel_width >= 2:
        nodes = _add_branch_join(nodes, rng)
    nodes = _terminal(nodes, chosen[-1], rng)
    nodes = _renumber(nodes)
    metadata = {
        "source": "kqa_pro_composition",
        "deep_dependency": {
            "composition_links": links,
            "fragment_count": fragment_count,
            "min_fragment_depth": min_fragment_depth,
            "max_fragment_depth": max_fragment_depth,
            "orchestration_depth": fragment_count,
            "parallel_width": min_parallel_width,
            "seed": seed,
        },
    }
    return ProgramIR(nodes, metadata=metadata)


def compose_candidate_stream(
    programs: Sequence[ProgramIR], *, count: int, seed: int, **kwargs
) -> Iterable[ProgramIR]:
    rng = random.Random(seed)
    for _ in range(max(0, int(count))):
        yield compose_deep_program(programs, seed=rng.randrange(1 << 31), **kwargs)


__all__ = ["compose_candidate_stream", "compose_deep_program"]
