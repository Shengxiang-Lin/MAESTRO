"""Compose dependency-closed, monotone KQA ProgramIR fragments.

Only genuine FIND/FIND_ALL roots are replaceable. Every selected fragment keeps
the COMPLETE ancestor closure of its output; no dependency is silently dropped
to satisfy a depth limit. A chained fragment replaces its own root with the
producer's output instead of introducing an unrelated FIND root.

Prepared producer pools support chains and genuine fork/join branches.
Terminal dependency closures are retained rather than reconstructed from arity.
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


def _terminal(nodes, template, rng):
    """Preserve the complete terminal closure, including QUERY_ATTR -> VERIFY."""
    roots = [n for n in template.nodes if not n.dependencies]
    if not roots or any(n.op not in _REPLACEABLE_ROOTS for n in roots):
        raise ValueError('terminal has no replaceable entity roots')
    remap = {root.node_id: nodes[-1].node_id for root in roots}
    result = list(nodes)
    for node_id in template.topological_order():
        if node_id in remap:
            continue
        node = template.node_map()[node_id]
        remap[node_id] = len(result)
        result.append(ProgramNode(len(result), node.op,
            tuple(remap[d] for d in node.dependencies), node.inputs, node.raw_function))
    return result


def prepare_composition_pools(programs, *, min_depth=2, max_depth=4):
    """Extract and deduplicate usable fragments before sampling compositions."""
    fragments, terminals = {}, {}
    for program in programs:
        node_map = program.node_map()
        if any(n.op not in _MONOTONE_ENTITY_OPS | _TERMINALS for n in program.nodes):
            continue
        order = program.topological_order()
        for node_id in order:
            if node_map[node_id].op not in _MONOTONE_ENTITY_OPS - _REPLACEABLE_ROOTS:
                continue
            closure = _ancestor_closure(node_map, node_id)
            chosen = [node_map[i] for i in order if i in closure]
            roots = [n for n in chosen if not n.dependencies]
            if len(roots) != 1 or roots[0].op not in _REPLACEABLE_ROOTS:
                continue
            if any(n.op not in _MONOTONE_ENTITY_OPS for n in chosen):
                continue
            fragment = ProgramIR(_renumber(chosen))
            if min_depth <= fragment.metrics()['depth'] <= max_depth:
                fragments.setdefault(fragment.canonical_signature(), fragment)
        if order and node_map[order[-1]].op in _TERMINALS:
            closure = _ancestor_closure(node_map, order[-1])
            chosen = [node_map[i] for i in order if i in closure]
            roots = [n for n in chosen if not n.dependencies]
            if roots and all(n.op in _REPLACEABLE_ROOTS for n in roots):
                terminal = ProgramIR(_renumber(chosen))
                terminals.setdefault(terminal.canonical_signature(), terminal)
    return tuple(fragments.values()), tuple(terminals.values())


def compose_deep_program(
    fragments, *, min_fragment_depth=2, max_fragment_depth=4,
    min_orchestration_depth=3, max_orchestration_depth=5,
    min_parallel_width=2, min_latent_anchors=2, seed=17,
    prepared_pools=None, motif=None,
):
    if min_orchestration_depth < 3 or max_orchestration_depth < min_orchestration_depth:
        raise ValueError('orchestration depth must satisfy 3 <= min <= max')
    if min_latent_anchors < 1 or min_latent_anchors >= max_orchestration_depth:
        raise ValueError('latent anchors must fit within orchestration depth')
    if min_parallel_width < 1:
        raise ValueError('parallel width must be positive')
    producers, terminals = prepared_pools or prepare_composition_pools(
        fragments, min_depth=min_fragment_depth, max_depth=max_fragment_depth)
    if not producers or not terminals:
        raise ValueError('no supported producer fragments or terminal programs')
    rng = random.Random(seed)
    motif = motif or ('chain' if min_parallel_width == 1 else rng.choice(('chain','fork_join','mixed_dag')))
    depth = rng.randint(max(min_orchestration_depth, min_latent_anchors + 1), max_orchestration_depth)
    terminal_pool = [p for p in terminals if sum(not n.dependencies for n in p.nodes)>=2] if motif=='mixed_dag' else terminals
    if not terminal_pool:
        motif, terminal_pool = 'fork_join', terminals
    terminal = rng.choice(terminal_pool)
    nodes, links, anchors = [], [], []

    def append_fragment(base, root=None, stage=0):
        fragment = rng.choice(producers)
        for _ in range(8):
            if fragment.canonical_signature() not in {x['source_signature'] for x in links}:
                break
            fragment = rng.choice(producers)
        result, remap = list(base), {}
        for node in fragment.nodes:
            if not node.dependencies and root is not None:
                remap[node.node_id] = root
                continue
            remap[node.node_id] = len(result)
            result.append(ProgramNode(len(result), node.op,
                tuple(remap[d] for d in node.dependencies), node.inputs, node.raw_function))
        output = remap[fragment.nodes[-1].node_id]
        links.append({'fragment':len(links), 'stage':stage,
            'producer_node_id':root, 'output_node_id':output,
            'source_signature':fragment.canonical_signature(), 'replaced_root':root is not None})
        return result, output

    current = None
    for stage in range(depth - 1):
        if motif != 'chain' and stage == depth - 2:
            outputs = []
            for _ in range(max(2, min_parallel_width)):
                nodes, output = append_fragment(nodes, root=current, stage=stage)
                outputs.append(output)
            nodes.append(ProgramNode(len(nodes), 'INTERSECT', tuple(outputs)))
            current = nodes[-1].node_id
        else:
            nodes, current = append_fragment(nodes, root=current, stage=stage)
        anchors.append(current)
    roots = [n for n in terminal.nodes if not n.dependencies]
    remap = {roots[0].node_id:current}
    for root in roots[1:]:
        if motif == 'mixed_dag':
            nodes, output = append_fragment(nodes, root=anchors[0], stage=depth - 2)
            remap[root.node_id] = output
        else:
            remap[root.node_id] = len(nodes)
            nodes.append(ProgramNode(len(nodes), root.op, (), root.inputs, root.raw_function))
    for node_id in terminal.topological_order():
        if node_id in remap:
            continue
        node = terminal.node_map()[node_id]
        remap[node_id] = len(nodes)
        nodes.append(ProgramNode(len(nodes), node.op,
            tuple(remap[d] for d in node.dependencies), node.inputs, node.raw_function))
    metadata = {'source':'kqa_pro_composition','deep_dependency':{
        'composition_links':links, 'fragment_count':len(links),
        'orchestration_depth':depth, 'orchestration_motif':motif,
        'parallel_width':1 if motif == 'chain' else max(2,min_parallel_width),
        'latent_anchor_node_ids':anchors, 'seed':seed}}
    return ProgramIR(nodes, metadata=metadata)


def compose_candidate_stream(programs, *, count, seed, **kwargs):
    rng = random.Random(seed)
    pools = prepare_composition_pools(programs,
        min_depth=kwargs.get('min_fragment_depth',2), max_depth=kwargs.get('max_fragment_depth',4))
    for _ in range(max(0,int(count))):
        yield compose_deep_program(programs, prepared_pools=pools, seed=rng.randrange(1<<31), **kwargs)


__all__ = ['compose_candidate_stream','compose_deep_program','prepare_composition_pools']
