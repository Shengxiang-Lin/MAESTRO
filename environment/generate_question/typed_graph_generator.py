"""Programmatic, typed partial-order Macro DAG generation.

Sample *information-goal nodes and typed handoffs first*, with no world
entities, YAGO predicates, media, or question strings. Never add a bare
random edge: every edge is a named, consumed information port.
"""
from __future__ import annotations

from collections import Counter
import random

from .typed_role_grammar import new_node

VERSION = "maestro.typed_macro_dag.v1"


def bind(source: dict, port: str) -> dict:
    return {
        "from_macro": source["macro_id"],
        "from_port": port,
        "type": source["output_ports"][port],
    }


def waves(nodes: list[dict]) -> list[list[str]]:
    remaining = {n["macro_id"] for n in nodes}
    done: set[str] = set()
    result = []
    while remaining:
        ready = [n["macro_id"] for n in nodes
                 if n["macro_id"] in remaining
                 and set(n["depends_on"]) <= done]
        if not ready:
            raise ValueError("macro_dag_cycle")
        result.append(ready)
        done.update(ready)
        remaining.difference_update(ready)
    return result


def _attempt(rng: random.Random, *, max_branches: int) -> dict:
    nodes = []
    n_discovery = rng.randint(2, max_branches)
    discovery = []
    for i in range(n_discovery):
        inputs = {}
        # A later discovery can refine an earlier candidate set; it remains
        # its own investigation with two independently scoped sources.
        if i >= 2 and rng.random() < 0.4:
            prior = rng.choice(discovery)
            inputs["prior_candidates"] = bind(prior, "candidates")
        node = new_node(
            f"M{len(nodes)+1}", "discovery", "discover_candidates", inputs
        )
        nodes.append(node)
        discovery.append(node)
    consumed_by_discovery = {
        link["from_macro"]
        for node in discovery for link in node["input_ports"].values()
    }
    leaves = [node for node in discovery
              if node["macro_id"] not in consumed_by_discovery]
    assert len(leaves) >= 2
    identify = new_node(
        f"M{len(nodes)+1}", "identification", "identify_entity", {
            f"candidate_{i+1}": bind(node, "candidates")
            for i, node in enumerate(leaves)
        },
    )
    nodes.append(identify)

    n_investigation = rng.randint(1, 3)
    investigations = []
    verifications = []
    for i in range(n_investigation):
        inputs = {"subject": bind(identify, "entity")}
        # Conditional follow-up: a later investigation consumes the earlier
        # verification workproduct, not merely its scheduling order.
        if verifications and rng.random() < 0.55:
            inputs["verified_context"] = bind(
                rng.choice(verifications), "verified"
            )
        research = new_node(
            f"M{len(nodes)+1}", "investigation",
            "investigate_relation", inputs,
        )
        nodes.append(research)
        investigations.append(research)
        if rng.random() < 0.45 and len(verifications) < 2:
            verification = new_node(
                f"M{len(nodes)+1}", "verification", "verify_finding",
                {"finding": bind(research, "finding")},
            )
            nodes.append(verification)
            verifications.append(verification)
    if len(investigations) > 1 and rng.random() < 0.65:
        left, right = rng.sample(investigations, 2)
        compare = new_node(
            f"M{len(nodes)+1}", "comparison", "compare_findings", {
                "left": bind(left, "finding"),
                "right": bind(right, "finding"),
            },
        )
        nodes.append(compare)

    # The Question requests all investigated links and any verification /
    # comparison findings. Do not silently drop an intermediate VerifiedClaim
    # simply because a later Investigation consumed it as context. Every
    # requested work product must reach the terminal goal with its own typed
    # port, even when the information is also used by another Macro.
    final_inputs = {"entity": bind(identify, "entity")}
    deliverables = [
        node for node in nodes
        if node["role"] in {"investigation", "verification", "comparison"}
    ]
    for i, node in enumerate(deliverables, 1):
        port = next(iter(node["output_ports"]))
        final_inputs[f"result_{i}"] = bind(node, port)
    assert "result_1" in final_inputs

    # A direct evidence edge from Wave 1 to the final synthesis is not an
    # empty transitive shortcut: synthesis consumes the source-specific
    # original witness, which Identity intentionally does NOT expose.
    if rng.random() < 0.65:
        first = rng.choice(discovery)
        final_inputs["original_evidence"] = bind(first, "evidence")
    synth = new_node(
        f"M{len(nodes)+1}", "synthesis", "synthesize_findings", final_inputs,
    )
    nodes.append(synth)
    return {
        "version": VERSION,
        "nodes": nodes,
        "terminal_macro_id": synth["macro_id"],
        "reference_waves": waves(nodes),
        "config": {"discovery_count": n_discovery,
                   "investigation_count": n_investigation},
    }


def generate_graph(*, seed: int, ordinal: int, max_branches: int = 3,
                   max_macros: int = 12, max_parallel: int = 3,
                   max_waves: int = 7) -> dict:
    """Reproducible bounded grammar sampler with explicit capacity constraints."""
    if not 2 <= max_branches <= 3:
        raise ValueError("max_branches_must_be_2_or_3")
    if not 5 <= max_macros <= 20 or not 1 <= max_parallel <= 5:
        raise ValueError("invalid_macro_graph_budget")
    if not 3 <= max_waves <= 12:
        raise ValueError("invalid_macro_wave_budget")
    # Stable independent random stream; changing World does not alter
    # the task skeleton selected for the same seed and ordinal.
    rng = random.Random(f"typed-macro-dag-v1:{seed}:{ordinal}")
    for _ in range(100):
        graph = _attempt(rng, max_branches=max_branches)
        if (len(graph["nodes"]) <= max_macros
                and len(graph["reference_waves"]) <= max_waves
                and max(map(len, graph["reference_waves"])) <= max_parallel):
            from .typed_graph_validator import validate_graph
            validate_graph(graph)
            return graph
    raise ValueError("unable_to_sample_graph_within_budgets")
