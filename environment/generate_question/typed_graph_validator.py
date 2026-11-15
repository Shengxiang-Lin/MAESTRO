"""Validate typed Macro DAGs without consulting the evidence World or an LLM."""
from __future__ import annotations

from collections import Counter
import hashlib
import json

from .typed_role_grammar import OUTPUTS, REQUIRED_INPUTS, allowed
from .typed_graph_generator import VERSION, waves


def validate_graph(graph: dict) -> dict:
    if graph.get("version") != VERSION:
        raise ValueError("invalid_typed_graph_version")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("missing_typed_graph_nodes")
    seen = {}
    all_ids = [n["macro_id"] for n in nodes]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("duplicate_macro_ids")
    for n in nodes:
        mid, role = n["macro_id"], n["role"]
        if role not in OUTPUTS or n.get("output_ports") != OUTPUTS[role]:
            raise ValueError("role_output_contract_mismatch:" + mid)
        ins = n.get("input_ports")
        if not isinstance(ins, dict):
            raise ValueError("invalid_input_ports:" + mid)
        if not set(REQUIRED_INPUTS[role]).issubset(ins):
            raise ValueError("missing_required_role_port:" + mid)
        if role == "identification":
            incoming = [p for p in ins if p.startswith("candidate_")]
            if len(incoming) < 2:
                raise ValueError("identity_needs_two_candidate_handoffs")
        if role == "comparison":
            if ins["left"]["from_macro"] == ins["right"]["from_macro"]:
                raise ValueError("comparison_needs_distinct_investigations")
        if role == "synthesis":
            results = [p for p in ins if p.startswith("result_")]
            if not results:
                raise ValueError("synthesis_needs_information_product")
        actual_parents = []
        for name, link in ins.items():
            accepted = allowed(role, name)
            if accepted is None or link.get("type") not in accepted:
                raise ValueError("illegal_macro_role_transition:" + mid + ":" + name)
            upstream = link.get("from_macro")
            source_port = link.get("from_port")
            if upstream not in seen:
                raise ValueError("future_or_undeclared_macro_handoff:" + mid)
            actual_type = seen[upstream]["output_ports"].get(source_port)
            if actual_type != link.get("type"):
                raise ValueError("macro_port_type_mismatch:" + mid + ":" + name)
            if upstream not in actual_parents:
                actual_parents.append(upstream)
        if n.get("depends_on") != actual_parents:
            raise ValueError("unconsumed_or_fabricated_dependency:" + mid)
        if n.get("operation") not in {
            "discover_candidates", "identify_entity", "investigate_relation",
            "verify_finding", "compare_findings", "synthesize_findings",
        }:
            raise ValueError("invalid_macro_operation:" + mid)
        seen[mid] = n

    if graph.get("terminal_macro_id") != nodes[-1]["macro_id"] or nodes[-1]["role"] != "synthesis":
        raise ValueError("invalid_macro_terminal")
    expected_waves = waves(nodes)
    if graph.get("reference_waves") != expected_waves:
        raise ValueError("macro_wave_schedule_mismatch")
    terminal = nodes[-1]["macro_id"]
    reachable = {terminal}
    changed = True
    while changed:
        old_count = len(reachable)
        for n in nodes:
            if n["macro_id"] in reachable:
                reachable.update(n["depends_on"])
        changed = len(reachable) > old_count
    if reachable != set(all_ids):
        raise ValueError("orphan_macro_not_needed_for_terminal")

    levels = {mid: level for level, wave in enumerate(expected_waves)
              for mid in wave}
    if any(levels[n["macro_id"]] <= levels[p]
           for n in nodes for p in n["depends_on"]):
        raise ValueError("non_causal_macro_edge")
    signatures = [
        (n["role"], n["operation"],
         tuple((port, link["from_macro"], link["from_port"])
               for port, link in sorted(n["input_ports"].items())))
        for n in nodes
    ]
    signature = hashlib.sha256(json.dumps(signatures, sort_keys=True).encode()).hexdigest()[:16]
    cross_wave = [
        {"from_macro": link["from_macro"], "to_macro": n["macro_id"],
         "to_port": port,
         "wave_span": levels[n["macro_id"]] - levels[link["from_macro"]]}
        for n in nodes for port, link in n["input_ports"].items()
        if levels[n["macro_id"]] - levels[link["from_macro"]] > 1
    ]
    return {
        "passed": True, "macro_count": len(nodes),
        "wave_count": len(expected_waves),
        "max_parallel": max(map(len, expected_waves)),
        "role_counts": dict(Counter(n["role"] for n in nodes)),
        "cross_wave_edges": cross_wave,
        "graph_signature": signature,
    }
