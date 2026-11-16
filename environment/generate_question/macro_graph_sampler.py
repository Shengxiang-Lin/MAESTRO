"""Graph-FIRST Macro-work sampler for the canonical Stage-1 pipeline.

The sampler never sees a YAGO entity, RDF relation, evidence row, or answer.
It selects task roles, typed handoffs and information work before grounding.
This intentionally replaces task-graph derivation from a preselected World.
"""
from __future__ import annotations

import random

from environment.schema import digest

VERSION = "maestro.macro_task_graph_first.v1"
SHAPES = ("parallel_association", "serial_refinement",
          "event_gated_association")

ROLE_OUTPUT = {
    "discover": ("discovery", "CandidateSet"),
    "identify": ("identification", "ProjectRef"),
    "event": ("investigation", "EventEvidence"),
    "directory_a": ("investigation", "CandidateReferenceSet"),
    "directory_b": ("investigation", "CandidateReferenceSet"),
    "associate": ("association", "CandidateReferenceSet"),
    "refine_b": ("association", "CandidateReferenceSet"),
    "crosscheck": ("comparison", "ConfirmedEntityRef"),
    "answer": ("synthesis", "FinalAnswer"),
}
INPUT_TYPES = {
    "identify": {"discover": "CandidateSet"},
    "event": {"identify": "ProjectRef"},
    "directory_a": {"identify": "ProjectRef", "event": "EventEvidence"},
    "directory_b": {"identify": "ProjectRef", "event": "EventEvidence"},
    "associate": {"directory_a": "CandidateReferenceSet",
                  "directory_b": "CandidateReferenceSet"},
    "refine_b": {"directory_a": "CandidateReferenceSet"},
    "crosscheck": {"associate": "CandidateReferenceSet",
                   "refine_b": "CandidateReferenceSet",
                   "event": "EventEvidence"},
    "answer": {"crosscheck": "ConfirmedEntityRef"},
}
WORK = {
    "discover": "Discover possible dossiers from a separately scoped evidence bundle",
    "identify": "Reconcile candidate clues and identify the fictional project",
    "event": "Investigate qualifying event details and eligible entity relationships",
    "directory_a": "Investigate project reference hypotheses in source A",
    "directory_b": "Independently investigate reference hypotheses in source B",
    "associate": "Associate references corroborated by both candidate directories",
    "refine_b": "Refine a candidate set by consulting a second information source",
    "crosscheck": "Compare fictional-event eligibility to genuine secondary YAGO relations",
    "answer": "Synthesize the verified subject through a real multi-hop YAGO relation query",
}


def sample_graph(*, seed: int, ordinal: int,
                 difficulty: str, forced_shape: str | None = None) -> dict:
    """Sample partial-order coarse information tasks, with no World access."""
    if difficulty not in {"medium", "hard"}:
        raise ValueError("graph_first_requires_medium_or_hard")
    rng = random.Random(f"maestro-graph-first:{seed}:{ordinal}")
    if forced_shape is not None and forced_shape not in SHAPES:
        raise ValueError("unsupported_graph_first_shape")
    # Cyclic balancing is intentional for small pilot samples; node roles
    # and DAG edges are chosen before the knowledge-grounding attempt.
    shape = forced_shape or SHAPES[(ordinal + seed) % len(SHAPES)]
    disc_count = 3 if shape == "serial_refinement" else 2
    result = []
    names = {}

    def node(kind: str, parents=(), *, role=None, description=None):
        mid = "M" + str(len(result)+1)
        definition = ROLE_OUTPUT[kind]
        incoming = {}
        for parent in parents:
            parent_mid = names[parent]
            origin = next(row for row in result if row["macro_id"] == parent_mid)
            expected = INPUT_TYPES.get(kind, {}).get(parent)
            if kind == "identify" and parent.startswith("discover"):
                expected = "CandidateSet"
            if expected != origin["output_type"]:
                raise ValueError("macro_graph_first_type_mismatch:" + kind)
            incoming[parent] = {
                "from_macro": parent_mid,
                "from_port": "output",
                "type": expected,
            }
        record = {
            "macro_id": mid, "task_key": kind,
            "role": role or definition[0],
            "information_goal": description or WORK[
                "discover" if kind.startswith("discover") else kind
            ],
            "input_ports": incoming,
            "output_port": "output", "output_type": definition[1],
            "depends_on": [link["from_macro"] for link in incoming.values()],
            "source_requirements": (
                ["cue_bundle"] if kind.startswith("discover") else
                ["project_registry"] if kind == "identify" else
                ["event_records"] if kind == "event" else
                ["simulated_directory_a"] if kind == "directory_a" else
                ["simulated_directory_b"] if kind in {"directory_b", "refine_b"} else
                ["real_secondary_fork"] if kind == "crosscheck" else
                ["real_relation_chain"] if kind == "answer" else []
            ),
        }
        result.append(record)
        names[kind] = mid

    for i in range(disc_count):
        key = f"discover{i+1}"
        node(key)
    node("identify", [f"discover{i+1}" for i in range(disc_count)])
    node("event", ["identify"])
    ref_input = ("event",) if shape == "event_gated_association" else ("identify",)
    node("directory_a", ref_input)
    if shape == "serial_refinement":
        node("refine_b", ["directory_a"])
        node("crosscheck", ["refine_b", "event"])
    else:
        node("directory_b", ref_input)
        node("associate", ["directory_a", "directory_b"])
        node("crosscheck", ["associate", "event"])
    node("answer", ["crosscheck"])
    graph = {
        "version": VERSION, "shape": shape,
        "nodes": result, "terminal_macro_id": result[-1]["macro_id"],
        "source_free": True, "relation_free": True,
        "difficulty_target": difficulty,
        "sampled_first": True,
    }
    validate_graph(graph)
    graph["graph_hash"] = digest(graph, "graph-first-frozen")
    return graph


def validate_graph(graph: dict) -> list[str]:
    """Strictly validate all information ports and task ordering."""
    if graph.get("version") != VERSION or graph.get("shape") not in SHAPES:
        raise ValueError("invalid_graph_first_version_or_shape")
    seen = {}
    nodes = graph.get("nodes", [])
    if len(nodes) != 9 or graph.get("terminal_macro_id") != nodes[-1]["macro_id"]:
        raise ValueError("graph_first_macro_budget_or_terminal")
    for row in nodes:
        if row["macro_id"] in seen:
            raise ValueError("graph_first_duplicate_macro_id")
        inputs = row["input_ports"]
        if row["depends_on"] != [
            edge["from_macro"] for edge in inputs.values()
        ]:
            raise ValueError("graph_first_missing_named_handoff")
        for edge in inputs.values():
            parent = seen.get(edge["from_macro"])
            if not parent or edge["type"] != parent["output_type"]:
                raise ValueError("graph_first_unbound_or_wrong_port")
        seen[row["macro_id"]] = row
    if sum(row["role"] == "discovery" for row in nodes) not in (2, 3):
        raise ValueError("graph_first_discovery_budget")
    if len(set(row["task_key"] for row in nodes)) != len(nodes):
        raise ValueError("graph_first_repeated_task_key")
    if not graph.get("sampled_first") or not graph.get("source_free"):
        raise ValueError("graph_first_has_early_world_dependency")
    if not graph.get("relation_free"):
        raise ValueError("graph_first_relation_sampled_too_soon")
    return []
