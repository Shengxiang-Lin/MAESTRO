"""Hybrid World grounding for task-first Macro relation sketches.

Grounding happens *after* the abstract Macro DAG and relation sketch have
been frozen. Reuse the proven v4 RDF motif / complete candidate-path
sampler as a fact constructor, NOT as the source of the new Macro graph.
Historical v4 specs remain unchanged and independently replayable.
"""
from __future__ import annotations

from environment.schema import digest
from . import hybrid_world_v4 as v4
from . import knowledge_macro as v1
from .macro_graph_sampler import validate_graph
from .relational_task_planner import validate_sketch

VERSION = "maestro.task_first_world_grounding.v1"


def ground_world(graph: dict, sketch: dict, kg, *,
                 seed: int, ordinal: int, difficulty: str,
                 min_hops: int, max_hops: int,
                 index: v1.YagoKnowledgeIndex | None = None) -> dict:
    validate_graph(graph)
    issues = validate_sketch(sketch, graph)
    if issues:
        raise ValueError("graph_first_bad_relation_sketch:" + ",".join(issues))
    index = index or v1.YagoKnowledgeIndex(kg)
    requested_disc = sum(row["role"] == "discovery" for row in graph["nodes"])
    # This world-first helper is *only* a grounded YAGO fact builder;
    # its built-in v4 task graph is discarded, never used in the new plan.
    for attempt in range(32):
        generated_ordinal = ordinal + attempt * 100003
        try:
            base = v4.make_spec(
                kg, seed=seed, ordinal=generated_ordinal,
                difficulty=difficulty, min_hops=min_hops,
                max_hops=max_hops, index=index,
            )
        except ValueError:
            continue
        if sum(t["role"] == "discovery"
               for t in base["macro_plan"]["macro_tasks"]) != requested_disc:
            continue
        facts = base["macro_plan"]["real_facts"]
        fork = base["macro_plan"]["real_fork"]
        if not facts or set(fork["candidate_forks"]) != set(
            base["macro_plan"]["all_cases"]
        ):
            continue
        grounded = {
            "version": VERSION,
            "graph_hash": graph["graph_hash"],
            "relation_sketch_hash": sketch["sketch_hash"],
            "selected_attempt": attempt,
            "base_v4": base,
            "motif": {
                "required_real_fork_predicate": fork["predicate"],
                "real_fork_count": len(fork["candidate_forks"]),
                "main_chain_predicates": [f["predicate"] for f in facts],
                "main_chain_hops": len(facts),
                "source_file": base["knowledge_witness"]["origin_file"]
                  if "origin_file" in base["knowledge_witness"] else
                  base["hybrid_backbone_v3"]["knowledge_witness"]["origin_file"],
                "context_only_v3_motif": base["macro_plan"]["real_motif"]["kind"],
            },
            "provenance": {
                "yago_relations": "genuine_frozen_rdf_witnesses",
                "project_event_relations": "fictional_generated_facts",
                "source_candidate_rows": "unverified_synthetic_claims",
            },
        }
        grounded["grounding_hash"] = digest(
            grounded, "task_first_grounding_frozen"
        )
        return grounded
    raise ValueError(
        "graph_first_no_complete_yago_motif_with_required_discovery_budget"
    )


def validate_grounding(grounded: dict, graph: dict,
                       sketch: dict) -> list[str]:
    try:
        if grounded["version"] != VERSION:
            return ["task_first_bad_grounding_version"]
        raw = {key: value for key, value in grounded.items()
               if key != "grounding_hash"}
        if (grounded["grounding_hash"] != digest(
                    raw, "task_first_grounding_frozen")
                or grounded["graph_hash"] != graph["graph_hash"]
                or grounded["relation_sketch_hash"] != sketch["sketch_hash"]):
            return ["task_first_grounding_hash_mismatch"]
        base = grounded["base_v4"]
        issues = v4.validate_spec(base)
        if issues:
            return ["task_first_invalid_yago_v4_witness:" + ",".join(issues)]
        if sum(t["role"] == "discovery"
               for t in base["macro_plan"]["macro_tasks"]) != sum(
            row["role"] == "discovery" for row in graph["nodes"]
        ):
            return ["task_first_wrong_discovery_budget"]
        fork = base["macro_plan"]["real_fork"]
        if (fork["predicate"] != grounded["motif"][
                "required_real_fork_predicate"]
                or len(base["macro_plan"]["real_facts"]) !=
                grounded["motif"]["main_chain_hops"]):
            return ["task_first_real_motif_mismatch"]
        return []
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return ["task_first_invalid_grounding:" + str(exc)[:140]]
