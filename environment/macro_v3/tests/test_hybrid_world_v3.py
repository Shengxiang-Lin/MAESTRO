"""Hybrid YAGO + fictional entities through the original Stage-1 pipeline."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from environment.generate_question import knowledge_macro as v1
from environment.generate_question import knowledge_macro_v2 as v2
from environment.generate_question import hybrid_world as hybrid
from environment.generate_question.world_motifs import (
    mine_motif, validate_frozen_motif,
)
from environment.generate_question.quality import (
    compile_and_validate, validate_replay,
)
from environment.generate_question.question import reference_question
from environment.generate_question.world import compile_program, generate_world
from environment.implement_subtask.realize import realize_scene
from environment.macro_v3.yago import Edge, KnowledgeGraph

try:
    import pyoxigraph  # noqa: F401
    RDF_READY = True
except ImportError:
    RDF_READY = False


def graph_fixture(n=36):
    domain, s = "https://example.org/", "http://schema.org/"
    edges = []
    for i in range(n):
        subject, location, region, country, institution = (
            domain + f"Researcher{i}", domain + f"Town{i}",
            domain + f"Region{i}", domain + f"Country{i}",
            domain + f"Institution{i}",
        )
        edges.extend([
            Edge(subject, s + "birthPlace", location),
            Edge(location, s + "containedInPlace", region),
            Edge(region, s + "containedInPlace", country),
            Edge(subject, s + "memberOf", institution),
        ])
    # Exact two-way route to test a *real* diamond motif:
    edges.extend([
        Edge(domain + "Researcher0", s + "knows", domain + "Friend0"),
        Edge(domain + "Friend0", s + "locatedIn", domain + "Region0"),
    ])
    return KnowledgeGraph(
        edges=edges, input_file="fixture-yago-4.6-facts.ttl",
        scanned=len(edges),
    )


class HybridWorldTests(unittest.TestCase):
    def test_real_motif_mining_distinguishes_fork_diamond(self):
        graph = graph_fixture()
        index = v1.YagoKnowledgeIndex(graph)
        s, d = "http://schema.org/", "https://example.org/"
        actual = [
            {"subject": d+"Researcher0", "predicate": s+"birthPlace",
             "object": d+"Town0", "object_label": "Town0"},
            {"subject": d+"Town0", "predicate": s+"containedInPlace",
             "object": d+"Region0", "object_label": "Region0"},
        ]
        motif = mine_motif(graph, index, actual, seed=3, ordinal=0)
        self.assertEqual(motif["kind"], "diamond")
        self.assertEqual(len(motif["extra_facts"]), 2)
        self.assertFalse(validate_frozen_motif(motif, actual))
        witnesses = set(graph.edges)
        for fact in motif["anchor_facts"] + motif["extra_facts"]:
            self.assertIn(Edge(fact["subject"], fact["predicate"],
                               fact["object"]), witnesses)
            self.assertEqual(fact["origin"], "yago")

    def test_hybrid_replay_causal_branches_and_source_lineage(self):
        graph = graph_fixture()
        index = v1.YagoKnowledgeIndex(graph)
        shapes = set()
        for ordinal in range(10):
            spec = hybrid.make_spec(
                graph, seed=17, ordinal=ordinal, difficulty="hard",
                min_hops=2, max_hops=3, index=index,
            )
            self.assertFalse(hybrid.validate_spec(spec))
            question = reference_question(spec)
            scene = compile_and_validate(spec, question)
            self.assertEqual(scene["gold_answer"], [{
                "answer": spec["macro_plan"]["real_facts"][-1]["object_label"]
            }])
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertEqual(compile_program(spec), scene["gold_program"])
            self.assertEqual(generate_world(spec), scene["world"])
            self.assertEqual(
                len(realize_scene(scene, strict=True)["macro_tasks"]),
                scene["macro_graph"]["macro_task_count"],
            )
            shapes.add(spec["macro_plan"]["hybrid_shape"])
            qa = scene["difficulty_audit"]["macro_first_audit"]
            self.assertTrue(qa["handoff_counterfactuals"][
                "all_handoffs_necessary"])
            self.assertTrue(qa["source_counterfactuals_pass"])
            necessity = qa["candidate_information_necessity"]
            self.assertTrue(necessity["passed"])
            self.assertTrue(necessity["identification_unique"])
            self.assertTrue(all(
                necessity["clue_relaxation_necessary"].values()
            ))
            self.assertTrue(all(
                necessity["discovery_relaxation_necessary"].values()
            ))
            self.assertTrue(all(
                necessity["synthetic_branch_necessity"].values()
            ))
            self.assertTrue(necessity["all_discovery_filters_removed"]["ambiguous"])
            self.assertIn("verification", {
                row["role"] for row in spec["macro_plan"]["macro_tasks"]
            })
            bank = scene["world"]["sources"]
            true_links = scene["world"]["synthetic_world"]["true_relations"]
            self.assertTrue(true_links)
            self.assertTrue(all(row["origin"] == "synthetic"
                                for row in true_links))
            self.assertTrue(all(row["origin"] == "yago"
                                for row in spec["macro_plan"]["real_motif"][
                                    "anchor_facts"]))
            for sid in ("link_a", "link_b"):
                self.assertTrue(all(
                    entry["claim_status"] == "unverified_candidate"
                    for entry in bank[sid]["observations"]
                ))
                self.assertEqual(bank[sid]["access_policy"]["mode"], "browse")
            self.assertTrue(all(
                source["evidence_kind"] == "real_yago_fact_excerpts"
                for name, source in bank.items() if name.startswith("relation_")
            ))
            self.assertLessEqual(len(spec["macro_plan"]["macro_tasks"]), 9)
        self.assertEqual(shapes, {"parallel_fork_join", "serial_refinement"})

    def test_tamper_provenance_and_fictional_truth_detected(self):
        graph = graph_fixture()
        spec = hybrid.make_spec(graph, seed=12, ordinal=3, difficulty="medium")
        changed = deepcopy(spec)
        changed["macro_plan"]["synthetic_truth"][0]["object"] = "urn:fake"
        self.assertTrue(hybrid.validate_spec(changed))
        scene = compile_and_validate(spec, reference_question(spec))
        self.assertEqual(validate_replay(scene)["status"], "valid")
        scene["world"]["synthetic_world"]["true_relations"][0][
            "object"] = "urn:fake"
        self.assertEqual(validate_replay(scene)["status"], "invalid")

    def test_frozen_v2_remains_valid(self):
        graph = graph_fixture()
        original = v2.make_spec(graph, seed=17, ordinal=0,
                                difficulty="medium")
        scene = compile_and_validate(original, reference_question(original))
        self.assertEqual(validate_replay(scene)["status"], "valid")
        self.assertEqual(original["reasoning_pattern"], "typed_knowledge_v2")


@unittest.skipUnless(RDF_READY, "requires RDF loader")
if __name__ == "__main__":
    unittest.main()
