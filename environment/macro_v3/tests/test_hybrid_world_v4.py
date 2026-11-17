"""Causal hybrid motif v4: end-to-end old Stage-1/2 and stronger information tests."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from environment.generate_question import knowledge_macro as v1
from environment.generate_question import hybrid_world as v3
from environment.generate_question import hybrid_world_v4 as v4
from environment.generate_question.real_motif_grounding import (
    mine_required_fork, validate_frozen_required_fork,
)
from environment.generate_question.question import reference_question
from environment.generate_question.quality import compile_and_validate, validate_replay
from environment.generate_question.world import compile_program, generate_world
from environment.implement_subtask.realize import realize_scene
from environment.macro_v3.yago import Edge, KnowledgeGraph

try:
    import pyoxigraph  # noqa: F401
    RDF_READY = True
except ImportError:
    RDF_READY = False


def example_yago(n=48):
    namespace = "https://example.org/"
    schema = "http://schema.org/"
    facts = []
    for i in range(n):
        person = namespace + f"Researcher{i}"
        town = namespace + f"Town{i}"
        region = namespace + f"Region{i}"
        country = namespace + f"Country{i}"
        institute = namespace + f"Institute{i}"
        facts.extend([
            Edge(person, schema+"birthPlace", town),
            Edge(town, schema+"containedInPlace", region),
            Edge(region, schema+"containedInPlace", country),
            Edge(person, schema+"memberOf", institute),
        ])
    return KnowledgeGraph(
        edges=facts, input_file="fixture-yago-facts.ttl",
        scanned=len(facts),
    )


class CausalHybridTests(unittest.TestCase):
    def test_provenance_and_motif_witness_not_fictional(self):
        kg = example_yago()
        index = v1.YagoKnowledgeIndex(kg)
        spec = v4.make_spec(
            kg, seed=17, ordinal=0, difficulty="hard", index=index,
        )
        self.assertFalse(v4.validate_spec(spec))
        paths = spec["macro_plan"]["candidate_paths"]
        fork = spec["macro_plan"]["real_fork"]
        self.assertFalse(validate_frozen_required_fork(fork, paths))
        corpus = set(kg.edges)
        for case, item in fork["candidate_forks"].items():
            self.assertIn(
                Edge(item["subject"], item["predicate"], item["object"]),
                corpus,
            )
            self.assertEqual(item["origin"], "yago")
        for source in spec["macro_plan"]["synthetic_graph"]["true_relations"]:
            self.assertEqual(source["origin"], "synthetic")
        self.assertTrue(spec["macro_plan"]["synthetic_graph"]["events"])
        self.assertTrue(spec["macro_plan"]["synthetic_graph"]["organizers"])
        self.assertTrue(spec["macro_plan"]["synthetic_graph"]["venues"])

    def test_all_handoff_source_and_counterfactual_gates(self):
        kg = example_yago()
        idx = v1.YagoKnowledgeIndex(kg)
        observed = set()
        for ordinal in range(8):
            spec = v4.make_spec(
                kg, seed=19, ordinal=ordinal, difficulty="hard",
                index=idx,
            )
            self.assertEqual(spec, v4.make_spec(
                kg, seed=19, ordinal=ordinal, difficulty="hard", index=idx,
            ))
            scene = compile_and_validate(spec, reference_question(spec))
            report = scene["difficulty_audit"]["macro_first_audit"]
            self.assertTrue(report["handoff_counterfactuals"][
                "all_handoffs_necessary"])
            self.assertTrue(report["source_counterfactuals_pass"])
            info = report["candidate_information_necessity"]
            self.assertTrue(info["passed"])
            self.assertTrue(info["identification_unique"])
            self.assertTrue(all(info["single_clue_necessary"].values()))
            self.assertTrue(all(info["discovery_macro_necessary"].values()))
            self.assertTrue(all(info["link_branch_necessary"].values()))
            self.assertTrue(info["synthetic_event_necessary"])
            self.assertTrue(info["grounded_real_fork_necessary"])
            self.assertTrue(info["all_discovery_bypass_ambiguous"])
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertEqual(generate_world(spec), scene["world"])
            self.assertEqual(compile_program(spec), scene["gold_program"])
            self.assertEqual(len(realize_scene(scene, strict=True)["macro_tasks"]),
                             len(spec["macro_plan"]["macro_tasks"]))
            self.assertEqual(len(scene["gold_answer"]), 1)
            self.assertEqual(scene["gold_answer"][0]["answer"],
                             spec["macro_plan"]["real_facts"][-1]["object_label"])
            world = scene["world"]
            self.assertIn("real_fork", world["sources"])
            self.assertIn("event_ledger", world["sources"])
            for row in world["sources"]["entity_index"]["observations"]:
                self.assertTrue(row["values"]["project_name"])
            for row in world["sources"]["event_ledger"]["observations"]:
                self.assertTrue(row["values"]["event_name"])
                self.assertTrue(row["values"]["organizer_name"])
                self.assertTrue(row["values"]["venue_name"])
            for row in world["sources"]["real_fork"]["observations"]:
                self.assertIn("knowledge_fact", row)
                self.assertTrue(row["evidence_origin_id"].startswith("YAGO-4.6:"))
            observed.add(
                (spec["macro_plan"]["hybrid_shape"],
                 len(spec["macro_plan"]["macro_tasks"]))
            )
        self.assertGreaterEqual(len(observed), 2)

    def test_frozen_truth_and_source_replay_tamper_detected(self):
        kg = example_yago()
        spec = v4.make_spec(kg, seed=13, ordinal=1, difficulty="medium")
        changed = deepcopy(spec)
        changed["macro_plan"]["real_fork"]["predicate"] = "urn:fake"
        self.assertTrue(v4.validate_spec(changed))
        changed = deepcopy(spec)
        changed["macro_plan"]["synthetic_graph"]["events"][
            spec["macro_plan"]["target_case"]]["status"] = "invented"
        self.assertTrue(v4.validate_spec(changed))
        scene = compile_and_validate(spec, reference_question(spec))
        broken = deepcopy(scene)
        broken["world"]["sources"]["real_fork"]["observations"][0][
            "values"]["motif_object"] = "urn:fake"
        self.assertEqual(validate_replay(broken)["status"], "invalid")

    def test_old_v3_frozen_replay_unchanged(self):
        kg = example_yago()
        spec = v3.make_spec(kg, seed=17, ordinal=0, difficulty="medium")
        scene = compile_and_validate(spec, reference_question(spec))
        self.assertEqual(validate_replay(scene)["status"], "valid")


if __name__ == "__main__":
    unittest.main()
