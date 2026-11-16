"""YAGO v2 causal necessity: unique identity and no Discovery bypass.

All tests run through the *original* canonical Stage-1 compiler/quality,
not the experimental v3.x composer.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from environment.schema import digest
from environment.generate_question import knowledge_macro as v1
from environment.generate_question import knowledge_macro_v2 as v2
from environment.generate_question.information_necessity import (
    _audit_knowledge_replanning, audit_candidate_information,
)
from environment.generate_question.question import reference_question
from environment.generate_question.quality import (
    compile_and_validate, validate_replay,
)
from environment.generate_question.world import compile_program, generate_world
from environment.implement_subtask.realize import realize_scene
from environment.macro_v3.yago import Edge, KnowledgeGraph

try:
    import pyoxigraph  # noqa
    RDF_READY = True
except ImportError:
    RDF_READY = False


def graph_fixture(n=28):
    e, p = "https://yago-knowledge.org/resource/", "http://schema.org/"
    edges = []
    for i in range(n):
        a, b, c, d = [e + f"{name}{i}" for name in
                       ("Subject", "City", "Region", "Country")]
        edges.extend([
            Edge(a, p + "birthPlace", b),
            Edge(b, p + "containedInPlace", c),
            Edge(c, p + "containedInPlace", d),
            Edge(a, p + "memberOf", e + f"Institution{i}"),
        ])
    return KnowledgeGraph(edges=edges,
                          input_file="fixture-yago-facts.ttl",
                          scanned=len(edges))


class KnowledgeV2QualityTests(unittest.TestCase):
    def test_real_paths_identification_and_bypass(self):
        kg = graph_fixture()
        index = v1.YagoKnowledgeIndex(kg)
        observed = set()
        for i in range(12):
            spec = v2.make_spec(
                kg, seed=17, ordinal=i, index=index,
                difficulty="hard" if i % 2 else "medium",
                min_hops=2, max_hops=3,
            )
            self.assertFalse(v2.validate_spec(spec))
            self.assertEqual(spec, v2.make_spec(
                kg, seed=17, ordinal=i, index=index,
                difficulty="hard" if i % 2 else "medium",
                min_hops=2, max_hops=3,
            ))
            scene = compile_and_validate(spec, reference_question(spec))
            audit = scene["difficulty_audit"]["macro_first_audit"][
                "candidate_information_necessity"
            ]
            self.assertEqual(audit["policy"], v2.NECESSITY_POLICY)
            self.assertTrue(audit["passed"])
            self.assertTrue(audit["identification_unique"])
            self.assertEqual(audit["identity_macro_workproduct_count"], 1)
            self.assertTrue(all(
                row["has_real_competing_answer"]
                for row in audit["source_counterfactuals"].values()
            ))
            self.assertTrue(all(
                row["has_real_competing_answer"]
                for row in audit["discovery_counterfactuals"].values()
            ))
            self.assertTrue(audit["all_discovery_filters_removed"]["ambiguous"])
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertEqual(generate_world(spec), scene["world"])
            self.assertEqual(compile_program(spec), scene["gold_program"])
            self.assertEqual(len(realize_scene(scene, strict=True)["macro_tasks"]),
                             scene["macro_graph"]["macro_task_count"])
            observed.add(len(spec["macro_plan"]["macro_tasks"]))
            # All *competing* source observations are real RDF facts.
            known = set(kg.edges)
            for case, chain in spec["macro_plan"]["candidate_paths"].items():
                for fact in chain:
                    self.assertIn(
                        Edge(fact["subject"], fact["predicate"], fact["object"]),
                        known,
                    )
        self.assertGreaterEqual(len(observed), 2)

    def test_old_v1_replay_still_supported(self):
        kg = graph_fixture()
        spec = v1.make_spec(kg, seed=17, ordinal=3, difficulty="medium")
        scene = compile_and_validate(spec, reference_question(spec))
        self.assertEqual(validate_replay(scene)["status"], "valid")
        self.assertEqual(spec["reasoning_pattern"], "typed_knowledge_v1")

    def test_forged_candidate_and_duplicate_answer_rejected(self):
        kg = graph_fixture()
        spec = v2.make_spec(kg, seed=8, ordinal=1, difficulty="medium")
        wrong = deepcopy(spec)
        case = next(k for k in wrong["macro_plan"]["candidate_paths"]
                    if k != wrong["macro_plan"]["target_case"])
        wrong["macro_plan"]["candidate_paths"][case][-1][
            "object_label"] = wrong["macro_plan"]["candidate_paths"][
                wrong["macro_plan"]["target_case"]][-1]["object_label"]
        wrong["knowledge_contract_hash"] = digest(
            wrong["macro_plan"], wrong["knowledge_witness"],
            wrong["seed"], wrong["ordinal"], v2.HASH_SALT,
        )
        self.assertIn("knowledge_v2_duplicate_terminal_answer",
                      v2.validate_spec(wrong))

    def test_old_failure_mode_rejected_by_new_audit(self):
        kg = graph_fixture()
        spec = v1.make_spec(kg, seed=17, ordinal=0, difficulty="medium")
        scene = compile_and_validate(spec, reference_question(spec))
        report = _audit_knowledge_replanning(scene)
        self.assertFalse(report["passed"])
        self.assertFalse(report["identification_unique"])


@unittest.skipUnless(RDF_READY, "requires RDF loader")
class KnowledgeV2CliTests(unittest.TestCase):
    def test_original_generate_questions_command_with_rdf(self):
        from environment.config import load_config, args_from_config
        from environment.generate_question.pipeline import generate_questions
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            rdf = home / "yago-4.6-facts.ttl"
            entries = [
                "@prefix x: <https://example.org/> .",
                "@prefix s: <http://schema.org/> .",
            ]
            for i in range(28):
                entries.extend([
                    f"x:Subject{i} s:birthPlace x:City{i} .",
                    f"x:City{i} s:containedInPlace x:Region{i} .",
                    f"x:Region{i} s:containedInPlace x:Country{i} .",
                    f"x:Subject{i} s:memberOf x:Institute{i} .",
                ])
            rdf.write_text("\n".join(entries), encoding="utf-8")
            config = load_config("environment/configs/questions_knowledge_v2.yaml")
            config["generation"]["knowledge_yago"] = str(rdf)
            config["generation"]["output_dir"] = str(home / "generated")
            config["generation"]["target"] = 3
            args = args_from_config(config)
            args.no_progress = True
            result = generate_questions(args)
            self.assertEqual(result["status"], "done", result)
            self.assertEqual(result["generated"], 3)
            quality = json.loads(
                (args.output_dir / "quality_report.json").read_text()
            )
            self.assertEqual(quality["replay_pass"], 3)
            for row in (args.output_dir /
                        "constructed" / "scenes.jsonl").read_text().splitlines():
                scene = json.loads(row)
                self.assertEqual(scene["problem_spec"]["reasoning_pattern"],
                                 "typed_knowledge_v2")
                self.assertTrue(scene["difficulty_audit"]["macro_first_audit"][
                    "candidate_information_necessity"]["passed"])


if __name__ == "__main__":
    unittest.main()
