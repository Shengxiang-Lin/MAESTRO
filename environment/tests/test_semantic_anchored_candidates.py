"""Regression tests for YAGO subject-coherent candidate construction."""
import random
import unittest

from environment.task_construction.binding.predicate_binder import (
    AttributeBinding, RelationBinding,
)
from environment.task_construction.binding.program_binder import BindingPlan
from environment.task_construction.binding.semantic_anchors import SemanticAnchorBank
from environment.task_construction.data_sources.taxonomy import TaxonomyIndex
from environment.task_construction.ir import ProgramIR, ProgramNode


class SemanticAnchoredCandidateTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = TaxonomyIndex({
            "schema:Film": {"schema:CreativeWork"},
            "schema:Person": {"schema:Thing"},
        })
        self.catalog = {"predicates": {
            "schema:director": {"count": 50, "kind": "relation"},
            "schema:datePublished": {"count": 50, "kind": "attribute"},
            "schema:producer": {"count": 50, "kind": "relation"},
            "schema:zip": {"count": 50, "kind": "attribute"},
            "schema:rare": {"count": 1, "kind": "attribute"},
        }}
        self.bank = SemanticAnchorBank({"profiles": [{
            "source_subject": "yago:SampleFilm",
            "entity_types": ["schema:Film"],
            "observed_predicates": [
                "schema:director", "schema:datePublished", "schema:rare",
            ],
        }]}, self.catalog, self.taxonomy)

    def _plan(self, *, filter_after_relation=False, direction="forward"):
        nodes = [
            ProgramNode(0, "FIND", (), ("type:schema:Film",)),
        ]
        if filter_after_relation:
            nodes += [
                ProgramNode(1, "RELATE", (0,), ("schema:director", direction)),
                ProgramNode(2, "FILTER_YEAR", (1,),
                            ("schema:datePublished", "<", "2010")),
                ProgramNode(3, "COUNT", (2,), ()),
            ]
            node_types = {0: "schema:Film", 1: "schema:Person",
                          2: "schema:Person"}
            attributes = {2: AttributeBinding(
                "schema:datePublished", "schema:Person", "year"
            )}
            relations = {1: RelationBinding(
                "schema:director", "schema:Film", "schema:Person", direction
            )}
        else:
            nodes += [
                ProgramNode(1, "FILTER_YEAR", (0,),
                            ("schema:datePublished", "<", "2010")),
                ProgramNode(2, "RELATE", (1,), ("schema:director", direction)),
                ProgramNode(3, "COUNT", (2,), ()),
            ]
            node_types = {0: "schema:Film", 1: "schema:Film",
                          2: "schema:Person"}
            attributes = {1: AttributeBinding(
                "schema:datePublished", "schema:Film", "year"
            )}
            relations = {2: RelationBinding(
                "schema:director", "schema:Film", "schema:Person", direction
            )}
        return BindingPlan(
            ProgramIR(nodes), node_types, relations, attributes
        )

    def test_real_joint_profile_supports_two_predicates_on_same_subject(self):
        anchor = self.bank.sample(self._plan().program, rng=random.Random(17))
        evidence = self.bank.evaluate(self._plan(), anchor)
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence["matched_count"], 2)
        self.assertEqual(evidence["matched_predicates"],
                         ["schema:datePublished", "schema:director"])
        self.assertNotIn("yago:SampleFilm", str(evidence))

    def test_matching_predicates_on_different_entities_does_not_qualify(self):
        anchor = self.bank.sample(self._plan().program, rng=random.Random(17))
        self.assertIsNone(self.bank.evaluate(
            self._plan(filter_after_relation=True), anchor
        ))

    def test_only_jointly_observed_predicates_are_eligible(self):
        self.assertNotIn("schema:rare", self.bank.profiles[0]["predicates"])
        self.assertEqual(len(self.bank.profiles), 1)

    def test_isolated_single_predicate_profiles_cannot_form_false_anchor(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            SemanticAnchorBank({"profiles": [
                {"source_subject": "a", "entity_types": ["schema:Film"],
                 "observed_predicates": ["schema:director"]},
                {"source_subject": "b", "entity_types": ["schema:Film"],
                 "observed_predicates": ["schema:datePublished"]},
            ]}, self.catalog, self.taxonomy)

    def test_sample_is_seed_stable(self):
        program = self._plan().program
        a = self.bank.sample(program, rng=random.Random(123))
        b = self.bank.sample(program, rng=random.Random(123))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
