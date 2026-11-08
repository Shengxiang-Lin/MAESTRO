"""Opt-in semantic-anchored v2 regressions and 30-task failure cases."""
import random
import unittest
from types import SimpleNamespace

from environment.task_construction.binding.semantic_anchors import SemanticAnchorBank
from environment.task_construction.data_sources.taxonomy import TaxonomyIndex
from environment.task_construction.ir import ProgramIR, ProgramNode
from environment.task_construction.quality.degeneracy import find_degenerate_verification
from environment.task_construction.source_preparation import compact_source_summary
from environment.question_generation.semantic_checks import explicit_global_join_contradictions


class V2AnchoringTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = TaxonomyIndex()
        self.catalog = {"predicates": {
            "schema:geo": {"count": 100, "kind": "attribute"},
            "schema:location": {"count": 100, "kind": "relation"},
            "schema:director": {"count": 100, "kind": "relation"},
            "schema:datePublished": {"count": 100, "kind": "attribute"},
        }}
        self.bank = SemanticAnchorBank({"profiles": [
            {"source_subject": "person-1", "entity_types": ["schema:Person"],
             "observed_predicates": ["schema:director", "schema:datePublished"]},
            {"source_subject": "place-1", "entity_types": ["schema:Place"],
             "observed_predicates": ["schema:geo", "schema:location"]},
        ]}, self.catalog, self.taxonomy)
        self.program = ProgramIR([
            ProgramNode(0, "FIND", (), ("type:schema:Person",)),
            ProgramNode(1, "FILTER_STRING", (0,),
                        ("<ATTR>", "<CMP:=>", "<SYMBOL>")),
            ProgramNode(2, "RELATE", (1,), ("<REL>", "forward")),
            ProgramNode(3, "QUERY_NAME", (2,), ()),
        ])

    def test_balancing_selects_underrepresented_yago_subject_type(self):
        usage = {
            "anchor_type:schema:Place": 12,
            "schema:geo": 12, "schema:location": 12,
        }
        a = self.bank.sample(self.program, rng=random.Random(17),
                             usage=usage, balanced=True)
        self.assertIn("schema:Person", a["types"])

    def test_original_sampling_still_available(self):
        a = self.bank.sample(self.program, rng=random.Random(17))
        self.assertIn("predicates", a)
        self.assertIn("types", a)


class V2DegeneracyTests(unittest.TestCase):
    def program(self, verify_cmp, filter_value="1915", attr="schema:startDate"):
        return ProgramIR([
            ProgramNode(0, "FIND", (), ("type:schema:Place",)),
            ProgramNode(1, "FILTER_YEAR", (0,),
                        ("schema:startDate", "<CMP:=>", filter_value)),
            ProgramNode(2, "QUERY_ATTR", (1,), (attr,)),
            ProgramNode(3, "VERIFY_YEAR", (2,), (verify_cmp, "1915")),
        ])

    def test_uploaded_self_negating_example(self):
        p = self.program("<CMP:!=>")
        self.assertEqual(find_degenerate_verification(p),
                         "verification_negates_upstream_equal_filter")

    def test_redundant_verification(self):
        self.assertEqual(
            find_degenerate_verification(self.program("<CMP:=>")),
            "verification_repeats_upstream_equal_filter",
        )

    def test_nonidentical_attribute_and_value_are_not_flagged(self):
        self.assertIsNone(find_degenerate_verification(
            self.program("<CMP:!=>", filter_value="1916")))
        self.assertIsNone(find_degenerate_verification(
            self.program("<CMP:!=>", attr="schema:endDate")))


class V2ReviewerGuards(unittest.TestCase):
    def plan(self, mixed=False):
        a = {"definition": "FILTER_NUMBER", "inputs": []}
        b = {"definition": "FILTER_NUMBER", "inputs": []}
        left = {"definition": "INTERSECT", "inputs": [a, b]}
        if mixed:
            left = {"definition": "UNION", "inputs": [left, a]}
        return SimpleNamespace(query={"definition": "COUNT", "inputs": [left]})

    def test_uploaded_and_to_at_least_one_or_case(self):
        question = ("How many people satisfy at least one of the following "
                    "criteria: a mass different from 62.45 kg or a height "
                    "greater than 1.6559 m?")
        self.assertIn("all_branch_intersection_rewritten_as_any_criterion",
                      explicit_global_join_contradictions(self.plan(), question))

    def test_normal_both_is_not_rejected(self):
        question = "How many people have a mass other than 62.45 kg and height over 1.6559 m?"
        self.assertEqual(explicit_global_join_contradictions(self.plan(), question), [])

    def test_mixed_union_is_not_rejected_by_high_precision_rule(self):
        question = "How many people satisfy at least one of the following criteria: A or B?"
        self.assertEqual(explicit_global_join_contradictions(self.plan(mixed=True), question), [])


class V2SourceSummaryTests(unittest.TestCase):
    def test_large_inline_catalog_is_not_carried_into_build_summary(self):
        report = {
            "artifacts": {"yago_catalog": "environment/data/yago_predicate_catalog.json"},
            "yago_catalog": {
                "total_triples": 1234, "predicate_count": 12,
                "predicates": {"schema:geo": {"samples": ["x"] * 200}},
                "graph_witness_support": {"motifs": ["large"] * 200},
            },
            "yago_profiles": {"profile_count": 50},
        }
        result = compact_source_summary(report)
        self.assertEqual(result["yago_catalog"]["predicate_count"], 12)
        self.assertEqual(result["yago_profiles"]["profile_count"], 50)
        self.assertNotIn("predicates", result["yago_catalog"])
        self.assertNotIn("graph_witness_support", result["yago_catalog"])
        self.assertEqual(result["yago_catalog"]["output"],
                         "environment/data/yago_predicate_catalog.json")


if __name__ == "__main__":
    unittest.main()
