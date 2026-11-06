"""Stage 2 public-program realization and semantic certification regressions."""
from __future__ import annotations

import json
import unittest
from unittest.mock import Mock,patch

from environment.question_generation.public_semantics import (
    public_semantic_scaffold, naturalization_metrics,
)
from environment.question_generation.prompts import author_prompt,critic_prompt
from environment.question_generation.pipeline import _realize_one
from environment.tests.test_stage2_surface_certification import (
    Stage2SurfaceCertificationTests,
)
from environment.trajectory_simulation.eligibility import digest


class PublicScaffoldTests(unittest.TestCase):
    def test_all_hops_dependencies_direction_join_and_terminal_are_explicit(self):
        world={"program":{"nodes":[
            {"node_id":0,"op":"FIND","inputs":["PRIVATE_ENTITY_CANARY"],
             "dependencies":[]},
            {"node_id":1,"op":"FILTER_YEAR",
             "inputs":["schema:birthDate","<","1840"],
             "dependencies":[0]},
            {"node_id":2,"op":"RELATE",
             "inputs":["schema:winner","backward"],"dependencies":[1]},
            {"node_id":3,"op":"FIND_ALL","inputs":[],"dependencies":[]},
            {"node_id":4,"op":"INTERSECT","inputs":[],"dependencies":[2,3]},
            {"node_id":5,"op":"QUERY_NAME","inputs":[],"dependencies":[4]},
        ]},"world":{"facts":[{"object":"PRIVATE_ANSWER_CANARY"}]}}
        canonical=(
            "Identify the entity names shared by the subjects whose winner "
            "relation points to a Person whose birth date is less than 1840 "
            "and the candidate entities discovered in the task sources?"
        )
        graph=public_semantic_scaffold(world,canonical)
        self.assertEqual(graph["output"],"S5")
        self.assertEqual(graph["steps"][2]["direction"],"backward")
        self.assertEqual(graph["steps"][2]["relation"],"winner")
        self.assertEqual(graph["steps"][2]["inputs_from"],["S1"])
        self.assertEqual(graph["steps"][4]["inputs_from"],["S2","S3"])
        self.assertIn("ALL",graph["steps"][4]["goal"])
        self.assertEqual(graph["steps"][1]["attribute"],"birth date")
        self.assertEqual(graph["steps"][1]["public_value"],"1840")
        self.assertEqual(graph["steps"][5]["operation"],"QUERY_NAME")
        serialized=json.dumps(graph,ensure_ascii=False)
        self.assertNotIn("PRIVATE_ENTITY_CANARY",serialized)
        self.assertNotIn("PRIVATE_ANSWER_CANARY",serialized)
        payload={"draft_question":canonical,"public_semantic_scaffold":graph}
        self.assertNotIn("PRIVATE",author_prompt(payload))
        self.assertNotIn("PRIVATE",critic_prompt(payload,canonical))

    def test_hidden_filter_value_not_disclosed_even_if_programir_contains_it(self):
        world={"program":{"nodes":[
            {"node_id":0,"op":"FIND_ALL","inputs":[],"dependencies":[]},
            {"node_id":1,"op":"FILTER_STRING",
             "inputs":["schema:address","=","SECRET_VALUE_CANARY"],
             "dependencies":[0]},
            {"node_id":2,"op":"COUNT","inputs":[],"dependencies":[1]},
        ]}}
        question="How many entities have a specified address in the task references?"
        graph=public_semantic_scaffold(world,question)
        self.assertIsNone(graph["steps"][1]["public_value"])
        self.assertNotIn("SECRET_VALUE_CANARY",str(graph))

    def test_literal_rendering_uses_only_public_text(self):
        world={"program":{"nodes":[
            {"node_id":0,"op":"FIND_ALL","inputs":[],"dependencies":[]},
            {"node_id":1,"op":"FILTER_NUMBER",
             "inputs":["schema:populationNumber","<","1765"],"dependencies":[0]},
            {"node_id":2,"op":"COUNT","inputs":[],"dependencies":[1]},
        ]}}
        graph=public_semantic_scaffold(
            world,"How many candidates have population number less than 1,765?",
        )
        self.assertEqual(graph["steps"][1]["public_value"],"1,765")
        self.assertEqual(graph["steps"][1]["comparison"],"less than")

    def test_trivial_edits_not_counted_as_naturalization(self):
        canonical=("What is the number of entities described by the "
                   "following conditions: entities drawn from candidates "
                   "in entities for which after requiring a value?")
        trivial=canonical.replace("number of entities","number of Event entities")
        natural="How many matching entities satisfy the requested conditions?"
        self.assertFalse(naturalization_metrics(canonical,trivial)["meaningful_rewrite_proxy"])
        self.assertTrue(naturalization_metrics(canonical,natural)["meaningful_rewrite_proxy"])


class GraphFirstTests(unittest.TestCase):
    def setup_case(self):
        return Stage2SurfaceCertificationTests().data()

    def test_full_graph_review_can_accept_a_candidate_without_critic_veto(self):
        payload,world,capability,rendered=self.setup_case()
        candidate=rendered.question+" Return the matching names."
        author=Mock()
        author.author.return_value={"question":candidate}
        critic=Mock()
        review={
            "eligible":True,"reviewed":True,"decision":"eligible",
            "issues":[],"method":"llm_semantic_screen",
            "question_sha256":digest(candidate),
            "program_sha256":digest(world["program"]),
        }
        certificate={"method":"llm_semantic_screen",
                     "question_sha256":digest(candidate)}
        with patch(
            "environment.question_generation.verification.certify_surface",
            return_value=(certificate,review),
        ) as verify:
            ok,row,audit=_realize_one(
                payload,world,capability,author,critic,
                question_max_attempts=1,use_critic=True,
            )
        self.assertTrue(ok)
        self.assertEqual(row["question"],candidate)
        self.assertEqual(verify.call_count,1)
        critic.review.assert_not_called()
        self.assertIn("public_semantic_scaffold",author.author.call_args.args[0])
        self.assertEqual(audit[0]["author_programir_review"],review)

    def test_repair_is_independently_graph_checked(self):
        payload,world,capability,rendered=self.setup_case()
        first=rendered.question+" Return the exact names."
        repair=rendered.question+" Return the matching names."
        author=Mock()
        author.author.return_value={"question":first}
        critic=Mock()
        critic.review.return_value={"accept":True,"semantic_fidelity":"pass",
                                   "issues":[],"repair_question":repair}
        expected_review={
            "eligible":True,"reviewed":True,"decision":"eligible",
            "issues":[],"method":"llm_semantic_screen",
        }
        with patch(
            "environment.question_generation.verification.certify_surface",
            side_effect=[
                (None,{"decision":"mismatch","review_failure_kind":"semantic_mismatch"}),
                ({"method":"llm_semantic_screen"},expected_review),
            ],
        ) as verify:
            ok,row,audit=_realize_one(
                payload,world,capability,author,critic,
                question_max_attempts=1,use_critic=True,
            )
        self.assertTrue(ok)
        self.assertEqual(verify.call_count,2)
        self.assertEqual(row["question"],repair)
        self.assertIn("repair_programir_review",audit[0])
        self.assertEqual(critic.review.call_count,1)


if __name__=="__main__":
    unittest.main()
