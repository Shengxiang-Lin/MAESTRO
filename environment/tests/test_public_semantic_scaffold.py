"""Stage 2 public-program realization and semantic certification regressions."""
from __future__ import annotations

import json
import unittest

from environment.question_generation.public_semantics import (
    public_semantic_scaffold,
)


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



if __name__ == "__main__":
    unittest.main()
