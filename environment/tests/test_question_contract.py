import unittest
from unittest.mock import patch

from environment.question_generation.contracts import contract_issues
from environment.question_generation.public_semantics import public_question_contract
from environment.question_generation.alignment import check_alignment
from environment.question_generation.backend import OpenAICompatibleBackend


class QuestionContractTests(unittest.TestCase):
    def world(self):
        return {"program": {"nodes": [
            {"node_id": 0, "op": "FIND_ALL", "inputs": [], "dependencies": []},
            {"node_id": 1, "op": "FILTER_NUMBER",
             "inputs": ["schema:population", "=", "1203"], "dependencies": [0]},
            {"node_id": 2, "op": "RELATE",
             "inputs": ["schema:neighbor", "backward"], "dependencies": [1]},
            {"node_id": 3, "op": "FILTER_NUMBER",
             "inputs": ["schema:malePopulation", "=", "488"], "dependencies": [2]},
            {"node_id": 4, "op": "COUNT", "inputs": [], "dependencies": [3]},
        ]}}

    def test_numeric_formats_and_whole_date(self):
        payload = {"draft_question": "Count entities with population 1,203, rank 6 and date 1997-08-01?",
                   "terminal_contract": {"required_numbers": ["1", "203", "6", "1997", "08", "01"]}}
        self.assertEqual(contract_issues(payload, "How many have population 1203, rank six and date 1997-08-01?"), [])
        for bad in ("1203 six 1997-01-08", "1203 six 1997 08 01", "203 six 1997-08-01"):
            self.assertIn("public_numeric_constraint_missing", contract_issues(payload, bad))

    def test_date_prose_and_string_code_have_different_normalization(self):
        from environment.question_generation.contracts import normalize_value
        payload = {'draft_question': 'death date 1997-08-01 telephone country code +650',
                   'terminal_contract': {'required_literals': ['1997-08-01', '+650']}}
        self.assertEqual(contract_issues(payload, 'death date August 1, 1997 telephone country code +650'), [])
        self.assertIn('public_literal_constraint_missing',
                      contract_issues(payload, 'death date August 1, 1997 telephone country code 650'))
        self.assertNotEqual(normalize_value('+650', 'FILTER_STRING'),
                            normalize_value('650', 'FILTER_STRING'))
        self.assertNotEqual(normalize_value('00123', 'FILTER_STRING'),
                            normalize_value('123', 'FILTER_STRING'))
        self.assertEqual(normalize_value('1,203', 'FILTER_NUMBER'),
                         normalize_value('1203', 'FILTER_NUMBER'))

    def test_filters_keep_relation_owner(self):
        contract = public_question_contract(
            self.world(), "How many neighbors have male population 488 and point to entities with population 1203?", {"fragments": []},
        )
        scopes = {s["id"]: s for s in contract["scopes"]}
        self.assertEqual(scopes["E0"]["conditions"][0]["public_value"], "1203")
        self.assertEqual(scopes["E2"]["conditions"][0]["public_value"], "488")
        self.assertEqual(scopes["E2"]["direction"], "backward")
        self.assertEqual(scopes["E2"]["inputs"], ["E0"])

    def test_shared_result_does_not_absorb_branch_filter(self):
        world = self.world()
        world["program"]["nodes"].insert(4, {
            "node_id": 5, "op": "QUERY_NAME", "inputs": [], "dependencies": [2],
        })
        contract = public_question_contract(world, "population 1203 male population 488 neighbor", {"fragments": []})
        scopes = {s["id"]: s for s in contract["scopes"]}
        self.assertEqual(scopes["E2"]["conditions"], [])
        self.assertEqual(scopes["E3"]["inputs"], ["E2"])

    def test_alignment_requires_every_obligation_and_valid_evidence(self):
        world = self.world()
        question = "How many neighbors have male population 488 and point to entities with population 1203?"
        client = OpenAICompatibleBackend("test://local", "test")
        items = [{"id": f"S{i}", "status": "match", "spans": [0], "reason": "matches"} for i in range(5)]
        for values, extras, eligible in (
            (items, [], True), (items[:-1], [], False),
            ([*items[:-1], items[0]], [], False),
            ([{**v, "spans": [99]} for v in items], [], False),
            (items, [{"span": 0, "reason": "invented restriction"}], False),
            ([{**v, "status": "uncertain"} for v in items], [], False),
        ):
            with patch.object(OpenAICompatibleBackend, "complete_json", side_effect=[
                {"alignments": values[:4], "extra_constraints": extras},
                {"alignments": values[4:], "extra_constraints": []},
            ]):
                result = check_alignment({"fragments": []}, world, question, question, client)
                self.assertEqual(result["eligible"], eligible)
                schema = OpenAICompatibleBackend.complete_json.call_args.kwargs["response_schema"]
                self.assertEqual(schema["properties"]["alignments"]["minItems"],
                                 schema["properties"]["alignments"]["maxItems"])

    def test_filter_after_ranking_keeps_its_own_scope(self):
        world = self.world()
        world["program"]["nodes"][2] = {
            "node_id": 2, "op": "SELECT_AMONG", "inputs": ["schema:age", "min"],
            "dependencies": [1],
        }
        contract = public_question_contract(world, "smallest age population 1203 male population 488", {"fragments": []})
        scopes = {s["id"]: s for s in contract["scopes"]}
        self.assertEqual(scopes["E2"]["conditions"], [])
        self.assertEqual(scopes["E3"]["inputs"], ["E2"])


if __name__ == "__main__":
    unittest.main()
