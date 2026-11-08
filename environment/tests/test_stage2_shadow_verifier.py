"""No-network regressions for the Stage 2B shadow verifier."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from environment.question_generation.semantic_equivalence import compare_graphs
from environment.question_generation.shadow_audit import collect_candidates
from environment.question_generation.shadow_verifier import verify_candidate


def graph(*scopes):
    return {"scopes": list(scopes), "answer_scope": scopes[-1]["id"]}


def scope(sid, operation, *inputs, **fields):
    return {"id": sid, "operation": operation, "inputs": list(inputs), **fields}


def number(comparison, value, attribute="mass"):
    return {
        "operation": "FILTER_NUMBER", "attribute": attribute,
        "comparison": comparison, "public_value": value,
    }


def event_path():
    return graph(
        scope("person", "FIND", identify_one_by_its_conditions="Person", conditions=[{
            "operation": "FILTER_YEAR", "attribute": "birth date",
            "comparison": "equal to", "public_value": "1675",
        }]),
        scope("related", "RELATE", "person", relation="organizer", subject="related", object="person"),
        scope("actor", "RELATE", "related", relation="plays in", subject="actor", object="related"),
        scope("event", "RELATE", "actor", relation="participant", subject="event", object="actor",
              conditions=[{
                  "operation": "FILTER_STRING", "attribute": "slogan",
                  "comparison": "equal to", "public_value": "Serving Communities",
              }]),
        scope("answer", "COUNT", "event", distinct=True),
    )


class ShadowEquivalenceTests(unittest.TestCase):
    def test_serial_and_same_range_intersection_match_flat_conjunction(self):
        lo, hi = number(">", "2"), number("<", "5.0")
        flat = graph(scope("x", "FIND_ALL", conditions=[hi, lo]),
                     scope("answer", "COUNT", "x", distinct=True))
        serial = graph(scope("base", "FIND_ALL", conditions=[lo]),
                       scope("filtered", "FILTER_NUMBER", "base", conditions=[hi]),
                       scope("return", "COUNT", "filtered", distinct=True))
        intersection = graph(
            scope("left", "FIND_ALL", conditions=[lo]),
            scope("right", "FIND_ALL", conditions=[hi]),
            scope("and", "INTERSECT", "right", "left"),
            scope("answer", "COUNT", "and", distinct=True),
        )
        self.assertEqual(compare_graphs(flat, serial)["status"], "match")
        self.assertEqual(compare_graphs(flat, intersection)["status"], "match")
        intersection["scopes"][-2]["operation"] = "UNION"
        self.assertEqual(compare_graphs(flat, intersection)["status"], "mismatch")
        intersection["scopes"][-2]["operation"] = "INTERSECT"
        intersection["scopes"][0].update(operation="FIND", public_anchor="Named entity")
        self.assertEqual(compare_graphs(flat, intersection)["status"], "mismatch")

    def test_relation_path_direction_owner_and_result_change_are_visible(self):
        gold = event_path()
        for mutation in ("collapse", "reverse", "move_condition", "names"):
            with self.subTest(mutation=mutation):
                observed = deepcopy(gold)
                scopes = observed["scopes"]
                if mutation == "collapse":
                    scopes[1:3] = []
                    scopes[1].update(inputs=["person"], relation="organizer", object="person")
                elif mutation == "reverse":
                    scopes[2].update(subject="related", object="actor")
                elif mutation == "move_condition":
                    scopes[1]["conditions"] = scopes[3].pop("conditions")
                else:
                    scopes[-1].update(operation="QUERY_NAME")
                    del scopes[-1]["distinct"]
                self.assertEqual(compare_graphs(gold, observed)["status"], "mismatch")

    def test_universe_condition_can_apply_directly_to_relation_endpoint(self):
        direct = event_path()
        intersection = deepcopy(direct)
        scopes = intersection["scopes"]
        condition = scopes[3].pop("conditions")
        scopes[4:4] = [
            scope("slogans", "FIND_ALL", conditions=condition),
            scope("joined", "INTERSECT", "slogans", "event"),
        ]
        scopes[-1]["inputs"] = ["joined"]
        self.assertEqual(compare_graphs(intersection, direct)["status"], "match")

    def test_numeric_direction_and_string_code_stay_distinct(self):
        for atom, changed in (
            (number("<", "5,174"), number("at most", "5174")),
            (number("equal to", "1190"), number("different from", "1190")),
            ({
                "operation": "FILTER_STRING", "attribute": "postal code",
                "comparison": "equal to", "public_value": "00123",
            }, {
                "operation": "FILTER_STRING", "attribute": "postal code",
                "comparison": "equal to", "public_value": "123",
            }),
        ):
            with self.subTest(atom=atom):
                a = graph(scope("x", "FIND_ALL", conditions=[atom]), scope("o", "QUERY_NAME", "x"))
                b = graph(scope("x", "FIND_ALL", conditions=[changed]), scope("o", "QUERY_NAME", "x"))
                self.assertEqual(compare_graphs(a, b)["status"], "mismatch")

    def test_rank_across_union_matches_explicit_union(self):
        a = scope("a", "FIND", public_anchor="A")
        b = scope("b", "FIND", public_anchor="B")
        rank = {"ranking": "largest", "rank_attribute": "mass", "include_ties": True}
        compact = graph(a, b, scope("r", "SELECT_BETWEEN", "a", "b", **rank))
        explicit = graph(a, b, scope("union", "UNION", "b", "a"),
                         scope("r", "SELECT_AMONG", "union", **rank))
        self.assertEqual(compare_graphs(compact, explicit)["status"], "match")

    def test_unbound_endpoints_and_cycles_abstain(self):
        gold = event_path()
        wrong = deepcopy(gold)
        wrong["scopes"][2]["subject"] = "unbound"
        self.assertEqual(compare_graphs(gold, wrong)["status"], "unresolved")
        wrong = deepcopy(gold)
        wrong["scopes"][2]["inputs"] = ["answer"]
        self.assertEqual(compare_graphs(gold, wrong)["status"], "unresolved")

    def test_compact_parser_result_preserves_directed_chain(self):
        class Backend:
            def complete_json(self, system, payload, **kwargs):
                def step(op, inputs, args, conditions=()):
                    return dict(op=op, inputs=inputs, args=args, conditions=list(conditions))

                return {"unresolved": [], "steps": [
                    step("ONE", [], ["Person"],
                         [{"kind": "year", "args": ["birth date", "=", "1675"]}]),
                    step("EDGE", [0], ["backward", "organizer"]),
                    step("EDGE", [1], ["backward", "plays in"]),
                    step("EDGE", [2], ["backward", "participant"], [
                        {"kind": "text", "args": ["slogan", "=", "Serving Communities"]},
                    ]),
                    step("COUNT", [3], []),
                ]}

        verdict = verify_candidate(Backend(), "candidate", event_path())
        self.assertEqual(verdict["status"], "match")
        self.assertEqual(verdict["parsed_graph"]["scopes"][3]["subject"], "3")

    def test_parser_receives_only_candidate_and_can_abstain_on_shared_value(self):
        gold = event_path()

        class Backend:
            def complete_json(self, system, payload, **kwargs):
                self.payload = json.loads(payload)
                return {
                    "steps": [],
                    "unresolved": ["requires one recorded value to meet both comparisons"],
                }

        backend = Backend()
        question = "How many events have a recorded value strictly between 2 and 5?"
        result = verify_candidate(backend, question, gold)
        self.assertEqual(backend.payload, {"candidate_question": question})
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["reason"], "parser_uncertainty")


class ShadowAuditTests(unittest.TestCase):
    def test_deduplicate_drafts_preserving_audit_provenance_and_labels(self):
        row = {
            "task_id": "task",
            "audit": [{"stage": "reference", "scoped_question_sha256": "hash"}] + [
                {"stage": "candidate", "question": question, "attempt": i + 1}
                for i, question in enumerate(["first", "middle", "first"])
            ],
        }
        case = {"task_id": "task", "question": "first", "label": "equivalent", "category": "AND"}
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.jsonl"
            labels = Path(directory) / "labels.jsonl"
            audit.write_text(json.dumps(row) + "\n")
            labels.write_text(json.dumps(case) + "\n")
            candidates = collect_candidates([audit], [labels])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["label"], "equivalent")
        self.assertEqual([x["attempt"] for x in candidates[0]["observations"]], [1, 3])


if __name__ == "__main__":
    unittest.main()
