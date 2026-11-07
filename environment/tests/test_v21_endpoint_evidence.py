"""Evidence grounding, local endpoints and the v20 ownership false acceptance."""
import copy
import json
import unittest

from environment.question_generation.coverage_review import (
    compile_coverage_probes, coverage_manifest, validate_coverage_manifest,
)
from environment.question_generation.scoped_surface import ScopedPlan
from environment.question_generation.semantic_probes import (
    probe_candidate_semantics, explicit_probe_contradictions,
)
from environment.tests.test_coverage_aware_certification import node, scripted_item


class EndpointEvidenceTests(unittest.TestCase):
    def test_intersection_cannot_be_certified_by_listing_two_entity_sets(self):
        probe = {"id": "P1", "kind": "group_join", "group": "second", "expected": "AND"}
        bad = (
            "Let the second group consist of entities in the first group with "
            "an elevation greater than 424.5 m, as well as administrative areas "
            "in the first group with a start date greater than 1865.",
            "The second group consists of entities recorded as locations of "
            "organizations with a start date less than 1962 AND entities "
            "recorded as locations of organizations with a lei code of XYZ.",
        )
        for text in bad:
            with self.subTest(text=text):
                self.assertEqual(explicit_probe_contradictions([probe], text),
                                 ["P1:intersection_replaced_by_entity_list"])

    def test_entity_list_guard_preserves_explicit_intersections_and_attribute_joins(self):
        probe = {"id": "P1", "kind": "group_join", "group": "second", "expected": "AND"}
        good = (
            "Let the second group consist of entities in both entities with "
            "elevation greater than 424.5 m and administrative areas with "
            "a start date greater than 1865.",
            "The second group consists of entities that have an elevation "
            "greater than 424.5 m as well as a start date greater than 1865.",
            "The second group consists of entities that have elevation "
            "greater than 424.5 m AND a start date greater than 1865.",
            "The second group consists of each entity satisfying the location "
            "condition AND entities qualifying through the second condition.",
        )
        for text in good:
            with self.subTest(text=text):
                self.assertEqual(explicit_probe_contradictions([probe], text), [])
        wrong = ("The second group consists of entities with elevation greater "
                 "than 424.5 m as well as administrative areas with start date "
                 "greater than 1865.")
        self.assertEqual(explicit_probe_contradictions([{**probe, "expected": "OR"}], wrong), [])

    def condition(self, attribute, value, comparison="less than", kind="number"):
        return {"attribute": attribute, "comparison": comparison,
                "value": {"type": kind, "value": value}}

    def owner_case(self, relation):
        base = node("FIND_ALL", conditions=[self.condition("male population", "489")])
        source = node("FILTER_NUMBER", base, conditions=[
            self.condition("elevation", "441", "different from")])
        edge = node("RELATE", source, relation=relation, direction="backward")
        result = node("FILTER_NUMBER", base, conditions=[
            self.condition("population number", "1770")])
        joined = node("INTERSECT", edge, result)
        return ScopedPlan([], [], node("QUERY_NAME", joined))

    def test_all_relation_names_have_both_endpoint_owner_obligations(self):
        for relation in ("neighbor", "location", "parent taxon", "winner", "consumes"):
            with self.subTest(relation=relation):
                probes = compile_coverage_probes(self.owner_case(relation), "", "")
                owners = {(p["attribute"], p["expected"]) for p in probes
                          if p["kind"] == "condition_owner"}
                self.assertIn(("elevation", "A"), owners)
                self.assertIn(("population number", "B"), owners)

    def test_rank_returns_entities_and_requires_argmax_attribute_and_ties(self):
        query = node("SELECT_BETWEEN", node("FIND_ALL"), node("FIND_ALL"))
        query.update(ranking="largest", rank_attribute="female population")
        probes = compile_coverage_probes(ScopedPlan([], [], query), "", "")
        self.assertEqual({p["kind"]: p["expected"] for p in probes}, {
            "answer_type": "ENTITIES", "selection_operation": "ARGMAX",
            "ranking_attribute": "female population", "ranking_ties": "INCLUDE",
        })

    def test_explicit_related_attachment_rejects_even_if_model_labels_match(self):
        for relation in ("neighbor", "location", "parent taxon"):
            for direction in ("forward", "backward"):
                plan = self.owner_case(relation)
                plan.query["inputs"][0]["inputs"][0]["direction"] = direction
                prefix = (f"entities whose recorded '{relation}' points to "
                          if direction == "backward" else
                          f"entities listed as targets of the '{relation}' relation from ")
                wrong = (prefix + "a member of the first group that has an elevation "
                         "different from 441 m AND a population number less than 1,770")
                probes = compile_coverage_probes(plan, "", wrong)
                population = next(p for p in probes if p.get("attribute") == "population number")
                self.assertEqual(explicit_probe_contradictions([population], wrong),
                                 [f"{population['id']}:result_filter_moved_to_input_endpoint"])
                correct = (prefix + "a member of the first group that has an elevation "
                           "different from 441 m, and are themselves entities with "
                           "a population number less than 1,770")
                self.assertEqual(explicit_probe_contradictions([population], correct), [])
                other_value = wrong.replace("1,770", "1,771")
                self.assertEqual(explicit_probe_contradictions([population], other_value), [])

    def test_owner_quote_cannot_hide_attachment_inside_full_relative_clause(self):
        candidate = ("entities whose recorded neighbor points to a member of the first "
                     "group that has elevation different from 441 AND population "
                     "number less than 1,770")
        probe = next(p for p in compile_coverage_probes(self.owner_case("neighbor"), "", candidate)
                     if p.get("attribute") == "population number")
        class Reader:
            def complete_json(self, *args, **kwargs):
                return {"answers": [{"id": probe["id"], "answer": "B",
                                     "owner": candidate, "evidence": candidate}]}
        self.assertEqual(probe_candidate_semantics(Reader(), "", candidate, [probe])[0]["answer"],
                         "UNCLEAR")

    def test_ranking_attribute_normalizes_operation_words_without_changing_attribute(self):
        candidate = "Which entities have the largest recorded female population? Include ties."
        probe = {"id": "P1", "kind": "ranking_attribute", "expected": "female population",
                 "question": "Which attribute is ranked?", "evidence_required": True}
        class Reader:
            def __init__(self, label): self.label = label
            def complete_json(self, *args, **kwargs):
                return {"answers": [{"id": "P1", "answer": self.label, "evidence": candidate}]}
        self.assertEqual(probe_candidate_semantics(Reader("largest recorded female population"),
                                                  "", candidate, [probe])[0]["answer"],
                         "female population")
        self.assertEqual(probe_candidate_semantics(Reader("male population"), "", candidate,
                                                  [probe])[0]["answer"], "UNCLEAR")

    def test_v20_wrong_neighbor_owner_is_not_rescued_by_correct_and_direction(self):
        candidate = ("entities whose recorded neighbor points to a member of the first "
                     "group that has elevation different from 441 AND population "
                     "number less than 1,770")
        probes = compile_coverage_probes(self.owner_case("neighbor"), "", candidate)
        records = []
        for p in probes:
            # Script the grammatical error observed by the reader; expected
            # labels are used only for unrelated fixture obligations.
            answer = ("A" if p.get("attribute") == "population number" else p["expected"])
            item = scripted_item(p["id"], answer, candidate, subject="entities",
                                 object="a member of the first group",
                                 subject_role="B", object_role="A",
                                 owner="a member of the first group")
            records.append({**item, "kind": p["kind"], "expected": p["expected"]})
        self.assertEqual(coverage_manifest(probes, records)["status"], "incomplete")
        with self.assertRaises(ValueError):
            validate_coverage_manifest(probes, records, coverage_manifest(probes, records),
                                       candidate=candidate)

    def test_relation_evidence_and_role_integrity_fail_closed(self):
        candidate = "cities listed as targets of named after from airport"
        plan = ScopedPlan([], [], node("QUERY_NAME", node(
            "RELATE", node("FIND", conditions=[]), relation="named after", direction="forward")))
        probe = next(p for p in compile_coverage_probes(plan, "", candidate)
                     if p["kind"] == "relation_path")
        good = scripted_item(probe["id"], "A_TO_B", candidate, subject="airport",
                             object="cities", subject_role="A", object_role="B")
        class Reader:
            def __init__(self, item): self.item = item
            def complete_json(self, system, user, **kwargs):
                self.request = json.loads(user)
                assert '"expected"' not in user
                return {"answers": [self.item]}
        result = probe_candidate_semantics(Reader(good), "PRIVATE REFERENCE", candidate, [probe])
        self.assertEqual(result[0]["answer"], "A_TO_B")
        for changes in ({"evidence": "invented clause"}, {"subject": "invented entity"},
                        {"subject_role": "B", "object_role": "A"},
                        {"evidence": "named after"}, {"object": "airport"}):
            with self.subTest(changes=changes):
                bad = {**good, **changes}
                result = probe_candidate_semantics(Reader(bad), "", candidate, [probe])
                self.assertEqual(result[0]["answer"], "UNCLEAR")

    def test_schema_separates_owner_and_relation_and_never_supplies_expected(self):
        from environment.question_generation.semantic_probes import _evidence_schema
        probes = compile_coverage_probes(self.owner_case("neighbor"), "", "")
        schema = _evidence_schema(probes)
        alternatives = schema["properties"]["answers"]["items"]["anyOf"]
        for p, item in zip(probes, alternatives, strict=True):
            self.assertEqual(item["properties"]["id"]["const"], p["id"])
            self.assertNotIn("expected", item["properties"])
            if p["kind"] == "condition_owner":
                self.assertIn("owner", item["required"])
                self.assertNotIn("subject", item["properties"])
                self.assertEqual(item["properties"]["answer"]["enum"], ["A", "B", "UNCLEAR"])
            if p["kind"] == "relation_path":
                self.assertIn("subject_role", item["required"])
                self.assertNotIn("owner", item["properties"])

    def test_replay_rejects_changed_quotes_even_when_labels_match(self):
        candidate = "Which cities are targets of named after from airport?"
        query = node("QUERY_NAME", node("RELATE", node("FIND_ALL"),
                     relation="named after", direction="forward"))
        probes = compile_coverage_probes(ScopedPlan([], [], query), "", candidate)
        records = [{
            **scripted_item(p["id"], p["expected"], candidate, subject="airport",
                            object="cities", subject_role="A", object_role="B"),
            "kind": p["kind"], "expected": p["expected"],
        } for p in probes]
        validate_coverage_manifest(probes, records, coverage_manifest(probes, records),
                                   candidate=candidate)
        bad = copy.deepcopy(records)
        bad[0]["evidence"] = "fabricated witness"
        with self.assertRaises(ValueError):
            validate_coverage_manifest(probes, bad, coverage_manifest(probes, bad),
                                       candidate=candidate)

    def test_numeric_anchor_uses_prose_and_string_filters_are_owned(self):
        base = node("FIND_ALL", conditions=[
            self.condition("population number", "1.19E+3"),
            self.condition("iata code", "BQG", "equal to", "string")])
        query = node("QUERY_NAME", node("RELATE", base, relation="terminus",
                                      direction="backward"))
        probes = compile_coverage_probes(ScopedPlan([], [], query), "", "")
        self.assertTrue(any("1,190" in p["question"] for p in probes))
        self.assertFalse(any("1.19E+3" in p["question"] for p in probes))
        self.assertIn(("iata code", "A"), [
            (p["attribute"], p["expected"]) for p in probes
            if p["kind"] == "condition_owner"])

    def test_disputed_probe_has_one_recheck_and_preserves_first_observation(self):
        from environment.tests.test_stage2_surface_certification import (
            FakeAuthor, FakeCritic, Stage2SurfaceCertificationTests)
        from environment.question_generation.realization import _realize_one
        class Reader(FakeCritic):
            def complete_json(self, system, user, **kwargs):
                response = super().complete_json(system, user, **kwargs)
                if ("independent reader of candidate text only" in system
                        and "ONE focused rereading" not in system):
                    for row in response["answers"]:
                        if row["answer"] == "A_TO_B":
                            row.update(answer="B_TO_A", subject_role="B", object_role="A")
                return response
        payload, world, capability, _ = Stage2SurfaceCertificationTests().data()
        ok, row, audit = _realize_one(payload, world, capability, FakeAuthor(),
                                    Reader(), max_revisions=0)
        self.assertTrue(ok, str(row))
        self.assertEqual(audit[-1]["candidate_probe_recheck_calls"], 1)
        self.assertTrue(any(r["answer"] != r["expected"]
                            for r in audit[-1]["candidate_probe_initial_answers"]))
        self.assertTrue(all(r["answer"] == r["expected"]
                            for r in audit[-1]["candidate_probe_answers"]))

    def test_unclear_recheck_cannot_loop_or_be_voted_into_acceptance(self):
        from environment.tests.test_stage2_surface_certification import (
            FakeAuthor, FakeCritic, Stage2SurfaceCertificationTests)
        from environment.question_generation.realization import _realize_one
        class Reader(FakeCritic):
            def complete_json(self, system, user, **kwargs):
                response = super().complete_json(system, user, **kwargs)
                if "independent reader of candidate text only" in system:
                    for row in response["answers"]:
                        if row["answer"] == "A_TO_B":
                            row["answer"] = "UNCLEAR"
                return response
        payload, world, capability, _ = Stage2SurfaceCertificationTests().data()
        ok, row, audit = _realize_one(payload, world, capability, FakeAuthor(),
                                    Reader(), max_revisions=0)
        self.assertFalse(ok)
        self.assertEqual(row["reason"], "rejected_uncertain")
        self.assertEqual(audit[-1]["candidate_probe_recheck_calls"], 1)
        self.assertEqual(audit[-1]["final_judge_calls"], 0)
