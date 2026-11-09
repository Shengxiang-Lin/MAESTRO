"""Goal invariant and relation-aware Semantic World execution regressions.

Critical distinctions:
  same source record != same entity;
  same category != same entity;
  explicit directed reference != same entity or category;
  relative chronology != calendar date.
All results derive only from simulated, source-cited World facts.
"""
from __future__ import annotations

import copy
import unittest

from test_taskfirst import KQA, scene
from test_semantic_world import proposal_for, pass_review
from envir.macro.builder import build_one
from envir.macro.goal_invariants import (
    GOAL_RULES, goal_rules, goal_text_issues, question_surface_issues,
)
from envir.macro.relational_world import relational_layout, world_relations
from envir.macro.semantic_world import (
    materialize_spec, compile_semantic_world_scene, validate_semantic_world_scene,
)
from envir.macro.validation import validate_clean_scene
from envir.macro.cli import parser


def compiled_for(intent: str):
    for n in range(100):
        plan = build_one(ordinal=n, seed=17, kqa=KQA, seed_record=scene(n))
        if plan["goal"]["intent"] == intent:
            return plan
    raise AssertionError("No " + intent + " fixture found")


class RelationalWorldTests(unittest.TestCase):
    def test_approved_goal_rules_are_frozen_not_model_authored(self):
        for intent in GOAL_RULES:
            plan = compiled_for(intent)
            spec = materialize_spec(plan, proposal_for(plan), pass_review())
            self.assertEqual(spec["goal_contract"]["execution_rules"],
                             goal_rules(intent))
            rules = spec["goal_contract"]["execution_rules"]
            if intent in {"comparison", "prioritization"}:
                self.assertEqual(rules["order"],
                                 "score_descending_then_candidate_id")
            if intent == "feasibility":
                self.assertEqual(rules["selection"]["combine"], "and")
                self.assertEqual(rules["selection"]["score_gte"], 50)
            if intent == "intersection":
                self.assertIn("NOT_identity", rules["relation"])
            if intent == "composition":
                self.assertIn("directed", rules["relation"])
            changed = copy.deepcopy(spec)
            changed["goal_contract"]["execution_rules"]["order"] = "ascending"
            from envir.macro.semantic_world import validate_spec
            self.assertIn("semantic_world_goal_drift",
                          validate_spec(plan, changed))

    def test_intersection_is_category_overlap_across_distinct_sources(self):
        base = compiled_for("intersection")
        spec = materialize_spec(base, proposal_for(base), pass_review())
        scene_new = compile_semantic_world_scene(base, spec)
        self.assertEqual([], validate_clean_scene(scene_new))
        world = scene_new["world"]
        for edge in world["relations"]:
            self.assertEqual(len(edge["source_ids"]), 2)
            self.assertTrue(set(edge["source_ids"]) <= set(world["sources"]))
            for ref in edge["fact_refs"]:
                sid, fid = ref.split(":", 1)
                self.assertIn(sid, world["sources"])
                self.assertIn(fid, [
                    row["fact_id"] for row in world["sources"][sid]["facts"]])
        terminal = base["macro_graph"]["terminal_handoffs"][0]
        final = scene_new["feedback_bank"][terminal["from_task"]][
            "answer"][terminal["output_slot"]]["operation_result"]
        self.assertEqual(final["matched_groups"], {
            group: sorted(row["candidate_key"] for row in records)
            for group, records in self._category_rows(world).items()
            if len({row["source_id"] for row in records}) >= 2
        })
        self.assertEqual(final["match_policy"],
                         "category_overlap_across_distinct_sources_not_entity_identity")
        self.assertEqual([], validate_semantic_world_scene(scene_new))

    @staticmethod
    def _category_rows(world):
        grouped = {}
        for sid, source in world["sources"].items():
            facts = {x["field"]: x["value"] for x in source["facts"]}
            grouped.setdefault(facts["synthetic_group_key"], []).append({
                "source_id": sid, "candidate_key": facts["candidate_key"]})
        return grouped

    def test_explicit_reference_links_have_source_and_fact_witnesses(self):
        base = compiled_for("composition")
        spec = materialize_spec(base, proposal_for(base), pass_review())
        new = compile_semantic_world_scene(base, spec)
        terminal = base["macro_graph"]["terminal_handoffs"][0]
        report = new["feedback_bank"][terminal["from_task"]][
            "answer"][terminal["output_slot"]]["operation_result"]
        self.assertGreaterEqual(len(report["reference_links"]), 1)
        self.assertEqual(
            report["linkage_policy"],
            "directed_reference_links_do_not_imply_shared_entity")
        self.assertEqual(
            len(report["reference_links"]),
            len([edge for edge in new["world"]["relations"]
                 if edge["kind"] == "references"]))
        for link in report["reference_links"]:
            self.assertNotEqual(link["from_source_id"], link["to_source_id"])
            self.assertIn(link["from_source_id"], new["world"]["sources"])
            self.assertIn(link["to_source_id"], new["world"]["sources"])
        self.assertEqual([], validate_clean_scene(new))

    def test_temporal_relations_are_forward_and_replayable(self):
        base = compiled_for("temporal")
        spec = materialize_spec(base, proposal_for(base), pass_review())
        new = compile_semantic_world_scene(base, spec)
        terminal = base["macro_graph"]["terminal_handoffs"][0]
        report = new["feedback_bank"][terminal["from_task"]][
            "answer"][terminal["output_slot"]]["operation_result"]
        self.assertIn("explicit_precedence_links", report)
        self.assertEqual(len(report["ordered_candidates"]),
                         len(new["world"]["sources"]))
        for edge in new["world"]["relations"]:
            if edge["kind"] != "precedes":
                continue
            before, after = edge["source_ids"]
            x = {f["field"]: f["value"] for f in
                 new["world"]["sources"][before]["facts"]}
            y = {f["field"]: f["value"] for f in
                 new["world"]["sources"][after]["facts"]}
            self.assertLess(x["synthetic_time_index"],
                            y["synthetic_time_index"])
        changed = copy.deepcopy(new)
        changed["world"]["relations"][0]["kind"] = "invented_relation"
        self.assertIn("semantic_world_relation_graph_drift",
                      validate_clean_scene(changed))

    def test_empty_intersection_is_a_valid_result(self):
        plan = compiled_for("intersection")
        spec = materialize_spec(plan, proposal_for(plan), pass_review())
        ids = sorted(x["source_id"] for x in spec["source_bindings"])
        found_empty = False
        for n in range(40):
            data = relational_layout(
                spec=spec, case_ref="case-" + str(n),
                source_ids=ids, goal_id="intersection")
            if len({row["group_key"] for row in data.values()}) == len(ids):
                found_empty = True
                break
        self.assertTrue(found_empty, "generator should cover no-match cases")

    def test_question_guards_filter_internal_names_and_goal_drift(self):
        self.assertIn(
            "question_exposes_internal_schema_or_artifact",
            question_surface_issues(
                "Please generate the PriorityReport for the archive entries."))
        self.assertIn(
            "question_exposes_internal_schema_or_artifact",
            question_surface_issues(
                "Sort by the synthetic relative_time_index value."))
        self.assertIn(
            "question_asserts_unobserved_media",
            question_surface_issues(
                "Which image shows a verified winner holding a trophy?"))
        self.assertIn(
            "goal_order_direction_changed",
            goal_text_issues("prioritization", "Rank entries in ascending order."))
        self.assertIn(
            "group_match_falsely_equated_with_entity_identity",
            goal_text_issues(
                "intersection", "Find the same players in both sources."))
        self.assertEqual(
            [], goal_text_issues(
                "intersection", "Which groups recur across source records?"))
        self.assertEqual(
            [], goal_text_issues(
                "feasibility", "Which entries meet both criteria, if any?"))

    def test_obsolete_grounding_cli_is_removed(self):
        cli = parser()
        try:
            with self.assertRaises(SystemExit):
                cli.parse_args(["ground-scenes"])
        except AssertionError:
            raise
        parsed = cli.parse_args([
            "realize-world", "--scenes", "input.jsonl",
            "--output-dir", "semantic-output",
            "--world-api-base", "http://127.0.0.1:8807/v1",
            "--world-model", "local-model",
        ])
        self.assertEqual(parsed.command, "realize-world")


if __name__ == "__main__":
    unittest.main()
