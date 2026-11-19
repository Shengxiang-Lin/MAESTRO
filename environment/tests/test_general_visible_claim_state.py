"""General Claim–Source–Dependency tests: visible-only, round-wise, conservative."""
from __future__ import annotations

import json
import unittest

from environment.untrusted_orchestration.inspection.visible_claim_state import (
    audit_visible_claim_states,
)
from environment.implement_subtask.visible_evidence import visible_year_claims


def scene(*, known_origins=True):
    def task(name, sources, parents, fields):
        return {
            "task_id": name, "source_ids": sources,
            "depends_on": parents,
            "outputs": {"typed_relation": fields},
        }
    return {
        "task_id": "generic-case",
        "question": "Compare the two source counts for Case-011.",
        "macro_graph": {"tasks": [
            task("a", ["alpha"], [], ["case_id", "completed_count"]),
            task("b", ["beta"], [], ["case_id", "completed_count"]),
            task("join", [], ["a", "b"], ["case_id", "completed_count"]),
        ]},
        "world": {"sources": {
            "alpha": ({
                "evidence_origin_id": "source-root-A",
                "rows": [{"completed_count": 9999}],
            } if known_origins else {}),
            "beta": ({
                "evidence_origin_id": "source-root-B",
                "rows": [{"completed_count": 8888}],
            } if known_origins else {}),
        }},
    }


def report(value=None, *, status="done", summary=None):
    result = json.dumps([
        {"values": {"case_id": "Case-011", "completed_count": value}}
    ]) if value is not None else ""
    return {
        "task_instruction": "Read assigned source record",
        "finish_result": (
            {"status": status, "result": result,
             "summary": summary or "Read the assigned source record."}
            if status != "missing" else None
        ),
    }


def replay(entries, world=None):
    world = world or scene()
    trajectory, bindings = [], {}
    for i, (tid, value, status, summary) in enumerate(entries, 1):
        bindings[i] = [tid]
        trajectory.append({
            "attempt": i, "action": "delegate_task",
            "result": {"subtask_results": [
                report(value, status=status, summary=summary)
            ]},
        })
    return audit_visible_claim_states(
        world, trajectory, task_ids_by_round=bindings,
    )


def match_state(round_):
    return next(x for x in round_["claim_states"].values()
                if x["attribute"] == "completed_count")


class GeneralVisibleClaimStateTests(unittest.TestCase):
    def test_two_independent_matching_sources_are_corroborated(self):
        data = replay([
            ("a", 4, "done", None),
            ("b", 4, "done", None),
        ])
        self.assertTrue(data["applicable"])
        self.assertEqual(data["schema"], "claim_source_dependency")
        self.assertEqual(
            match_state(data["rounds"][0])["status"], "unverified",
        )
        latest = data["rounds"][1]
        state = match_state(latest)
        self.assertEqual(state["status"], "corroborated")
        self.assertEqual(state["independent_origins"], 2)
        self.assertTrue(state["sufficient_to_accept"])
        self.assertEqual(latest["main_can_act_from_round"], 3)
        self.assertEqual(len(latest["claim_nodes"]), 2)
        self.assertEqual(
            latest["claim_relations"][0]["relation"], "corroborates",
        )
        self.assertNotIn("9999", json.dumps(data))
        self.assertNotIn("8888", json.dumps(data))
        self.assertNotIn("gold_answer", json.dumps(data))
        self.assertTrue(data["private_only"])

    def test_distinct_delivered_values_are_disputed_not_truth_corrected(self):
        data = replay([
            ("a", 4, "done", None),
            ("b", 7, "done", None),
        ])
        state = match_state(data["rounds"][-1])
        self.assertEqual(state["status"], "disputed")
        self.assertFalse(state["sufficient_to_accept"])
        self.assertEqual(state["evidence_posture"], "reconcile")
        self.assertEqual(
            data["rounds"][-1]["claim_relations"][0]["relation"],
            "contradicts",
        )

    def test_retry_missing_revokes_prior_confirmation(self):
        data = replay([
            ("a", 4, "done", None),
            ("b", 4, "done", None),
            ("b", None, "missing", None),
        ])
        self.assertEqual(
            match_state(data["rounds"][1])["status"], "corroborated",
        )
        self.assertEqual(
            match_state(data["rounds"][2])["status"], "unverified",
        )
        self.assertEqual(len(data["rounds"][2]["claim_nodes"]), 1)
        self.assertTrue(
            data["rounds"][2]["returned_source_reports"][0]
                ["replaced_earlier_report"]
        )
        self.assertGreaterEqual(
            data["summary"].get("source_report_replacements", 0), 1,
        )

    def test_retried_claim_retains_visible_revision_lineage(self):
        data = replay([
            ("a", 4, "done", None),
            ("a", 7, "done", None),
        ])
        first = data["rounds"][0]["claim_nodes"][0]
        second = data["rounds"][1]["claim_nodes"][0]
        self.assertEqual(first["revision_seq"], 1)
        self.assertEqual(second["revision_seq"], 2)
        self.assertEqual(second["revision_of"], first["claim_id"])
        event = data["rounds"][1]["returned_source_reports"][0]
        self.assertEqual(event["superseded_claim_ids"], [first["claim_id"]])
        self.assertEqual(second["value"], 7)
        self.assertEqual(len(data["rounds"][1]["claim_nodes"]), 1)

    def test_unknown_source_origin_never_counts_as_independent(self):
        data = replay([
            ("a", 4, "done", None),
            ("b", 4, "done", None),
        ], world=scene(known_origins=False))
        state = match_state(data["rounds"][-1])
        self.assertEqual(state["status"], "unverified")
        self.assertFalse(state["sufficient_to_accept"])
        self.assertEqual(state["evidence_posture"], "obtain_independent_evidence")

    def test_partial_delivery_is_observed_but_not_accepted(self):
        data = replay([("a", 4, "partial", None)])
        state = match_state(data["rounds"][0])
        self.assertEqual(state["status"], "partial")
        self.assertFalse(state["sufficient_to_accept"])
        self.assertEqual(state["evidence_posture"], "complete_partial_delivery")
        self.assertEqual(data["rounds"][0]["claim_nodes"][0]
                         ["delivery_status"], "partial")

    def test_result_summary_conflict_is_intra_report_conflict(self):
        data = replay([
            ("a", 4, "done", "Case-011: completed_count=7")
        ])
        state = match_state(data["rounds"][0])
        self.assertEqual(state["status"], "intra_report_conflict")
        self.assertFalse(state["sufficient_to_accept"])
        self.assertTrue(state["internal_disagreement"])

    def test_task_dependencies_do_not_become_independent_confirmation(self):
        data = replay([
            ("a", 4, "done", None),
            ("join", 4, "done", None),
        ])
        result = data["rounds"][-1]
        state = match_state(result)
        self.assertEqual(state["status"], "unverified")
        self.assertFalse(state["sufficient_to_accept"])
        self.assertTrue(result["task_dependency_edges"])
        self.assertTrue(result["claim_dependency_edges"])
        self.assertEqual(result["claim_dependency_edges"][0]["relation"],
                         "same_subject_task_dependency")

    def test_derived_duplicate_does_not_revoke_independent_confirmation(self):
        data = replay([
            ("a", 4, "done", None),
            ("b", 4, "done", None),
            ("join", 4, "done", None),
        ])
        state = match_state(data["rounds"][-1])
        self.assertEqual(state["status"], "corroborated")
        self.assertEqual(state["independent_origins"], 2)
        self.assertTrue(state["sufficient_to_accept"])
        self.assertTrue(data["rounds"][-1]["claim_dependency_edges"])
        # An actually different derived observation remains a visible
        # discrepancy requiring investigation, not a new independent root.
        wrong = replay([
            ("a", 4, "done", None),
            ("b", 4, "done", None),
            ("join", 9, "done", None),
        ])
        self.assertEqual(
            match_state(wrong["rounds"][-1])["status"], "disputed",
        )

    def test_missing_or_unparseable_feedback_does_not_create_claim(self):
        data = replay([
            ("a", None, "done", "Maybe the count was probably 7."),
        ])
        self.assertEqual(data["rounds"][0]["claim_states"], {})
        self.assertEqual(data["rounds"][0]["claim_nodes"], [])
        self.assertEqual(
            data["summary"]["unparsed_completed_report_slots"], 1,
        )

    def test_state_is_recomputed_from_final_4b_feedback_and_visible_4c_prefix(self):
        from environment.untrusted_orchestration.offline_pipeline import (
            _refresh_visible_claim_supervision,
        )
        world = scene()
        world["macro_graph"]["reference_waves"] = [
            ["a"], ["b"], ["join"],
        ]
        episode = {
            "trajectory": [
                {"attempt": 1, "action": "delegate_task",
                 "result": {"subtask_results": [report(4)]}},
                {"attempt": 2, "action": "delegate_task",
                 "result": {"subtask_results": [report(7)]}},
            ]
        }
        audit = {
            "branch_round": 2,
            "decision_audit": [{
                "round": 2, "decision_kind": "delegate", "task_ids": ["b"]
            }],
        }
        _refresh_visible_claim_supervision(world, episode, audit)
        self.assertEqual(
            match_state(audit["visible_claim_state"]["rounds"][-1])["status"],
            "disputed",
        )
        # 4B's final visible report, not 4A's stale preliminary text.
        episode["trajectory"][1]["result"]["subtask_results"][0] = report(4)
        _refresh_visible_claim_supervision(world, episode, audit)
        self.assertEqual(
            match_state(audit["visible_claim_state"]["rounds"][-1])["status"],
            "corroborated",
        )
        # If Main 4C failed before dispatching round 2, the simulated
        # future report must not appear in its private visible-state label.
        _refresh_visible_claim_supervision(
            world, episode, audit, observed_rounds={1},
        )
        self.assertEqual(len(audit["visible_claim_state"]["rounds"]), 1)
        self.assertEqual(
            match_state(audit["visible_claim_state"]["rounds"][0])["status"],
            "unverified",
        )

    def test_explicit_field_claim_in_result_prose_is_visible(self):
        from environment.untrusted_orchestration.inspection.visible_claim_extraction import (
            extract_visible_claims,
        )
        claims, parse = extract_visible_claims({
            "finish_result": {
                "status": "done",
                "result": "Found Case-011, completed_count=4.",
                "summary": "The search returned the matching record.",
            }
        }, output_fields=["case_id", "completed_count"],
           source_id="alpha")
        self.assertTrue(parse["parsed"])
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["surface"], "result")
        self.assertEqual(claims[0]["subject"], "Case-011")
        self.assertEqual(claims[0]["value"], 4)

    def test_source_only_metric_is_retained_without_entity_invention(self):
        from environment.untrusted_orchestration.inspection.visible_claim_extraction import (
            extract_visible_claims,
        )
        claims, parse = extract_visible_claims({
            "finish_result": {
                "status": "done",
                "result": '[{"values":{"metric_a":7}}]',
                "summary": "Metric returned.",
            }
        }, output_fields=["metric_a"], source_id="stat_a")
        self.assertTrue(parse["parsed"])
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["subject"], "source:stat_a")
        self.assertEqual(claims[0]["attribute"], "metric_a")
        self.assertEqual(claims[0]["value"], 7)

    def test_current_dossier_identifier_has_observation_provenance(self):
        from environment.untrusted_orchestration.inspection.visible_claim_extraction import (
            extract_visible_claims,
        )
        world = scene(known_origins=False)
        world["macro_graph"]["tasks"][0]["outputs"]["typed_relation"] = [
            "case_id",
        ]
        world["macro_graph"]["tasks"][1]["outputs"]["typed_relation"] = [
            "case_id",
        ]
        for sid, origin in (("alpha", "actual-root-A"),
                            ("beta", "actual-root-B")):
            world["world"]["sources"][sid]["observations"] = [{
                "observation_id": sid + ".OBS001",
                "fact_refs": [sid + ":F001"],
                "evidence_origin_id": origin,
                "values": {"case_id": "Dossier-a1b2c3d4e5"},
            }]

        def delivered(sid):
            return {
                "finish_result": {
                    "status": "done",
                    "result": json.dumps([{
                        "values": {"case_id": "Dossier-a1b2c3d4e5"},
                        "source_fact_refs": [sid + ":F001"],
                    }]),
                    "summary": "Read the assigned dossier source.",
                },
            }

        found, diagnostics = extract_visible_claims(
            delivered("alpha"), output_fields=["case_id"],
            source_id="alpha",
        )
        self.assertTrue(diagnostics["parsed"])
        self.assertEqual(found[0]["subject"], "Dossier-a1b2c3d4e5")
        self.assertEqual(found[0]["fact_refs"], ["alpha:F001"])
        trajectory = [{
            "attempt": i, "action": "delegate_task",
            "result": {"subtask_results": [delivered(sid)]},
        } for i, sid in enumerate(("alpha", "beta"), 1)]
        audit = audit_visible_claim_states(
            world, trajectory, task_ids_by_round={1: ["a"], 2: ["b"]},
        )
        current = next(iter(audit["rounds"][-1]["claim_states"].values()))
        self.assertEqual(current["status"], "corroborated")
        self.assertEqual(current["independent_origins"], 2)

    def test_natural_code_owned_result_remains_parseable(self):
        from environment.untrusted_orchestration.inspection.visible_claim_extraction import (
            extract_visible_claims,
        )
        found, diagnostics = extract_visible_claims({
            "finish_result": {
                "status": "done",
                "result": "The current task returned 1 record(s): "
                          "project id: PRJ-001.",
                "summary": "Matched a documented project.",
            },
        }, output_fields=["project_id"], source_id="entity_index")
        self.assertTrue(diagnostics["parsed"])
        self.assertEqual(found[0]["attribute"], "project_id")
        self.assertEqual(found[0]["value"], "PRJ-001")

    def test_composed_claim_preserves_multi_source_lineage_without_warrant(self):
        world = scene(known_origins=False)
        world["macro_graph"]["tasks"] = [{
            "task_id": "joined", "depends_on": [],
            "source_ids": ["alpha", "beta"],
            "outputs": {"typed_relation": ["case_id", "completed_count"]},
        }]
        for sid, origin in (("alpha", "publisher:A"),
                            ("beta", "publisher:B")):
            world["world"]["sources"][sid]["observations"] = [{
                "fact_refs": [sid + ":F001"],
                "evidence_origin_id": origin,
            }]
        returned = {
            "finish_result": {
                "status": "done",
                "result": json.dumps([{
                    "values": {
                        "case_id": "Case-011", "completed_count": 4,
                    },
                    "source_fact_refs": ["alpha:F001", "beta:F001"],
                }]),
                "summary": "The combined result cites both source records.",
            },
        }
        data = audit_visible_claim_states(world, [{
            "attempt": 1, "action": "delegate_task",
            "result": {"subtask_results": [returned]},
        }], task_ids_by_round={1: ["joined"]})
        label = match_state(data["rounds"][0])
        claim = data["rounds"][0]["claim_nodes"][0]
        self.assertEqual(claim["evidence_origin_ids"],
                         ["publisher:A", "publisher:B"])
        self.assertEqual(claim["provenance_source_ids"],
                         ["alpha", "beta"])
        self.assertIsNone(claim["evidence_origin_id"])
        self.assertEqual(label["status"], "unverified")
        self.assertEqual(label["independent_origins"], 0)
        self.assertEqual(label["cited_source_ids"], ["alpha", "beta"])
        self.assertEqual(
            data["summary"]["active_multi_origin_claims"], 1,
        )

    def test_round_alignment_failure_is_detected(self):
        world = scene()
        with self.assertRaisesRegex(
            ValueError, "visible_claim_dispatch_alignment_invalid"
        ):
            audit_visible_claim_states(
                world, [{
                    "attempt": 1,
                    "action": "delegate_task",
                    "result": {"subtask_results": [report(4)]},
                }], task_ids_by_round={1: ["a", "b"]},
            )


if __name__ == "__main__":
    unittest.main()
