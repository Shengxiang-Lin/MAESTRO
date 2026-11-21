"""Independent, same-fact source claims across Stage-1 sibling Macros."""
from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from environment.generate_question.planning.evidence_plan import (
    compose_evidence_plan,
)
from environment.generate_question.execution.program_ir import Relation
from environment.untrusted_orchestration.state_machine import (
    Attempt, ExecutionState,
)
from environment.untrusted_orchestration.reporting.surface import render_records
from environment.untrusted_orchestration.scenario.observable_scenarios import (
    classify_observable_scenario, comparable_incoming_tasks,
)
from environment.tests.stage4_fixtures import sample


class CorroborationPlanTests(unittest.TestCase):
    def test_independent_sources_deliver_sibling_reference_claims(self):
        needs = {
            "version": "maestro.evidence_plan.v1",
            "requested": {
                "terminal_relation_keys": ["birthplace", "containedinplace"],
                "event_eligibility": {
                    "field": "event_year", "op": "filter_gte", "value": 2025,
                },
                "independent_corroboration": True,
            },
            "available_sources": {
                "link_a": True, "link_b": True, "event_ledger": True,
                "real_fork": True,
            },
            "selected_producer": "corroborated_reference",
            "source_access": ["browse", "browse"],
            "sampling_attempts": 1,
        }
        with patch(
            "environment.generate_question.planning.evidence_plan."
            "_plan_information_needs", return_value=needs,
        ):
            plan = compose_evidence_plan(
                seed=17, ordinal=0, difficulty="medium", hops=2,
                goal_keys=("birthplace", "containedinplace"),
            )
        refs = [
            task for task in plan["macro_tasks"]
            if task["operation"] == "read_reference_candidates"
        ]
        self.assertEqual(len(refs), 2)
        self.assertEqual(
            {tuple(t["source_scope"]) for t in refs},
            {("link_a",), ("link_b",)},
        )
        self.assertEqual(refs[0]["depends_on"], refs[1]["depends_on"])
        self.assertEqual(
            refs[0]["output_schema"], {"route_key": "str", "entity_0": "str"},
        )
        verify = next(t for t in plan["macro_tasks"]
                      if t["operation"] == "verify_real_fork")
        self.assertEqual(
            verify["depends_on"][1:],
            [refs[0]["macro_id"], refs[1]["macro_id"]],
        )
        self.assertLessEqual(len(plan["macro_tasks"]), 9)


def _report(value: str, sid: str):
    return {
        "finish_result": {
            "status": "done",
            "result": render_records([{
                "values": {"route_key": "route:abc", "entity_0": value},
                "source_fact_refs": ["scope:F001", sid + ":F001"],
            }]),
            "summary": "One candidate route was returned.",
        },
        "trace_summary": "Inspected the allocated source.",
    }


class PublicCorroborationTests(unittest.TestCase):
    def _state(self):
        scene, reports, _ = sample(0)
        scene = deepcopy(scene)
        # A common upstream project scope is not independent evidence.
        # Only the two separately cited source rows support entity_0.
        scene["world"]["sources"] = {
            "scope": {"observations": [{
                "values": {"project_id": "P1"},
                "fact_refs": ["scope:F001"],
                "evidence_origin_id": "origin:scope",
            }]},
            "link_a": {"observations": [{
                "values": {"route_key": "route:abc", "entity_0": "ENT-A"},
                "fact_refs": ["link_a:F001"],
                "evidence_origin_id": "origin:link_a",
            }]},
            "link_b": {"observations": [{
                "values": {"route_key": "route:abc", "entity_0": "ENT-A"},
                "fact_refs": ["link_b:F001"],
                "evidence_origin_id": "origin:link_b",
            }]},
        }
        tasks = []
        for sid in ("link_a", "link_b"):
            tasks.append({
                "task_id": sid, "source_ids": [sid],
                "depends_on": [], "macro_role": "association",
                "outputs": {"typed_relation": ["route_key", "entity_0"]},
                "operator_chain": [],
            })
        scene["macro_graph"]["tasks"] = tasks
        state = ExecutionState(scene, {})
        state.cross_source_checks_enabled = True
        for sid in ("link_a", "link_b"):
            state.current[sid] = Attempt(
                task_id=sid, version=1, status="done", product=Relation(
                    ("route_key", "entity_0"), ()),
                available_fields=("route_key", "entity_0"),
                used_parent_versions={}, runtime={},
            )
        return state

    def test_clean_independent_claims_are_not_conflicts(self):
        state = self._state()
        for sid in ("link_a", "link_b"):
            state.set_public_report(sid, _report("ENT-A", sid), round_number=2)
        self.assertEqual(len(state._explicit_report_claims()), 2)
        state._detect_explicit_cross_report_disagreements()
        self.assertEqual(state.detected_conflicts, [])

    def test_one_faulted_report_yields_independent_b3(self):
        state = self._state()
        state.set_public_report("link_a", _report("ENT-A", "link_a"), round_number=2)
        state.set_public_report("link_b", _report("ENT-B", "link_b"), round_number=2)
        state._detect_explicit_cross_report_disagreements()
        independent = [
            x for x in state.detected_conflicts
            if x["kind"] == "explicit_independent_claim_disagreement"
        ]
        self.assertEqual(len(independent), 1)
        self.assertEqual(
            independent[0]["claim"]["attribute"], "entity_0",
        )
        realized = classify_observable_scenario([], independent)
        self.assertEqual(realized["scenario_id"], "B3")


if __name__ == "__main__":
    unittest.main()
