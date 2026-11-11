"""Focused causal/event-fault regression tests, independent of LLM calls."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from environment.tests.test_untrusted_orchestration import sample
from environment.generate_question.program_ir import Relation
from environment.untrusted_orchestration.atomic_faults import (
    AtomicFault, probe_atoms,
)
from environment.untrusted_orchestration.event_scheduler import (
    EvidenceEvent, EventScheduler,
)
from environment.untrusted_orchestration.scenario_generator import (
    make_scenario, delayed_event_candidate,
)
from environment.untrusted_orchestration.faults import Fault, relation_rows
from environment.untrusted_orchestration.report_author import naturalize_report
from environment.untrusted_orchestration.state_machine import (
    ExecutionState, Attempt,
)
from environment.untrusted_orchestration.trajectory_builder import build_branch
from environment.untrusted_orchestration.generate_turns import (
    generate_turns, _plan_branches, _publish,
)
from environment.untrusted_orchestration.publication_guard import (
    review_turn_surface,
)


class EventEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene, cls.reports, cls.episode = sample(0)

    def test_typed_faults_and_mode_validation(self):
        atom = AtomicFault(
            "wrong_entity_binding", "taskA", 2, seed=17,
        )
        compiled = atom.as_legacy_fault()
        self.assertEqual(compiled.mode, "wrong")
        self.assertEqual(compiled.atom_kind, "wrong_entity_binding")
        self.assertEqual(atom.surface, "processing")
        with self.assertRaises(ValueError):
            AtomicFault("random_number_invention", "taskA", 2)
        with self.assertRaises(ValueError):
            Fault("taskA", "partial", atom_kind="wrong_entity_binding")

    def test_legacy_fork_is_one_atom_event_scenario(self):
        plans, _, _ = _plan_branches(
            [self.scene], {self.scene["task_id"]: self.episode},
            branches_per_scene=1, seed=17,
            fault_modes=["missing"], persistence="once",
            multi_target_probability=0, max_rounds=16,
        )
        self.assertEqual(len(plans), 1)
        plan = plans[0]["scenario_plan"]
        self.assertEqual(plan["mode"], "round_fork")
        self.assertEqual(plan["blueprint"], "immediate_feedback_fault")
        self.assertEqual(
            [a["kind"] for a in plan["fault_atoms"]],
            ["no_result"],
        )
        self.assertEqual(plan["evidence_events"], [])

    def test_report_missing_is_distinct_from_acquisition_failure(self):
        tid = self.scene["macro_graph"]["reference_waves"][0][0]
        fault = Fault(tid, "report", atom_kind="report_missing")
        episode, turns, audit = build_branch(
            self.scene, self.reports, self.episode,
            branch_round=2, faults=[fault],
        )
        initial = episode["trajectory"][0]["result"]["subtask_results"][0]
        self.assertTrue(initial["done"])
        self.assertIsNone(initial["finish_result"])
        self.assertEqual(audit["faults"][0]["mode"], "report")
        self.assertEqual(audit["execution_outcome"], "recovered")
        self.assertTrue(turns)

    def test_report_summary_conflict_uses_only_public_values(self):
        state = ExecutionState(self.scene, {})
        tid = self.scene["macro_graph"]["reference_waves"][0][0]
        state.current[tid] = Attempt(
            task_id=tid, version=1, status="done",
            product=Relation(("marker_code",), ({
                "values": {"marker_code": "MK-0531"},
                "support": ("image:F001",),
            },)),
            available_fields=("marker_code",),
            used_parent_versions={}, runtime={},
        )
        state.public_reports[tid] = {
            "finish_result": {
                "status": "done",
                "result": json.dumps([{"marker_code": "MK-0531"}]),
                "summary": (
                    "The summary records marker code = MK-6243 "
                    "for the returned record."
                ),
            },
        }
        self.assertIn(
            "disagree", state.visible_discrepancy(tid),
        )

    def test_report_fabrication_consumes_communicated_not_hidden_value(self):
        candidates = probe_atoms(self.scene, ["fabricated_claim"])
        choices = [
            tid for wave in self.scene["macro_graph"]["reference_waves"]
            for tid in wave if "fabricated_claim" in candidates[tid]
        ]
        if not choices:
            self.skipTest("Scene has no source-backed report distractor")
        target = choices[0]
        state = ExecutionState(self.scene, {})
        tasks = state.tasks
        for wave in self.scene["macro_graph"]["reference_waves"]:
            for tid in wave:
                atom = (AtomicFault("fabricated_claim", target, 1)
                        .as_legacy_fault() if tid == target else None)
                attempt = state.register(tid, fault=atom)
                report, meta = naturalize_report(
                    tasks[tid], attempt, instruction="Read the source.",
                    model_alias="model_1", reviewed=None,
                )
                state.set_public_report(tid, report)
                if tid == target:
                    self.assertEqual(
                        meta["mode"], "controlled_untrusted_report"
                    )
                    self.assertNotEqual(
                        relation_rows(attempt.product),
                        relation_rows(state.communicated_product(tid)),
                    )
                    # The actual operator trace remains independently
                    # auditable: a report fault is not executor corruption.
                    self.assertEqual(
                        attempt.runtime["report_fault_kind"],
                        "fabricated_claim",
                    )
                    return
        self.fail("Target task was not executed")

    def test_event_hold_release_does_not_rewrite_prior_history(self):
        class State:
            tasks = {
                "previous": {}, "lookup": {}, "other_source": {},
            }
            current = {"previous": object()}
            def _dependents(self, task_id):
                return {"lookup"} if task_id == "previous" else set()
        event = EvidenceEvent(
            "hold_independent_evidence", "other_source", 3,
            after_task_id="lookup",
        )
        schedule = EventScheduler(
            [event.to_private_dict()], state=State(), fork_round=2,
        )
        self.assertEqual(
            schedule.available(["lookup", "other_source"], 2),
            ["lookup"],
        )
        schedule.note_execution("lookup", 2)
        self.assertEqual(
            schedule.available(["other_source"], 3),
            ["other_source"],
        )
        self.assertEqual(schedule.audit()[0]["status"], "released")
        with self.assertRaises(ValueError):
            EventScheduler([EvidenceEvent(
                "hold_independent_evidence", "previous", 3,
            ).to_private_dict()], state=State(), fork_round=2)
        with self.assertRaises(ValueError):
            EvidenceEvent("message_delayed", "other_source", 3)

    def test_delayed_candidate_requires_independent_shared_downstream(self):
        scene = {
            "macro_graph": {
                "reference_waves": [
                    ["A"], ["B", "C"], ["D"], ["E"]
                ],
                "tasks": [
                    {"task_id": "A", "depends_on": []},
                    {"task_id": "B", "depends_on": ["A"]},
                    {"task_id": "C", "depends_on": []},
                    {"task_id": "D", "depends_on": ["C"]},
                    {"task_id": "E", "depends_on": ["B", "D"]},
                ],
            },
        }
        event = delayed_event_candidate(
            scene, fork_round=2, initial_fault_task="A",
        )
        self.assertEqual(event["task_id"], "C")
        self.assertEqual(event["after_task_id"], "B")
        self.assertEqual(event["earliest_round"], 3)

    def test_event_mode_one_report_atom_is_executable(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            scenes = root / "scenes.jsonl"
            reports = root / "reports.jsonl"
            base = root / "episodes.jsonl"
            scenes.write_text(json.dumps(self.scene) + "\n")
            reports.write_text(
                "".join(json.dumps(r) + "\n" for r in self.reports)
            )
            base.write_text(json.dumps(self.episode) + "\n")
            result = generate_turns(
                scenes_path=scenes, reports_path=reports,
                base_episodes_path=base, output_dir=root / "dataset",
                max_scenes=1, branches_per_scene=1, workers=2,
                no_progress=True, scenario_mode="event_driven",
                atomic_fault_kinds=["report_missing"],
                max_fault_atoms=1, delayed_evidence_probability=0.0,
            )
            self.assertEqual(result["status"], "done")
            self.assertEqual(
                result["atomic_fault_kinds_planned"]["report_missing"], 1,
            )
            self.assertTrue(
                (root / "dataset" / "round_index.jsonl").is_file()
            )
            self.assertTrue(
                (root / "dataset" / "private" / "publication_guard.jsonl")
                .is_file()
            )

    def test_publication_guard_excludes_reviewer_leak(self):
        self.assertEqual(
            review_turn_surface({
                "output": {"action": "delegate_task",
                           "reasoning": (
                               "The critic correctly noted that the "
                               "earlier action was insufficient."
                           ),
                           "params": {"tasks": []}}
            }),
            ["generation_process_leak:reasoning"],
        )
        self.assertEqual(
            review_turn_surface({
                "output": {"action": "complete",
                           "reasoning": (
                               "The previously returned field is "
                               "sufficient to answer the question."
                           ),
                           "params": {"answer": "42"}}
            }),
            [],
        )


if __name__ == "__main__":
    unittest.main()
