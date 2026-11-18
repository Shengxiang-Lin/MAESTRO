"""Focused causal/event-fault regression tests, independent of LLM calls."""
from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from environment.tests.test_untrusted_orchestration import sample
from environment.generate_question.execution.program_ir import Relation
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
                    {"task_id": "A", "depends_on": [],
                     "source_ids": ["image"]},
                    {"task_id": "B", "depends_on": ["A"],
                     "source_ids": ["index"]},
                    {"task_id": "C", "depends_on": [],
                     "source_ids": ["subtitle"]},
                    {"task_id": "D", "depends_on": ["C"],
                     "source_ids": ["sub_index"]},
                    {"task_id": "E", "depends_on": ["B", "D"],
                     "source_ids": []},
                ],
            },
        }
        event = delayed_event_candidate(
            scene, fork_round=2, initial_fault_task="A",
        )
        self.assertEqual(event["task_id"], "C")
        self.assertEqual(event["after_task_id"], "B")
        self.assertEqual(event["earliest_round"], 3)
        plan = {
            "index": 0, "scene_index": 0, "task_id": "Q001",
            "sample_id": "Q001-fork-02", "branch_round": 2,
            "faults": [Fault("A", "wrong")],
        }
        scenario = make_scenario(
            scene, plan, mode="event_driven",
            eligible_atoms={"B": ["wrong_retrieval"]},
            max_fault_atoms=2, delayed_evidence_probability=1.0,
        )
        self.assertEqual(len(scenario["fault_atoms"]), 2)
        self.assertEqual(scenario["fault_atoms"][0]["origin_round"], 1)
        self.assertEqual(scenario["fault_atoms"][1]["origin_round"], 2)
        self.assertEqual(
            scenario["blueprint"], "delayed_independent_evidence"
        )
        self.assertEqual(
            scenario["evidence_events"][0]["task_id"], "C"
        )

    def test_late_conflict_uses_public_reports_and_consumed_version(self):
        state = ExecutionState(self.scene, {})
        state.tasks = {
            "old": {
                "task_id": "old", "depends_on": [],
                "source_ids": ["image"],
                "operator_chain": [],
            },
            "independent": {
                "task_id": "independent", "depends_on": [],
                "source_ids": ["subtitle"],
                "operator_chain": [],
            },
            "lookup": {
                "task_id": "lookup", "depends_on": ["old"],
                "operator_chain": [],
            },
            "join": {
                "task_id": "join",
                "depends_on": ["old", "independent"],
                "operator_chain": [{
                    "op": "join", "left_on": "marker_code",
                    "right_on": "marker_code",
                }],
            },
        }
        state.current = {
            "old": Attempt(
                "old", 1, "done",
                Relation(("marker_code",), ({
                    "values": {"marker_code": "MK-6243"},
                    "support": ("image:F002",),
                },)),
                ("marker_code",), {}, {},
            ),
            "independent": Attempt(
                "independent", 1, "done",
                Relation(("marker_code",), ({
                    "values": {"marker_code": "MK-0531"},
                    "support": ("subtitle:F001",),
                },)),
                ("marker_code",), {}, {},
            ),
            "lookup": Attempt(
                "lookup", 1, "done",
                Relation(("record_key_1",), ({
                    "values": {"record_key_1": "KEY1-012"},
                    "support": ("index:F012",),
                },)),
                ("record_key_1",), {"old": 1}, {},
            ),
        }
        state.set_public_report(
            "old", {"finish_result": {
                "status": "done",
                "result": json.dumps([{"marker_code": "MK-6243"}]),
                "summary": "One candidate key was returned.",
            }}, round_number=1,
        )
        state.set_public_report(
            "independent", {"finish_result": {
                "status": "done",
                "result": json.dumps([{"marker_code": "MK-0531"}]),
                "summary": "A second independent key was returned.",
            }}, round_number=4,
        )
        state.cross_source_checks_enabled = True
        state._detect_late_cross_source_conflicts()
        self.assertEqual(state.current["old"].status, "disputed")
        self.assertIn(
            "MK-6243", state.current["old"].runtime["visible_issue"],
        )
        self.assertTrue(state.detected_conflicts)
        state._detect_late_cross_source_conflicts()
        self.assertEqual(len(state.detected_conflicts), 1)
        # A conflict with no earlier downstream consumer is a reason
        # to compare evidence, NOT automatically to name a wrong source.
        state2 = ExecutionState(self.scene, {})
        state2.tasks = state.tasks
        state2.current = {
            k: v for k, v in state.current.items() if k != "lookup"
        }
        state2.public_reports = dict(state.public_reports)
        state2.cross_source_checks_enabled = True
        state2._detect_late_cross_source_conflicts()
        self.assertFalse(state2.detected_conflicts)

    def test_cited_late_disagreement_requires_independent_public_claims(self):
        scene = deepcopy(self.scene)
        scene["world"]["sources"].update({
            sid: {"observations": [{
                "fact_refs": [sid + ":F001"],
                "evidence_origin_id": "independent-root-" + sid,
            }]} for sid in ("alpha", "beta")
        })

        def fixture(late_year=2004, late_refs=None):
            state = ExecutionState(scene, {})
            state.tasks = {
                "old": {
                    "task_id": "old", "depends_on": [],
                    "source_ids": ["alpha"],
                    "operator_chain": [],
                    "outputs": {"typed_relation": [
                        "case_id", "reported_year_a",
                    ]},
                },
                "late": {
                    "task_id": "late", "depends_on": [],
                    "source_ids": ["beta"],
                    "operator_chain": [],
                    "outputs": {"typed_relation": [
                        "case_id", "reported_year_b",
                    ]},
                },
                "consumer": {
                    "task_id": "consumer", "depends_on": ["old"],
                    "source_ids": [], "operator_chain": [],
                    "outputs": {"typed_relation": ["case_id"]},
                },
            }
            state.current = {
                "old": Attempt(
                    "old", 1, "done",
                    Relation(("case_id", "reported_year_a"), ({
                        "values": {
                            "case_id": "Case-011", "reported_year_a": 2003,
                        },
                        "support": ("alpha:F001",),
                    },)),
                    ("case_id", "reported_year_a"), {}, {},
                ),
                "late": Attempt(
                    "late", 1, "done",
                    Relation(("case_id", "reported_year_b"), ({
                        "values": {
                            "case_id": "Case-011", "reported_year_b": late_year,
                        },
                        "support": ("beta:F001",),
                    },)),
                    ("case_id", "reported_year_b"), {}, {},
                ),
                "consumer": Attempt(
                    "consumer", 1, "done",
                    Relation(("case_id",), ({
                        "values": {"case_id": "Case-011"},
                        "support": ("alpha:F001",),
                    },)),
                    ("case_id",), {"old": 1}, {},
                ),
            }
            state.public_reports = {
                "old": {"finish_result": {
                    "status": "done",
                    "result": json.dumps([{
                        "values": {
                            "case_id": "Case-011", "reported_year_a": 2003,
                        },
                        "source_fact_refs": ["alpha:F001"],
                    }]), "summary": "Recorded the first event year.",
                }},
                "late": {"finish_result": {
                    "status": "done",
                    "result": json.dumps([{
                        "values": {
                            "case_id": "Case-011",
                            "reported_year_b": late_year,
                        },
                        "source_fact_refs": (
                            late_refs if late_refs is not None
                            else ["beta:F001"]
                        ),
                    }]), "summary": "Recorded the later event year.",
                }},
            }
            state.public_report_rounds = {"old": 1, "late": 4}
            state.cross_source_checks_enabled = True
            return state

        observed = fixture()
        decisions = observed.scheduling_state()
        self.assertIn("old", decisions["repair"])
        self.assertEqual(
            observed.detected_conflicts[-1]["kind"],
            "explicit_independent_claim_disagreement",
        )
        self.assertEqual(
            observed.observable_issues()[0]["kind"],
            "cross_source_conflict",
        )
        self.assertIn(
            "Case-011", observed.detected_conflicts[-1]["claim"]["subject"],
        )
        # Repeated checks do not schedule fresh invisible verifications.
        observed.scheduling_state()
        self.assertEqual(len(observed.detected_conflicts), 1)

        # Private WorkProduct divergence is invisible if the delivered
        # Result and the independent source still agree.
        hidden = fixture(late_year=2003)
        hidden.current["old"].product = Relation(
            ("case_id", "reported_year_a"), ({
                "values": {
                    "case_id": "Case-011", "reported_year_a": 1999,
                }, "support": ("alpha:F001",),
            },),
        )
        self.assertFalse(hidden.scheduling_state()["repair"])
        self.assertEqual(hidden.observable_issues(), [])

        # A derived/multi-origin return cannot act as a second
        # independent observation even if it cites a second source.
        mixed = fixture(late_refs=["alpha:F001", "beta:F001"])
        self.assertFalse(mixed.scheduling_state()["repair"])
        self.assertEqual(mixed.detected_conflicts, [])

    def test_frozen_publication_rejects_critic_process_leak(self):
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
            out = root / "output"
            result = generate_turns(
                scenes_path=scenes, reports_path=reports,
 output_dir=out,
                max_scenes=1, branches_per_scene=1,
                fault_modes=["missing"], workers=2,
                no_progress=True,
            )
            self.assertGreater(result["reviewed_turns"], 0)
            saved = out / "journal" / "000000.json"
            journal = json.loads(saved.read_text())
            journal["turns"][0]["output"]["reasoning"] = (
                "The critic correctly noted this action was needed."
            )
            saved.write_text(json.dumps(journal))
            plans, _, _ = _plan_branches(
                [self.scene], {self.scene["task_id"]: self.episode},
                branches_per_scene=1, seed=17,
                fault_modes=["missing"], persistence="once",
                multi_target_probability=0, max_rounds=16,
            )
            published = _publish(out, plans)
            self.assertEqual(published["reviewed_turns"], 0)
            self.assertEqual(
                published["publication_generation_process_leaks"], 1,
            )
            self.assertEqual(
                json.loads((out / "failed.jsonl").read_text().splitlines()[0])
                ["status"],
                "publication_generation_process_leak",
            )

    def test_reconfirmed_evidence_preserves_unrelated_executed_work(self):
        state = ExecutionState(self.scene, {})
        for wave in self.scene["macro_graph"]["reference_waves"]:
            for tid in wave:
                attempt = state.register(tid)
                state.set_public_report(
                    tid, {"finish_result": {
                        "status": "done", "result": "Grounded evidence.",
                        "summary": "Completed observation.",
                    }},
                )
        earliest = self.scene["macro_graph"]["reference_waves"][0][0]
        prior_attempt_versions = {
            tid: item.version for tid, item in state.current.items()
        }
        replacement = state.register(earliest)
        self.assertEqual(replacement.version, 2)
        self.assertEqual(state.invalidations, [])
        self.assertEqual(len(state.revalidations), 1)
        self.assertEqual(
            state.revalidations[0]["task_id"], earliest,
        )
        for tid in state._dependents(earliest):
            if tid in state.current:
                self.assertEqual(
                    state.current[tid].version, prior_attempt_versions[tid]
                )
                self.assertEqual(state.current[tid].status, "done")

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
 output_dir=root / "dataset",
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
