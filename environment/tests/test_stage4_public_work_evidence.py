"""Public evidence, truthful partial reporting, and visible recovery checks."""
from __future__ import annotations

from copy import deepcopy
import unittest

from environment.generate_question.execution.program_ir import Relation
from environment.untrusted_orchestration.reporting.public_work import (
    observed_work_checks, public_work_clauses, visible_work_discrepancy,
)
from environment.untrusted_orchestration.reporting.report_author import (
    naturalize_report,
)
from environment.untrusted_orchestration.runtime.faults import Fault, FaultUnsupported
from environment.untrusted_orchestration.runtime.atomic_faults import (
    AtomicFault, probe_atoms,
)
from environment.untrusted_orchestration.trajectory_builder import build_branch
from environment.untrusted_orchestration.state_machine import (
    ExecutionState, deterministic_report,
)
from environment.tests.test_stage4_history_first import _scene
from environment.tests.test_untrusted_orchestration import sample


def relation(values, refs=("source:F1",)):
    return Relation(tuple(values), ({
        "values": dict(values), "support": tuple(refs),
    },))


class PublicWorkEvidenceTests(unittest.TestCase):
    def test_arithmetic_is_recomputable_from_public_report_without_gold(self):
        src = relation({"a": 120, "b": 75})
        wrong = relation({"difference": 46})
        step = {
            "op": "derive_arithmetic", "left_field": "a",
            "right_field": "b", "output_field": "difference",
            "operation": "absolute_difference", "round_digits": 0,
        }
        checks = observed_work_checks(step, [src], wrong)
        self.assertEqual(len(checks), 1)
        clause = public_work_clauses([{"public_work_checks": checks}])[0]
        self.assertIn("120", clause)
        self.assertIn("75", clause)
        self.assertIn("46", clause)
        self.assertIn("rather than 45", visible_work_discrepancy(clause))
        clean = relation({"difference": 45})
        clean_clause = public_work_clauses([{
            "public_work_checks": observed_work_checks(step, [src], clean)
        }])[0]
        self.assertIsNone(visible_work_discrepancy(clean_clause))
        self.assertIsNone(visible_work_discrepancy(
            "An unexplained 46 was returned; source values are not known."
        ))

    def test_wrong_entity_join_has_real_input_key_evidence(self):
        left = relation({"record_id": "B02", "name": "Beta"}, ("A:F2",))
        right = relation({"record_id": "A01", "revenue": 100}, ("B:F1",))
        combined = Relation(
            ("record_id", "name", "revenue"), ({
                "values": {
                    "record_id": "B02", "name": "Beta", "revenue": 100,
                },
                "support": ("A:F2", "B:F1"),
            },),
        )
        node = {
            "op": "join", "left_on": "record_id",
            "right_on": "record_id",
        }
        clauses = public_work_clauses([{"public_work_checks":
            observed_work_checks(node, [left, right], combined)
        }])
        self.assertEqual(len(clauses), 1)
        self.assertIn("B02", visible_work_discrepancy(clauses[0]))
        matched_right = relation(
            {"record_id": "B02", "revenue": 100}, ("B:F2",)
        )
        matched = Relation(
            ("record_id", "name", "revenue"), ({
                "values": {"record_id": "B02", "name": "Beta", "revenue": 100},
                "support": ("A:F2", "B:F2"),
            },),
        )
        clean_clause = public_work_clauses([{"public_work_checks":
            observed_work_checks(node, [left, matched_right], matched)
        }])[0]
        self.assertIsNone(visible_work_discrepancy(clean_clause))

    def test_completed_attempt_can_admit_failed_extraction_in_natural_text(self):
        scene, parent, child = _scene()
        state = ExecutionState(scene, {})
        parent_attempt = state.register("parent")
        state.set_public_report("parent", deterministic_report(
            parent, parent_attempt,
            instruction="Find the named place.",
            model_alias="model_1",
        ))
        attempted = state.register(
            "child",
            fault=Fault(
                "child", "partial", atom_kind="done_but_incomplete",
            ),
        )
        report, status = naturalize_report(
            child, attempted,
            instruction="Look up the place and collect the available evidence.",
            model_alias="model_1", reviewed=None,
        )
        self.assertEqual(report["finish_result"]["status"], "done")
        self.assertIn("could not extract",
                      report["finish_result"]["summary"])
        self.assertNotIn("observation year",
                         report["task_instruction"].casefold())
        state.set_public_report("child", report)
        self.assertIn("unfinished extraction", state.visible_discrepancy("child"))
        self.assertEqual(status["status"], "accepted")

    def test_done_but_incomplete_is_really_llm_authored(self):
        scene, parent, child = _scene()
        state = ExecutionState(scene, {})
        state.register("parent")
        partial = state.register(
            "child", fault=Fault(
                "child", "partial", atom_kind="done_but_incomplete"
            ),
        )

        class ReportAuthor:
            def __init__(self):
                self.calls = 0

            def complete_json(self, system, payload):
                self.calls += 1
                return {
                    "summary": (
                        "The lookup returned the place name but I could "
                        "not extract the observation year from this result."
                    ),
                    "trace_summary": (
                        "I inspected the authorized lookup and returned "
                        "only the available observations."
                    ),
                }

        class ReportCritic:
            def complete_json(self, system, payload):
                return {
                    "verdict": "pass", "grounded": True,
                    "scope_preserved": True, "natural": True,
                    "useful": True,
                }

        author = ReportAuthor()
        report, status = naturalize_report(
            child, partial,
            instruction="Find the place and the observation year.",
            model_alias="model_1", reviewed=None,
            author=author, critic=ReportCritic(), max_revisions=0,
        )
        self.assertGreater(author.calls, 0)
        self.assertEqual(status["status"], "accepted", status)
        self.assertEqual(status["mode"], "llm_reviewed")
        self.assertTrue(status["controlled_done_partial_overlay"])
        self.assertEqual(report["finish_result"]["status"], "done")
        self.assertIn("could not extract",
                      report["finish_result"]["summary"])

    def test_real_computation_error_produces_visible_recovery_decision(self):
        """Full 4A branch, not just a parser fixture: the changed numeric
        output must be observable before the teacher decides completion.
        """
        found = False
        for i in range(10):
            scene, rows, base = sample(i)
            eligible = probe_atoms(scene, ["computation_error"])
            for wave_index, wave in enumerate(
                scene["macro_graph"]["reference_waves"]
            ):
                for tid in wave:
                    if "computation_error" not in eligible.get(tid, []):
                        continue
                    episode, turns, audit = build_branch(
                        scene, rows, base,
                        branch_round=wave_index + 2,
                        faults=[AtomicFault(
                            "computation_error", tid, wave_index + 1,
                        ).as_legacy_fault()],
                    )
                    self.assertIn(
                        audit["execution_outcome"], {
                            "recovered",
                            "short_replay_horizon",
                            "incorrect_result_after_actual_propagation",
                        },
                    )
                    delivered = next(
                        report for report in episode["trajectory"][
                            wave_index]["result"]["subtask_results"]
                        if report is not None and
                        "For the" in (report.get("finish_result") or {})
                        .get("summary", "")
                    )
                    self.assertTrue(
                        "calculation" in delivered["finish_result"]["summary"]
                        or "count" in delivered["finish_result"]["summary"],
                    )
                    if audit["execution_outcome"] == "recovered":
                        self.assertTrue(turns)
                        self.assertGreater(
                            len(audit["runtime"]["attempt_versions"][tid]), 1,
                        )
                        found = True
                        break
                if found:
                    break
            if found:
                break
        self.assertTrue(found,
                        "No actual recoverable calculation error was produced")

    def test_real_wrong_entity_binding_stays_an_executable_fault(self):
        # The public association witness must come from actual attempted
        # Join evidence, not a synthetic wrong-ID label.
        found = False
        for i in range(10):
            scene, rows, _ = sample(i)
            state = ExecutionState(scene, {})
            for wave in scene["macro_graph"]["reference_waves"]:
                for tid in wave:
                    task = state.tasks[tid]
                    fault = (Fault(tid, "wrong",
                                   atom_kind="wrong_entity_binding")
                             if any(step["op"] == "join"
                                    for step in task["operator_chain"])
                             else None)
                    try:
                        attempt = state.register(tid, fault=fault)
                    except FaultUnsupported:
                        attempt = state.register(tid)
                        fault = None
                    report = deterministic_report(
                        task, attempt,
                        instruction="Inspect the permitted evidence.",
                        model_alias="model_1",
                    )
                    state.set_public_report(tid, report)
                    if fault is not None:
                        found = True
                        self.assertIn(
                            "For the record association",
                            report["finish_result"]["summary"],
                        )
                        self.assertIn(
                            "identifiers disagree",
                            state.visible_discrepancy(tid),
                        )
                        break
                if found:
                    break
            if found:
                break
        self.assertTrue(found, "No genuine wrong join available")


if __name__ == "__main__":
    unittest.main()
