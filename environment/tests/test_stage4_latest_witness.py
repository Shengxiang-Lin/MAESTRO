"""Current Stage-4 latest-batch witness and visible-history regressions."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from environment.tests.test_untrusted_orchestration import sample
from environment.untrusted_orchestration.atomic_faults import AtomicFault
from environment.untrusted_orchestration.latest_witness import (
    make_latest_pair, extend_clean_prefix, extend_local_scene,
)
from environment.untrusted_orchestration.offline_pipeline import (
    _public_done_null_fields, _visibility_issues,
    author_branch_main, author_branch_reports,
)
from environment.untrusted_orchestration.trajectory_builder import build_branch
from environment.untrusted_orchestration.generate_turns import generate_turns
from environment.untrusted_orchestration.scenario_generator import make_scenario
from environment.untrusted_orchestration.main_decision_contract import (
    build_main_decision_contract,
)


class FlattenedO1MissingFieldTests(unittest.TestCase):
    def test_done_null_field_is_seen_in_actual_flat_o1_batches(self):
        batches = [{"attempt": 1, "subtask_results": [{
            "status": "done",
            "result": json.dumps([{
                "values": {"case_id": "Case-002", "metric_b": None},
                "source_fact_refs": ["metric_b:F001"],
            }]),
            "summary": "The record was delivered.",
            "instruction": "Return case_id and metric_b.",
        }]}]
        requests = [{
            "request_id": "R02-Eone",
            "operation": "complete_partial_evidence",
            "target_fields": ["metric_b"],
            "known_identifiers": [],
        }]
        self.assertEqual(_public_done_null_fields(batches), {"metric_b"})
        self.assertEqual(_visibility_issues(
            requests, "[Attempt 1] Result: metric_b null",
            public_batches=batches,
        ), [])
        # The private planner's belief is NOT enough if no such field was
        # actually missing from any delivered report.
        clean = deepcopy(batches)
        clean[0]["subtask_results"][0]["result"] = json.dumps([{
            "values": {"case_id": "Case-002", "metric_b": 17},
            "source_fact_refs": ["metric_b:F001"],
        }])
        self.assertEqual(
            _visibility_issues(
                requests, "[Attempt 1] Result: metric_b 17",
                public_batches=clean,
            ), ["R02-Eone:missing_status"],
        )

    def test_structured_nested_legacy_report_is_not_confused_with_flat(self):
        rows = [{"attempt": 2, "subtask_results": [{
            "finish_result": {
                "status": "done",
                "result": '[{"values":{"a":null,"b":11}}]',
            },
        }]}]
        self.assertEqual(_public_done_null_fields(rows), {"a"})

    def test_public_cross_report_conflict_is_evidenced_not_assumed(self):
        def record(value, ref):
            return {
                "status": "done",
                "result": json.dumps([{
                    "values": {"case_id": "Case-013", "metric": value},
                    "source_fact_refs": [ref],
                }]),
                "summary": "The record was delivered.",
            }
        goals = [{
            "operation": "verify_conflicting_evidence",
            "target_fields": ["metric"],
            "main_visible_justification": (
                "Two cited reports disagree about Case-013 metric."
            ),
        }]
        history = [
            {"attempt": 1, "subtask_results": [record(87, "s1:F001")]},
            {"attempt": 2, "subtask_results": [record(55, "s2:F003")]},
        ]
        evidence = build_main_decision_contract(
            history, goals, number_of_tasks=1,
        )["visible_evidence"]
        self.assertEqual(
            evidence["confirmed_cross_report_disagreements"][0][
                "reported_values"
            ], [87, 55],
        )
        history[1]["subtask_results"][0] = record(55, "s1:F002")
        evidence = build_main_decision_contract(
            history, goals, number_of_tasks=1,
        )["visible_evidence"]
        self.assertEqual(
            evidence["confirmed_cross_report_disagreements"], [],
        )

    def test_public_dossier_result_summary_conflict(self):
        history = [{"attempt": 1, "subtask_results": [{
            "status": "done",
            "result": '[{"values":{"case_id":"Dossier-0123A"}}]',
            "summary": "The summary records case id = Dossier-4567B.",
        }]}]
        requests = [{
            "operation": "verify_conflicting_evidence",
            "target_fields": ["case_id"],
            "main_visible_justification": (
                "The result and summary disagree about case id: "
                "Dossier-0123A versus Dossier-4567B."
            ),
        }]
        contract = build_main_decision_contract(
            history, requests, number_of_tasks=1,
        )
        found = contract["visible_evidence"][
            "confirmed_result_summary_conflicts"
        ]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["result_value"], "dossier-0123a")
        self.assertEqual(found[0]["summary_value"], "dossier-4567b")


class LatestBatchWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene, cls.reports, cls.clean_episode = sample(0)

    def _find_case(self):
        waves = self.scene["macro_graph"]["reference_waves"]
        delegates = [
            step for step in self.clean_episode["trajectory"]
            if step["action"] == "delegate_task"
        ]
        for round_number in range(3, len(delegates) + 2):
            if round_number - 2 >= len(waves):
                continue
            extension = make_latest_pair(
                self.scene, feedback_round=round_number,
                max_parallel_tasks=4,
            )
            if extension is not None:
                return round_number, extension
        return None, None

    def test_only_latest_batch_changes_and_added_reports_are_executable(self):
        round_number, extension = self._find_case()
        self.assertIsNotNone(extension, "fixture lacks latest-batch witness")
        original = deepcopy(self.scene)
        adjusted = extend_local_scene(self.scene, extension)
        base = extend_clean_prefix(self.clean_episode, extension)
        self.assertEqual(self.scene, original)
        for old, new in zip(
            self.clean_episode["trajectory"],
            base["trajectory"],
        ):
            if old["attempt"] < round_number - 1:
                self.assertEqual(old, new)
        latest = next(
            row for row in base["trajectory"]
            if row["attempt"] == round_number - 1
            and row["action"] == "delegate_task"
        )
        self.assertEqual(
            len(latest["params"]["tasks"]),
            len(self.scene["macro_graph"]["reference_waves"][
                round_number - 2
            ]) + 2,
        )
        self.assertEqual(
            len(latest["params"]["tasks"]),
            len(latest["result"]["subtask_results"]),
        )
        self.assertLessEqual(len(latest["params"]["tasks"]), 4)
        self.assertEqual(
            len(adjusted["macro_graph"]["reference_waves"][
                round_number - 2
            ]), len(latest["params"]["tasks"]),
        )
        self.assertTrue(extension["private_generation_basis"][
            "not_real_world_independent_confirmation"
        ])

    def test_prefer_one_new_report_disagreeing_with_prior_natural_report(self):
        delegates = [
            step for step in self.clean_episode["trajectory"]
            if step["action"] == "delegate_task"
        ]
        single = None
        for round_number in range(3, len(delegates) + 2):
            candidate = make_latest_pair(
                self.scene, feedback_round=round_number,
                base_episode=self.clean_episode, max_parallel_tasks=4,
            )
            if candidate is not None and candidate[
                "historical_anchor_used"
            ]:
                single = (round_number, candidate)
                break
        self.assertIsNotNone(
            single, "fixture lacks explicit prior source pair"
        )
        round_number, extension = single
        self.assertEqual(len(extension["entries"]), 1)
        self.assertEqual(
            extension["private_generation_basis"][
                "historical_anchor"]["report_round"]
            ] < round_number - 1, True,
        )
        original_history = deepcopy(self.clean_episode["trajectory"])
        expanded = extend_clean_prefix(self.clean_episode, extension)
        for before, after in zip(
            original_history, expanded["trajectory"]
        ):
            if before["attempt"] < round_number - 1:
                self.assertEqual(before, after)
        fault = AtomicFault(
            "fabricated_claim", extension["faulted_task_id"],
            round_number - 1, variant_index=1,
        ).as_legacy_fault()
        plan = {
            "index": 0, "scene_index": 0,
            "task_id": self.scene["task_id"],
            "sample_id": "single-latest-witness",
            "branch_round": round_number, "faults": [fault],
        }
        scenario = make_scenario(
            self.scene, plan, mode="event_driven",
            max_fault_atoms=1,
        )
        episode, turns, audit = build_branch(
            self.scene, self.reports, self.clean_episode,
            branch_round=round_number, faults=[fault],
            scenario_plan=scenario,
            witness_extension=extension,
            max_parallel_tasks=4, max_rounds=20,
        )
        self.assertTrue(any(
            issue["kind"] == "cross_source_conflict"
            for snap in audit["observable_issue_snapshots"]
            for issue in snap["issues"]
        ))
        self.assertEqual(audit["execution_outcome"], "recovered")
        self.assertTrue(turns)

    def test_direct_generator_exports_cross_report_candidate_diagnostics(self):
        round_number, extension = self._find_case()
        self.assertIsNotNone(extension)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            scenes = root / "scenes.jsonl"
            reports = root / "reports.jsonl"
            base = root / "clean_episodes.jsonl"
            for path, records in (
                (scenes, [self.scene]),
                (reports, self.reports),
                (base, [self.clean_episode]),
            ):
                path.write_text(
                    "".join(json.dumps(item) + "\n" for item in records),
                    encoding="utf-8",
                )
            result = generate_turns(
                scenes_path=scenes, reports_path=reports,
                clean_episodes_path=base,
                output_dir=root / "output",
                max_scenes=1, branches_per_scene=4,
                round_sampling_strategy="chronological",
                candidate_strategy="enumerate",
                scenario_mode="event_driven",
                atomic_fault_kinds=["no_result"],
                latest_cross_report_pairs=True,
                max_candidates_per_round=4,
                max_distractors_per_fault=1,
                max_selected_per_round=1,
                max_language_attempts_per_round=1,
                workers=1, no_progress=True,
                main_mode="template", report_mode="template",
            )
            self.assertGreater(
                result["latest_pair_candidates_planned"], 0
            )
            selections = [
                json.loads(line) for line in (
                    root / "output/private/candidate_selection.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(
                row["observable_issue_family"] == "inter_report_conflict"
                for row in selections
            ))
            self.assertTrue(any(
                row["execution_outcome"] == "recovered"
                and row["observable_issue_family"] == "inter_report_conflict"
                for row in selections
            ))

    def test_actual_replay_detects_rechecks_and_reviews_new_reports(self):
        round_number, extension = self._find_case()
        self.assertIsNotNone(extension)
        fault = AtomicFault(
            "fabricated_claim", extension["faulted_task_id"],
            origin_round=round_number - 1, variant_index=1,
        ).as_legacy_fault()
        plan = {
            "index": 0, "scene_index": 0,
            "task_id": self.scene["task_id"],
            "sample_id": "test-latest-witness",
            "branch_round": round_number,
            "faults": [fault],
        }
        scenario = make_scenario(
            self.scene, plan, mode="event_driven", max_fault_atoms=1,
        )
        scenario["blueprint"] = "latest_batch_cross_report_conflict"
        episode, _turns, audit = build_branch(
            self.scene, self.reports, self.clean_episode,
            branch_round=round_number, faults=[fault],
            scenario_plan=scenario, witness_extension=extension,
            max_parallel_tasks=4, max_rounds=20,
        )
        all_issues = [
            issue for snapshot in audit["observable_issue_snapshots"]
            for issue in snapshot["issues"]
        ]
        self.assertIn(
            "cross_source_conflict", {x["kind"] for x in all_issues}
        )
        self.assertTrue(audit["runtime"]["detected_conflicts"])
        self.assertIn(audit["execution_outcome"], {
            "recovered", "unresolved_visible_claim_conflict",
        })
        reviewed_ep, reviewed_audit = author_branch_reports(
            self.scene, self.reports, episode, audit,
        )
        witness_ids = {
            entry["task"]["task_id"] for entry in extension["entries"]
        }
        self.assertTrue(witness_ids <= {
            row["task_id"] for row in
            reviewed_audit["subagent_report_reviews"]
        })
        self.assertTrue(all(
            row["authoring"]["status"] == "accepted"
            for row in reviewed_audit["subagent_report_reviews"]
        ))
        # A strict no-model Main run still reconstructs its input from
        # the augmented public catalog. Stage-1/2 original validation
        # is performed against the unmodified original Scene above.
        output, _labels, last_audit = author_branch_main(
            extend_local_scene(self.scene, extension),
            reviewed_ep, reviewed_audit,
            max_rounds=20,
        )
        self.assertEqual(output["branch_round"], round_number)
        self.assertIsNotNone(last_audit["visible_claim_state"])


if __name__ == "__main__":
    unittest.main()
