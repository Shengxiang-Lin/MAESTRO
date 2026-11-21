"""Single-turn scenario-first generation: public witnesses, no Gold leaks."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from environment.tests.test_untrusted_orchestration import sample
from environment.untrusted_orchestration.generate_turns import generate_turns, _plan_branches
from environment.untrusted_orchestration.reporting.surface import render_records
from environment.untrusted_orchestration.scenario.candidate_exploration import enumerate_round_candidates
from environment.untrusted_orchestration.scenario.observable_scenarios import (
    TARGETS, classify_observable_scenario, comparable_incoming_tasks,
    feasible_fault_targets,
)
from environment.untrusted_orchestration.scenario.single_turn_selection import select_single_turn_candidates


class ScenarioTaxonomyTests(unittest.TestCase):
    def test_two_families_four_scenarios(self):
        mapping = {
            "missing_delivery": ("A", "evidence_insufficiency", None),
            "partial_delivery": ("A", "evidence_insufficiency", None),
            "intra_report_conflict": ("B1", "evidence_inconsistency", "internal"),
            "lookup_key_disagreement": ("B2", "evidence_inconsistency", "constraint"),
            "arithmetic_inconsistency": ("B2", "evidence_inconsistency", "constraint"),
            "scope_constraint_mismatch": ("B2", "evidence_inconsistency", "constraint"),
            "historical_report_conflict": ("B3", "evidence_inconsistency", "inter_report"),
            "cross_source_conflict": ("B3", "evidence_inconsistency", "inter_report"),
        }
        for kind, expected in mapping.items():
            with self.subTest(kind=kind):
                result = classify_observable_scenario([{"kind": kind}], [])
                self.assertEqual(
                    (result["scenario_id"], result["scenario_family"],
                     result["comparison_type"]), expected,
                )

    def test_public_conflicts_have_temporal_and_origin_attributes(self):
        state = classify_observable_scenario([], [{
            "kind": "explicit_independent_claim_disagreement",
            "older_round": 1, "latest_round": 3,
            "conflict_timing": "delayed",
        }])
        self.assertEqual(state["scenario_id"], "B3")
        self.assertEqual(state["temporal_relations"], ["cross_round"])
        self.assertEqual(state["source_relations"], ["independent"])
        unknown = classify_observable_scenario(
            [{"kind": "private_fabricated_claim"}], [],
        )
        self.assertIsNone(unknown["scenario_id"])
        self.assertEqual(unknown["observability"], "no_observable_issue")

    def test_multiple_issues_preserve_all_tags_with_one_primary(self):
        state = classify_observable_scenario([
            {"kind": "partial_delivery"},
            {"kind": "intra_report_conflict"},
            {"kind": "arithmetic_inconsistency"},
        ], [])
        self.assertEqual(state["scenario_id"], "B2")
        self.assertEqual(state["matched_scenarios"], ["B2", "A", "B1"])


class ScenarioFeasibilityTests(unittest.TestCase):
    @staticmethod
    def report(field="date_a", value="2015-06-12", ref="image_a:F001"):
        return {"finish_result": {
            "status": "done",
            "result": render_records([{
                "values": {"entity_final": "Project-X", field: value},
                "source_fact_refs": [ref],
            }]),
            "summary": "Completed investigation.",
        }}

    def test_b3_requires_two_delivered_comparable_reports(self):
        tasks = [
            {"task_id": tid, "depends_on": [], "source_ids": [source],
             "outputs": {"typed_relation": ["entity_final", "date_a"]},
             "operator_chain": [{"op": "scan"}]}
            for tid, source in (("A", "image_a"), ("B", "audio_b"))
        ]
        scene = {"macro_graph": {
            "tasks": tasks, "reference_waves": [["A"], ["B"]],
        }}
        clean = {"trajectory": [
            {"action": "delegate_task", "attempt": 1,
             "result": {"subtask_results": [self.report()]}},
            {"action": "delegate_task", "attempt": 2,
             "result": {"subtask_results": [self.report(ref="audio_b:F002")]}},
        ]}
        round_plan = {"incoming_task_ids": ["B"], "branch_round": 3}
        self.assertEqual(comparable_incoming_tasks(scene, clean, round_plan), {"B"})
        self.assertIn(("B3", "B", "fabricated_claim"),
                      feasible_fault_targets(
                          scene, round_plan, kinds=["fabricated_claim"],
                          base_episode=clean,
                      ))
        opposite = deepcopy(clean)
        opposite["trajectory"][0]["result"]["subtask_results"][0] = self.report(
            field="date_b",
        )
        self.assertEqual(comparable_incoming_tasks(scene, opposite, round_plan),
                         set())
        self.assertEqual(comparable_incoming_tasks(scene, None, round_plan),
                         set())

    def test_single_turn_constructor_freezes_stage3_and_uses_one_atom(self):
        scene, _, clean = sample(0)
        kinds = [kind for kind in TARGETS["A"][2]] + [
            "result_summary_conflict", "fabricated_claim",
            "wrong_retrieval", "wrong_entity_binding", "computation_error",
        ]
        rounds, _, _ = _plan_branches(
            [scene], {scene["task_id"]: clean},
            branches_per_scene=1, seed=17, fault_modes=["missing", "partial", "wrong"],
            persistence="once", multi_target_probability=0,
            max_rounds=16, scenario_mode="event_driven",
            atomic_fault_kinds=kinds,
        )
        candidates = enumerate_round_candidates(
            scene, rounds[0], atomic_fault_kinds=kinds,
            max_candidates=15, max_distractors=2,
            seed=17, base_episode=clean, preserve_stage3_history=True,
        )
        self.assertTrue(candidates)
        for item in candidates:
            self.assertIn(item["target_scenario"], TARGETS)
            self.assertEqual(len(item["faults"]), 1)
            self.assertEqual(len(item["scenario_plan"]["fault_atoms"]), 1)
            self.assertNotIn("same_source_recheck_plan", item["scenario_plan"])
            self.assertNotIn("history_first_plan", item["scenario_plan"])
        with self.assertRaisesRegex(
            ValueError, "single_turn_requires_one_immediate_fault",
        ):
            enumerate_round_candidates(
                scene, rounds[0], atomic_fault_kinds=kinds,
                max_candidates=15, max_distractors=2,
                seed=17, base_episode=clean, preserve_stage3_history=True,
                delayed_evidence_probability=0.5,
            )


class ScenarioSelectionTests(unittest.TestCase):
    def test_target_mismatch_is_not_a_training_label(self):
        plans = [{
            "index": i, "task_id": "Q", "branch_round": 2,
            "target_scenario": scenario,
            "scenario_plan": {"fault_atoms": [{"kind": "fabricated_claim"}]},
        } for i, scenario in enumerate(("B3", "B1"))]
        journal = {
            "status": "single_turn_planned",
            "episode": {"trajectory": [{
                "attempt": 2, "action": "delegate_task",
            }]},
            "private_audit": {
                "observable_issue_snapshots": [{
                    "decision_round": 2, "issues": [{
                        "task_id": "M1", "reported_round": 1,
                        "kind": "intra_report_conflict",
                        "evidence": "Result contradicts Summary",
                    }],
                }],
                "runtime": {"detected_conflicts": []},
                "parallel_batches_verified": True,
                "evidence_plan": [{"evidence_requests": [{
                    "operation": "verify_conflicting_evidence",
                    "target_fields": ["date_a"],
                }]}],
            },
        }
        selected, rows, stats = select_single_turn_candidates(
            plans, [journal, journal], target_main_turns=2,
            max_selected_per_round=None, seed=17,
        )
        self.assertEqual(selected, {1})
        self.assertEqual(rows[0]["selection_reason"],
                         "target_scenario_not_publicly_realized")
        self.assertEqual(stats["selected_scenarios"], {"B1": 1})

    def test_publication_reports_scenarios_and_preserves_prefix(self):
        scene, reports, clean = sample(0)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / n for n in ("scene.jsonl", "report.jsonl", "clean.jsonl")]
            paths[0].write_text(json.dumps(scene) + "\n")
            paths[1].write_text("".join(json.dumps(r) + "\n" for r in reports))
            paths[2].write_text(json.dumps(clean) + "\n")
            out = root / "out"
            summary = generate_turns(
                scenes_path=paths[0], reports_path=paths[1],
                clean_episodes_path=paths[2], output_dir=out,
                max_scenes=1, branches_per_scene=1,
                decision_unit="single_turn", scenario_mode="event_driven",
                candidate_strategy="enumerate", selection_strategy="scenario_first",
                atomic_fault_kinds=["no_result", "result_summary_conflict"],
                max_candidates_per_round=2, max_distractors_per_fault=1,
                target_main_turns=2, workers=1, no_progress=True,
            )
            self.assertEqual(summary["selection_strategy"], "scenario_first")
            # Never replace observable screening reasons with the legacy
            # multi-round "not_execution_recovered" label.
            selection = [
                json.loads(line) for line in
                (out / "private/candidate_selection.jsonl").read_text().splitlines()
            ]
            self.assertTrue(selection)
            self.assertEqual(
                sum(summary["candidate_screening_outcomes"].values()),
                len(selection),
            )
            self.assertEqual(
                sum(summary["target_realization_distribution"].values()),
                len(selection),
            )
            self.assertEqual(
                sum(summary["no_public_witness_by_fault_kind"].values()),
                summary["candidate_screening_outcomes"].get(
                    "no_public_witness", 0
                ),
            )
            self.assertEqual(
                sum(summary["budget_reserve_by_scenario"].values()),
                summary["candidate_screening_outcomes"].get(
                    "budget_reserve", 0
                ),
            )
            self.assertTrue(all(
                row["selection_reason"] not in {
                    "not_execution_recovered", "language_review_primary",
                }
                and "language_review_selection_reason" in row
                for row in selection
            ))
            self.assertEqual(sum(summary["published_scenario_types"].values()),
                             summary["single_turn_accepted"])
            self.assertEqual(sum(summary["published_scenario_families"].values()),
                             summary["single_turn_accepted"])
            self.assertEqual(
                summary["published_single_turn_stage3_prior_history"],
                {"preserved": summary["single_turn_accepted"]},
            )
            public = (out / "main_turns.jsonl").read_text()
            self.assertNotIn('"fault_kind"', public)
            self.assertNotIn('"target_scenario"', public)


if __name__ == "__main__":
    unittest.main()
