"""V10: bounded exhaustive 4A exploration before costly 4B/4C LLM calls."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from environment.tests.test_untrusted_orchestration import sample
from environment.untrusted_orchestration.config import (
    DEFAULT_CONFIG, load_config,
)
from environment.untrusted_orchestration.scenario.candidate_exploration import (
    enumerate_round_candidates, expand_all_rounds,
)
from environment.untrusted_orchestration.scenario.candidate_selection import (
    select_recovered_candidates, classify_trial,
)
from environment.untrusted_orchestration.inspection.evidence_affordances import (
    analyze_scene_evidence_affordances,
)
from environment.untrusted_orchestration.generate_turns import (
    generate_turns, _plan_branches,
)


KINDS = [
    "no_result", "incomplete_retrieval",
    "wrong_retrieval", "wrong_entity_binding",
    "report_missing", "result_summary_conflict",
    "fabricated_claim",
]


class CandidateExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene, cls.reports, cls.base = sample(0)

    def _rounds(self):
        plans, _, _ = _plan_branches(
            [self.scene], {self.scene["task_id"]: self.base},
            branches_per_scene=1, seed=17,
            fault_modes=["missing", "partial", "wrong"],
            persistence="once", multi_target_probability=0,
            max_rounds=16, scenario_mode="event_driven",
            atomic_fault_kinds=KINDS,
        )
        return plans

    def test_enumerate_real_fault_instances_and_unique_ids(self):
        originals = self._rounds()
        expanded = expand_all_rounds(
            [self.scene], originals,
            atomic_fault_kinds=KINDS,
            max_candidates_per_round=24,
            max_distractors_per_fault=2,
            max_fault_atoms=1,
            delayed_evidence_probability=0,
            seed=17,
        )
        self.assertGreater(len(expanded), len(originals))
        self.assertEqual(
            len({plan["sample_id"] for plan in expanded}),
            len(expanded),
        )
        self.assertEqual(
            [plan["index"] for plan in expanded],
            list(range(len(expanded))),
        )
        self.assertTrue(all(
            plan["faults"][0].atom_kind in KINDS
            for plan in expanded if plan["faults"]
        ))
        self.assertTrue(all(
            len(plan["scenario_plan"]["fault_atoms"]) == 1
            for plan in expanded if plan["faults"]
        ))
        self.assertTrue(all(
            plan["scenario_plan"]["scenario_id"] == plan["sample_id"]
            for plan in expanded
        ))
        repeat = expand_all_rounds(
            [self.scene], originals,
            atomic_fault_kinds=KINDS,
            max_candidates_per_round=24,
            max_distractors_per_fault=2,
            max_fault_atoms=1,
            delayed_evidence_probability=0,
            seed=17,
        )
        self.assertEqual(
            [(p["fault_mode"], p["faults"][0].task_id,
              p["candidate_variant_index"]) for p in expanded],
            [(p["fault_mode"], p["faults"][0].task_id,
              p["candidate_variant_index"]) for p in repeat],
        )
        limited = enumerate_round_candidates(
            self.scene, originals[0], atomic_fault_kinds=KINDS,
            max_candidates=2, max_distractors=2, seed=17,
        )
        self.assertEqual(len(limited), 2)
        self.assertGreater(
            limited[0]["candidate_count_pruned"], 0
        )

    def test_all_feedback_rounds_and_all_fault_candidates_have_no_caps(self):
        from environment.untrusted_orchestration.scenario.round_sampling import (
            select_feedback_rounds,
        )
        all_rounds = [(r, {"action": "delegate_task"}) for r in range(2, 12)]
        self.assertEqual(
            select_feedback_rounds(
                all_rounds, limit=None, strategy="chronological", seed=17
            ), all_rounds,
        )
        original = self._rounds()[0]
        full = enumerate_round_candidates(
            self.scene, original, atomic_fault_kinds=KINDS,
            max_candidates=None, max_distractors=2, seed=17,
        )
        capped = enumerate_round_candidates(
            self.scene, original, atomic_fault_kinds=KINDS,
            max_candidates=2, max_distractors=2, seed=17,
        )
        self.assertGreaterEqual(len(full), len(capped))
        self.assertEqual(full[0]["candidate_count_pruned"], 0)
        self.assertEqual(len({p["sample_id"] for p in full}), len(full))

    def test_no_selection_limit_retains_every_distinct_recovery(self):
        plans = [{
            "index": i, "sample_id": f"no-cap-{i}",
            "task_id": "q", "branch_round": 3, "faults": [],
            "scenario_plan": {"fault_atoms": [{
                "kind": ("no_result" if i == 0
                         else "incomplete_retrieval" if i == 1
                         else "wrong_retrieval"),
            }]},
        } for i in range(3)]
        journals = [{
            "status": "recovered",
            "episode": {"trajectory": [
                {"attempt": 3, "action": "delegate_task"}
            ]},
            "private_audit": {
                "execution_outcome": "recovered",
                "runtime": {},
                "evidence_plan": [{"evidence_requests": [{
                    "operation": f"recovery_{i}",
                    "target_fields": ["value"],
                    "authorized_source_ids": ["source"],
                }]}],
            },
        } for i in range(3)]
        selected, rows, stats = select_recovered_candidates(
            plans, journals, max_selected_per_round=None,
        )
        self.assertEqual(selected, {0, 1, 2})
        self.assertEqual(stats["candidates_selected_for_language_review"], 3)
        self.assertTrue(all(
            row["selection_purpose"] == "positive" for row in rows
        ))

    def test_selection_prefers_actual_recovered_invalidation(self):
        plans = [{
            "index": i,
            "sample_id": f"Q-candidate-{i}",
            "task_id": "Q", "branch_round": 2,
            "faults": [],
            "scenario_plan": {"fault_atoms": [{
                "kind": "wrong_retrieval"
            }]},
        } for i in range(3)]
        def journal(outcome, *, invalidations=0, conflict=0, op=""):
            return {
                "status": outcome,
                "episode": {"trajectory": [
                    {"attempt": 2, "action": "delegate_task"}
                ]},
                "private_audit": {
                    "execution_outcome": outcome,
                    "fault_events_applied": [{}],
                    "runtime": {
                        "invalidations": [{}] * invalidations,
                        "detected_conflicts": [{}] * conflict,
                    },
                    "evidence_plan": [{
                        "evidence_requests": [{
                            "operation": op, "target_fields": ["marker_code"],
                            "authorized_source_ids": ["visual"],
                        }]
                    }],
                },
            }
        journals = [
            journal("recovered", op="retry_missing_evidence"),
            journal("recovered", invalidations=1, conflict=1,
                    op="verify_conflicting_evidence"),
            journal("unverified_report_claim"),
        ]
        selected, rows, stats = select_recovered_candidates(
            plans, journals, max_selected_per_round=1,
        )
        self.assertEqual(selected, {1})
        self.assertEqual(
            rows[1]["category"],
            "recovered_with_downstream_invalidation",
        )
        self.assertFalse(rows[2]["selected_for_language_review"])
        self.assertEqual(stats["candidates_simulated"], 3)
        self.assertEqual(stats["candidates_filtered_before_llm"], 2)


    def test_selects_distinct_historical_conflicts_before_routine_retries(self):
        """N can conflict with N-1, N-3 and the first feedback round."""
        plans = [{
            "index": i,
            "sample_id": f"Q-history-{i}",
            "task_id": "Q",
            "branch_round": 7,
            "faults": [],
            "scenario_plan": {"fault_atoms": [{
                "kind": "wrong_retrieval" if i in (1, 2, 3)
                else "incomplete_retrieval"
            }]},
        } for i in range(5)]

        def trial(old_round=None, *, outcome="recovered"):
            conflicts = [] if old_round is None else [{
                "kind": "cross_round_report_disagreement",
                "older_round": old_round,
                "latest_round": 6,
                "compared_parent_tasks": ["old", "new"],
            }]
            return {
                "status": outcome,
                "episode": {"trajectory": [
                    {"attempt": 7, "action": "delegate_task"},
                ]},
                "private_audit": {
                    "execution_outcome": outcome,
                    "runtime": {"detected_conflicts": conflicts},
                    "evidence_plan": [{"evidence_requests": [{
                        "operation": "verify_conflicting_evidence"
                        if old_round is not None else "retry_missing_evidence",
                        "target_fields": ["marker_code"],
                        "authorized_source_ids": ["record"],
                    }]}],
                },
            }

        journals = [trial(), trial(1), trial(3), trial(5), trial()]
        selected, rows, stats = select_recovered_candidates(
            plans, journals, max_selected_per_round=3,
            strategy="stratified",
        )
        self.assertEqual(selected, {1, 2, 3})
        self.assertEqual(rows[1]["history_conflict_older_rounds"], [1])
        self.assertEqual(rows[2]["history_conflict_round_gaps"], [3])
        self.assertEqual(rows[3]["history_conflict_round_gaps"], [1])
        self.assertEqual(stats["selected_recovered_historical_conflicts"], 3)
        self.assertEqual(
            stats["historical_conflict_older_round_distribution"],
            {1: 1, 3: 1, 5: 1},
        )
        self.assertEqual(
            stats["selected_historical_conflict_round_gap_distribution"],
            {1: 1, 3: 1, 5: 1},
        )
        # The history priority also applies at capacity one; increasing the
        # cap is optional, never a prerequisite for historical coverage.
        one, _, _ = select_recovered_candidates(
            plans, journals, max_selected_per_round=1,
            strategy="stratified",
        )
        self.assertEqual(len(one), 1)
        self.assertTrue(one <= {1, 2, 3})

    def test_nonrecovered_history_conflict_is_diagnostic_only(self):
        plans = [{
            "index": i, "sample_id": f"Q-history-negative-{i}",
            "task_id": "Q", "branch_round": 4, "faults": [],
            "scenario_plan": {"fault_atoms": [{
                "kind": "wrong_retrieval" if i else "incomplete_retrieval"
            }]},
        } for i in range(2)]

        def journal(status, conflict):
            return {
                "status": status,
                "episode": {"trajectory": [
                    {"attempt": 4, "action": "delegate_task"}
                ]},
                "private_audit": {
                    "execution_outcome": status,
                    "runtime": {"detected_conflicts": ([{
                        "kind": "cross_round_report_disagreement",
                        "older_round": 1, "latest_round": 3,
                    }] if conflict else [])},
                    "evidence_plan": [],
                },
            }

        chosen, rows, _ = select_recovered_candidates(
            plans, [
                journal("recovered", False),
                journal("incorrect_result_after_actual_propagation", True),
            ], max_selected_per_round=3, strategy="stratified",
            max_diagnostics_per_scene=1,
        )
        self.assertIn(0, chosen)
        self.assertEqual(rows[0]["selection_purpose"], "positive")
        self.assertEqual(rows[1]["selection_purpose"], "diagnostic")
        self.assertNotEqual(rows[1]["selection_purpose"], "positive")

    def test_template_end_to_end_enumeration_selects_before_reports(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            scenes = root / "scenes.jsonl"
            reports = root / "reports.jsonl"
            base = root / "base.jsonl"
            scenes.write_text(json.dumps(self.scene) + "\n")
            reports.write_text("".join(
                json.dumps(r) + "\n" for r in self.reports
            ))
            base.write_text(json.dumps(self.base) + "\n")
            out = root / "run"
            summary = generate_turns(
                scenes_path=scenes, reports_path=reports,
 output_dir=out,
                max_scenes=1, branches_per_scene=1, workers=2,
                no_progress=True, scenario_mode="event_driven",
                candidate_strategy="enumerate", atomic_fault_kinds=KINDS,
                max_candidates_per_round=8, max_distractors_per_fault=2,
                max_selected_per_round=1,
                delayed_evidence_probability=0.0,
            )
            self.assertEqual(summary["status"], "done")
            self.assertEqual(summary["requested_feedback_rounds"], 1)
            round_index = [
                json.loads(line) for line in
                (out / "feedback_round_index.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(round_index), 1)
            self.assertEqual(
                round_index[0]["candidate_count"],
                summary["candidate_trials"],
            )
            self.assertGreater(summary["candidate_trials"], 1)
            self.assertEqual(
                summary["candidate_trials"],
                summary["candidates_simulated"],
            )
            self.assertLessEqual(
                summary["candidates_selected_for_language_review"], 1
            )
            self.assertEqual(
                summary["screened_out_before_llm"],
                summary["candidate_trials"] -
                summary["candidates_selected_for_language_review"],
            )
            choices = [
                json.loads(line) for line in
                (out / "private" / "candidate_selection.jsonl")
                .read_text().splitlines()
            ]
            self.assertEqual(len(choices), summary["candidate_trials"])
            self.assertTrue(
                (out / "private" /
                 "scene_evidence_affordances.jsonl").is_file()
            )
            self.assertIn("potential_cross_check_macro_pairs", summary)
            self.assertEqual(
                len({row["round_sample_id"] for row in choices}),
                len(choices),
            )
            unselected = [
                row for row in choices
                if not row["selected_for_language_review"]
            ]
            self.assertTrue(unselected)
            for row in unselected:
                index = row["candidate_index"]
                stage_a = json.loads(
                    (out / "stage4a" / "journal" /
                     f"{index:06d}.json").read_text()
                )
                stage_b = json.loads(
                    (out / "stage4b" / "journal" /
                     f"{index:06d}.json").read_text()
                )
                self.assertEqual(
                    stage_b["status"], "not_selected_for_language_review"
                )
                self.assertIsNone(stage_b["episode"])
                self.assertEqual(
                    stage_b["private_audit"],
                    stage_a["private_audit"],
                )
            turns = [
                json.loads(line) for line in
                (out / "main_turns.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                len(turns), len({t["sample_id"] for t in turns})
            )
            self.assertTrue(all(
                set(turn["input"]) == {"subtask_history"}
                for turn in turns
            ))
            self.assertTrue(all(
                set(turn["output"]) == {"action", "reasoning", "params"}
                for turn in turns
            ))

    def test_independent_sources_can_be_complementary_not_verifiers(self):
        scene = {
            "task_id": "Q-affordance",
            "macro_graph": {
                "reference_waves": [["A", "B"], ["C"]],
                "tasks": [{
                    "task_id": "A", "depends_on": [],
                    "source_ids": ["visual"],
                    "outputs": {"typed_relation": [
                        "candidate_id", "marker_code"
                    ]}, "operator_chain": [],
                }, {
                    "task_id": "B", "depends_on": [],
                    "source_ids": ["caption"],
                    "outputs": {"typed_relation": ["candidate_id"]},
                    "operator_chain": [],
                }, {
                    "task_id": "C", "depends_on": ["A", "B"],
                    "source_ids": [],
                    "outputs": {"typed_relation": ["marker_code"]},
                    "operator_chain": [{"op": "join"}],
                }],
            },
        }
        facts = analyze_scene_evidence_affordances(scene)
        self.assertEqual(facts["independent_comparable_pairs"], 1)
        self.assertEqual(facts["complementary_fusion_pairs"], 1)
        self.assertEqual(facts["potential_cross_check_pairs"], 0)
        self.assertEqual(facts["same_wave_comparable_pairs"], 1)
        self.assertEqual(
            facts["provisional_downstream_before_late_witness_pairs"], 0
        )
        self.assertEqual(
            facts["candidate_relationships"][0]
            ["verification_status"], "schema_only_not_proven",
        )

    def test_macro_internal_join_and_verification_are_not_same_claim_proof(self):
        scene = {
            "task_id": "Q-existing-verifier",
            "world": {"sources": {
                "left": {"observations": [{
                    "evidence_origin_id": "publisher:A",
                    "fact_refs": ["left:F001"],
                }]},
                "right": {"observations": [{
                    "evidence_origin_id": "publisher:B",
                    "fact_refs": ["right:F001"],
                }]},
            }},
            "task_feedback": [{
                "task_id": "work_M1",
                "workproduct": [{
                    "values": {"entity_0": "ENT-10"},
                    "source_fact_refs": ["left:F001", "right:F001"],
                }],
            }],
            "macro_graph": {
                "reference_waves": [["work_M1"]],
                "tasks": [{
                    "task_id": "work_M1",
                    "macro_role": "verification",
                    "depends_on": [],
                    "source_ids": ["left", "right"],
                    "outputs": {"typed_relation": ["entity_0"]},
                    "operator_chain": [
                        {"op": "scan", "source_id": "left"},
                        {"op": "scan", "source_id": "right"},
                        {"op": "join", "left_on": "entity_0",
                         "right_on": "entity_0"},
                    ],
                }],
            },
        }
        profile = analyze_scene_evidence_affordances(scene)
        self.assertEqual(profile["potential_cross_check_pairs"], 0)
        self.assertEqual(profile["internal_multi_source_macros"], 1)
        self.assertEqual(profile["internal_verification_macros"], 1)
        self.assertEqual(
            profile["internal_multi_source_supported_outputs"], 1,
        )
        internal = profile["internal_macro_evidence"][0]
        self.assertEqual(
            internal["declared_origin_distinctness"],
            "distinct_observation_origins",
        )
        self.assertEqual(
            internal["claim_comparison_status"],
            "not_established_by_macro_structure",
        )
        self.assertEqual(
            internal["executed_cited_sources"], ["left", "right"],
        )

    def test_config_one_supported_single_fault_path(self):
        config = load_config()
        self.assertEqual(config["candidate_strategy"], "enumerate")
        self.assertNotIn("max_fault_atoms", config)
        self.assertIn("computation_error", config["atomic_fault_kinds"])
        self.assertIn("done_but_incomplete", config["atomic_fault_kinds"])
        self.assertIsNone(config["branches_per_scene"])
        self.assertIsNone(config["max_candidates_per_round"])
        self.assertIsNone(config["max_selected_per_round"])
        self.assertIsNone(config["max_language_attempts_per_round"])
        content = DEFAULT_CONFIG.read_text(encoding="utf-8")
        with TemporaryDirectory() as td:
            f = Path(td) / "config.yaml"
            f.write_text(content.replace(
                "  mode: llm",
                "  candidate_strategy: sample_one\n  mode: llm",
                1,
            ))
            with self.assertRaisesRegex(
                ValueError, "untrusted_unknown_config_key"
            ):
                load_config(f)
            f.write_text(content.replace(
                "max_selected_per_round: null",
                "max_selected_per_round: 0"
            ))
            with self.assertRaisesRegex(
                ValueError, "untrusted_invalid_max_selected_per_round"
            ):
                load_config(f)


if __name__ == "__main__":
    unittest.main()
