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
from environment.untrusted_orchestration.candidate_exploration import (
    enumerate_round_candidates, expand_all_rounds,
)
from environment.untrusted_orchestration.candidate_selection import (
    select_recovered_candidates, classify_trial,
)
from environment.untrusted_orchestration.evidence_affordances import (
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
        self.assertEqual(config["max_fault_atoms"], 1)
        self.assertEqual(config["max_candidates_per_round"], 12)
        self.assertEqual(config["max_selected_per_round"], 1)
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
                "max_selected_per_round: 1",
                "max_selected_per_round: 0"
            ))
            with self.assertRaisesRegex(
                ValueError, "untrusted_invalid_max_selected_per_round"
            ):
                load_config(f)


if __name__ == "__main__":
    unittest.main()
