"""Round-local Main decision publication and anti-oracle regressions."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
import yaml

from environment.untrusted_orchestration.runtime.faults import Fault
from environment.untrusted_orchestration.generate_turns import generate_turns
from environment.untrusted_orchestration.config import generate_from_config
from environment.untrusted_orchestration.trajectory_builder import build_branch
from environment.untrusted_orchestration.offline_pipeline import (
    author_branch_reports, author_branch_main,
)
from environment.untrusted_orchestration.scenario.single_turn_selection import (
    select_single_turn_candidates, coordination_pattern, evidence_relation,
)
from environment.tests.stage4_fixtures import sample


class SingleTurnBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene, cls.reports, cls.clean = sample(0)

    def test_fork_emits_one_main_decision_without_executing_future_reports(self):
        task = self.scene["macro_graph"]["reference_waves"][0][0]
        episode, turns, audit = build_branch(
            self.scene, self.reports, self.clean, branch_round=2,
            faults=[Fault(task, "missing")],
        )
        self.assertEqual(audit["outcome"], "single_turn_planned")
        self.assertEqual(audit["decision_unit"], "single_turn")
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["round"], 2)
        self.assertEqual(len(audit["evidence_plan"]), 1)
        self.assertTrue(audit["parallel_batches_verified"])
        self.assertIn("no result", turns[0]["input"]["subtask_history"])
        self.assertEqual(len(audit["runtime"]["attempt_versions"][task]), 1)
        self.assertEqual(len(episode["trajectory"]), 2)
        current = episode["trajectory"][-1]
        self.assertEqual(current["action"], "delegate_task")
        self.assertTrue(current["result"]["pending"])
        self.assertEqual(current["result"]["subtask_results"], [])
        self.assertNotIn("recovered", episode)

    def test_real_discovery_retrieval_fault_has_public_filter_witness(self):
        # The sampled frozen discovery task executes two full case-id
        # filters before projecting their intersection to the final Result.
        # A wrong case delivered after those filters is now publicly
        # checkable from the attempted filter outputs, not hidden Gold.
        task = self.scene["macro_graph"]["reference_waves"][0][0]
        self.assertEqual(
            next(x for x in self.scene["macro_graph"]["tasks"]
                 if x["task_id"] == task)["macro_role"], "discovery",
        )
        _, _, audit = build_branch(
            self.scene, self.reports, self.clean, branch_round=2,
            faults=[Fault(task, "wrong",
                          atom_kind="wrong_retrieval", variant_index=0)],
        )
        self.assertEqual(audit["outcome"], "single_turn_planned")
        self.assertTrue(any(
            issue.get("kind") == "scope_constraint_mismatch"
            for snapshot in audit.get("observable_issue_snapshots", [])
            for issue in snapshot.get("issues", [])
        ))

    def test_original_prior_clean_dispatches_and_reports_are_unchanged(self):
        target = self.scene["macro_graph"]["reference_waves"][1][0]
        episode, _, audit = build_branch(
            self.scene, self.reports, self.clean, branch_round=3,
            faults=[Fault(target, "missing")],
        )
        self.assertTrue(audit["stage3_prior_history_preserved"])
        assert episode["trajectory"]
        original = self.clean["trajectory"][0]
        previous = episode["trajectory"][0]
        self.assertEqual(original["params"], previous["params"])
        self.assertEqual(
            [r.get("finish_result") for r in
             original["result"]["subtask_results"]],
            [r.get("finish_result") for r in
             previous["result"]["subtask_results"]],
        )

    def test_stage4b_c_publish_one_turn_without_terminal_gold_recovery(self):
        task = self.scene["macro_graph"]["reference_waves"][0][0]
        episode, _, audit = build_branch(
            self.scene, self.reports, self.clean, branch_round=2,
            faults=[Fault(task, "missing")],
        )
        authored_episode, authored_audit = author_branch_reports(
            self.scene, self.reports, episode, audit,
        )
        final_episode, turns, final_audit = author_branch_main(
            self.scene, authored_episode, authored_audit,
        )
        self.assertEqual(final_audit["outcome"], "single_turn_accepted")
        self.assertEqual(len(turns), 1)
        self.assertEqual(final_episode["trajectory"][-1]["result"]["subtask_results"], [])
        self.assertEqual(turns[0]["round"], 2)
        self.assertIsNone(final_audit["decision_reward_replay_error"])
        self.assertEqual(len(final_audit["decision_reward_states"]), 1)


    def test_full_pipeline_publishes_each_reviewed_fork_as_exactly_one_turn(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = root / "scenes.jsonl"
            reports = root / "subtasks.jsonl"
            clean = root / "clean.jsonl"
            scenes.write_text(json.dumps(self.scene) + "\n")
            reports.write_text("".join(json.dumps(row) + "\n" for row in self.reports))
            clean.write_text(json.dumps(self.clean) + "\n")
            output = root / "roundwise"
            summary = generate_turns(
                scenes_path=scenes, reports_path=reports,
                clean_episodes_path=clean, output_dir=output,
                max_scenes=1, branches_per_scene=1,
                atomic_fault_kinds=["no_result", "result_summary_conflict"],
                max_candidates_per_round=2, max_distractors_per_fault=1,
                target_main_turns=2, workers=1, no_progress=True,
            )
            self.assertEqual(summary["decision_unit"], "single_turn")
            self.assertGreaterEqual(summary["single_turn_accepted"], 1)
            self.assertEqual(summary["decision_unit"], "single_turn")
            self.assertEqual(summary["main_training_turns"],
                             summary["single_turn_accepted"])
            self.assertEqual(summary["published_episodes"],
                             summary["single_turn_accepted"])
            self.assertEqual(summary[
                "published_single_turn_stage3_prior_history"
            ], {"preserved": summary["single_turn_accepted"]})
            self.assertEqual(
                sum(summary["published_single_turn_evidence_relations_4a"].values()),
                summary["single_turn_accepted"],
            )
            self.assertEqual(
                sum(summary["published_single_turn_coordination_patterns"].values()),
                summary["single_turn_accepted"],
            )
            rows = [json.loads(line) for line in
                    (output / "main_turns.jsonl").read_text().splitlines()]
            episodes = [json.loads(line) for line in
                        (output / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), len(episodes))
            self.assertTrue(all(ep["format"] == "maestro_untrusted_single_turn.v1"
                                and ep["trajectory"][-1]["result"]["pending"]
                                and ep["trajectory"][-1]["result"]["subtask_results"] == []
                                for ep in episodes))
            self.assertTrue(all(len([
                step for step in ep["trajectory"]
                if step["attempt"] >= ep["branch_round"]
            ]) == 1 for ep in episodes))
            original_waves = self.scene["macro_graph"]["reference_waves"]
            # No fabricated source recheck, new historical Main dispatch or
            # extra Sub-Agent report in either clean or latest batch.
            for ep in episodes:
                for index, step in enumerate(ep["trajectory"][:-1]):
                    self.assertEqual(
                        len(step["params"]["tasks"]),
                        len(original_waves[index]),
                    )
            index_rows = [
                json.loads(line) for line in
                (output / "round_index.jsonl").read_text().splitlines()
            ]
            self.assertTrue(all(
                row["latest_subagent_task_ids"] ==
                original_waves[row["main_decision_round"] - 2]
                for row in index_rows
            ))


    def test_pilot_yaml_entrypoint_executes_one_turn_without_old_kwargs(self):
        """Regression for the actual -m environment YAML adapter, not only
        direct calls to generate_turns in the other pipeline test.
        """
        with TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = root / "scenes.jsonl"
            reports = root / "subtasks.jsonl"
            clean = root / "clean.jsonl"
            scenes.write_text(json.dumps(self.scene) + "\n", encoding="utf-8")
            reports.write_text(
                "".join(json.dumps(row) + "\n" for row in self.reports),
                encoding="utf-8",
            )
            clean.write_text(json.dumps(self.clean) + "\n", encoding="utf-8")

            pilot = (Path(__file__).resolve().parents[1] / "configs"
                     / "untrusted_orchestration_llm_pilot_10.yaml")
            cfg = yaml.safe_load(pilot.read_text(encoding="utf-8"))
            cfg["input"].update({
                "scenes": str(scenes),
                "reports": str(reports),
                "clean_episodes": str(clean),
            })
            cfg["generation"].update({
                "output_dir": str(root / "from-yaml"),
                "max_scenes": 1,
                "branches_per_scene": 1,
                "workers": 1,
                "mode": "template",
                "atomic_fault_kinds": [
                    "no_result", "result_summary_conflict",
                ],
                "max_candidates_per_round": 2,
                "max_distractors_per_fault": 1,
                "target_main_turns": 2,
                "no_progress": True,
            })
            config_path = root / "pilot.yaml"
            config_path.write_text(
                yaml.safe_dump(cfg), encoding="utf-8",
            )
            summary = generate_from_config(config_path)
            self.assertEqual(summary["decision_unit"], "single_turn")
            self.assertGreaterEqual(summary["single_turn_accepted"], 1)
            self.assertEqual(
                len((root / "from-yaml" / "main_turns.jsonl")
                    .read_text(encoding="utf-8").splitlines()),
                summary["main_training_turns"],
            )


def trial(i, kind="wrong_retrieval"):
    return {"index": i, "task_id": "Q", "branch_round": 2,
            "scenario_plan": {"fault_atoms": [{"kind": kind}]}}


def journal(issue="Different returned keys", *, status="single_turn_planned"):
    return {"status": status,
            "episode": {"trajectory": [{
                "attempt": 2, "action": "delegate_task",
                "result": {"pending": True, "subtask_results": []},
            }]},
            "private_audit": {
                "parallel_batches_verified": True,
                "observable_issue_snapshots": [{
                    "decision_round": 2,
                    "issues": ([] if issue is None else [{
                        "reported_round": 1,
                        "task_id": "A",
                        "kind": "lookup_key_disagreement",
                        "evidence": issue,
                    }]),
                }],
                "runtime": {"detected_conflicts": []},
                "evidence_plan": [{
                    "evidence_requests": [{
                        "operation": "verify_conflicting_evidence",
                        "target_fields": ["entity_id"],
                    }]
                }],
            }}


class SingleTurnSelectionTests(unittest.TestCase):
    def test_selects_distinct_publicly_observable_cases_not_future_recovery(self):
        planned = [trial(i) for i in range(4)]
        results = [
            journal("Different returned keys"),
            journal("Different returned keys"),
            journal("Different returned years"),
            journal(None),
        ]
        selected, rows, stats = select_single_turn_candidates(
            planned, results, target_main_turns=3,
            max_selected_per_round=None, seed=17,
        )
        self.assertEqual(len(selected), 2)
        self.assertNotIn(3, selected)
        self.assertEqual(rows[1]["selection_reason"], "duplicate_public_issue_and_action")
        self.assertEqual(stats["single_turn_actionable"], 2)

    def test_independent_cross_source_and_same_source_are_not_conflated(self):
        independent = [{
            "kind": "explicit_independent_claim_disagreement",
            "latest_round": 1,
            "claim": {"subject": "Project-X", "attribute": "owner"},
        }]
        same_source = [{
            "kind": "cross_round_report_disagreement",
            "latest_round": 1,
            "comparison_kind": "same_query_result_set",
        }]
        self.assertEqual(
            evidence_relation([], independent),
            "independent_source_claim_conflict",
        )
        self.assertEqual(
            evidence_relation([], same_source),
            "same_source_repeated_query_conflict",
        )
        self.assertNotEqual(
            evidence_relation([], independent),
            evidence_relation([], same_source),
        )

    def test_utility_identifies_real_parallel_repair_and_continuation(self):
        plan = [{"dispatch": [{"assignment_id": "a"}, {"assignment_id": "b"}],
                 "evidence_requests": [
                     {"operation": "verify_conflicting_evidence"},
                     {"operation": "retrieve_required_evidence"},
                 ]}]
        self.assertEqual(coordination_pattern(plan)[0], "verify_and_advance")
        self.assertEqual(coordination_pattern([])[0], "single_task")

    def test_selector_balances_relations_before_repeating_easy_variants(self):
        plans = [trial(i, "fabricated_claim") for i in range(4)]
        audits = [journal(f"local variation {i}") for i in range(3)]
        other = journal(None)
        other["private_audit"]["runtime"]["detected_conflicts"] = [{
            "kind": "explicit_independent_claim_disagreement",
            "latest_round": 1,
        }]
        audits.append(other)
        chosen, rows, stats = select_single_turn_candidates(
            plans, audits, target_main_turns=2,
            max_selected_per_round=None, seed=17,
        )
        selected_relations = {
            row["evidence_relation"] for row in rows
            if row["candidate_index"] in chosen
        }
        self.assertEqual(
            selected_relations,
            {"lookup_key_disagreement", "independent_source_claim_conflict"},
        )
        self.assertEqual(sum(stats["selected_evidence_relations"].values()), 2)

    def test_budget_one_and_missing_reports_are_separate_faults(self):
        planned = [trial(0, "no_result"), trial(1, "result_summary_conflict")]
        results = [
            journal("No result returned"),
            journal("Result and Summary disagree"),
        ]
        selected, rows, stats = select_single_turn_candidates(
            planned, results, target_main_turns=1,
            max_selected_per_round=None, seed=17,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(stats["estimated_selected_main_turns"], 1)
        self.assertEqual(
            [r["selection_purpose"] for r in rows].count("budget_reserve"), 1
        )


if __name__ == "__main__":
    unittest.main()
