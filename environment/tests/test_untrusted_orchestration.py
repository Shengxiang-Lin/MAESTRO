"""Regression tests for round-local faulty evidence and dynamic Main recovery."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

from environment.__main__ import main
from environment.generate_question.spec import make_spec
from environment.generate_question.question import reference_question
from environment.generate_question.quality import compile_and_validate
from environment.implement_subtask.realize import realize_scene
from environment.base_orchestration.trajectory_builder import build_episode
from environment.untrusted_orchestration.faults import (
    Fault, FaultUnsupported, probe_eligible_fault_modes,
)
from environment.untrusted_orchestration.trajectory_builder import (
    build_branch, dependency_safe_batch, batch_has_independent_dependencies,
)
from environment.untrusted_orchestration.generate_turns import (
    generate_turns, _plan_branches,
)
from environment.untrusted_orchestration.config import DEFAULT_CONFIG, load_config
from environment.untrusted_orchestration.state_machine import ExecutionState
from environment.generate_question.program_ir import Relation


def sample(i=0):
    spec = make_spec(
        seed_record={
            "record_id": "untrusted-" + str(i), "topic": "Pomona, California",
            "domain_hint": "geography",
        },
        ordinal=i, run_seed=97, difficulty="hard",
        macro_first_v2_share=100,
    )
    scene = compile_and_validate(spec, reference_question(spec))
    reports = [{
        **deepcopy(row),
        "finish_result": {
            "status": "done",
            "result": "Observed: " + json.dumps(row["answer"]),
            "summary": "Grounded observations " + json.dumps(row["answer"]),
        },
        "report_authoring": {"status": "accepted"},
    } for row in realize_scene(scene)["macro_tasks"]]
    episode, _, _, status = build_episode(scene, reports)
    if status["status"] != "accepted":
        raise AssertionError("invalid_clean_fixture")
    return scene, reports, episode


class RoundLocalRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene, cls.reports, cls.episode = sample(0)

    def test_config_only_entrypoint_and_model_paths(self):
        config = load_config()
        self.assertEqual(DEFAULT_CONFIG.name, "untrusted_orchestration.yaml")
        self.assertEqual(config["mode"], "template")
        self.assertTrue(config["scenes_path"].is_absolute())
        self.assertEqual(config["fault_modes"], ["missing", "partial", "wrong"])
        with patch("environment.__main__.generate_untrusted_from_config",
                   return_value={"status": "partial"}) as runner:
            self.assertEqual(main(["generate-untrusted-turns",
                                   "--config", str(DEFAULT_CONFIG),
                                   "--mode", "template", "--max-scenes", "5"]), 0)
            self.assertEqual(runner.call_args.kwargs["max_scenes"], 5)

    def test_missing_report_has_outer_done_and_recovers_by_targeted_retry(self):
        tid = self.scene["macro_graph"]["reference_waves"][0][0]
        ep, turns, audit = build_branch(
            self.scene, self.reports, self.episode,
            branch_round=2, faults=[Fault(tid, "missing")],
        )
        self.assertEqual(audit["outcome"], "recovered")
        self.assertTrue(turns)
        self.assertEqual(turns[0]["round"], 2)
        self.assertIn("no result", turns[0]["input"]["subtask_history"])
        original = ep["trajectory"][0]["result"]["subtask_results"][0]
        self.assertTrue(original["done"])
        self.assertIsNone(original["finish_result"])
        self.assertEqual(len(audit["runtime"]["attempt_versions"][tid]), 2)
        self.assertEqual(audit["counterfactual_unrepaired"]["status"], "blocked")
        self.assertEqual(ep["trajectory"][-1]["action"], "complete")
        self.assertNotIn("private", json.dumps(turns))
        self.assertNotIn("fault", json.dumps(turns))

    def test_persistent_missing_is_not_magically_repaired(self):
        tid = self.scene["macro_graph"]["reference_waves"][0][0]
        episode, turns, audit = build_branch(
            self.scene, self.reports, self.episode, branch_round=2,
            faults=[Fault(tid, "missing", persistence="same_path")],
            max_retries=1,
        )
        self.assertEqual(turns, [])
        self.assertEqual(audit["outcome"], "unresolved_dependencies")
        self.assertNotEqual(episode["trajectory"][-1]["action"], "complete")

    def test_partial_preserves_fields_and_restores_upstream(self):
        tasks = {t["task_id"]: t for t in self.scene["macro_graph"]["tasks"]}
        tid = next(tid for tid in self.scene["macro_graph"]["reference_waves"][0]
                   if len(tasks[tid]["outputs"]["typed_relation"]) > 1)
        ep, turns, audit = build_branch(
            self.scene, self.reports, self.episode, branch_round=2,
            faults=[Fault(tid, "partial")],
        )
        self.assertEqual(audit["outcome"], "recovered")
        versions = audit["runtime"]["attempt_versions"][tid]
        self.assertEqual(versions[0]["status"], "partial")
        self.assertTrue(versions[0]["runtime"]["available_fields"])
        self.assertTrue(versions[0]["runtime"]["missing_fields"])
        self.assertEqual(versions[1]["status"], "done")
        self.assertTrue(turns)
        self.assertEqual(ep["trajectory"][-1]["action"], "complete")

    def test_wrong_join_uses_real_distractor_and_exposes_propagation(self):
        found = False
        for i in range(12):
            scene, rows, episode = (
                (self.scene, self.reports, self.episode)
                if i == 0 else sample(i)
            )
            targets = [task["task_id"] for task in scene["macro_graph"]["tasks"]
                       if any(x["op"] == "join"
                              for x in task["operator_chain"])]
            if not targets:
                continue
            tid = targets[0]
            wave = next(i for i, w in enumerate(
                scene["macro_graph"]["reference_waves"]) if tid in w)
            if wave + 2 > len([
                row for row in episode["trajectory"]
                if row["action"] == "delegate_task"
            ]) + 1:
                continue
            try:
                ep, turns, audit = build_branch(
                    scene, rows, episode, branch_round=wave + 2,
                    faults=[Fault(tid, "wrong")],
                )
            except FaultUnsupported:
                continue
            if audit["faults"][0]["execution"].get("kind") != (
                    "misjoined_real_candidates"):
                continue
            found = True
            self.assertEqual(audit["counterfactual_unrepaired"]["status"],
                             "completed")
            self.assertFalse(audit["counterfactual_unrepaired"]["correct"])
            self.assertEqual(audit["outcome"], "recovered")
            self.assertIn("Recheck only", json.dumps(turns[0]["output"]))
            versions = audit["runtime"]["attempt_versions"][tid]
            self.assertEqual(len(versions), 2)
            self.assertNotEqual(versions[0]["rows"], versions[1]["rows"])
            self.assertEqual(ep["trajectory"][-1]["action"], "complete")
            break
        self.assertTrue(found, "no valid wrong-join fixture found")

    def test_parallel_batches_are_independent_and_snapshot_safe(self):
        # This used to produce [work_ordered, work_distinct_answer]
        # in one O1 parallel batch, silently running the latter after
        # the former's retry inside the Python loop.
        for i in range(12):
            scene, rows, base = (
                (self.scene, self.reports, self.episode)
                if i == 0 else sample(i)
            )
            first = scene["macro_graph"]["reference_waves"][0]
            tid = first[0]
            episode, labels, audit = build_branch(
                scene, rows, base, branch_round=2,
                faults=[Fault(tid, "missing")],
            )
            self.assertEqual(audit["outcome"], "recovered")
            self.assertTrue(audit["parallel_batches_verified"])
            actual = [
                record["task_ids"] for record in audit["decision_audit"]
                if record.get("decision_kind") == "delegate"
            ]
            by_id = {task["task_id"]: task for task in
                     scene["macro_graph"]["tasks"]}
            for batch in actual:
                for child in batch:
                    self.assertFalse(
                        set(by_id[child]["depends_on"]) & set(batch),
                        (child, batch),
                    )
        # Unit-level direct dependency guard: independently available
        # downstream values still cannot be sent alongside parent repair.
        state = ExecutionState(self.scene, {})
        wave = self.scene["macro_graph"]["reference_waves"]
        all_tasks = state.tasks
        parent = next(t for t in wave[0]
                      if any(t in task["depends_on"]
                             for task in all_tasks.values()))
        child = next(t["task_id"] for t in all_tasks.values()
                     if parent in t["depends_on"])
        self.assertFalse(batch_has_independent_dependencies(
            state, [parent, child],
        ))
        self.assertEqual(
            dependency_safe_batch(state, [parent], [child], 4), [parent],
        ) if state.ready(parent) else None

    def test_public_lookup_key_mismatch_is_detected_without_gold(self):
        # Mock the current WorkProduct identity to *demonstrate* that
        # detection depends only on the two public report strings.
        state = ExecutionState(self.scene, {})
        parent, child = "parent", "lookup"
        state.tasks = {
            parent: {
                "task_id": parent, "depends_on": [],
                "source_ids": [], "operator_chain": [],
            },
            child: {
                "task_id": child, "depends_on": [parent],
                "source_ids": ["stat_a"],
                "operator_chain": [{
                    "id": "lookup-1", "op": "lookup_by_key",
                    "inputs": ["input_0"],
                    "input_key_field": "record_key_1",
                    "source_key_field": "record_key_1",
                }],
            },
        }
        from environment.untrusted_orchestration.state_machine import Attempt
        state.current = {
            child: Attempt(
                task_id=child, version=1, status="done",
                product=Relation(("record_key_1", "completed_count"), ({
                    "values": {"record_key_1": "KEY1-96979",
                               "completed_count": 13},
                    "support": ("stat_a:F001",),
                },)),
                available_fields=("record_key_1", "completed_count"),
                used_parent_versions={parent: 1}, runtime={},
            ),
        }
        state.public_reports = {
            parent: {"finish_result": {
                "status": "done",
                "result": "The indexed record key is KEY1-94705.",
                "summary": "Confirmed for this entity.",
            }},
            child: {"finish_result": {
                "status": "done",
                "result": '[{"values":{"record_key_1":"KEY1-96979",'
                          '"completed_count":13}}]',
                "summary": "The current keyed lookup returned one record.",
            }},
        }
        self.assertIn("KEY1-96979", state.visible_discrepancy(child))
        self.assertIn("KEY1-94705", state.visible_discrepancy(child))
        state.public_reports[child]["finish_result"]["result"] = (
            '[{"values":{"record_key_1":"KEY1-94705",'
            '"completed_count":13}}]'
        )
        self.assertIsNone(state.visible_discrepancy(child))
        # Private relation still wrong, but without observable conflict
        # the supervisor must not pretend it knows the hidden fault.
        state.public_reports[child]["finish_result"]["result"] = (
            "The lookup returned one supported record."
        )
        self.assertIsNone(state.visible_discrepancy(child))

    def test_partial_field_liveness_and_row_completeness(self):
        state = ExecutionState(self.scene, {})
        # If terminal only takes/projects entry_label, it does not need
        # a preceding already-used relative_step ordering field.
        parent, downstream = "work_ordered", "work_distinct_answer"
        state.tasks = {
            parent: {
                "task_id": parent, "depends_on": [],
                "source_ids": [], "operator_chain": [],
                "outputs": {"typed_relation": ["entry_label", "relative_step"]},
            },
            downstream: {
                "task_id": downstream, "depends_on": [parent],
                "source_ids": [],
                "external_inputs": ["ordered"],
                "input_schema": {
                    "input_0": ["entry_label", "relative_step"],
                },
                "operator_chain": [
                    {"id":"take","op":"take","inputs":["ordered"],"limit":1},
                    {"id":"project","op":"project","inputs":["take"],
                     "fields":["entry_label"]},
                ],
                "program_node_id": "project",
                "outputs": {"typed_relation": ["entry_label"]},
            },
        }
        state.scene = {
            **self.scene,
            "macro_graph": {
                **self.scene["macro_graph"],
                "terminal_task_id": downstream,
            },
        }
        self.assertEqual(
            state.required_parent_fields(state.tasks[downstream], parent),
            {"entry_label"},
        )
        # A real frozen Task uses arbitrary external node IDs, never input_0.
        for real_task in self.scene["macro_graph"]["tasks"]:
            if real_task["depends_on"] and any(
                op["op"] == "lookup_by_key"
                for op in real_task["operator_chain"]
            ):
                required = state.required_parent_fields(
                    real_task, real_task["depends_on"][0],
                )
                lookup = next(
                    op for op in real_task["operator_chain"]
                    if op["op"] == "lookup_by_key"
                )
                self.assertIn(lookup["input_key_field"], required)
                break
        from environment.untrusted_orchestration.state_machine import Attempt
        state.current[parent] = Attempt(
            task_id=parent, version=1, status="partial",
            product=Relation(("entry_label", "relative_step"), ({
                "values": {"entry_label": "Entry-03",
                           "relative_step": None},
                "support": ("register:F001",),
            },)),
            available_fields=("entry_label",),
            used_parent_versions={},
            runtime={"rows_complete": True, "missing_fields": ["relative_step"]},
        )
        self.assertTrue(state.ready(downstream))
        self.assertFalse(state.repair_required(parent))
        state.current[parent].runtime["rows_complete"] = False
        self.assertFalse(state.ready(downstream))
        self.assertTrue(state.repair_required(parent))

    def test_fault_sampler_only_selects_preflight_supported_modes(self):
        all_tasks = {t["task_id"]: t for t in self.scene["macro_graph"]["tasks"]}
        possible = probe_eligible_fault_modes(
            self.scene, ["missing", "partial", "wrong"],
        )
        self.assertEqual(set(possible), set(all_tasks))
        self.assertTrue(any("missing" in v for v in possible.values()))
        planned, no_base, no_eligible = _plan_branches(
            [self.scene], {self.scene["task_id"]: self.episode},
            branches_per_scene=3, seed=17,
            fault_modes=["missing", "partial", "wrong"],
            persistence="once", multi_target_probability=0.0,
            max_rounds=16,
        )
        self.assertFalse(no_base)
        self.assertFalse(no_eligible)
        for plan in planned:
            for fault in plan["faults"]:
                self.assertIn(fault.mode, possible[fault.task_id])
                self.assertIn(
                    fault.task_id,
                    self.scene["macro_graph"]["reference_waves"][
                        plan["branch_round"] - 2],
                )

    def test_journal_resume_positive_publication_and_manifest(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            scenes, reports, episodes = (
                root / "scenes.jsonl", root / "subtasks.jsonl",
                root / "episodes.jsonl",
            )
            scenes.write_text(json.dumps(self.scene) + "\n")
            reports.write_text("".join(json.dumps(x) + "\n"
                                       for x in self.reports))
            episodes.write_text(json.dumps(self.episode) + "\n")
            args = dict(
                scenes_path=scenes, reports_path=reports,
                base_episodes_path=episodes, output_dir=root / "output",
                max_scenes=1, branches_per_scene=1,
                seed=17, fault_modes=["missing"], workers=2,
                no_progress=True,
            )
            summary = generate_turns(**args)
            self.assertEqual(summary["requested_branches"], 1)
            self.assertEqual(summary["recovered"], 1)
            self.assertEqual(summary["fault_mode_outcomes"]["missing"],
                             {"recovered": 1})
            self.assertGreater(summary["main_training_turns"], 0)
            output = root / "output"
            lines = [
                json.loads(s) for s in
                (output / "main_turns.jsonl").read_text().splitlines()
            ]
            self.assertTrue(all(
                set(turn["input"]) == {"subtask_history"} for turn in lines
            ))
            self.assertNotIn("faults", json.dumps(lines))
            self.assertTrue((output / "private" /
                             "fault_alignment.jsonl").exists())
            self.assertEqual(generate_turns(**args), summary)
            self.assertEqual(len(lines),
                             len((output / "main_turns.jsonl")
                                 .read_text().splitlines()))
            with self.assertRaisesRegex(ValueError, "signature_mismatch"):
                generate_turns(**{**args, "seed": 99})
            journal = output / "journal" / "000000.json"
            recorded = json.loads(journal.read_text(encoding="utf-8"))
            recorded["private_audit"]["parallel_batches_verified"] = False
            journal.write_text(json.dumps(recorded), encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "unsafe_parallel_batch_cannot_publish",
            ):
                generate_turns(**args)


if __name__ == "__main__":
    unittest.main()
