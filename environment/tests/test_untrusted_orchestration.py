"""Regression tests for round-local faulty evidence and dynamic Main recovery."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

from environment.__main__ import main
from environment.tests.current_stage1 import current_scene
from environment.implement_subtask.realize import realize_scene
from environment.base_orchestration.trajectory_builder import build_episode
from environment.untrusted_orchestration.runtime.faults import (
    Fault, FaultUnsupported, probe_eligible_fault_modes,
)
from environment.untrusted_orchestration.trajectory_builder import (
    build_branch, dependency_safe_batch, batch_has_independent_dependencies,
)
from environment.untrusted_orchestration.generate_turns import (
    generate_turns, _plan_branches,
)
from environment.untrusted_orchestration.config import DEFAULT_CONFIG, load_config
from environment.untrusted_orchestration.state_machine import ExecutionState, Attempt
from environment.untrusted_orchestration.reporting.author import naturalize_report
from environment.untrusted_orchestration.config import generate_from_config
from environment.generate_question.execution.program_ir import Relation


def sample(i=0):
    scene = current_scene(i)
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
        raise AssertionError("invalid_clean_fixture:" + str(status))
    return scene, reports, episode


class RoundLocalRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene, cls.reports, cls.episode = sample(0)

    def test_config_only_entrypoint_and_model_paths(self):
        config = load_config()
        self.assertEqual(DEFAULT_CONFIG.name, "untrusted_orchestration.yaml")
        self.assertEqual(config["mode"], "llm")
        self.assertTrue(config["scenes_path"].is_absolute())
        self.assertEqual(config["candidate_strategy"], "enumerate")
        self.assertNotIn("max_fault_atoms", config)
        self.assertNotIn("base_episodes_path", config)
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

    def test_wrong_join_uses_real_distractor_and_records_propagation(self):
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
            # A wrong join may be recovered or may propagate into a
            # completed incorrect answer in the current Task Plan. The
            # simulator must report the actual outcome, not fabricate
            # recovery merely because an independent wrong candidate exists.
            self.assertIn(audit["outcome"], {
                "recovered", "incorrect_result_after_actual_propagation",
            })
            versions = audit["runtime"]["attempt_versions"][tid]
            self.assertGreaterEqual(len(versions), 1)
            if audit["outcome"] == "recovered":
                self.assertIn("Reconcile the visible conflicting",
                              json.dumps(turns[0]["output"]))
                self.assertEqual(len(versions), 2)
                self.assertNotEqual(versions[0]["rows"], versions[1]["rows"])
                self.assertEqual(ep["trajectory"][-1]["action"], "complete")
            else:
                self.assertTrue(audit["counterfactual_unrepaired"]["status"])
                self.assertFalse(audit["counterfactual_unrepaired"]["correct"])
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
        delegates = [
            x for x in self.episode["trajectory"]
            if x["action"] == "delegate_task"
        ]
        # A diagnostic cap returns the first 3 chronological feedback
        # rounds; this is never random branch sampling.
        self.assertEqual(
            [p["branch_round"] for p in planned],
            list(range(2, min(3, len(delegates)) + 2)),
        )
        for plan in planned:
            self.assertEqual(plan["unit"], "main_feedback_round")
            self.assertEqual(plan["received_from_delegate_round"],
                             plan["branch_round"] - 1)
            self.assertEqual(
                plan["incoming_task_ids"],
                self.scene["macro_graph"]["reference_waves"][
                    plan["branch_round"] - 2
                ],
            )
            for fault in plan["faults"]:
                self.assertIn(fault.mode, possible[fault.task_id])
                self.assertIn(
                    fault.task_id,
                    self.scene["macro_graph"]["reference_waves"][
                        plan["branch_round"] - 2],
                )

    def test_every_base_feedback_round_is_extractable(self):
        """No fixed 2-branch cap; every received Sub-Agent batch is a unit."""
        delegates = [
            x for x in self.episode["trajectory"]
            if x["action"] == "delegate_task"
        ]
        self.assertGreater(len(delegates), 2)
        kw = dict(
            scenes=[self.scene],
            base={self.scene["task_id"]: self.episode},
            branches_per_scene=None, seed=17,
            fault_modes=["missing", "partial", "wrong"],
            persistence="once", multi_target_probability=0.0,
            max_rounds=16,
        )
        first, no_base, _ = _plan_branches(**kw)
        again, _, _ = _plan_branches(**kw)
        self.assertFalse(no_base)
        self.assertEqual(len(first), len(delegates))
        self.assertEqual(
            [p["branch_round"] for p in first],
            list(range(2, len(delegates) + 2)),
        )
        self.assertEqual(
            [(p["sample_id"], [(f.task_id, f.mode, f.seed)
                               for f in p["faults"]]) for p in first],
            [(p["sample_id"], [(f.task_id, f.mode, f.seed)
                               for f in p["faults"]]) for p in again],
        )
        self.assertEqual(
            len({p["sample_id"] for p in first}), len(first),
        )
        self.assertTrue(all(
            p["incoming_subtask_count"] == len(p["incoming_task_ids"])
            for p in first
        ))
        # Future addition of a shorter diagnostic cap must never alter
        # which (question, round) is at the start of the run.
        capped, _, _ = _plan_branches(
            **{**kw, "branches_per_scene": 2},
        )
        self.assertEqual(
            [p["sample_id"] for p in capped],
            [p["sample_id"] for p in first[:2]],
        )

    def test_unsupported_fault_round_is_preserved_in_plan(self):
        # Preflight must never silently discard an accepted feedback round,
        # even if no configured fault can execute on its most recent batch.
        delegates = [x for x in self.episode["trajectory"]
                     if x["action"] == "delegate_task"]
        planned, _, _ = _plan_branches(
            [self.scene], {self.scene["task_id"]: self.episode},
            branches_per_scene=None, seed=17, fault_modes=["wrong"],
            persistence="once", multi_target_probability=0.0,
            max_rounds=16,
        )
        self.assertEqual(len(planned), len(delegates))
        for plan in planned:
            if plan["skip_reason"]:
                self.assertEqual(
                    plan["skip_reason"],
                    "no_executable_fault_on_returned_batch",
                )
                self.assertEqual(plan["faults"], [])
            else:
                self.assertTrue(plan["faults"])
                self.assertTrue(all(f.mode == "wrong" for f in plan["faults"]))

    def test_exhaustive_round_generation_has_one_journal_per_feedback_round(self):
        """Stage 4A runs over rounds, 4B/4C consume its completed units."""
        delegates = [x for x in self.episode["trajectory"]
                     if x["action"] == "delegate_task"]
        with TemporaryDirectory() as td:
            root = Path(td)
            scenes = root / "scenes.jsonl"
            reports = root / "reports.jsonl"
            episodes = root / "episodes.jsonl"
            scenes.write_text(json.dumps(self.scene) + "\n")
            reports.write_text(
                "".join(json.dumps(r) + "\n" for r in self.reports)
            )
            episodes.write_text(json.dumps(self.episode) + "\n")
            summary = generate_turns(
                scenes_path=scenes, reports_path=reports,
                output_dir=root / "all_rounds",
                max_scenes=None, branches_per_scene=None,
                seed=17, fault_modes=["missing", "partial", "wrong"],
                workers=2, no_progress=True,
            )
            output = root / "all_rounds"
            rows = [
                json.loads(line) for line in
                (output / "round_index.jsonl").read_text().splitlines()
            ]
            self.assertEqual(summary["rounds_extracted_from_base"],
                             len(delegates))
            self.assertEqual(summary["candidate_trials"],
                             len(delegates))
            self.assertEqual(summary["completed_branches"],
                             len(delegates))
            self.assertEqual(summary["status"], "done")
            self.assertEqual(summary["questions_with_extracted_rounds"], 1)
            self.assertEqual(summary["rounds_per_question"],
                             {self.scene["task_id"]: len(delegates)})
            self.assertEqual(len(rows), len(delegates))
            self.assertEqual(
                [row["main_decision_round"] for row in rows],
                list(range(2, len(delegates) + 2)),
            )
            self.assertEqual(
                [row["received_from_delegate_round"] for row in rows],
                list(range(1, len(delegates) + 1)),
            )
            for stage in ("stage4a/journal", "stage4b/journal", "journal"):
                self.assertEqual(
                    len(list((output / stage).glob("*.json"))),
                    len(delegates),
                )
            # Resume all stages without repeating already completed
            # simulation/authoring or changing the published output.
            self.assertEqual(generate_turns(
                scenes_path=scenes, reports_path=reports,
                output_dir=output, max_scenes=None, branches_per_scene=None,
                seed=17, fault_modes=["missing", "partial", "wrong"],
                workers=2, no_progress=True,
            ), summary)

    def test_untrusted_subagent_llm_preserves_wrong_record_not_gold(self):
        task = {
            "task_id": "faulty-lookup",
            "source_ids": ["ledger"],
            "operator_chain": [{
                "op": "lookup_by_key", "source_id": "ledger",
                "input_key_field": "record_key_1",
                "source_key_field": "record_key_1",
            }],
        }
        from environment.generate_question.execution.program_ir import Relation
        observed = Attempt(
            task_id="faulty-lookup", version=1, status="done",
            product=Relation(("record_key_1", "completed_count"), ({
                "values": {"record_key_1": "KEY1-96979",
                           "completed_count": 13},
                "support": ("ledger:F013",),
            },)),
            available_fields=("record_key_1", "completed_count"),
            used_parent_versions={}, runtime={},
        )

        class Author:
            def __init__(self, wrong=False):
                self.wrong = wrong
                self.payloads = []

            def complete_json(self, system, request):
                observed_attempt = json.loads(request)[
                    "observed_current_attempt"
                ]
                self.payloads.append(observed_attempt)
                key = "KEY1-94705" if self.wrong else "KEY1-96979"
                return {
                    "result": ("The returned ledger entry " + key +
                               " has a completed count of 13."),
                    "summary": (
                        "I checked the assigned keyed ledger and recorded "
                        "the returned count for this entry."
                    ),
                    "trace_summary": (
                        "I looked up the record key in the supplied "
                        "ledger and returned the observed measurement."
                    ),
                    "status": observed_attempt["delivery_status"],
                }

        class Critic:
            def complete_json(self, system, request):
                return {
                    "verdict": "pass", "grounded": True,
                    "scope_preserved": True, "natural": True,
                    "useful": True,
                }

        author = Author()
        report, meta = naturalize_report(
            task, observed, instruction="Read the returned ledger key.",
            model_alias="model_1", reviewed=None,
            author=author, critic=Critic(), max_revisions=0,
        )
        self.assertEqual(meta["status"], "accepted")
        self.assertEqual(meta["mode"], "llm_reviewed")
        self.assertIn("KEY1-96979", report["finish_result"]["result"])
        self.assertIn("13", report["finish_result"]["result"])
        self.assertIn("checked", report["finish_result"]["summary"])
        self.assertIn("looked up", report["trace_summary"])
        self.assertNotIn("gold_answer", json.dumps(author.payloads))
        self.assertNotIn("fault", json.dumps(author.payloads).lower())

        rejected, meta = naturalize_report(
            task, observed, instruction="Read the returned ledger key.",
            model_alias="model_1", reviewed=None,
            author=Author(wrong=True), critic=Critic(), max_revisions=0,
        )
        self.assertEqual(meta["status"], "fallback")
        self.assertEqual(meta["mode"], "grounded_template_fallback")
        self.assertIn("result_unobserved_numeric_or_entity",
                      meta["failures"][0]["errors"])
        self.assertIn("KEY1-96979", rejected["finish_result"]["result"])
        self.assertNotIn("KEY1-94705", rejected["finish_result"]["result"])

    def test_4b_critic_boundary_rereview_does_not_rewrite_results(self):
        from environment.generate_question.execution.program_ir import Relation
        task = {
            "task_id": "lookup-record",
            "source_ids": ["index"],
            "operator_chain": [{
                "op": "lookup_by_key", "source_id": "index",
                "input_key_field": "record_key_1",
                "source_key_field": "record_key_1",
            }],
        }
        attempt = Attempt(
            task_id="lookup-record", version=1, status="done",
            product=Relation(("record_key_1", "completed_count"), ({
                "values": {
                    "record_key_1": "KEY1-96979",
                    "completed_count": 13,
                }, "support": ("index:F013",),
            },)),
            available_fields=("record_key_1", "completed_count"),
            used_parent_versions={}, runtime={},
        )

        class Author:
            def complete_json(self, system, request):
                return {
                    "result": (
                        "The returned ledger entry KEY1-96979 has a "
                        "completed count of 13."
                    ),
                    "summary": (
                        "The supplied ledger returned this observation "
                        "without an independent source check."
                    ),
                    "trace_summary": (
                        "I examined the assigned keyed record and "
                        "reported the fields actually delivered."
                    ),
                }

        class Critic:
            def __init__(self, second_pass):
                self.calls = 0
                self.second_pass = second_pass
            def complete_json(self, system, request):
                self.calls += 1
                if "SECOND REVIEW" in system and self.second_pass:
                    return {
                        "verdict": "pass", "grounded": True,
                        "scope_preserved": True, "natural": True,
                        "useful": True,
                    }
                return {
                    "verdict": "fail", "grounded": False,
                    "scope_preserved": False, "natural": True,
                    "useful": False,
                    "reason": (
                        "An intermediate lookup allegedly showed another "
                        "key, so the final delivered value should be "
                        "replaced with the earlier observation."
                    ),
                }

        critic = Critic(second_pass=True)
        report, meta = naturalize_report(
            task, attempt, instruction="Read the assigned ledger record.",
            model_alias="model_1", reviewed=None,
            author=Author(), critic=critic, max_revisions=0,
        )
        self.assertEqual(critic.calls, 2)
        self.assertEqual(meta["mode"], "llm_reviewed")
        self.assertTrue(meta["critic_boundary_rereviewed"])
        self.assertIn("KEY1-96979", report["finish_result"]["result"])

        rejecting = Critic(second_pass=False)
        _, failed = naturalize_report(
            task, attempt, instruction="Read the assigned ledger record.",
            model_alias="model_1", reviewed=None,
            author=Author(), critic=rejecting, max_revisions=0,
        )
        self.assertEqual(rejecting.calls, 2)
        self.assertEqual(failed["mode"], "grounded_template_fallback")

    def test_llm_author_preserves_partial_and_missing_status(self):
        from environment.generate_question.execution.program_ir import Relation
        task = {
            "task_id": "read-ledger", "source_ids": ["ledger"],
            "operator_chain": [{
                "op": "lookup_by_key", "source_id": "ledger",
                "input_key_field": "record_key_1",
                "source_key_field": "record_key_1",
            }],
        }
        class Author:
            def complete_json(self, system, request):
                status = json.loads(request)["observed_current_attempt"][
                    "delivery_status"
                ]
                if status == "missing":
                    return {
                        "status": "missing", "result": None,
                        "summary": (
                            "The assigned attempt ended without a usable "
                            "result to send back."
                        ),
                        "trace_summary": (
                            "The evidence acquisition finished but no "
                            "inspectable record was delivered."
                        ),
                    }
                return {
                    "status": "partial",
                    "result": (
                        "The record KEY1-96979 was returned, but its "
                        "completed count remains unresolved."
                    ),
                    "summary": (
                        "The completed_count field is unresolved although "
                        "the requested record key was delivered."
                    ),
                    "trace_summary": (
                        "I received the requested record identity without "
                        "a corresponding completed count value."
                    ),
                }
        class Critic:
            def complete_json(self, system, request):
                return {
                    "verdict": "pass", "grounded": True,
                    "scope_preserved": True, "natural": True,
                    "useful": True,
                }

        partial = Attempt(
            task_id="read-ledger", version=1, status="partial",
            product=Relation(("record_key_1", "completed_count"), ({
                "values": {
                    "record_key_1": "KEY1-96979", "completed_count": None,
                },
                "support": ("ledger:F013",),
            },)),
            available_fields=("record_key_1",),
            used_parent_versions={},
            runtime={"missing_fields": ["completed_count"],
                     "rows_complete": True},
        )
        report, meta = naturalize_report(
            task, partial, instruction="Read the completed count.",
            model_alias="model_1", reviewed=None,
            author=Author(), critic=Critic(),
        )
        self.assertEqual(meta["mode"], "llm_reviewed")
        self.assertEqual(report["finish_result"]["status"], "partial")
        self.assertIn("KEY1-96979", report["finish_result"]["result"])

        missing = Attempt(
            task_id="read-ledger", version=2, status="missing",
            product=None, available_fields=(),
            used_parent_versions={}, runtime={},
        )
        report, meta = naturalize_report(
            task, missing, instruction="Read the completed count.",
            model_alias="model_1", reviewed=None,
            author=Author(), critic=Critic(),
        )
        self.assertEqual(meta["mode"], "llm_reviewed")
        self.assertIsNone(report["finish_result"])
        self.assertTrue(report["done"])
        self.assertIn("without a usable", report["trace_summary"])

    def test_one_mode_configures_both_report_and_main_clients(self):
        instances = []
        class Backend:
            def __init__(self, *, model, **kwargs):
                instances.append(model)
        with patch(
            "environment.generate_question.authoring.backend.OpenAICompatibleBackend",
            Backend,
        ), patch(
            "environment.untrusted_orchestration.generate_turns.generate_turns",
            return_value={"status": "done"},
        ) as writer:
            generate_from_config(DEFAULT_CONFIG, mode="llm")
            self.assertEqual(writer.call_args.kwargs["report_mode"], "llm")
            self.assertEqual(writer.call_args.kwargs["main_mode"], "llm")
            self.assertIsNotNone(writer.call_args.kwargs["report_author"])
            self.assertIsNotNone(writer.call_args.kwargs["author"])
            self.assertEqual(len(instances), 4)

    def test_unreviewed_report_fallback_excluded_from_llm_sft(self):
        tid = self.scene["macro_graph"]["reference_waves"][0][0]
        class InvalidAuthor:
            def complete_json(self, system, request):
                return {"status": "done", "result": "Invented observation"}
        class Critic:
            def complete_json(self, system, request):
                return {
                    "verdict": "pass", "grounded": True,
                    "scope_preserved": True, "natural": True,
                    "useful": True,
                }
        episode, turns, audit = build_branch(
            self.scene, self.reports, self.episode,
            branch_round=2, faults=[Fault(tid, "missing")],
            report_author=InvalidAuthor(), report_critic=Critic(),
            report_max_revisions=0,
        )
        self.assertEqual(audit["outcome"], "language_review_unaccepted")
        self.assertEqual(turns, [])
        self.assertEqual(
            audit["subagent_report_reviews"][0]["authoring"]["mode"],
            "grounded_template_fallback",
        )
        self.assertTrue(audit["parallel_batches_verified"])

    def test_main_author_uses_recovery_specific_stage3_prompt(self):
        tid = self.scene["macro_graph"]["reference_waves"][0][0]
        seen = []
        class Author:
            def complete_json(self, system, request):
                seen.append(system)
                return json.loads(request)["reference_action_for_structure_only"]
        class Critic:
            def complete_json(self, system, request):
                seen.append(system)
                return {
                    "verdict": "pass", "grounded": True,
                    "assignment_fidelity": True, "o1_style": True,
                }
        episode, turns, audit = build_branch(
            self.scene, self.reports, self.episode,
            branch_round=2, faults=[Fault(tid, "missing")],
            author=Author(), critic=Critic(),
            main_max_revisions=0,
        )
        self.assertEqual(audit["outcome"], "recovered")
        self.assertTrue(any("UNTRUSTED ROUND-LOCAL RECOVERY" in x
                            for x in seen))
        self.assertTrue(any("UNTRUSTED RECOVERY REVIEW" in x
                            for x in seen))
        self.assertIn("llm_reviewed", {
            row["generation_mode"] for row in turns
        })
        self.assertTrue(audit["main_action_reviews"])

    def test_end_to_end_dual_llm_review_preserves_program_planner(self):
        tid = self.scene["macro_graph"]["reference_waves"][0][0]

        class Author:
            def complete_json(self, system, request):
                data = json.loads(request)
                if "observed_current_attempt" not in data:
                    return data["reference_action_for_structure_only"]
                observed = data["observed_current_attempt"]
                if not observed["finish_result_exists"]:
                    return {
                        "result": None,
                        "summary": (
                            "The investigation ended without a usable "
                            "result for the delegated task."
                        ),
                        "trace_summary": (
                            "The assigned evidence acquisition did not "
                            "produce an inspectable record."
                        ),
                        "status": "missing",
                    }
                records = observed["actual_delivered_rows"]
                if len(records) == 1:
                    values = records[0]["values"]
                    facts = ", ".join(
                        str(k).replace("_", " ") + " " + str(v)
                        for k, v in values.items()
                    )
                    result = "The returned observation contains " + facts + "."
                else:
                    result = (
                        "The completed task returned " +
                        str(observed["total_delivered_rows"]) +
                        " records for the next investigation."
                    )
                summary = (
                    "I checked the requested records and provided the "
                    "actual returned findings without further filtering."
                )
                if observed["delivery_status"] == "partial":
                    summary += (
                        " Still unresolved: " +
                        ", ".join(observed["unresolved_fields"]) + "."
                    )
                # Current 4B Author writes only Summary/Trace: the executor
                # inserts the full typed Result and FactRefs unchanged.
                return {
                    "summary": summary,
                    "trace_summary": (
                        "I performed the assigned operations and reported "
                        "their observed result for downstream use."
                    ),
                    "status": observed["delivery_status"],
                }

        class Critic:
            def complete_json(self, system, request):
                if "candidate_main_facing_report" in json.loads(request):
                    return {
                        "verdict": "pass", "grounded": True,
                        "scope_preserved": True, "natural": True,
                        "useful": True,
                    }
                return {
                    "verdict": "pass", "grounded": True,
                    "assignment_fidelity": True, "o1_style": True,
                }

        episode, turns, audit = build_branch(
            self.scene, self.reports, self.episode,
            branch_round=2, faults=[Fault(tid, "missing")],
            author=Author(), critic=Critic(),
            report_author=Author(), report_critic=Critic(),
            main_max_revisions=0, report_max_revisions=0,
        )
        self.assertEqual(
            audit["outcome"], "recovered",
            repr(audit["subagent_report_reviews"]) + " / " +
            repr(audit["main_action_reviews"]),
        )
        self.assertTrue(turns)
        self.assertTrue(all(
            review["authoring"]["mode"] == "llm_reviewed"
            for review in audit["subagent_report_reviews"]
        ))
        self.assertTrue(any(
            review["authoring"]["mode"] == "llm_reviewed"
            for review in audit["main_action_reviews"]
        ))
        self.assertTrue(audit["parallel_batches_verified"])
        self.assertEqual(episode["trajectory"][-1]["action"], "complete")
        self.assertTrue(all(
            set(turn["input"]) == {"subtask_history"}
            for turn in turns
        ))


    def test_failed_reviewed_episodes_remain_private_not_public(self):
        from environment.untrusted_orchestration.generate_turns import _publish

        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "journal").mkdir()
            plans = []
            for index, status in enumerate((
                "language_review_unaccepted", "main_action_not_accepted",
            )):
                task_id = f"question-failed-{index}"
                plans.append({
                    "index": index, "task_id": task_id,
                    "sample_id": f"{task_id}-round-2",
                    "branch_round": 2, "faults": [],
                    "scenario_plan": {
                        "blueprint": "test_fixture", "fault_atoms": [],
                    },
                })
                # The journal deliberately contains a full Episode even
                # though its language/action review failed.
                (root / "journal" / f"{index:06d}.json").write_text(
                    json.dumps({
                        "task_id": task_id, "status": status,
                        "episode": {
                            "task_id": task_id,
                            "trajectory": [{"action": "complete"}],
                            "outcome": "recovered",
                        },
                        "turns": [], "private_audit": None,
                    }), encoding="utf-8",
                )
            metrics = _publish(root, plans)
            self.assertEqual(metrics["published_episodes"], 0)
            self.assertEqual(metrics["main_training_turns"], 0)
            self.assertEqual(
                (root / "episodes.jsonl").read_text(encoding="utf-8"), ""
            )
            self.assertEqual(
                (root / "main_turns.jsonl").read_text(encoding="utf-8"), ""
            )
            failures = [
                json.loads(line) for line in
                (root / "failed.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(failures), 2)
            self.assertTrue(all(
                (root / "journal" / f"{i:06d}.json").exists()
                for i in range(2)
            ))

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
                output_dir=root / "output",
                max_scenes=1, branches_per_scene=1,
                seed=17, fault_modes=["missing"], workers=2,
                no_progress=True,
            )
            summary = generate_turns(**args)
            self.assertEqual(summary["candidate_trials"], 1)
            self.assertEqual(summary["recovered"], 1,
                             (root / "output" / "journal" /
                              "000000.json").read_text()[:1400])
            self.assertEqual(summary["fault_mode_outcomes"]["missing"],
                             {"recovered": 1})
            self.assertGreater(summary["main_training_turns"], 0)
            self.assertGreaterEqual(
                summary["local_decisions_qualified_diagnostic"],
                summary["main_training_turns"],
            )
            self.assertEqual(
                summary["local_decisions_reviewed"],
                summary["local_decisions_qualified_diagnostic"]
                + summary["local_decisions_rejected_diagnostic"],
            )
            diagnostic = root / "output" / "private" / "main_turns_local_review.jsonl"
            self.assertTrue(diagnostic.is_file())
            self.assertEqual(
                len(diagnostic.read_text(encoding="utf-8").splitlines()),
                summary["local_decisions_reviewed"],
            )
            self.assertEqual(
                summary["execution_recovered_branches"], 1,
            )
            self.assertEqual(
                summary["execution_unresolved_branches"], 0,
            )
            self.assertEqual(summary["round_sampling_unit"],
                             "question_x_main_feedback_round")
            self.assertEqual(summary["rounds_extracted_from_base"], 1)
            round_index = (root / "output" / "round_index.jsonl")
            indexes = [json.loads(row) for row in
                       round_index.read_text().splitlines()]
            self.assertEqual(len(indexes), 1)
            self.assertEqual(indexes[0]["main_decision_round"], 2)
            self.assertEqual(indexes[0]["received_from_delegate_round"], 1)
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


    def test_actual_attempt_trace_matches_post_fault_relation(self):
        """The report-facing trace must come from executed, changed rows."""
        from environment.untrusted_orchestration.runtime.faults import relation_rows
        first = self.scene["macro_graph"]["reference_waves"][0]
        tasks = {t["task_id"]: t for t in
                 self.scene["macro_graph"]["tasks"]}
        partial_tid = next(
            tid for tid in first
            if len(tasks[tid]["outputs"]["typed_relation"]) > 1
        )
        state = ExecutionState(self.scene, {})
        partial = state.register(partial_tid, fault=Fault(partial_tid, "partial"))
        steps = partial.runtime["observed_operator_steps"]
        self.assertTrue(steps)
        self.assertEqual(
            steps[-1]["sampled_result_rows"],
            relation_rows(partial.product)[:4],
        )
        self.assertEqual(
            steps[-1]["output_row_count"], len(partial.product.rows)
        )
        self.assertEqual(
            partial.runtime["delivery_fault_kind"], "field_not_delivered"
        )
        self.assertTrue(all(
            "evidence_atoms" in step and "supporting_fact_refs" in step
            for step in steps
        ))

        missing_tid = first[0]
        other = ExecutionState(self.scene, {})
        missing = other.register(missing_tid, fault=Fault(missing_tid, "missing"))
        records = missing.runtime["observed_operator_steps"]
        self.assertEqual(records[-1]["execution_status"], "missing")
        self.assertTrue(all(
            row["execution_status"] == "done" for row in records[:-1]
        ))
        self.assertFalse(any(
            row.get("sampled_result_rows") for row in records[-1:]
        ))

    def test_main_guard_distinguishes_subagent_attempt_from_author_draft(self):
        from environment.base_orchestration.action_author import _GENERATION_META
        self.assertIsNone(_GENERATION_META.search(
            "The previous attempt returned an incomplete result, so retry."
        ))
        self.assertIsNone(_GENERATION_META.search(
            "A repair attempt should cross-check the returned key."
        ))
        self.assertIsNotNone(_GENERATION_META.search(
            "My previous draft was rejected by the critic."
        ))

    def test_status_is_executor_owned_with_real_missing_attempt(self):
        """A legacy model status cannot override actual missing delivery."""
        tid = self.scene["macro_graph"]["reference_waves"][0][0]
        state = ExecutionState(self.scene, {})
        attempt = state.register(tid, fault=Fault(tid, "missing"))
        task = state.tasks[tid]
        captured = []
        class Author:
            def complete_json(self, system, request):
                contract = json.loads(request)["observed_current_attempt"]
                captured.append(contract)
                return {
                    "status": "done",  # Legacy Author misclassifies missing.
                    "result": None,
                    "summary": "No usable source record was returned for this task.",
                    "trace_summary": (
                        "The source acquisition failed to deliver any "
                        "inspectable observations for this attempt."
                    ),
                }
        class Critic:
            def complete_json(self, system, request):
                return {
                    "verdict": "pass", "grounded": True,
                    "scope_preserved": True, "natural": True, "useful": True,
                }
        report, review = naturalize_report(
            task, attempt, instruction="Investigate the assigned source.",
            model_alias="model_1", reviewed=None,
            author=Author(), critic=Critic(), max_revisions=0,
        )
        self.assertEqual(review["mode"], "llm_reviewed")
        self.assertTrue(review["author_status_ignored"])
        self.assertTrue(captured[0]["trace_from_actual_execution"])
        self.assertEqual(
            captured[0]["observed_operator_steps"][-1]["execution_status"],
            "missing",
        )
        self.assertIsNone(report["finish_result"])

    def test_execution_success_is_separate_from_review_acceptance(self):
        tid = self.scene["macro_graph"]["reference_waves"][0][0]
        class InvalidAuthor:
            def complete_json(self, system, request):
                return {"result": "Invented data", "summary": "No evidence"}
        class Critic:
            def complete_json(self, system, request):
                return {
                    "verdict": "pass", "grounded": True,
                    "scope_preserved": True, "natural": True, "useful": True,
                }
        _, turns, audit = build_branch(
            self.scene, self.reports, self.episode,
            branch_round=2, faults=[Fault(tid, "missing")],
            report_author=InvalidAuthor(), report_critic=Critic(),
            report_max_revisions=0,
        )
        self.assertEqual(audit["execution_outcome"], "recovered")
        self.assertTrue(audit["execution_recovered"])
        self.assertFalse(audit["language_review_accepted"])
        self.assertEqual(turns, [])



    def test_attempt_contract_separates_delivery_from_preview(self):
        from environment.untrusted_orchestration.reporting.contract import (
            build_report_contract,
        )
        tasks = {task["task_id"]: task for task in
                 self.scene["macro_graph"]["tasks"]}
        target = next(tid for tid in
                      self.scene["macro_graph"]["reference_waves"][0]
                      if len(tasks[tid]["outputs"]["typed_relation"]) > 1)
        state = ExecutionState(self.scene, {})
        attempt = state.register(target, fault=Fault(target, "partial"))
        contract = build_report_contract(
            tasks[target], attempt, "Inspect the assigned records."
        )
        self.assertEqual(contract["delivery_status"], "partial")
        self.assertTrue(contract["evidence_samples_are_preview_only"])
        self.assertTrue(contract["delivered_relation_is_complete_in_prompt"])
        self.assertEqual(contract["total_delivered_rows"],
                         len(contract["actual_delivered_rows"]))
        self.assertTrue(contract["observed_operator_steps"])
        self.assertTrue(all("remaining_rows_omitted" not in step
                            and "evidence_atoms_truncated" not in step
                            for step in contract["observed_operator_steps"]))
        self.assertTrue(all("sampled_evidence_atoms" in step for step
                            in contract["observed_operator_steps"]))

    def test_partial_null_prose_result_normalizes_from_actual_workproduct(self):
        tasks = {task["task_id"]: task for task in
                 self.scene["macro_graph"]["tasks"]}
        target = next(tid for tid in
                      self.scene["macro_graph"]["reference_waves"][0]
                      if len(tasks[tid]["outputs"]["typed_relation"]) > 1)
        state = ExecutionState(self.scene, {})
        attempt = state.register(target, fault=Fault(target, "partial"))
        class Author:
            def complete_json(self, system, request):
                return {
                    "result": None,  # Frequently observed in the 47-branch run
                    "summary": (
                        "The assigned scan returned only part of the "
                        "required result; the final delivery is incomplete."
                    ),
                    "trace_summary": (
                        "I examined the assigned records but could not "
                        "provide every requested field in the handoff."
                    ),
                }
        class Critic:
            def complete_json(self, system, request):
                payload = json.loads(request)["observed_current_attempt"]
                assert payload["delivery_status"] == "partial"
                return {
                    "verdict": "pass", "grounded": True,
                    "scope_preserved": True, "natural": True, "useful": True,
                }
        report, meta = naturalize_report(
            tasks[target], attempt,
            instruction="Read the assigned source records.",
            model_alias="model_1", reviewed=None,
            author=Author(), critic=Critic(), max_revisions=0,
        )
        self.assertEqual(meta["mode"], "llm_reviewed", meta)
        self.assertTrue(meta["executor_result_normalized"])
        self.assertEqual(report["finish_result"]["status"], "partial")
        self.assertIsInstance(report["finish_result"]["result"], str)
        self.assertIn("not delivered", report["finish_result"]["result"])

    def test_result_context_ids_are_grounded_but_invented_ids_still_fail(self):
        from environment.untrusted_orchestration.reporting.contract import (
            build_report_contract,
        )
        from environment.untrusted_orchestration.reporting.validation import (
            validate_report_draft, claim_tokens,
        )
        task = {"task_id": "lookup", "source_ids": ["stat_a"],
                "depends_on": [], "operator_chain": []}
        attempt = Attempt(
            task_id="lookup", version=1, status="done",
            product=Relation(("marker_code", "completed_count"), ({
                "values": {"marker_code": "MK-0531",
                           "completed_count": 36},
                "support": ("stat_a:F001",),
            },)),
            available_fields=("marker_code", "completed_count"),
            used_parent_versions={}, runtime={
                "observed_operator_steps": [{
                    "operation": "lookup_by_key",
                    "execution_status": "done",
                    "output_schema": ["marker_code", "completed_count"],
                    "supporting_fact_refs": ["stat_a:F001"],
                    "sampled_result_rows": [{
                        "values": {"marker_code": "MK-0531",
                                   "candidate_id": "Case-016"},
                    }],
                    "evidence_atoms": [],
                }],
            },
        )
        contract = build_report_contract(
            task, attempt, "Retrieve stat_a for Case-016 using MK-0531."
        )
        draft = {
            "result": "MK-0531 has count 36 for Case-016.",
            "summary": "I retrieved the requested record and its count.",
            "trace_summary": (
                "The assigned keyed lookup returned the corresponding "
                "stat_a record for the current case."
            ),
        }
        self.assertEqual(validate_report_draft(draft, contract, attempt), [])
        malicious = dict(draft, result="MK-0531 has count 36 for Case-999.")
        self.assertIn("result_unobserved_numeric_or_entity",
                      validate_report_draft(malicious, contract, attempt))
        self.assertEqual(claim_tokens("70,000 ms"),
                         claim_tokens("70000 ms"))
        self.assertEqual(claim_tokens("MK-0531"),
                         claim_tokens("mk-0531"))

    def test_branch_notifies_distinct_agent_progress_events(self):
        tid = self.scene["macro_graph"]["reference_waves"][0][0]
        report_events, main_events = [], []
        _, turns, audit = build_branch(
            self.scene, self.reports, self.episode,
            branch_round=2, faults=[Fault(tid, "missing")],
            on_report_review=lambda status: report_events.append(status),
            on_main_review=lambda status: main_events.append(status),
        )
        self.assertTrue(turns)
        self.assertEqual(len(report_events),
                         len(audit["subagent_report_reviews"]))
        self.assertEqual(len(main_events),
                         len(audit["main_action_reviews"]))
        self.assertTrue(all("mode" in status for status in report_events))
        self.assertTrue(all("mode" in status for status in main_events))


    def test_live_dual_progress_bars_track_report_and_main_calls(self):
        bars, messages = [], []
        class Bar:
            def __init__(self, *, total, initial, desc, **kwargs):
                self.total, self.n, self.desc = total, initial, desc
                self.postfix = {}
                self.snapshots = []
                bars.append(self)
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_value, traceback):
                return False
            def update(self, amount):
                self.n += amount
            def set_postfix(self, fields=None, **kwargs):
                self.postfix = dict(fields or {})
                self.snapshots.append((self.n, dict(self.postfix)))
            def refresh(self):
                pass
            @staticmethod
            def write(message):
                messages.append(message)

        with TemporaryDirectory() as temp:
            root = Path(temp)
            scenes, reports, episodes = (
                root / "scenes.jsonl", root / "subtasks.jsonl",
                root / "episodes.jsonl",
            )
            scenes.write_text(json.dumps(self.scene) + "\n")
            reports.write_text("".join(json.dumps(x) + "\n"
                                       for x in self.reports))
            episodes.write_text(json.dumps(self.episode) + "\n")
            with patch("environment.untrusted_orchestration.generate_turns.tqdm",
                       Bar):
                result = generate_turns(
                    scenes_path=scenes, reports_path=reports,
                output_dir=root / "dual_progress",
                    max_scenes=1, branches_per_scene=1,
                    seed=17, fault_modes=["missing"], workers=2,
                    no_progress=False,
                )
        self.assertEqual(result["candidate_trials"], 1)
        self.assertEqual([bar.desc for bar in bars],
                         ["Evidence simulation",
                          "Sub-Agent feedback",
                          "Main orchestration"])
        self.assertGreater(bars[0].n, 0)
        self.assertGreater(bars[1].n, 0)
        self.assertGreater(bars[2].n, 0)
        self.assertTrue(all(bar.total == bar.n for bar in bars))
        # Round-level 4A outcomes, report-level 4B reviews and action-level
        # 4C reviews are live counters, not a single misleading status bar.
        self.assertEqual(
            list(bars[0].postfix),
            ["recovered", "incorrect", "unresolved", "skipped", "written"],
        )
        self.assertEqual(bars[0].postfix["written"],
                         result["completed_branches"])
        self.assertEqual(
            bars[0].postfix["recovered"],
            result["execution_recovered_branches"],
        )
        self.assertEqual(
            list(bars[1].postfix),
            ["accepted", "fallback", "failed", "written"],
        )
        self.assertEqual(
            bars[1].postfix["written"],
            sum(result["subagent_report_review_statuses"].values()),
        )
        self.assertEqual(
            bars[2].postfix["written"],
            sum(result["main_action_review_statuses"].values()),
        )
        self.assertEqual(
            bars[2].postfix["processed"], result["main_actions_processed"],
        )
        self.assertEqual(
            bars[2].postfix["planned"], result["main_actions_planned"],
        )
        self.assertEqual(
            bars[2].postfix["skipped_downstream"],
            result["main_actions_skipped_downstream"],
        )
        self.assertEqual(
            result["main_actions_planned"],
            result["main_actions_processed"]
            + result["main_actions_skipped_downstream"],
        )
        self.assertEqual(result["recovered"], result["published_episodes"])
        self.assertEqual(
            result["main_training_turns"],
            len((root / "dual_progress" / "main_turns.jsonl")
                .read_text(encoding="utf-8").splitlines()),
        )
        # Summary has one canonical key per publication quantity, not
        # historical duplicates with permanently equal values.
        self.assertFalse({
            "requested_branches", "requested_feedback_rounds",
            "reviewed_branches", "reviewed_turns",
            "screened_out_before_llm", "publication_surface_leaks",
            "candidate_backfill_diversity", "limitations",
        } & set(result))
        self.assertGreater(len(bars[0].snapshots), 1)
        self.assertGreater(len(bars[1].snapshots), 1)
        self.assertGreater(len(bars[2].snapshots), 1)
        self.assertTrue(any(
            "reviewed branches=" in m and "reviewed turns=" in m
            and "SFT" not in m for m in messages
        ))


    def test_main_progress_accounts_for_skipped_downstream_actions(self):
        class InvalidMainAuthor:
            def complete_json(self, system, request):
                # The first Main decision fails O1 reasoning validation.
                return {
                    "action": "delegate_task",
                    "reasoning": "No",
                    "params": {"tasks": []},
                }

        class AlwaysPassCritic:
            def complete_json(self, system, request):
                return {
                    "verdict": "pass", "grounded": True,
                    "assignment_fidelity": True, "o1_style": True,
                }

        with TemporaryDirectory() as td:
            root = Path(td)
            files = {
                "scenes_path": root / "scenes.jsonl",
                "reports_path": root / "reports.jsonl",
            }
            files["scenes_path"].write_text(json.dumps(self.scene) + "\n")
            files["reports_path"].write_text(
                "".join(json.dumps(r) + "\n" for r in self.reports)
            )

            summary = generate_turns(
                **files, output_dir=root / "failed_main",
                max_scenes=1, branches_per_scene=1,
                seed=17, fault_modes=["missing"],
                workers=2, no_progress=True,
                author=InvalidMainAuthor(), critic=AlwaysPassCritic(),
                main_max_revisions=0,
            )
            self.assertGreater(summary["main_actions_planned"], 1)
            self.assertEqual(summary["main_actions_processed"], 1)
            self.assertEqual(
                summary["main_actions_skipped_downstream"],
                summary["main_actions_planned"] - 1,
            )
            self.assertEqual(summary["main_training_turns"], 0)
            self.assertEqual(
                summary["outcomes"]["main_action_not_accepted"], 1,
            )

    def test_offline_evidence_pipeline_stage_order_and_dynamic_requests(self):
        """4A must finish before *all* 4B calls, then 4C calls commence."""
        events = []
        main_review_requests = []

        class ReportAuthor:
            def complete_json(self, system, request):
                events.append("report")
                observed = json.loads(request)["observed_current_attempt"]
                result = (
                    [row["values"]
                     for row in observed["actual_delivered_rows"]]
                    if observed["finish_result_exists"] else None
                )
                return {
                    "result": result,
                    "summary": (
                        "The assigned evidence operation returned the "
                        "actual observed rows without further inference. "
                        + ("The handoff remains partial and unresolved."
                           if observed["delivery_status"] == "partial" else
                           "The returned evidence is available for the "
                           "next dependent investigation.")
                    ),
                    "trace_summary": (
                        "I used the authorized observation for this "
                        "assignment and reported the returned records "
                        "without performing any later dependency step."
                    ),
                }

        class MainAuthor:
            def complete_json(self, system, request):
                events.append("main")
                payload = json.loads(request)
                assert "reference_action_for_structure_only" not in payload
                assert "previous_feedback" not in payload
                return {
                    "reasoning": (
                        "The previous feedback does not fully establish the "
                        "requested fields. I should obtain the outstanding "
                        "evidence from the authorized source before proceeding."
                    )
                }

        class Critic:
            def complete_json(self, system, request):
                if "candidate_main_facing_report" in json.loads(request):
                    return {
                        "verdict": "pass", "grounded": True,
                        "scope_preserved": True, "natural": True,
                        "useful": True,
                    }
                payload = json.loads(request)
                if "candidate_reasoning_only" in payload:
                    main_review_requests.append(payload)
                    assert "candidate_prose_only" not in payload
                    assert "expected_current_evidence_goals" in payload
                    assert isinstance(payload["candidate_reasoning_only"], str)
                    assert all(
                        "target_instruction" not in goal
                        for goal in payload["expected_current_evidence_goals"]
                    )
                return {
                    "verdict": "pass", "grounded": True,
                    "assignment_fidelity": True, "o1_style": True,
                }

        with TemporaryDirectory() as td:
            root = Path(td)
            files = {
                "scenes_path": root / "scenes.jsonl",
                "reports_path": root / "reports.jsonl",
                            }
            files["scenes_path"].write_text(json.dumps(self.scene) + "\n")
            files["reports_path"].write_text(
                "".join(json.dumps(r) + "\n" for r in self.reports)
            )

            summary = generate_turns(
                **files, output_dir=root / "dataset", max_scenes=1,
                branches_per_scene=1, fault_modes=["missing"], seed=17,
                workers=2, no_progress=True,
                author=MainAuthor(), critic=Critic(),
                report_author=ReportAuthor(), report_critic=Critic(),
                max_revisions=0,
            )
            a = json.loads((root / "dataset" / "stage4a" /
                            "journal" / "000000.json").read_text())
            b = json.loads((root / "dataset" / "stage4b" /
                            "journal" / "000000.json").read_text())
            c = json.loads((root / "dataset" / "journal" /
                            "000000.json").read_text())
            requests = [
                req for p in a["private_audit"]["evidence_plan"]
                for req in p["evidence_requests"]
            ]
            self.assertTrue(requests)
            self.assertTrue(any(r["operation"] ==
                                "retry_missing_evidence" for r in requests))
            self.assertTrue(all("target_fields" in req and
                                "main_visible_justification" in req
                                for req in requests))
            self.assertTrue(all(r["operation"] != "fixed_macro_replay"
                                for r in requests))
            for plan in a["private_audit"]["evidence_plan"]:
                assigned = plan["dispatch"]
                self.assertEqual(len(assigned),
                                 len({d["execution_binding"]["macro_task_id"]
                                      for d in assigned}))
                self.assertEqual(
                    {r["request_id"] for r in plan["evidence_requests"]},
                    {identifier for d in assigned
                     for identifier in d["evidence_request_ids"]},
                )
                self.assertTrue(all(len(r["target_fields"]) == 1
                                    for r in plan["evidence_requests"]))
            self.assertEqual(
                len(b["private_audit"]["subagent_report_reviews"]),
                len(a["private_audit"]["subagent_report_reviews"]),
            )
            self.assertEqual(
                len(c["private_audit"]["main_action_reviews"]),
                events.count("main"),
            )
            self.assertEqual(summary["completed_branches"], 1)
            self.assertTrue((root / "dataset" / "main_turns.jsonl").exists())
            for step in c["episode"]["trajectory"]:
                if (step["action"] != "delegate_task"
                        or step["attempt"] < c["private_audit"]["branch_round"]):
                    continue
                for assignment, report in zip(
                    step["params"]["tasks"],
                    step["result"]["subtask_results"],
                ):
                    self.assertEqual(assignment["task_instruction"],
                                     report["task_instruction"])
                    self.assertEqual(assignment["model"], report["model"])
        self.assertTrue(main_review_requests)
        self.assertIn("report", events)
        self.assertIn("main", events)
        first_main = events.index("main")
        self.assertTrue(all(x == "report" for x in events[:first_main]))
        self.assertTrue(all(x == "main" for x in events[first_main:]))

    def test_stage4c_freezes_off_target_author_assignments(self):
        """A fluent off-target Main draft cannot redefine the executor goal."""
        from environment.base_orchestration.action_author import (
            author_delegate_action, proposed_template_action,
        )
        tid = self.scene["macro_graph"]["reference_waves"][0][0]
        _, _, audit = build_branch(
            self.scene, self.reports, self.episode,
            branch_round=2, faults=[Fault(tid, "missing")],
        )
        expected = [
            dispatch["assignment_contract"]
            for dispatch in audit["evidence_plan"][0]["dispatch"]
        ]
        self.assertTrue(expected)

        class OffTarget:
            def complete_json(self, system, request):
                return {
                    "action": "delegate_task",
                    "reasoning": (
                        "The previous Sub-Agent returned an incomplete "
                        "observation, so I should delegate the next "
                        "authorized evidence acquisition now."
                    ),
                    "params": {"tasks": [{
                        "task_instruction": (
                            "Determine the final winner and score, rather "
                            "than the evidence explicitly requested by "
                            "the current dependency."
                        ),
                        "context": "The final answer must be obtained now.",
                        "model": "model_1", "tools": [],
                    } for _ in expected]},
                }

        class Critic:
            def complete_json(self, system, request):
                return {
                    "verdict": "pass", "grounded": True,
                    "assignment_fidelity": True, "o1_style": True,
                }

        action, status = author_delegate_action(
            assignments=expected, prompt="User question",
            author=OffTarget(), critic=Critic(),
            max_revisions=0, freeze_assignment_prose=True,
        )
        self.assertEqual(
            action["params"],
            proposed_template_action(expected)["params"],
        )
        self.assertNotIn("final winner and score", json.dumps(action))
        self.assertTrue(status["assignment_contract_frozen"])
        self.assertEqual(status["authored_fields"], ["reasoning"])
        self.assertEqual(status["mode"], "llm_reviewed_contract_frozen")

    def test_stage4b_intermediate_numeric_is_not_final_delivered_value(self):
        from environment.untrusted_orchestration.reporting.validation import (
            explicit_final_value_issues, validate_report_draft,
        )
        delivered = [{
            "values": {"marker_code": None, "candidate_id": "Case-008"},
            "source_fact_refs": ["visual:F008"],
        }]
        contract = {
            "output_fields": ["marker_code", "candidate_id"],
            "actual_delivered_rows": delivered,
            "observed_operator_steps": [{
                "operation": "filter_gte", "execution_status": "done",
                "output_schema": ["marker_code", "frame_time_ms"],
                "output_row_count": 1, "threshold": 60000,
                "supporting_fact_refs": ["visual:F008"],
            }],
            "total_delivered_rows": 1,
            "executed_operations_in_order": ["filter_gte"],
            "assigned_task_instruction": "Inspect the supplied visual records.",
            "upstream_handoffs": [],
            "available_provenance_sources": ["visual"],
            "unresolved_fields": ["marker_code"],
        }
        self.assertEqual(
            explicit_final_value_issues(
                "The delivered marker code is unavailable, after the "
                "observed frame threshold of 60000.", contract,
            ), [],
        )
        self.assertIn(
            "claimed_final_field_value_not_delivered",
            explicit_final_value_issues(
                "The final marker code: MK-4010.", contract,
            ),
        )
        self.assertIn(
            "claimed_final_field_value_not_delivered",
            explicit_final_value_issues(
                "The final candidate id = Case-999.", contract,
            ),
        )

    def test_evidence_request_retargets_missing_and_partial_fields(self):
        from environment.untrusted_orchestration.runtime.evidence_planning import (
            evidence_request,
        )
        task = next(
            t for t in self.scene["macro_graph"]["tasks"]
            if len(t["outputs"]["typed_relation"]) > 1
            and not t["depends_on"]
        )
        state = ExecutionState(self.scene, {})
        initial = evidence_request(
            state, task["task_id"], round_number=1, retry=False,
        )
        self.assertEqual(initial["operation"], "retrieve_required_evidence")
        state.register(task["task_id"],
                       fault=Fault(task["task_id"], "partial"))
        updated = evidence_request(
            state, task["task_id"], round_number=2, retry=True,
        )
        self.assertEqual(updated["operation"], "complete_partial_evidence")
        self.assertTrue(updated["unresolved_fields"])
        self.assertLessEqual(len(updated["target_fields"]),
                             len(initial["target_fields"]))
        self.assertNotIn("gold_answer", json.dumps(updated))


if __name__ == "__main__":
    unittest.main()
