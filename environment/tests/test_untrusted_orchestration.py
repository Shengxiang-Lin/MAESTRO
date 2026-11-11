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
from environment.untrusted_orchestration.state_machine import ExecutionState, Attempt
from environment.untrusted_orchestration.report_author import naturalize_report
from environment.untrusted_orchestration.config import generate_from_config
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
            self.assertIn("Reconcile the visible conflicting",
                      json.dumps(turns[0]["output"]))
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
        from environment.generate_question.program_ir import Relation
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

    def test_llm_author_preserves_partial_and_missing_status(self):
        from environment.generate_question.program_ir import Relation
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

    def test_stage2_and_stage3_llm_clients_configured_separately(self):
        instances = []
        class Backend:
            def __init__(self, *, model, **kwargs):
                instances.append(model)
        with patch(
            "environment.generate_question.backend.OpenAICompatibleBackend",
            Backend,
        ), patch(
            "environment.untrusted_orchestration.generate_turns.generate_turns",
            return_value={"status": "done"},
        ) as writer:
            generate_from_config(
                DEFAULT_CONFIG, report_mode="llm", main_mode="template",
            )
            self.assertEqual(writer.call_args.kwargs["report_mode"], "llm")
            self.assertEqual(writer.call_args.kwargs["main_mode"], "template")
            self.assertIsNotNone(writer.call_args.kwargs["report_author"])
            self.assertIsNone(writer.call_args.kwargs["author"])
            self.assertEqual(len(instances), 2)
            instances.clear()
            generate_from_config(
                DEFAULT_CONFIG, report_mode="template", main_mode="llm",
            )
            self.assertEqual(writer.call_args.kwargs["main_mode"], "llm")
            self.assertEqual(writer.call_args.kwargs["report_mode"], "template")
            self.assertIsNotNone(writer.call_args.kwargs["author"])
            self.assertIsNone(writer.call_args.kwargs["report_author"])
            self.assertEqual(len(instances), 2)

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
                return {
                    "result": result,
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
            self.assertEqual(summary["recovered"], 1,
                             (root / "output" / "journal" /
                              "000000.json").read_text()[:1400])
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


    def test_actual_attempt_trace_matches_post_fault_relation(self):
        """The report-facing trace must come from executed, changed rows."""
        from environment.untrusted_orchestration.faults import relation_rows
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
        from environment.untrusted_orchestration.report_author import (
            _observed_contract,
        )
        tasks = {task["task_id"]: task for task in
                 self.scene["macro_graph"]["tasks"]}
        target = next(tid for tid in
                      self.scene["macro_graph"]["reference_waves"][0]
                      if len(tasks[tid]["outputs"]["typed_relation"]) > 1)
        state = ExecutionState(self.scene, {})
        attempt = state.register(target, fault=Fault(target, "partial"))
        contract = _observed_contract(
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
        from environment.untrusted_orchestration.report_author import (
            _observed_contract, _guard, _claim_tokens,
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
        contract = _observed_contract(
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
        self.assertEqual(_guard(draft, contract, attempt), [])
        malicious = dict(draft, result="MK-0531 has count 36 for Case-999.")
        self.assertIn("result_unobserved_numeric_or_entity",
                      _guard(malicious, contract, attempt))
        self.assertEqual(_claim_tokens("70,000 ms"),
                         _claim_tokens("70000 ms"))
        self.assertEqual(_claim_tokens("MK-0531"),
                         _claim_tokens("mk-0531"))

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
        bars = []
        class Bar:
            def __init__(self, *, total, initial, desc, **kwargs):
                self.total, self.n, self.desc = total, initial, desc
                bars.append(self)
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_value, traceback):
                return False
            def update(self, amount):
                self.n += amount
            def set_postfix(self, *args, **kwargs):
                pass
            def refresh(self):
                pass
            @staticmethod
            def write(message):
                pass

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
                    base_episodes_path=episodes,
                    output_dir=root / "dual_progress",
                    max_scenes=1, branches_per_scene=1,
                    seed=17, fault_modes=["missing"], workers=2,
                    no_progress=False,
                )
        self.assertEqual(result["requested_branches"], 1)
        self.assertEqual([bar.desc for bar in bars],
                         ["Stage 4A evidence simulation",
                          "Stage 4B Sub-Agent feedback",
                          "Stage 4C Main orchestration"])
        self.assertGreater(bars[0].n, 0)
        self.assertGreater(bars[1].n, 0)
        self.assertGreater(bars[2].n, 0)
        self.assertTrue(all(bar.total == bar.n for bar in bars))


    def test_offline_evidence_pipeline_stage_order_and_dynamic_requests(self):
        """4A must finish before *all* 4B calls, then 4C calls commence."""
        events = []

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
                return json.loads(request)["reference_action_for_structure_only"]

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

        with TemporaryDirectory() as td:
            root = Path(td)
            files = {
                "scenes_path": root / "scenes.jsonl",
                "reports_path": root / "reports.jsonl",
                "base_episodes_path": root / "base.jsonl",
            }
            files["scenes_path"].write_text(json.dumps(self.scene) + "\n")
            files["reports_path"].write_text(
                "".join(json.dumps(r) + "\n" for r in self.reports)
            )
            files["base_episodes_path"].write_text(
                json.dumps(self.episode) + "\n"
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
        self.assertIn("report", events)
        self.assertIn("main", events)
        first_main = events.index("main")
        self.assertTrue(all(x == "report" for x in events[:first_main]))
        self.assertTrue(all(x == "main" for x in events[first_main:]))

    def test_evidence_request_retargets_missing_and_partial_fields(self):
        from environment.untrusted_orchestration.evidence_planning import (
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
