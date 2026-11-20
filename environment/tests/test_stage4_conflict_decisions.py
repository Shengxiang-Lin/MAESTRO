"""Stage-4 public conflict scheduling, strict replay and semantic regressions."""
from copy import deepcopy
import unittest

from environment.tests.test_stage4_history_first import _scene as base_scene
from environment.untrusted_orchestration.state_machine import ExecutionState, deterministic_report, public_history_item
from environment.untrusted_orchestration.runtime.faults import Fault, relation_rows
from environment.untrusted_orchestration.scenario.same_source_recheck import _clone_source_query
from environment.untrusted_orchestration.scenario.handoff_conflicts import expose_prior_handoff
from environment.untrusted_orchestration.reporting.result_text import render_records
from environment.untrusted_orchestration.reporting.public_language import sanitize_public_report
from environment.untrusted_orchestration.rewards.decision_reward_state import capture_decision_state
from environment.untrusted_orchestration.rewards.orchestration_reward import score_decision
from environment.untrusted_orchestration.runtime.evidence_planning import evidence_request, evidence_assignment
from environment.untrusted_orchestration.trajectory_builder import source_catalog
from environment.base_orchestration.action_author import proposed_template_action
from environment.untrusted_orchestration.decision_reward_replay import replay_reward_states
from environment.untrusted_orchestration.orchestration.reasoning_fidelity import derivation_instructions, reasoning_action_issues


def _scene():
    scene, parent, child = base_scene()
    scene["problem_spec"]["sources"] = [
        {"source_id": sid, "modality": "text", "title": sid,
         "access_mode": source["access_policy"]["mode"]}
        for sid, source in scene["world"]["sources"].items()
    ]
    scene["public"]["available_source_descriptions"] = [
        {"source_id": sid, "modality_hint": "text", "title": sid,
         "access_mode": source["access_policy"]["mode"]}
        for sid, source in scene["world"]["sources"].items()
    ]
    return scene, parent, child


def public(state, tid, rnd):
    attempt = state.current[tid]
    rows = attempt.runtime.get("communicated_rows")
    return {
        "subtask_index": 0, "model": "model_1", "done": True,
        "steps_taken": 1, "task_instruction": "Return the requested observations.",
        "finish_result": {"status": "done", "summary": "Returned the selected records.",
                          "result": render_records(rows if rows is not None else relation_rows(attempt.product))},
        "trace_summary": "Inspected the assigned source.",
    }


def fixture(independent=False, fault_on_old=False):
    scene, parent, child = _scene()
    scene["world"]["sources"]["parent_src"]["observations"].append({
        "values": {"case_id": "Case-002", "entity_final": "Dunhaven District"},
        "fact_refs": ["parent_src:F002"], "observation_id": "parent_src.OBS002",
        "evidence_origin_id": "synthetic:parent002",
    })
    parent["operator_chain"].insert(1, {
        "id": "parent_filter", "op": "filter_eq", "inputs": ["parent_scan"],
        "field": "case_id", "value": "Case-001",
    })
    parent["operator_chain"][-1]["inputs"] = ["parent_filter"]
    clone = _clone_source_query(parent, 5)
    scene["macro_graph"]["tasks"].append(clone)
    if independent:
        other = _clone_source_query(parent, 9)
        other.pop("history_recheck_of")
        other["task_id"] = "independent"
        scene["macro_graph"]["tasks"].append(other)
    state = ExecutionState(scene, {})
    state.register("parent", fault=(Fault("parent", "report", atom_kind="fabricated_claim") if fault_on_old else None))
    state.set_public_report("parent", public(state, "parent", 1), round_number=1)
    tid = clone["task_id"]
    state.register(tid, fault=(None if fault_on_old else Fault(tid, "report", atom_kind="fabricated_claim")))
    state.set_public_report(tid, public(state, tid, 4), round_number=4)
    state.cross_source_checks_enabled = True
    return scene, state, tid


def snapshot(scene, state):
    scheduling = state.scheduling_state()
    return capture_decision_state(
        state, round_number=5, repairs=scheduling["repair"],
        ready=scheduling["ready"], blocked=scheduling["blocked"],
        max_parallel_tasks=4, catalog=source_catalog(scene), model_alias="model_1",
        event_pending=False, can_complete=False,
    )


class PublicConflictDecisionTests(unittest.TestCase):
    def test_history_conflict_is_scorable_and_suspends_original_consumers(self):
        scene, state, clone = fixture()
        snap = snapshot(scene, state)
        self.assertTrue(snap["scorable"])
        self.assertEqual([a["task_id"] for a in snap["actionable"]], [clone])
        self.assertIn("child", snap["public_conflict_blocked_task_ids"])
        self.assertFalse(state.ready("child"))
        self.assertIn("child", state.decision_dependents(clone))
        request = evidence_request(state, clone, round_number=5, retry=True)
        assignment = evidence_assignment(state, clone, catalog=source_catalog(scene),
                                         model_alias="model_1", request=request)
        action = proposed_template_action([assignment])
        self.assertEqual(score_decision(action, snap)["reward"], 1.0)
        # Gold changes cannot change a public conflict or its reward visibility.
        scene["gold_answer"] = [{"entity_final": "irrelevant hidden answer"}]
        self.assertTrue(snapshot(scene, state)["scorable"])
        # A real agreeing report releases the paused consumer.
        state.register(clone)
        state.set_public_report(clone, public(state, clone, 5), round_number=5)
        state.scheduling_state()
        self.assertFalse(state.unresolved_cited_conflicts())
        self.assertTrue(state.ready("child"))

    def test_rechecking_the_older_report_keeps_the_actual_conflicting_pair(self):
        scene, state, clone = fixture(fault_on_old=True)
        self.assertTrue(snapshot(scene, state)["scorable"])
        state.register(clone)
        state.set_public_report(clone, public(state, clone, 5), round_number=5)
        snap = snapshot(scene, state)
        self.assertTrue(snap["scorable"])
        self.assertEqual([x["task_id"] for x in snap["actionable"]], ["parent"])
        self.assertEqual(set(state.detected_conflicts[-1]["compared_parent_tasks"]),
                         {"parent", clone})
        self.assertFalse(state.ready("child"))
        state.register("parent")
        state.set_public_report("parent", public(state, "parent", 6), round_number=6)
        state.scheduling_state()
        self.assertTrue(state.ready("child"))

    def test_independent_task_remains_parallel_ready(self):
        scene, state, clone = fixture(independent=True)
        snap = snapshot(scene, state)
        self.assertEqual({a["task_id"] for a in snap["actionable"]}, {clone, "independent"})
        self.assertNotIn("independent", snap["public_conflict_blocked_task_ids"])

    def test_private_stale_conflict_flag_is_not_a_public_witness(self):
        scene, state, clone = fixture()
        state.register(clone)
        state.set_public_report(clone, public(state, clone, 5), round_number=5)
        state.current[clone].status = "disputed"
        state.current[clone].runtime["history_conflict_recheck_requested"] = True
        state.current[clone].runtime["visible_issue"] = "private diagnostic only"
        snap = snapshot(scene, state)
        self.assertFalse(snap["scorable"])
        self.assertIn(clone, snap["unobservable_private_repairs"])

    def test_keyed_handoff_metadata_is_restored_for_strict_replay(self):
        scene, parent, child = _scene()
        state = ExecutionState(scene, {})
        state.register("parent")
        earlier = public(state, "parent", 1)
        state.set_public_report("parent", earlier, round_number=1)
        state.register("child", fault=Fault("child", "wrong", atom_kind="wrong_retrieval"))
        latest = public(state, "child", 2)
        state.set_public_report("child", latest, round_number=2)
        prefix = [{"attempt": n, "action": "delegate_task",
                   "result": {"subtask_results": [report]}}
                  for n, report in [(1, earlier), (2, latest)]]
        entries = [{"subtask_results": [public_history_item(report, 0)], "total_steps": 1}
                   for report in [earlier, latest]]
        anchor = expose_prior_handoff(state, entries, prefix,
            {"older_task": "parent", "latest_task": "child", "key_field": "entity_final"})
        state.cross_source_checks_enabled = True
        state.scheduling_state()
        state.register("child")
        repaired = public(state, "child", 3)
        state.set_public_report("child", repaired, round_number=3)
        episode = {"task_id": scene["task_id"], "branch_round": 3,
                   "trajectory": prefix + [
                       {"attempt": 3, "action": "delegate_task", "result": {"subtask_results": [repaired]}},
                       {"attempt": 4, "action": "complete"}]}
        audit = {"branch_round": 3, "history_conflict_anchor": anchor,
                 "runtime": state.audit(), "event_scenario": {"mode": "event_driven"},
                 "decision_audit": [
                     {"round": 3, "decision_kind": "delegate", "task_ids": ["child"]},
                     {"round": 4, "decision_kind": "complete", "task_ids": []}]}
        states = replay_reward_states(scene, episode, audit)
        self.assertEqual([s["decision_round"] for s in states], [3, 4])
        self.assertTrue(all(s["scorable"] for s in states))
        self.assertTrue(states[1]["can_complete"])
        self.assertNotIn("history_handoff_parent_task_id", child)


class Stage4SemanticTests(unittest.TestCase):
    def test_date_inputs_cannot_be_replaced_by_observation_extrema(self):
        task = {"operator_chain": [{
            "op": "goal_compute", "input_fields": ["date_a", "date_b"],
            "output_field": "duration_days", "goal_operation": {"op": "temporal_difference"},
        }]}
        instruction = " ".join(derivation_instructions(task))
        assignments = [{"target_instruction": instruction, "source_ids": [], "tools": []}]
        self.assertIn("date_a, date_b", instruction)
        self.assertIn("date_pair_replaced_with_observation_date_extrema",
            reasoning_action_issues("Compute elapsed days between the earliest and latest dates.", assignments))
        self.assertEqual(reasoning_action_issues(
            "Compute elapsed days using date_a and date_b already returned.", assignments), [])

    def test_source_free_work_and_intermediate_fields(self):
        assignments = [{"target_instruction": "Recompute the ratio.", "source_ids": [], "tools": []}]
        self.assertIn("source_free_action_claims_external_acquisition",
            reasoning_action_issues("I will query the original source again.", assignments))
        self.assertEqual(reasoning_action_issues(
            "I will inspect the already returned records and recompute the ratio.", assignments), [])
        self.assertIn("intermediate_field_claimed_as_question_output",
            reasoning_action_issues("The question explicitly requires entity_0.",
                assignments, question="Find the ratio of the two measurements."))

    def test_join_witness_rereview_can_pass_or_still_fail(self):
        import json
        from environment.generate_question.execution.program_ir import Relation
        from environment.untrusted_orchestration.runtime.faults import run_attempt
        from environment.untrusted_orchestration.offline_pipeline import _attempt_from_audit
        from environment.untrusted_orchestration.reporting.report_author import naturalize_report
        left = Relation(("case_id", "name"), (
            {"values": {"case_id": "A01", "name": "Alpha"}, "support": ("left:F1",)},
            {"values": {"case_id": "B02", "name": "Beta"}, "support": ("left:F2",)},
        ))
        right = Relation(("case_id", "metric"), (
            {"values": {"case_id": "A01", "metric": 7}, "support": ("right:F1",)},
        ))
        task = {
            "task_id": "join", "source_ids": [], "depends_on": ["L", "R"],
            "external_inputs": ["left_input", "right_input"],
            "input_schema": {"input_0": ["case_id", "name"], "input_1": ["case_id", "metric"]},
            "operator_chain": [
                {"id": "joined", "op": "join", "inputs": ["left_input", "right_input"],
                 "left_on": "case_id", "right_on": "case_id"},
                {"id": "projected", "op": "project", "inputs": ["joined"], "fields": ["case_id"]},
            ], "program_node_id": "projected", "outputs": {"typed_relation": ["case_id"]},
        }
        product, runtime = run_attempt(task, parents=[left, right], sources={},
            fault=Fault("join", "wrong", atom_kind="wrong_entity_binding"))
        attempt = _attempt_from_audit(task, {
            "version": 1, "status": "done", "runtime": runtime,
            "rows": relation_rows(product), "used_parent_versions": {"L": 1, "R": 1}, "cost": 1,
        })
        class Author:
            def complete_json(self, system, request):
                return {"summary": "Returned one identifier from the attempted association.",
                        "trace_summary": "Combined the two supplied inputs and selected the requested field."}
        class Critic:
            def __init__(self, allow_second):
                self.calls = 0
                self.allow_second = allow_second
            def complete_json(self, system, request):
                self.calls += 1
                data = json.loads(request)
                witness = data["executed_work_witnesses"][0]
                if (witness["left"], witness["right"]) != ("B02", "A01"):
                    raise AssertionError("Actual input binding must remain visible")
                passed = self.allow_second and self.calls == 2
                return {"verdict": "pass" if passed else "fail",
                        "grounded": passed, "scope_preserved": passed,
                        "natural": passed, "useful": passed,
                        "reason": "Join delivered rows contain only B02; the pairing is unsupported.",
                        "repair_hint": "Inspect the actual input binding."}
        for allowed in [True, False]:
            critic = Critic(allowed)
            report, status = naturalize_report(task, attempt,
                instruction="Match the supplied identifiers.", model_alias="model_1",
                reviewed=None, author=Author(), critic=critic, max_revisions=0)
            self.assertEqual(critic.calls, 2)
            self.assertEqual(status["status"], "accepted" if allowed else "fallback")
            self.assertTrue(status["critic_witness_rereviewed"])

    def test_ratio_mistakes_execute_from_unchanged_inputs(self):
        from environment.generate_question.execution.program_ir import Relation
        from environment.untrusted_orchestration.runtime.faults import run_attempt
        from environment.untrusted_orchestration.reporting.public_work import public_work_clauses, visible_work_discrepancy
        parent = Relation(("a", "b"), ({"values": {"a": 209, "b": 76}, "support": ("source:F1",)},))
        task = {
            "task_id": "ratio", "source_ids": [], "depends_on": ["input"],
            "external_inputs": ["input_node"], "input_schema": {"input_0": ["a", "b"]},
            "operator_chain": [{"id": "calculated", "op": "derive_arithmetic",
                "inputs": ["input_node"], "left_field": "a", "right_field": "b",
                "operation": "ratio", "round_digits": 2, "output_field": "answer"}],
            "program_node_id": "calculated", "outputs": {"typed_relation": ["answer"]},
        }
        before = relation_rows(parent)
        for variant, expected, mistake in [
            (2, 0.36, "swapped_ratio_operands"), (3, 275.0, "ratio_percent_scale")
        ]:
            product, audit = run_attempt(task, parents=[parent], sources={},
                fault=Fault("ratio", "wrong", atom_kind="computation_error", variant_index=variant))
            self.assertEqual(product.rows[0]["values"]["answer"], expected)
            self.assertEqual(audit["calculation_mistake"], mistake)
            clauses = " ".join(public_work_clauses(audit["observed_operator_steps"]))
            self.assertIsNotNone(visible_work_discrepancy(clauses))
            self.assertEqual(relation_rows(parent), before)

    def test_report_narration_normalization_preserves_result_values(self):
        report = {"trace_summary": "The WorkProduct retained the Set of evidence.",
                  "finish_result": {"result": "WorkProduct", "summary": "A WorkProduct was delivered."}}
        fixed = sanitize_public_report(report)
        self.assertEqual(fixed["finish_result"]["result"], "WorkProduct")
        self.assertNotIn("WorkProduct", fixed["trace_summary"])
        self.assertNotIn("Set of evidence", fixed["trace_summary"])
        self.assertEqual(report["finish_result"]["summary"], "A WorkProduct was delivered.")


if __name__ == "__main__":
    unittest.main()
