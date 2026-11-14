"""Deterministic action-quality reward regression tests; no LLM or GPU."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from environment.untrusted_orchestration.decision_reward_state import (
    SCHEMA, capture_decision_state,
)
from environment.untrusted_orchestration.orchestration_reward import (
    orchestration_reward, score_decision,
)


def option(tid, field, source, *, priority=0.7, descendants=(), tools=None):
    return {
        "task_id": tid, "operation": "retrieve_required_evidence",
        "kind": "new", "target_fields": [field], "source_ids": [source] if source else [],
        "known_identifiers": [], "required_parent_fields": {},
        "depends_on": [], "descendants": list(descendants),
        "tools": ["GoogleSearchAction"] if tools is None else tools,
        "model": "model_1", "priority": priority,
        "evidence_units": [tid + "::" + field],
    }


def state(*opts, complete=False):
    return {
        "schema": SCHEMA, "decision_round": 3, "max_parallel_tasks": 4,
        "can_complete": complete, "event_pending": False,
        "scorable": True, "actionable": list(opts),
    }


def delegate(*specs):
    return json.dumps({
        "action": "delegate_task",
        "reasoning": "These evidence requests can proceed independently.",
        "params": {"tasks": [
            {"task_instruction": f"Use {src} to return {field} evidence.",
             "context": "Current question requires this evidence.",
             "model": "model_1", "tools": ["GoogleSearchAction"]}
            for src, field in specs
        ]},
    })


class DeterministicRewardTests(unittest.TestCase):
    def setUp(self):
        self.a = option("A", "project_id", "source_A", priority=1.1)
        self.b = option("B", "status_code", "source_B", priority=0.7)
        self.c = option("C", "entry_label", "source_C", priority=0.6)
        self.env = state(self.a, self.b, self.c)

    def test_parallel_coverage_and_priority_ranking(self):
        abc = delegate(("source_A","project_id"), ("source_B","status_code"),
                       ("source_C","entry_label"))
        ab = delegate(("source_A","project_id"), ("source_B","status_code"))
        a = delegate(("source_A","project_id"))
        b = delegate(("source_B","status_code"))
        scores = [score_decision(x,self.env) for x in (abc,ab,a,b)]
        self.assertAlmostEqual(scores[0]["reward"], 1.0)
        self.assertGreater(scores[0]["reward"], scores[1]["reward"])
        self.assertGreater(scores[1]["reward"], scores[2]["reward"])
        self.assertGreater(scores[2]["reward"], scores[3]["reward"])
        self.assertAlmostEqual(scores[0]["parallel"], 1.0)
        self.assertAlmostEqual(scores[2]["parallel"], 1/3, places=7)
        self.assertAlmostEqual(
            scores[1]["reward"],
            score_decision(delegate(("source_B","status_code"),
                                    ("source_A","project_id")),self.env)["reward"],
        )

    def test_unknown_task_duplicate_and_wrong_tool_are_hard_invalid(self):
        for prompt in (
            delegate(("source_Z","unknown_field")),
            delegate(("source_A","project_id"),("source_A","project_id")),
            delegate(("source_A","project_id"),
                     ("source_B","status_code"),
                     ("source_C","entry_label"),
                     ("source_Z","unknown_field")),
        ):
            result = score_decision(prompt, self.env)
            self.assertEqual(result["reward"], -1)
            self.assertFalse(result["valid"])
        wrong_tool = json.loads(delegate(("source_A","project_id")))
        wrong_tool["params"]["tasks"][0]["tools"] = ["ExecuteCodeAction"]
        self.assertEqual(score_decision(wrong_tool,self.env)["reward"], -1)

    def test_same_batch_dependency_is_illegal(self):
        a = option("A","project_id","source_A",descendants=["D"])
        d = option("D","detail","source_D")
        joined = delegate(("source_A","project_id"),("source_D","detail"))
        result = score_decision(joined,state(a,d))
        self.assertEqual(result["reason"],"same_batch_dependency_violation")
        self.assertEqual(result["reward"],-1)

    def test_field_scoping_penalizes_excess_not_teacher_paraphrase(self):
        short = delegate(("source_A","project_id"))
        long = json.loads(short)
        long["params"]["tasks"][0]["task_instruction"] = (
            "Consult source_A. Return project_id but also obtain status_code."
        )
        self.assertGreater(
            score_decision(short,self.env)["reward"],
            score_decision(long,self.env)["reward"],
        )
        paraphrase = json.loads(short)
        paraphrase["params"]["tasks"][0]["task_instruction"] = (
            "Retrieve project id from source_A; give independent evidence."
        )
        self.assertEqual(
            score_decision(short,self.env)["reward"],
            score_decision(paraphrase,self.env)["reward"],
        )

    def test_complete_and_private_oracle_fail_closed(self):
        done = {"action":"complete","params":{"answer":"A supported finding"}}
        self.assertEqual(score_decision(done,state(complete=True))["reward"],1)
        self.assertEqual(score_decision(done,self.env)["reward"],-1)
        self.assertEqual(
            score_decision({"action":"complete","params":None},
                           state(complete=True))["reward"],-1
        )
        private = state(self.a)
        private["scorable"] = False
        with self.assertRaisesRegex(ValueError,"unobservable"):
            score_decision(delegate(("source_A","project_id")),private)

    def test_indistinguishable_goals_do_not_guess_private_task_id(self):
        x = option("X","project_id","source_A")
        y = option("Y","project_id","source_A")
        result = score_decision(delegate(("source_A","project_id")),state(x,y))
        self.assertEqual(result["reward"],-1)
        self.assertIn("ambiguous",result["reason"])

    def test_full_reward_api_ignores_reference_action(self):
        candidate = delegate(("source_A","project_id"))
        r1 = orchestration_reward(candidate, {"decision_state":self.env,
                                             "reference_action":{"action":"complete"}})
        r2 = orchestration_reward(candidate, {"decision_state":self.env,
                                             "reference_action":{}})
        self.assertEqual(r1,r2)

    def test_logged_attempt_replay_matches_live_stage4_snapshots(self):
        # Real deterministic Stage4 branch, without any LLM calls.
        from environment.tests.test_untrusted_orchestration import sample
        from environment.untrusted_orchestration.faults import Fault
        from environment.untrusted_orchestration.trajectory_builder import build_branch
        from train_maestro_8b.backfill_reward_states import replay_reward_states

        scene, reports, base = sample(0)
        target = scene["macro_graph"]["reference_waves"][0][0]
        episode, turns, audit = build_branch(
            scene, reports, base, branch_round=2,
            faults=[Fault(target, "missing")],
        )
        self.assertEqual(audit["outcome"], "recovered")
        self.assertTrue(turns)
        reconstructed = replay_reward_states(scene, episode, audit)
        expected = audit["decision_reward_states"]
        self.assertEqual(len(reconstructed), len(expected))
        for old, new in zip(expected, reconstructed):
            self.assertEqual(old["decision_round"], new["decision_round"])
            self.assertEqual(old["can_complete"], new["can_complete"])
            self.assertEqual(old["scorable"], new["scorable"])
            self.assertEqual(old["actionable"], new["actionable"])

    def test_capture_excludes_descendant_of_repair(self):
        class FakeState:
            def __init__(self):
                self.tasks = {
                    t: {"depends_on":[]}
                    for t in ("A","B","C")
                }
                self.current = {
                    "A": SimpleNamespace(status="disputed")
                }
            def ready(self, tid):
                return True
            def _dependents(self, tid):
                return {"C"} if tid=="A" else set()
            def public_status(self, tid):
                return "done"
            def visible_discrepancy(self, tid):
                return "Report and independent source disagree" if tid=="A" else None
        def fake_req(_,tid,*,round_number,retry):
            return {"operation":"verify_conflicting_evidence" if tid=="A" else
                    "retrieve_required_evidence",
                    "target_fields":["project_id" if tid=="A" else "status_code"],
                    "authorized_source_ids":["source_"+tid],
                    "known_identifiers":[],
                    "required_parent_fields":{}}
        def fake_assignment(_,tid,*,catalog,model_alias,request):
            return {"model":model_alias,"tools":["GoogleSearchAction"]}
        with patch(
            "environment.untrusted_orchestration.decision_reward_state.evidence_request",
            side_effect=fake_req
        ), patch(
            "environment.untrusted_orchestration.decision_reward_state.evidence_assignment",
            side_effect=fake_assignment
        ):
            reward_state = capture_decision_state(
                FakeState(), round_number=3, repairs=["A"], ready=["B","C"],
                blocked=[], max_parallel_tasks=4, catalog={},
                model_alias="model_1", event_pending=False,
                can_complete=False,
            )
        self.assertTrue(reward_state["scorable"])
        self.assertEqual({x["task_id"] for x in reward_state["actionable"]},{"A","B"})


if __name__ == "__main__":
    unittest.main()
