"""Trusted Stage-3 simulator: no future evidence leaks, correct state/cost."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from envir.generate_question.spec import make_spec
from envir.generate_question.question import reference_question
from envir.generate_question.quality import compile_and_validate
from envir.implement_subtask.realize import realize_scene
from envir.stage3.simulator import TrustedSimulator, MainAgentSession


def fixture(i=0):
    spec = make_spec(
        seed_record={"record_id": "stage3-smoke-"+str(i),
                     "topic": "Premier League", "domain_hint": "sports"},
        ordinal=i, run_seed=19, difficulty="hard",
        macro_first_v2_share=100,
    )
    return compile_and_validate(spec, reference_question(spec))


def accepted_reports(scene):
    result = realize_scene(scene)
    return [{
        **deepcopy(record),
        "finish_result": {
            "status": "done",
            "result": deepcopy(record["answer"]),
            "summary": record["trace_summary"]["summary"],
        },
        "report_authoring": {"status": "accepted"},
    } for record in result["macro_tasks"]]


class TrustedSimulatorTests(unittest.TestCase):
    def test_reset_hides_full_dag_gold_and_future_reports(self):
        scene = fixture()
        engine = TrustedSimulator(scene, mode="oracle_baseline")
        session = engine.public_session()
        self.assertIsInstance(session, MainAgentSession)
        self.assertFalse(hasattr(session, "evaluate"))
        self.assertEqual(session.observe(), session.reset())
        seen = session.observe()
        self.assertLess(len(seen["ready_tasks"]),
                        len(scene["macro_graph"]["tasks"]))
        for prohibited in ("world", "gold_answer", "gold_program",
                           "macro_graph", "gold_trace", "task_feedback"):
            self.assertNotIn(prohibited, seen)
        self.assertFalse(seen["reports"])
        ready = {t["macro_task_id"] for t in seen["ready_tasks"]}
        hidden = next(t["task_id"] for t in scene["macro_graph"]["tasks"]
                      if t["task_id"] not in ready)
        with self.assertRaisesRegex(ValueError, "dependency_not_unlocked"):
            session.step({"type": "delegate", "macro_task_id": hidden})
        self.assertEqual(session.observe()["total_cost"], 0)

    def test_actions_unlock_only_after_accept_and_charge_exactly(self):
        scene = fixture(1)
        engine = TrustedSimulator(scene, mode="oracle_baseline")
        agent = engine.public_session()
        initial = agent.reset()
        tid = initial["ready_tasks"][0]["macro_task_id"]
        delegated = agent.step({"type": "delegate", "macro_task_id": tid})
        self.assertEqual(delegated["cost"], 1.0)
        self.assertIn(tid, {
            report["macro_task_id"]
            for report in delegated["observation"]["reports"]
        })
        self.assertEqual(delegated["observation"]["reports"][0][
            "finish_result"]["status"], "done")
        self.assertFalse(delegated["observation"]["ready_tasks"]
                         and any(row["macro_task_id"] == tid
                                 for row in delegated["observation"]["ready_tasks"]))
        asked = agent.step({"type": "request_evidence",
                           "macro_task_id": tid})
        self.assertEqual(asked["cost"], 0.5)
        self.assertTrue(asked["event"]["requested_evidence"])
        with self.assertRaisesRegex(ValueError, "already_requested"):
            agent.step({"type": "request_evidence", "macro_task_id": tid})
        retry = agent.step({"type": "retry_same", "macro_task_id": tid})
        self.assertTrue(retry["event"]["same_report_as_before"])
        agent.step({"type": "hold", "macro_task_id": tid})
        accepted = agent.step({"type": "accept", "macro_task_id": tid})
        self.assertIn(tid, accepted["observation"]["accepted_macro_task_ids"])
        self.assertAlmostEqual(accepted["observation"]["total_cost"], 2.7)

    def test_successful_reference_schedule_and_reset(self):
        for i in range(5):
            scene = fixture(i)
            engine = TrustedSimulator(scene, mode="oracle_baseline", budget=40)
            api = engine.public_session()
            public = api.reset()
            while not public["done"]:
                ready = public["ready_tasks"]
                self.assertTrue(ready)
                tid = ready[0]["macro_task_id"]
                api.step({"type": "delegate", "macro_task_id": tid})
                public = api.step({
                    "type": "accept", "macro_task_id": tid,
                })["observation"]
            self.assertEqual(public["status"], "completed")
            self.assertTrue(engine.evaluate()["success"])
            self.assertEqual(engine.evaluate()["accepted_macro_count"],
                             scene["macro_graph"]["macro_task_count"])
            self.assertNotIn("gold_answer", public)
            with self.assertRaisesRegex(ValueError, "already_finished"):
                api.step({"type": "terminate"})
            refreshed = api.reset()
            self.assertFalse(refreshed["done"])
            self.assertEqual(refreshed["total_cost"], 0)
            self.assertEqual(len(refreshed["reports"]), 0)

    def test_incorrect_declared_answer_is_only_private_failure(self):
        scene = fixture(7)
        engine = TrustedSimulator(scene, mode="oracle_baseline")
        api = engine.public_session()
        state = api.reset()
        while not state["done"]:
            tid = state["ready_tasks"][0]["macro_task_id"]
            api.step({"type": "delegate", "macro_task_id": tid})
            action = {"type": "accept", "macro_task_id": tid}
            if tid == scene["macro_graph"]["terminal_task_id"]:
                action["declared_answer"] = [{"not": "the answer"}]
            state = api.step(action)["observation"]
        self.assertEqual(state["status"], "completed")
        self.assertFalse(engine.evaluate()["success"])
        self.assertNotIn("success", state)

    def test_budget_and_unknown_action_do_not_change_state(self):
        engine = TrustedSimulator(fixture(3), mode="oracle_baseline",
                                  budget=1.5)
        api = engine.public_session()
        tid = api.observe()["ready_tasks"][0]["macro_task_id"]
        with self.assertRaisesRegex(ValueError, "invalid_action"):
            api.step({"type": "peek_world"})
        with self.assertRaisesRegex(ValueError, "requires_delivered_report"):
            api.step({"type": "accept", "macro_task_id": tid})
        api.step({"type": "delegate", "macro_task_id": tid})
        with self.assertRaisesRegex(ValueError, "insufficient_budget"):
            api.step({"type": "retry_same", "macro_task_id": tid})
        self.assertEqual(api.observe()["total_cost"], 1.0)
        self.assertEqual(api.step({"type": "terminate"})["done"], True)
        self.assertFalse(engine.evaluate()["success"])

    def test_reviewed_reports_rejected_if_missing_or_tampered(self):
        scene = fixture(4)
        reports = accepted_reports(scene)
        reviewed = TrustedSimulator(scene, reports)
        self.assertTrue(reviewed.public_session().observe()["ready_tasks"])
        with self.assertRaisesRegex(ValueError, "complete_reviewed_scene"):
            TrustedSimulator(scene, reports[:-1])
        altered = deepcopy(reports)
        altered[0]["answer"] = [{"made_up": 100}]
        with self.assertRaisesRegex(ValueError, "not_oracle_grounded"):
            TrustedSimulator(scene, altered)
        altered = deepcopy(reports)
        altered[0]["report_authoring"]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "unapproved_report"):
            TrustedSimulator(scene, altered)


if __name__ == "__main__":
    unittest.main()
