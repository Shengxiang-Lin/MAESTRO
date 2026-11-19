"""CPU/offline contract checks for all five modules (no API and no model weights)."""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reliability_core import load_reviewer, normalize_review, public_reports


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
    async def __call__(self, prompt, **kwargs):
        if not self.responses:
            raise RuntimeError("Unexpected reviewer LLM call")
        return json.dumps(self.responses.pop(0))


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.reports = public_reports([
            {"attempt": 1, "status": "done", "instruction": "Find date",
             "model": "worker", "result": "April 2", "summary": "Date April 2"},
            {"attempt": 2, "is_parallel_batch": True, "subtask_results": [
                {"status": "done", "instruction": "Check date", "model": "worker",
                 "result": "April 3", "summary": "Date April 3"},
                {"status": "partial", "instruction": "Find source",
                 "model": "worker", "result": "-", "summary": "No source"}]},
        ])

    def test_public_history_and_normalizer(self):
        self.assertEqual([x["id"] for x in self.reports],
                         ["r1.1", "r2.1", "r2.2"])
        self.assertEqual(self.reports[0]["result"], "April 2")
        result = normalize_review({"evidence": [
            {"report_id": "r1.1", "risk": 2, "reason": "high"},
            {"report_id": "oracle", "risk": 0, "reason": "bad"}],
            "suggestions": []}, self.reports)
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["evidence"][0]["risk"], 1.0)

    def test_propagation(self):
        from importlib import util
        meth = load_reviewer("propuq-mas")
        result = asyncio.run(meth(
            reports=self.reports, llm=None, options={
                "local_uncertainty": {"r1.1": 0.5, "r2.1": 0.2, "r2.2": 0.1},
                "edges": [{"parent": "r1.1", "child": "r2.1", "uptake": 0.5}]}))
        values = {x["report_id"]: x["risk"] for x in result["evidence"]}
        self.assertAlmostEqual(values["r2.1"], 0.4)
        self.assertEqual(result["details"]["estimator"], "provided")

    def test_ledger_conflict(self):
        fake = FakeLLM([{"claims": [
            {"report_id": "r1.1", "subject": "event", "attribute": "date",
             "scope": "same", "value": "April 2"},
            {"report_id": "r2.1", "subject": "event", "attribute": "date",
             "scope": "same", "value": "April 3"}], "open_needs": [], "issues": []}])
        result = asyncio.run(load_reviewer("omni-decision")(
            reports=self.reports, llm=fake, options={}))
        self.assertGreater(result["evidence"][0]["risk"], 0.8)
        self.assertEqual(result["details"]["ledger"][0]["state"], "conflict")

    def test_verify_hypotheses(self):
        fake = FakeLLM([
            {"hypotheses": [{"report_id": "r1.1", "error_type": "conflict",
                             "claim": "A conflicts with B"}]},
            {"verdicts": [{"index": 0, "label": "entailed",
                           "reason": "Visible disagreement"}]}])
        result = asyncio.run(load_reviewer("verifymas")(
            reports=self.reports, llm=fake, options={}))
        self.assertEqual(result["evidence"][0]["risk"], 0.9)
        self.assertEqual(result["suggestions"][0]["action"], "verify")

    def test_hassum_clusters(self):
        fake = FakeLLM([
            {"equivalent": True, "similarity": 0.9},
            {"equivalent": False, "similarity": 0.1},
            {"equivalent": False, "similarity": 0.1},
            {"equivalent": True, "similarity": 0.9}])
        result = asyncio.run(load_reviewer("hassum")(
            reports=self.reports[:1], llm=fake,
            options={"samples_by_report": {"r1.1": ["A", "A", "B", "B"]}}))
        self.assertEqual(result["details"]["r1.1"]["group_sizes"], [2, 2])
        self.assertGreater(result["evidence"][0]["risk"], 0.0)

    def test_a_trust_fails_without_whitebox(self):
        with self.assertRaises(ValueError):
            asyncio.run(load_reviewer("a-trust")(
                reports=self.reports, llm=None, options={}))


if __name__ == "__main__":
    unittest.main()
