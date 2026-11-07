"""No-network isolation, per-fragment repair and deterministic DAG assembly."""
from __future__ import annotations

from collections import Counter
import copy
import json
from threading import Lock
import unittest

from environment.delegation_blueprint import (
    VERSION, naturalize_delegation_blueprint,
    validate_delegation_blueprint, goal_for_fragment,
)


def _outline():
    def task(fid, wave, deps, operation, value):
        return {
            "fragment_id": fid,
            "wave": wave,
            "depends_on": deps,
            "input_slots": [
                {"from_fragment": d, "kind": "completed_result"} for d in deps
            ],
            "output_kind": "entity_set",
            "public_steps": [{
                "id": "S" + fid[-1] if fid[-1].isdigit() else "S3",
                "operation": operation,
                "inputs_from": [],
                "attribute": "population",
                "comparison": "equal to",
                "public_value": value,
                "goal": "Retain exactly the public candidates in this branch.",
            }],
            "capabilities": ["ExtractUrlContentAction"],
        }
    return {
        "schema": "public_delegation_outline.v1",
        "schedule": "compact",
        "waves": [
            ["fragment:0", "fragment:1"],
            ["fragment:terminal"],
        ],
        "task_semantics": {"goal": "Which entities meet both conditions?"},
        "public_modalities": [{"medium": "text", "availability": "provided"}],
        "tasks": [
            task("fragment:0", 1, [], "FILTER_NUMBER", "10"),
            task("fragment:1", 1, [], "FILTER_NUMBER", "20"),
            task("fragment:terminal", 2,
                 ["fragment:0", "fragment:1"], "INTERSECT", ""),
        ],
    }


class LocalGoalBackend:
    def __init__(self, bad_once=None, always_bad=None):
        self.calls = []
        self.counts = Counter()
        self.lock = Lock()
        self.bad_once = bad_once
        self.always_bad = always_bad

    def complete_json(self, system, user, **kwargs):
        packet = json.loads(user)
        current = packet["current_fragment"]
        fid = current["fragment_id"]
        with self.lock:
            self.calls.append(packet)
            self.counts[fid] += 1
            attempt = self.counts[fid]
        if fid == self.always_bad:
            return {"goal": "Use the preceding task result to answer."}
        if fid == self.bad_once and attempt == 1:
            return {"goal": "Use the previous task result to answer."}
        if current["depends_on"]:
            return {"goal": (
                "Intersect the two separately specified upstream entity sets "
                "and return the matching entities with public evidence."
            )}
        return {"goal": (
            "Find candidates meeting the local public population condition "
            "and return the verified entity set."
        )}


class IndependentFragmentGoalTests(unittest.TestCase):
    def test_one_request_per_fragment_and_dag_order_preserved(self):
        outline = _outline()
        backend = LocalGoalBackend()
        audit = {}
        blueprint = naturalize_delegation_blueprint(
            backend, outline, "Which entities meet both conditions?",
            max_workers=3, audit=audit,
        )
        self.assertEqual(blueprint["version"], VERSION)
        self.assertEqual(
            [g["fragment_id"] for g in blueprint["goals"]],
            [t["fragment_id"] for t in outline["tasks"]],
        )
        self.assertEqual(blueprint["outline"]["waves"], outline["waves"])
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual(audit["author_calls"], 3)
        self.assertEqual(
            {packet["current_fragment"]["fragment_id"] for packet in backend.calls},
            {"fragment:0", "fragment:1", "fragment:terminal"},
        )
        self.assertEqual(
            {dep["input_slot"]["from_fragment"]
             for packet in backend.calls
             if packet["current_fragment"]["fragment_id"] == "fragment:terminal"
             for dep in packet["dependency_contracts"]},
            {"fragment:0", "fragment:1"},
        )
        for packet in backend.calls:
            fragment = packet["current_fragment"]
            self.assertEqual(
                set(fragment["depends_on"]),
                {d["input_slot"]["from_fragment"] for d in packet["dependency_contracts"]},
            )
            self.assertNotIn("goals", packet)
            self.assertNotIn("verified_public_outline", packet)
            self.assertNotIn("expected_answer", json.dumps(packet))
            if fragment["fragment_id"] in ("fragment:0", "fragment:1"):
                self.assertEqual(packet["dependency_contracts"], [])
        self.assertTrue(
            validate_delegation_blueprint(blueprint, outline)
        )
        self.assertIn("Intersect", goal_for_fragment(blueprint, "fragment:terminal"))

    def test_retry_is_only_on_failed_fragment(self):
        outline = _outline()
        backend = LocalGoalBackend(bad_once="fragment:1")
        audit = {}
        blueprint = naturalize_delegation_blueprint(
            backend, outline, "Which entities meet both conditions?",
            max_workers=2, audit=audit,
        )
        self.assertEqual(backend.counts["fragment:0"], 1)
        self.assertEqual(backend.counts["fragment:1"], 2)
        self.assertEqual(backend.counts["fragment:terminal"], 1)
        self.assertEqual(audit["author_calls"], 4)
        self.assertEqual(
            next(x for x in audit["fragments"]
                 if x["fragment_id"] == "fragment:1")["calls"], 2,
        )
        self.assertTrue(validate_delegation_blueprint(blueprint, outline))

    def test_persistent_ambiguity_fails_closed(self):
        backend = LocalGoalBackend(always_bad="fragment:1")
        audit = {}
        with self.assertRaisesRegex(
            ValueError, "blueprint_fragment_failed:fragment:1",
        ):
            naturalize_delegation_blueprint(
                backend, _outline(), "Which entities meet both conditions?",
                max_workers=2, audit=audit,
            )
        self.assertEqual(backend.counts["fragment:1"], 2)
        self.assertEqual(audit["author_calls"], sum(backend.counts.values()))

    def test_tampered_positional_goal_is_invalid_on_replay(self):
        outline = _outline()
        blueprint = naturalize_delegation_blueprint(
            LocalGoalBackend(), outline, "Which entities meet both conditions?",
        )
        forged = copy.deepcopy(blueprint)
        forged["goals"][-1]["goal"] = (
            "Use the preceding task result and return the matching entities."
        )
        with self.assertRaisesRegex(
            ValueError, "blueprint_ambiguous_positional_dependency",
        ):
            validate_delegation_blueprint(forged, outline)


if __name__ == "__main__":
    unittest.main()
