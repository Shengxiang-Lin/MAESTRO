"""Current cross-wave trust-state, guarded evidence IR and Question paraphrases."""
from __future__ import annotations

import copy
import unittest

from environment.generate_question.spec import make_spec
from environment.generate_question.question import (
    reference_question, guard_question,
)
from environment.generate_question.surface_quality import is_explicit_user_request
from environment.generate_question.quality import (
    compile_and_validate, validate_replay,
)
from environment.generate_question.pipeline import quality_report
from environment.generate_question.program_ir import (
    Relation, ProgramError, _apply, validate_program,
)
from environment.implement_subtask.realize import realize_scene
from environment.implement_subtask.stage_local_claims import compile_claim_contract


def seeded_spec(pattern, mode=None, kind=None):
    for i in range(250):
        spec = make_spec(
            seed_record={
                "record_id": f"v14-witness-{i:04d}",
                "topic": "Collection event records",
                "domain_hint": "general",
                "source_modalities": ["text", "table"],
            },
            ordinal=i, run_seed=17, difficulty="hard",
        )
        plan = spec["macro_plan"]
        if (plan["pattern"] == pattern
                and (mode is None or plan["claim_verification_mode"] == mode)
                and (kind is None or plan["goal_resolution"]["kind"] == kind)):
            return spec
    raise AssertionError(f"no_{pattern}_{mode}_{kind}")


class QuestionGuardPrecisionTests(unittest.TestCase):
    def test_locate_is_a_genuine_action_request(self):
        self.assertTrue(is_explicit_user_request(
            "In the regional event archive, locate the event with the "
            "recorded year discrepancy and determine its confirmed venue."
        ))
        self.assertFalse(is_explicit_user_request(
            "In the regional event archive there is an event with a "
            "recorded year discrepancy and an unconfirmed venue."
        ))

    def test_higher_one_comparison_paraphrase(self):
        spec = seeded_spec("disputed_reports", kind="comparison")
        scene = compile_and_validate(spec, reference_question(spec))
        original = scene["question"]
        self.assertIn("higher count, first or second?", original)
        paraphrase = original.replace(
            "which of the two independent sources has the higher count, "
            "first or second?",
            "which of the two independently recorded counts is the higher one?",
        )
        problems = guard_question(paraphrase, spec, scene["macro_graph"])
        self.assertNotIn("macro_first_v2_goal_semantics_missing", problems)
        unrelated = original.replace(
            "which of the two independent sources has the higher count, "
            "first or second?",
            "what is the reported case identifier?",
        )
        self.assertIn(
            "macro_first_v2_goal_semantics_missing",
            guard_question(unrelated, spec, scene["macro_graph"]),
        )


class ClaimRelationOperatorTests(unittest.TestCase):
    def test_real_claim_filter_must_have_both_fact_sources(self):
        node = {
            "op": "claim_relation_filter", "inputs": ["joined"],
            "subject_field": "case_id",
            "left_field": "reported_year_a",
            "right_field": "reported_year_b",
            "left_source_id": "report_a", "right_source_id": "report_b",
            "attribute": "reported_event_year",
            "valid_scope": "same_reported_event",
            "relation": "contradicts",
        }
        columns = ("case_id", "reported_year_a", "reported_year_b")
        result = _apply(node, [Relation(columns, ({
            "values": {"case_id": "Case-001", "reported_year_a": 2018,
                       "reported_year_b": 2019},
            "support": ("report_a:F003", "report_b:F001"),
        },))])
        self.assertEqual(len(result.rows), 1)
        with self.assertRaisesRegex(ProgramError, "right_provenance"):
            _apply(node, [Relation(columns, ({
                "values": {"case_id": "Case-001", "reported_year_a": 2018,
                           "reported_year_b": 2019},
                "support": ("report_a:F003",),
            },))])
        wrong_types = Relation(columns, ({
            "values": {"case_id": "Case-001", "reported_year_a": 2018,
                       "reported_year_b": "2019"},
            "support": ("report_a:F003", "report_b:F001"),
        },))
        with self.assertRaisesRegex(ProgramError, "incomparable_types"):
            _apply(node, [wrong_types])
        invalid = dict(node, id="one", inputs=["source"], relation="corroborates")
        program = {
            "nodes": [
                {"id": "source", "op": "scan", "inputs": [], "source_id": "x"},
                invalid,
            ],
            "output": "one",
        }
        self.assertIn(
            "claim_relation_filter_invalid_contract",
            validate_program(program),
        )

    def test_keyed_second_report_is_real_late_macro(self):
        spec = seeded_spec("disputed_reports", mode="keyed_followup")
        scene = compile_and_validate(spec, reference_question(spec))
        self.assertEqual(validate_replay(scene)["status"], "valid")
        tasks = {x["task_id"]: x for x in scene["macro_graph"]["tasks"]}
        self.assertEqual(tasks["work_M2"]["depends_on"], ["work_M1"])
        self.assertIn("lookup_by_key", [
            n["op"] for n in tasks["work_M2"]["operator_chain"]
        ])
        self.assertIn("claim_relation_filter", [
            n["op"] for n in tasks["work_M3"]["operator_chain"]
        ])
        self.assertEqual(
            scene["world"]["sources"]["report_b"]["access_policy"]["mode"],
            "keyed_lookup",
        )
        report_b = next(
            role for role in spec["sources"]
            if role["source_id"] == "report_b"
        )
        self.assertIn(
            report_b["acquisition_route"],
            {"search_only", "search_then_extract"},
        )
        self.assertNotIn(
            "resource_url", scene["world"]["sources"]["report_b"],
        )
        self.assertEqual(
            report_b["evidence_access"]["tool_steps"][0]["input_binding"],
            "upstream_key_query",
        )
        m1 = next(r for r in scene["task_feedback"] if r["task_id"] == "work_M1")
        m2 = next(r for r in scene["task_feedback"] if r["task_id"] == "work_M2")
        self.assertLessEqual(
            {x["values"]["case_id"] for x in m2["workproduct"]},
            {x["values"]["case_id"] for x in m1["workproduct"]},
        )
        ledger = scene["evidence_ledger"]
        trust = ledger["reference_wave_trust"]
        self.assertEqual(trust["mode"], "keyed_followup")
        self.assertEqual(ledger["source_first_wave"]["report_a"], 1)
        self.assertEqual(ledger["source_first_wave"]["report_b"], 2)
        self.assertGreater(trust["cross_wave_claim_pairs"], 0)
        self.assertGreater(trust["answer_relevant_cross_wave_pairs"], 0)
        relevant = [item for item in trust["transitions"]
                    if item["answer_supporting_claim_pair"]]
        self.assertTrue(any(
            row["before_status"] == "missing"
            and row["after_status"] == "unverified" for row in relevant
        ))
        self.assertTrue(any(
            row["before_status"] == "unverified"
            and row["after_status"] == "disputed" for row in relevant
        ))
        oracle = realize_scene(scene)
        dispute = next(x for x in oracle["macro_tasks"]
                       if x["macro_task_id"] == "work_M3")
        self.assertTrue(compile_claim_contract(dispute)[
            "reported_year_disagreement_checked_locally"
        ])
        stats = quality_report([scene], no_progress=True)
        self.assertGreater(
            stats["evidence_relations"]["reference_wave_trust"]
                 ["answer_relevant_cross_wave_pairs"], 0,
        )

    def test_parallel_second_report_remains_an_independent_wave_one(self):
        spec = seeded_spec("corroborated_dispute", mode="parallel")
        scene = compile_and_validate(spec, reference_question(spec))
        self.assertEqual(validate_replay(scene)["status"], "valid")
        tasks = {x["task_id"]: x for x in scene["macro_graph"]["tasks"]}
        self.assertEqual(tasks["work_M2"]["depends_on"], [])
        self.assertEqual(
            scene["world"]["sources"]["report_b"]["access_policy"]["mode"],
            "browse",
        )
        self.assertEqual(
            scene["evidence_ledger"]["reference_wave_trust"][
                "cross_wave_claim_pairs"
            ], 0,
        )
        self.assertTrue(all(
            item["before_status"] == "missing"
            for item in scene["evidence_ledger"]["reference_wave_trust"][
                "transitions"
            ]
        ))

    def test_tampering_claim_or_source_origin_rejected(self):
        spec = seeded_spec("disputed_reports", mode="keyed_followup")
        scene = compile_and_validate(spec, reference_question(spec))
        tampered = copy.deepcopy(scene)
        tampered["evidence_ledger"]["reference_wave_trust"][
            "cross_wave_claim_pairs"
        ] = 0
        self.assertIn(
            "evidence_ledger_not_reproducible",
            validate_replay(tampered)["issues"],
        )
        tampered = copy.deepcopy(scene)
        tampered["world"]["sources"]["report_b"]["evidence_origin_id"] = (
            "independent_stream:report_a"
        )
        self.assertEqual(validate_replay(tampered)["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
