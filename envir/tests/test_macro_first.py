"""Macro-first integration: frozen Macro boundaries and local Solver replay."""
from __future__ import annotations

import copy
import unittest

from envir.generate_question.spec import make_spec, spec_issues
from envir.generate_question.question import reference_question, guard_question
from envir.generate_question.quality import compile_and_validate, validate_replay
from envir.generate_question.compiler import QuestionSession, oracle_schedule
from envir.generate_question.macro_first import (
    PATTERNS, compile_macro_solvers, sample_macro_plan, validate_handoffs,
)
from envir.generate_question.program_ir import MissingSourceError
from envir.implement_subtask.realize import realize_scene


def make(i: int, level: str = "medium"):
    return make_spec(
        seed_record={
            "record_id": f"macro-seed-{i}",
            "topic": f"Reference Topic {i}",
            "domain_hint": "history",
            "source_modalities": ["image", "text", "table"],
        },
        ordinal=i, run_seed=17, difficulty=level,
        macro_first_share=100,
    )


class MacroFirstTests(unittest.TestCase):
    def test_opt_in_disabled_preserves_legacy_and_easy_modes(self):
        seed = {"record_id": "legacy", "topic": "Legacy Collection"}
        old = make_spec(seed_record=seed, ordinal=7, run_seed=17,
                        difficulty="medium")
        self.assertEqual(old, make_spec(
            seed_record=seed, ordinal=7, run_seed=17,
            difficulty="medium", macro_first_share=0,
        ))
        easy = make_spec(seed_record=seed, ordinal=7, run_seed=17,
                         difficulty="easy", macro_first_share=100)
        self.assertNotEqual(easy["reasoning_pattern"], "macro_first")
        with self.assertRaisesRegex(ValueError, "macro_first_share"):
            make_spec(seed_record=seed, ordinal=7, run_seed=17,
                      difficulty="medium", macro_first_share=101)

    def test_frozen_macro_plan_and_local_oracle_for_90_seeds(self):
        patterns, structures, target_labels = set(), set(), set()
        for i in range(90):
            difficulty = "hard" if i % 2 else "medium"
            spec = make(i, difficulty)
            self.assertEqual(spec_issues(spec), [])
            self.assertEqual(spec, make(i, difficulty))
            question = reference_question(spec)
            self.assertEqual(guard_question(question, spec), [])
            scene = compile_and_validate(spec, question)
            self.assertEqual(validate_replay(scene)["status"], "valid")
            plan = spec["macro_plan"]
            patterns.add(plan["pattern"])
            graph = scene["macro_graph"]
            self.assertEqual(graph["plan_origin"], "macro_first")
            self.assertEqual(graph["macro_task_count"],
                             len(plan["macro_tasks"]))
            self.assertEqual(
                [t["task_id"] for t in graph["tasks"]],
                ["work_" + t["macro_id"] for t in plan["macro_tasks"]]
            )
            self.assertEqual(
                [t["depends_on"] for t in graph["tasks"]],
                [["work_" + d for d in t["depends_on"]]
                 for t in plan["macro_tasks"]]
            )
            self.assertTrue(all(len(t["operator_chain"]) >= 2
                                for t in graph["tasks"]))
            self.assertEqual(oracle_schedule(scene)["answer"],
                             scene["gold_answer"])
            audit = scene["difficulty_audit"]["macro_first_audit"]
            self.assertTrue(audit["macro_boundary_preserved"])
            self.assertTrue(audit["handoff_counterfactuals"][
                "all_handoffs_necessary"])
            self.assertTrue(audit["source_counterfactuals_pass"])
            self.assertEqual(len(scene["gold_answer"]), 1)
            self.assertEqual(
                set(scene["gold_answer"][0]),
                {plan["goal"]["answer_field"]}
            )
            self.assertEqual({sid for row in
                scene["difficulty_audit"]["answer_provenance"]
                for ref in row["source_fact_refs"]
                for sid in [ref.split(":")[0]]},
                set(scene["world"]["sources"])
            )
            structures.add((
                plan["pattern"], plan["index_hops"],
                tuple(tuple(t["depends_on"]) for t in plan["macro_tasks"])
            ))
            if "visual" in scene["world"]["sources"]:
                witness = scene["world"]["sources"]["visual"]["observations"]
                color = plan["clues"]["emblem_color"]
                for row in witness:
                    if row["values"]["emblem_color"] == color:
                        target_labels.add(row["values"]["candidate_id"])
        self.assertEqual(patterns, set(PATTERNS))
        self.assertGreaterEqual(len(structures), 5)
        self.assertGreater(len(target_labels), 2)

    def test_stage_two_oracle_for_every_pattern(self):
        covered = {}
        for i in range(80):
            spec = make(i)
            mode = spec["macro_plan"]["pattern"]
            if mode in covered:
                continue
            scene = compile_and_validate(spec, reference_question(spec))
            oracle = realize_scene(scene)
            self.assertEqual(oracle["macro_task_count"],
                             len(spec["macro_plan"]["macro_tasks"]))
            self.assertEqual(oracle["reference_waves"],
                             scene["macro_graph"]["reference_waves"])
            terminal = next(t for t in oracle["macro_tasks"]
                            if t["macro_task_id"] == oracle["terminal_macro_task_id"])
            self.assertEqual(terminal["answer"], scene["gold_answer"])
            self.assertTrue(all(t["trace_summary"]["steps"]
                                for t in oracle["macro_tasks"]))
            covered[mode] = True
        self.assertEqual(set(covered), set(PATTERNS))

    def test_macro_contract_tampering_is_rejected(self):
        spec = make(4)
        altered = copy.deepcopy(spec)
        altered["macro_plan"]["macro_tasks"][1]["depends_on"] = ["M1"]
        self.assertIn("macro_plan_not_reproducible_from_frozen_seed",
                      spec_issues(altered))
        altered = copy.deepcopy(spec)
        altered["macro_plan"]["macro_tasks"][0]["source_scope"] = ["archive"]
        self.assertIn("macro_plan_not_reproducible_from_frozen_seed",
                      spec_issues(altered))
        scene = compile_and_validate(spec, reference_question(spec))
        tampered = copy.deepcopy(scene)
        tampered["macro_graph"]["tasks"][0]["depends_on"] = ["work_M3"]
        self.assertEqual(validate_replay(tampered)["status"], "invalid")

    def test_keyed_sources_are_not_browsable(self):
        spec = make(8, "hard")
        scene = compile_and_validate(spec, reference_question(spec))
        session = QuestionSession(scene)
        self.assertNotIn("macro_plan", session.public_view)
        self.assertNotIn("gold_program", session.public_view)
        for sid, source in scene["world"]["sources"].items():
            if source["access_policy"]["mode"] == "keyed_lookup":
                with self.assertRaises(MissingSourceError):
                    session.read_source(sid)

    def test_direct_handoff_interventions_change_answer(self):
        for i in range(15):
            spec = make(i)
            scene = compile_and_validate(spec, reference_question(spec))
            report = validate_handoffs(scene)
            self.assertGreaterEqual(report["edge_count"], 2)
            self.assertEqual(report["edge_count"], report["sensitive_edges"])


if __name__ == "__main__":
    unittest.main()
