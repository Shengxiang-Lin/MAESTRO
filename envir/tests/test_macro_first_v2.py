"""Macro-first V2 regression: varied realistic goals, evidence and modalities."""
from __future__ import annotations

import copy
import unittest

from envir.generate_question.spec import make_spec, spec_issues
from envir.generate_question.question import reference_question, guard_question
from envir.generate_question.quality import compile_and_validate, validate_replay
from envir.generate_question.compiler import QuestionSession, oracle_schedule
from envir.generate_question.pipeline import quality_report
from envir.implement_subtask.realize import realize_scene
from envir.generate_question.program_ir import MissingSourceError


def spec_for(i, difficulty=None):
    topics = (
        ("Premier League", "sports"),
        ("The Office", "film"),
        ("Pearl Jam", "music"),
        ("Library catalog", "book"),
        ("Science outreach project", "science"),
        ("Survey map", "geography"),
        ("Operations review", "business"),
        ("Historical exhibition", "history"),
    )
    topic, domain = topics[i % len(topics)]
    return make_spec(
        seed_record={
            "record_id": f"diverse-{i}",
            "topic": topic,
            "domain_hint": domain,
            "source_modalities": ["image", "text", "table"],
        },
        ordinal=i, run_seed=28,
        difficulty=difficulty or ("hard" if i % 2 else "medium"),
        macro_first_v2_share=100,
    )


class DiverseMacroTests(unittest.TestCase):
    def test_new_mode_is_opt_in_without_changing_v1(self):
        seed = {"record_id": "source-3", "topic": "Premier League",
                "domain_hint": "sports"}
        old = make_spec(seed_record=seed, ordinal=12, run_seed=28,
                        difficulty="medium", macro_first_share=100)
        unchanged = make_spec(seed_record=seed, ordinal=12, run_seed=28,
                              difficulty="medium", macro_first_share=100,
                              macro_first_v2_share=0)
        self.assertEqual(old, unchanged)
        v2 = make_spec(seed_record=seed, ordinal=12, run_seed=28,
                       difficulty="medium", macro_first_share=100,
                       macro_first_v2_share=100)
        self.assertEqual(v2["reasoning_pattern"], "macro_first_v2")
        self.assertEqual(v2, make_spec(seed_record=seed, ordinal=12, run_seed=28,
                       difficulty="medium", macro_first_v2_share=100))
        easy = make_spec(seed_record=seed, ordinal=12, run_seed=28,
                         difficulty="easy", macro_first_v2_share=100)
        self.assertNotEqual(easy["reasoning_pattern"], "macro_first_v2")

    def test_120_frozen_scenes_and_natural_questions(self):
        modes, goals, macro_counts = set(), set(), set()
        sizes, scopes, local_lengths, domains = set(), set(), set(), set()
        has_temporal, has_audio = False, False
        for i in range(120):
            spec = spec_for(i)
            self.assertEqual(spec_issues(spec), [])
            self.assertEqual(spec, spec_for(i))
            plan = spec["macro_plan"]
            modes.add(plan["pattern"])
            goals.add(plan["goal"]["answer_field"])
            domains.add(plan["domain"])
            q = reference_question(spec)
            self.assertEqual(guard_question(q, spec), [])
            self.assertNotIn("fictional", q.lower())
            self.assertNotIn("simulated", q.lower())
            scene = compile_and_validate(spec, q)
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertEqual(oracle_schedule(scene)["answer"], scene["gold_answer"])
            self.assertEqual(len(scene["gold_answer"]), 1)
            self.assertEqual(
                set(scene["gold_answer"][0]), {plan["goal"]["answer_field"]}
            )
            self.assertEqual(scene["macro_graph"]["macro_task_count"],
                             len(plan["macro_tasks"]))
            self.assertEqual(
                [task["task_id"] for task in scene["macro_graph"]["tasks"]],
                ["work_" + t["macro_id"] for t in plan["macro_tasks"]]
            )
            self.assertEqual(
                [task["depends_on"] for task in scene["macro_graph"]["tasks"]],
                [["work_" + v for v in t["depends_on"]]
                 for t in plan["macro_tasks"]]
            )
            macro_counts.add(len(plan["macro_tasks"]))
            for m in plan["macro_tasks"]:
                scopes.add(len(m["source_scope"]))
                sizes.update(m["evidence_plan"]["source_observation_targets"].values())
            for task in scene["macro_graph"]["tasks"]:
                local_lengths.add(len(task["operator_chain"]))
            for sid, source in scene["world"]["sources"].items():
                expected = next(
                    m["evidence_plan"]["source_observation_targets"][sid]
                    for m in plan["macro_tasks"] if sid in m["source_scope"]
                )
                self.assertEqual(len(source["observations"]), expected)
                self.assertEqual(
                    set(source["schema"]), set(plan["source_schemas"][sid])
                )
                if source["access_policy"]["mode"] == "keyed_lookup":
                    with self.assertRaises(MissingSourceError):
                        QuestionSession(scene).read_source(sid)
            self.assertEqual(
                scene["public"]["status"], "ready_for_investigation"
            )
            self.assertNotIn(
                "synthetic", str(scene["public"]).lower()
            )
            self.assertTrue(scene["difficulty_audit"][
                "macro_first_audit"]["source_counterfactuals_pass"])
            self.assertTrue(scene["difficulty_audit"][
                "macro_first_audit"]["handoff_counterfactuals"][
                    "all_handoffs_necessary"])
            if plan["pattern"] == "multimodal_event":
                has_temporal = True
                first = scene["macro_graph"]["tasks"][0]
                self.assertEqual(first["source_ids"], ["footage", "commentary"])
                self.assertIn("compare_fields", [x["op"]
                                for x in first["operator_chain"]])
                self.assertTrue(any(term in q.lower() for term in
                                    ("followed by", "later", "after", "before")))  # explicit ordering
            if any(s["modality"] == "audio" for s in spec["sources"]):
                has_audio = True
        self.assertEqual(modes, {
            "serial_identity", "parallel_cues", "disputed_reports",
            "multimodal_event",
        })
        self.assertGreaterEqual(len(goals), 8)
        self.assertGreaterEqual(len(domains), 6)
        self.assertGreaterEqual(len(macro_counts), 4)
        self.assertGreaterEqual(len(sizes), 8)
        self.assertEqual(scopes, {0, 1, 2})
        self.assertGreaterEqual(max(local_lengths), 6)
        self.assertTrue(has_temporal and has_audio)

    def test_stage_two_reports_and_quality_metrics(self):
        scenes = [compile_and_validate(spec_for(i), reference_question(spec_for(i)))
                  for i in range(20)]
        for scene in scenes:
            sub = realize_scene(scene)
            self.assertEqual(sub["macro_task_count"],
                             scene["macro_graph"]["macro_task_count"])
            terminal = next(record for record in sub["macro_tasks"]
                            if record["macro_task_id"] ==
                            sub["terminal_macro_task_id"])
            self.assertEqual(terminal["answer"], scene["gold_answer"])
        metrics = quality_report(scenes, no_progress=True)
        self.assertEqual(metrics["replay_pass"], 20)
        self.assertEqual(metrics["macro_first_v2"]["scenes"], 20)
        self.assertGreater(metrics["macro_first_v2"][
            "video_audio_joint_macros"], 0)
        self.assertGreater(len(metrics["macro_first_v2"][
            "source_observation_count_histogram"]), 5)

    def test_ranked_oracle_evidence_sampling_prioritizes_top_row(self):
        from envir.implement_subtask.realize import _operator_evidence
        from envir.generate_question.program_ir import Relation
        product = Relation(("entry_label", "priority"), (
            {"values": {"entry_label": "Entry-09", "priority": 915},
             "support": ("register:F009",)},
            {"values": {"entry_label": "Entry-01", "priority": 300},
             "support": ("register:F001",)},
        ))
        facts = {
            "register:F001": {"fact_ref": "register:F001",
                              "source_id": "register", "values": {"priority": 300}},
            "register:F009": {"fact_ref": "register:F009",
                              "source_id": "register", "values": {"priority": 915}},
        }
        result = _operator_evidence(
            {"id": "ranked", "op": "sort", "inputs": ["rows"],
             "field": "priority", "descending": True},
            product,
            {"source_fact_refs": ["register:F001", "register:F009"]},
            facts,
            {"register"},
            evidence_limit=1,
        )
        self.assertEqual(result["evidence_atoms"][0]["fact_ref"],
                         "register:F009")
        self.assertEqual(result["supporting_fact_refs"],
                         ["register:F001", "register:F009"])

    def test_tampered_local_plan_and_world_fail_closed(self):
        spec = spec_for(4)
        changed = copy.deepcopy(spec)
        changed["macro_plan"]["macro_tasks"][0]["evidence_plan"][
            "source_observation_targets"] = {"visual": 1}
        self.assertIn("macro_first_v2_plan_not_reproducible",
                      spec_issues(changed))
        scene = compile_and_validate(spec, reference_question(spec))
        changed = copy.deepcopy(scene)
        sid = next(iter(changed["world"]["sources"]))
        changed["world"]["sources"][sid]["observations"].pop()
        self.assertEqual(validate_replay(changed)["status"], "invalid")
        changed = copy.deepcopy(scene)
        changed["macro_graph"]["tasks"][-1]["depends_on"] = []
        self.assertEqual(validate_replay(changed)["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
