"""Macro-first V2 regression: varied realistic goals, evidence and modalities."""
from __future__ import annotations

import copy
import unittest

from environment.generate_question.spec import make_spec, spec_issues
from environment.generate_question.question import reference_question, guard_question
from environment.generate_question.quality import compile_and_validate, validate_replay
from environment.generate_question.compiler import QuestionSession, oracle_schedule
from environment.generate_question.pipeline import quality_report
from environment.implement_subtask.realize import realize_scene
from environment.generate_question.program_ir import MissingSourceError


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
    def test_current_medium_hard_use_macro_v2_and_easy_stays_simple(self):
        seed = {"record_id": "source-3", "topic": "Premier League",
                "domain_hint": "sports"}
        current = make_spec(
            seed_record=seed, ordinal=12, run_seed=28, difficulty="medium",
        )
        self.assertEqual(current["reasoning_pattern"], "macro_first_v2")
        self.assertEqual(current["reasoning_pattern"], make_spec(
            seed_record=seed, ordinal=12, run_seed=28, difficulty="medium",
            macro_first_v2_share=100,
        )["reasoning_pattern"])
        easy = make_spec(
            seed_record=seed, ordinal=12, run_seed=28, difficulty="easy",
        )
        self.assertEqual(easy["reasoning_pattern"], "approval")
        with self.assertRaisesRegex(ValueError, "macro_planning_share"):
            make_spec(seed_record=seed, ordinal=12, run_seed=28,
                      difficulty="medium", macro_first_v2_share=0)

    def test_frozen_output_schema_key_order_is_semantically_irrelevant(self):
        # Scenario round-trip serialization may sort JSON object keys; a
        # typed relation's field *set* must not depend on dict insertion order.
        found = False
        for i in range(60):
            spec = spec_for(i)
            target = next((
                task for task in spec["macro_plan"]["macro_tasks"]
                if task["role"] == "metric_a_lookup"
            ), None)
            if target is None:
                continue
            schema = target["output_schema"]
            altered = dict(reversed(list(schema.items())))
            if list(altered) == list(schema):
                continue
            target["output_schema"] = altered
            scene = compile_and_validate(spec, reference_question(spec))
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertEqual(
                set(scene["macro_graph"]["tasks"][
                    next(j for j, t in enumerate(scene["macro_graph"]["tasks"])
                         if t["macro_role"] == "metric_a_lookup")
                ]["outputs"]["typed_relation"]),
                set(schema),
            )
            found = True
            break
        self.assertTrue(found, "test must exercise a two-column metric Macro")

    def test_declared_input_schema_ignores_order_but_checks_types(self):
        # Regresses 41 genuine candidate examples incorrectly rejected as
        # macro_declared_input_type_mismatch when JSON sorted schema keys.
        from environment.generate_question.program_ir import ProgramError
        compiled = 0
        for i in range(100):
            spec = spec_for(i)
            plan = spec["macro_plan"]["macro_tasks"]
            dependent = next((task for task in plan
                              if task["input_contract"]
                              and any(len(fields) > 1 for fields in
                                      task["input_contract"].values())), None)
            if dependent is None:
                continue
            reference = reference_question(spec)
            scene = compile_and_validate(spec, reference)
            self.assertEqual(validate_replay(scene)["status"], "valid")
            compiled += 1
            if compiled == 1:
                bad = copy.deepcopy(spec)
                dep = next(task for task in bad["macro_plan"]["macro_tasks"]
                           if task["macro_id"] == dependent["macro_id"])
                parent = next(iter(dep["input_contract"]))
                field = next(iter(dep["input_contract"][parent]))
                before = dep["input_contract"][parent][field]
                dep["input_contract"][parent][field] = (
                    "int" if before != "int" else "str"
                )
                with self.assertRaises(Exception):
                    compile_and_validate(bad, reference)
        self.assertGreater(compiled, 15)

    def test_120_frozen_scenes_and_natural_questions(self):
        modes, goals, macro_counts, distinct_answers = set(), set(), set(), set()
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
            distinct_answers.add(str(scene["gold_answer"]))
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
            "multimodal_event", "corroborated_dispute",
        })
        self.assertGreaterEqual(len(goals), 8)
        self.assertGreater(len(distinct_answers), 30)
        self.assertGreaterEqual(len(domains), 6)
        self.assertGreaterEqual(len(macro_counts), 4)
        self.assertGreaterEqual(len(sizes), 8)
        self.assertEqual(scopes, {0, 1, 2})
        self.assertGreaterEqual(max(local_lengths), 6)
        self.assertTrue(has_temporal and has_audio)

    def test_natural_goal_paraphrase_does_not_change_answer_contract(self):
        for i in range(150):
            spec = spec_for(i)
            if spec["macro_plan"]["goal"]["answer_field"] == "match_result":
                q = reference_question(spec)
                alternate = q.replace("match result", "final score")
                self.assertEqual(guard_question(alternate, spec), [])
                bad = q.replace("match result", "match year")
                self.assertIn("macro_first_v2_goal_semantics_missing",
                              guard_question(bad, spec))
                break
        else:
            self.fail("no match-result test scene found")

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
        from environment.implement_subtask.realize import _operator_evidence
        from environment.generate_question.program_ir import Relation
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
