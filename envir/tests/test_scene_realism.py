"""Guard one-output questions, frozen source titles and variable evidence."""
from __future__ import annotations
import copy
import json
import unittest
from envir.generate_question.spec import make_spec, spec_issues
from envir.generate_question.question import reference_question, guard_question
from envir.generate_question.quality import compile_and_validate, validate_replay
from envir.generate_question.scenario import enrich_one
from envir.generate_question.source_title_contract import (
    SCENE_REALISM_VERSION, expected_source_titles,
)
from envir.generate_question.information_necessity import (
    VARIABLE_POLICY, audit_candidate_information,
)
from envir.generate_question.pipeline import quality_report


def sample(i, difficulty="hard", topic="Premier League", domain="sports"):
    return make_spec(
        seed_record={"record_id": f"v3-realism-{i}", "topic": topic,
                     "domain_hint": domain},
        ordinal=i, run_seed=73, difficulty=difficulty,
        macro_first_v2_share=100,
    )


class SchemaAuthor:
    def complete_json(self, prompt, request):
        frozen = json.loads(request)["frozen_contract"]
        return {
            "scenario": {
                "setting": f"An information review of {frozen['topic']} materials",
                "user_role": "investigator",
                "purpose": "confirm the requested outcome",
                "motivation": "compare independent working evidence",
                "entity_kind": "events and records",
            },
            "source_titles": [
                {"source_id": "visual", "title": "Audio transcript"},
                {"source_id": "archive", "title": "Video report"},
            ],
        }


class GroundedCritic:
    def complete_json(self, prompt, request):
        self.candidate = json.loads(request)["candidate"]
        return {"verdict": "pass", "topic_coherent": True,
                "same_relation_semantics": True, "source_faithful": True,
                "no_unlicensed_facts": True, "reason": "Grounded."}


class SceneRealismTests(unittest.TestCase):
    def test_80_seeded_scenes_replay_with_schema_titles_and_one_goal(self):
        scenes = []
        for i in range(80):
            spec = sample(i, "hard" if i % 2 else "medium")
            self.assertEqual(spec["scene_realism_version"],
                             SCENE_REALISM_VERSION)
            self.assertEqual(spec_issues(spec), [])
            self.assertEqual(
                {s["source_id"]: s["title"] for s in spec["sources"]},
                expected_source_titles(spec),
            )
            q = reference_question(spec)
            self.assertEqual(guard_question(q, spec), [])
            self.assertNotIn("fictional", q.lower())
            scene = compile_and_validate(spec, q)
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertEqual(len(scene["gold_answer"]), 1)
            self.assertIn("structural_difficulty_profile",
                          scene["difficulty_audit"]["macro_first_audit"])
            if i < 16:
                scenes.append(scene)
        metrics = quality_report(scenes, no_progress=True)["macro_first_v2"]
        self.assertEqual(metrics["schema_locked_scene_count"], 16)
        self.assertEqual(sum(v["scenes"] for v in
                            metrics["structural_difficulty_by_target"].values()), 16)

    def test_author_cannot_publish_invented_media_titles(self):
        spec = sample(0)
        critic = GroundedCritic()
        reviewed, decision = enrich_one(
            spec, author=SchemaAuthor(), critic=critic, max_revisions=0,
        )
        self.assertEqual(decision["status"], "accepted")
        self.assertEqual(
            {s["source_id"]: s["title"] for s in reviewed["sources"]},
            expected_source_titles(spec),
        )
        self.assertEqual(
            {row["source_id"]: row["title"]
             for row in critic.candidate["source_titles"]},
            expected_source_titles(spec),
        )
        tampered = copy.deepcopy(reviewed)
        tampered["sources"][0]["title"] = "Invented audio recording"
        self.assertIn("scene_source_title_not_schema_frozen",
                      spec_issues(tampered))

    def test_explicit_second_answer_is_rejected(self):
        for i in range(15):
            spec = sample(i, topic="The Office", domain="film")
            q = reference_question(spec)
            bad = q.replace(
                "what is the confirmed",
                "which case is this, and what is the confirmed",
            )
            self.assertNotEqual(q, bad)
            self.assertIn("semantic_additional_output_requested",
                          guard_question(bad, spec))
            self.assertEqual(guard_question(q, spec), [])

    def test_triangle_and_triangular_are_equivalent(self):
        for i in range(450):
            spec = sample(i)
            plan = spec["macro_plan"]
            if (plan["pattern"] == "serial_identity"
                    and plan["clues"]["visual_modality"] == "image"
                    and plan["clues"]["emblem_shape"] == "triangle"):
                q = reference_question(spec)
                alt = q.replace("triangle motif", "triangular motif")
                self.assertNotEqual(q, alt)
                self.assertNotIn("macro_first_v2_clue_missing",
                                 guard_question(alt, spec))
                return
        self.fail("No triangle image case")

    def test_medium_hard_have_variable_indispensable_branches(self):
        by_difficulty = {"medium": [], "hard": []}
        seen = set()
        for i in range(200):
            for difficulty in by_difficulty:
                spec = sample(i, difficulty)
                plan = spec["macro_plan"]
                if plan["pattern"] != "multimodal_event":
                    continue
                self.assertEqual(plan["information_necessity_policy"],
                                 VARIABLE_POLICY)
                scene = compile_and_validate(spec, reference_question(spec))
                report = audit_candidate_information(scene)
                self.assertTrue(report["passed"])
                profile = plan["candidate_profile"]
                self.assertEqual(report["left_candidate_count"],
                                 profile["left_candidate_count"])
                self.assertEqual(report["right_candidate_count"],
                                 profile["right_candidate_count"])
                sizes = (report["left_candidate_count"],
                         report["right_candidate_count"])
                seen.add((difficulty, sizes))
                by_difficulty[difficulty].append(sum(sizes))
        self.assertGreaterEqual(len(by_difficulty["medium"]), 25)
        self.assertGreaterEqual(len(by_difficulty["hard"]), 25)
        self.assertGreaterEqual(len(seen), 5)
        self.assertGreater(
            sum(by_difficulty["hard"])/len(by_difficulty["hard"]),
            sum(by_difficulty["medium"])/len(by_difficulty["medium"]),
        )


if __name__ == "__main__":
    unittest.main()
