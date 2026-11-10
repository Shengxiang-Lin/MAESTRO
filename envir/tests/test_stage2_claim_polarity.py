"""Regression of stage-local claim polarity and real V2 lookup accounting."""
from __future__ import annotations
import unittest

from envir.implement_subtask.stage_local_claims import (
    classify_disagreement_claim, stage_local_claim_issues,
)
from envir.generate_question.spec import make_spec
from envir.generate_question.question import reference_question
from envir.generate_question.quality import compile_and_validate
from envir.generate_question.pipeline import quality_report


class ClaimPolarityTests(unittest.TestCase):
    def test_unknown_does_not_become_verified_fact(self):
        unknown = [
            "No independent report disagreement has been verified yet.",
            "The two report years have not yet been compared.",
            "This step has not confirmed a disagreement between reports.",
            "Report disagreements still need to be checked.",
            "The reported-year conflict is currently unknown.",
        ]
        for text in unknown:
            with self.subTest(text=text):
                self.assertEqual(stage_local_claim_issues(
                    text, {"reported_year_disagreement_established": False},
                ), [])
        assertive = [
            "The two reports disagree on the reported year.",
            "This stage identified five cases where independent reports disagree.",
            "The reports were confirmed to have conflicting years.",
        ]
        for text in assertive:
            with self.subTest(text=text):
                self.assertEqual(
                    stage_local_claim_issues(
                        text, {"reported_year_disagreement_established": False}
                    ), ["claims_unverified_independent_report_disagreement"],
                )

    def test_mixed_unknown_and_assertion_still_rejected(self):
        sentence = (
            "No independent report disagreement was verified; "
            "however the two reports disagree on the year."
        )
        self.assertEqual(classify_disagreement_claim(sentence), "established")
        self.assertEqual(
            stage_local_claim_issues(
                sentence, {"reported_year_disagreement_established": False}
            ), ["claims_unverified_independent_report_disagreement"],
        )

    def test_actual_v2_lookup_nodes_are_counted(self):
        scenes = []
        for i in range(12):
            spec = make_spec(
                seed_record={
                    "record_id": "key-count-" + str(i),
                    "topic": "Premier League", "domain_hint": "sports",
                },
                ordinal=i, run_seed=4,
                difficulty="hard", macro_first_v2_share=100,
            )
            scenes.append(compile_and_validate(spec, reference_question(spec)))
        count = sum(
            node["op"] == "lookup_by_key"
            for scene in scenes for node in scene["gold_program"]["nodes"]
        )
        report = quality_report(scenes, no_progress=True)
        self.assertGreater(count, 0)
        self.assertEqual(report["macro_first_v2"]["keyed_lookup_count"], count)
        self.assertEqual(report["evidence_acquisition"]["keyed_lookup_count"], count)
        self.assertEqual(report["evidence_acquisition"]["v2_keyed_lookup_count"], count)


if __name__ == "__main__":
    unittest.main()
