"""Stage-1 V1.3 deterministic evidence relations and replay regressions."""
from __future__ import annotations

import copy
import unittest

from environment.generate_question.evidence_operators import (
    extract_claim, align_claims, relate_claims, resolve_claim,
    check_warrant, evidence_states,
)
from environment.generate_question.spec import make_spec
from environment.generate_question.question import reference_question
from environment.generate_question.quality import (
    compile_and_validate, validate_replay,
)
from environment.generate_question.pipeline import quality_report


def _claim(source_id: str, origin: str, value: int, *,
           subject: str = "Case-001", seq: int = 1,
           revision_of=None, authoritative=False):
    source = {"source_id": source_id, "evidence_origin_id": origin}
    obs = {
        "observation_id": source_id + ".OBS001",
        "fact_refs": [source_id + ":F001"],
        "values": {"case_id": subject, "reported_year": value},
    }
    return extract_claim(
        source, obs, field="reported_year",
        attribute="reported_event_year", valid_scope="same_reported_event",
        revision_seq=seq, revision_of=revision_of,
        authoritative=authoritative,
    )


def _spec_of(pattern: str):
    for index in range(100):
        spec = make_spec(
            seed_record={
                "record_id": "v13-operator-%03d" % index,
                "topic": "Regional event records",
                "domain_hint": "sports",
                "source_modalities": ["text", "table", "image"],
            },
            ordinal=index, run_seed=17, difficulty="medium",
        )
        if spec["macro_plan"]["pattern"] == pattern:
            return spec
    raise AssertionError("pattern_not_sampled:" + pattern)


class EvidenceOperatorTests(unittest.TestCase):
    def test_independent_claims_corrobate_and_are_warranted(self):
        a = _claim("report_a", "independent_a", 2018)
        b = _claim("report_b", "independent_b", 2018)
        self.assertTrue(align_claims(a, b)["comparable"])
        self.assertEqual(relate_claims(a, b)["relation"], "corroborates")
        self.assertEqual(check_warrant([a])["status"], "unverified")
        verified = check_warrant([a, b])
        self.assertEqual(verified["status"], "corroborated")
        self.assertTrue(verified["sufficient_to_accept"])
        self.assertEqual(resolve_claim([a, b])["value"], 2018)
        self.assertEqual(
            evidence_states([a, b])["Case-001|reported_event_year|same_reported_event"]["status"],
            "corroborated",
        )

    def test_conflicting_claims_require_resolution(self):
        a = _claim("report_a", "independent_a", 2018)
        b = _claim("report_b", "independent_b", 2019)
        self.assertEqual(relate_claims(a, b)["relation"], "contradicts")
        self.assertEqual(check_warrant([a, b])["status"], "disputed")
        self.assertFalse(resolve_claim([a, b])["sufficient_to_accept"])
        self.assertIsNone(resolve_claim([a, b])["value"])
        official = _claim(
            "official", "official_register", 2019, authoritative=True,
        )
        resolution = resolve_claim(
            [a, b, official], policy="authoritative_record",
        )
        self.assertEqual(resolution["status"], "resolved")
        self.assertEqual(resolution["value"], 2019)

    def test_lineage_duplicates_are_not_independent(self):
        a = _claim("report_a", "copied_root", 2018)
        b = _claim("report_b", "copied_root", 2018)
        self.assertEqual(relate_claims(a, b)["relation"], "dependent_duplicate")
        self.assertEqual(check_warrant([a, b])["status"], "unverified")
        self.assertFalse(check_warrant([a, b])["sufficient_to_accept"])

    def test_supersession_requires_explicit_successor_with_same_lineage(self):
        a = _claim("old", "single_agency", 2018)
        b = _claim("new", "single_agency", 2019, seq=2,
                   revision_of=a["claim_id"], authoritative=True)
        relation = relate_claims(a, b)
        self.assertEqual(relation["relation"], "supersedes")
        self.assertEqual(relation["superseding_claim_id"], b["claim_id"])
        result = resolve_claim([a, b], policy="authoritative_record")
        self.assertEqual(result["value"], 2019)
        self.assertEqual(result["status"], "resolved")
        unlicensed = _claim("new", "different_agency", 2019, seq=2,
                            revision_of=a["claim_id"])
        self.assertEqual(relate_claims(a, unlicensed)["relation"], "contradicts")

    def test_same_field_different_entity_or_time_not_a_conflict(self):
        a = _claim("report_a", "root_a", 2018)
        b = _claim("report_b", "root_b", 2019, subject="Case-002")
        self.assertEqual(relate_claims(a, b)["relation"], "incomparable")
        c = dict(b, subject="Case-001", valid_scope="next_year")
        self.assertEqual(relate_claims(a, c)["relation"], "incomparable")

    def test_provenance_and_mixed_warrant_fail_closed(self):
        a = _claim("report_a", "root_a", 2018)
        with self.assertRaisesRegex(ValueError, "mixed_claim_keys"):
            check_warrant([a, _claim("report_b", "root_b", 2019,
                                     subject="Case-002")])
        with self.assertRaisesRegex(ValueError, "claim_source_origin_required"):
            extract_claim({"source_id": "unknown"}, {
                "observation_id": "unknown.OBS001", "fact_refs": ["unknown:F001"],
                "values": {"case_id": "Case-001", "year": 2020},
            }, field="year", attribute="event_year")


class SceneEvidenceLedgerTests(unittest.TestCase):
    def test_real_world_claim_pairs_and_source_ablation(self):
        spec = _spec_of("disputed_reports")
        scene = compile_and_validate(spec, reference_question(spec))
        ledger = scene["evidence_ledger"]
        self.assertEqual(validate_replay(scene)["status"], "valid")
        self.assertTrue(ledger["independent_comparable_pairs"] > 0)
        self.assertEqual(
            len(ledger["relations"]), ledger["independent_comparable_pairs"],
        )
        self.assertEqual(
            ledger["verification_sensitive_pairs"],
            ledger["independent_comparable_pairs"],
        )
        self.assertGreater(ledger["relation_counts"].get("contradicts", 0), 0)
        self.assertGreater(ledger["relation_counts"].get("corroborates", 0), 0)
        self.assertTrue(all(
            rel["source_fact_refs"] and len(rel["source_fact_refs"]) == 2
            for rel in ledger["relations"]
        ))
        self.assertNotIn("evidence_ledger", scene["public"])
        self.assertEqual(
            scene["difficulty_audit"]["evidence_relation_audit"]
            ["verification_sensitive_pairs"],
            ledger["verification_sensitive_pairs"],
        )
        self.assertTrue(all(
            item["comparison_scope"] ==
            "source_ablation_not_real_arrival_order"
            for item in ledger["warrant_transitions"]
        ))

    def test_late_context_pattern_uses_same_source_grounded_claims(self):
        spec = _spec_of("corroborated_dispute")
        scene = compile_and_validate(spec, reference_question(spec))
        ledger = scene["evidence_ledger"]
        self.assertEqual(validate_replay(scene)["status"], "valid")
        self.assertGreater(ledger["independent_comparable_pairs"], 0)
        self.assertGreater(ledger["relation_counts"].get("contradicts", 0), 0)
        self.assertTrue(set(ledger["source_first_wave"]) ==
                        {"report_a", "report_b"})

    def test_versioned_nondispute_has_empty_not_fabricated_relations(self):
        spec = _spec_of("parallel_cues")
        scene = compile_and_validate(spec, reference_question(spec))
        ledger = scene["evidence_ledger"]
        self.assertEqual(ledger["claims"], [])
        self.assertEqual(ledger["relation_counts"], {})
        self.assertEqual(ledger["independent_comparable_pairs"], 0)
        self.assertEqual(validate_replay(scene)["status"], "valid")

    def test_tampering_private_ledger_invalidates_strict_replay(self):
        spec = _spec_of("disputed_reports")
        scene = compile_and_validate(spec, reference_question(spec))
        bad = copy.deepcopy(scene)
        bad["evidence_ledger"]["relations"][0]["relation"] = "supersedes"
        audit = validate_replay(bad)
        self.assertEqual(audit["status"], "invalid")
        self.assertIn("evidence_ledger_not_reproducible", audit["issues"])

    def test_legacy_scene_without_marker_does_not_get_a_new_ledger(self):
        spec = _spec_of("disputed_reports")
        spec.pop("evidence_relation_version")
        scene = compile_and_validate(spec, reference_question(spec))
        self.assertNotIn("evidence_ledger", scene)
        self.assertNotIn("evidence_origin_id",
                         scene["world"]["sources"]["report_a"])
        self.assertEqual(validate_replay(scene)["status"], "valid")

    def test_quality_report_exposes_relations_not_gold_or_claims(self):
        spec = _spec_of("disputed_reports")
        scene = compile_and_validate(spec, reference_question(spec))
        report = quality_report([scene], no_progress=True)
        aggregate = report["evidence_relations"]
        self.assertEqual(aggregate["versioned_scenes"], 1)
        self.assertEqual(
            aggregate["independent_comparable_pairs"],
            scene["evidence_ledger"]["independent_comparable_pairs"],
        )
        self.assertNotIn("claims", aggregate)


if __name__ == "__main__":
    unittest.main()
