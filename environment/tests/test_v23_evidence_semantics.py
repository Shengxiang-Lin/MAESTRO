"""v23 candidate-only evidence vs semantics regressions from real v22 audit.

No private World/Gold and no online model is used.
"""
from __future__ import annotations

import copy
import json
import unittest

from environment.question_generation.coverage_review import (
    conflicting_endpoint_evidence, coverage_manifest, validate_coverage_manifest,
)
from environment.question_generation.semantic_probes import probe_candidate_semantics
from environment.question_generation.realization import _realize_one
from environment.tests.test_stage2_surface_certification import (
    FakeAuthor, FakeCritic, Stage2SurfaceCertificationTests,
)

CANDIDATE_11 = (
    "How many distinct events are there in which the recorded 'winner' points "
    "to a person who has a birth date less than 1840, a mass less than "
    "94.9 kg, and a height of 1.7612 m in the relevant records?"
)
CANDIDATE_29 = (
    "Let the first group consist of organizations that are targets of the "
    "'published on' relation from events with an end date greater than 1912, "
    "have a logo of the crimson mountain emblem, whose recorded 'published "
    "on' relation points to Northstar Holdings, and have a number of employees "
    "different from 544. For this first group, is any recorded address equal "
    "to 238 Maple Road?"
)


def record(probe, *, evidence, answer=None, **extra):
    out = {
        "id": probe["id"], "kind": probe["kind"],
        "expected": probe["expected"],
        "answer": answer if answer is not None else probe["expected"],
        "evidence": evidence,
        "subject": "", "object": "",
        "subject_role": "", "object_role": "", "owner": "",
    }
    out.update(extra)
    return out


class EvidenceVsSemanticTests(unittest.TestCase):
    def test_case11_hallucinated_quote_is_evidence_failure_not_semantic_vote(self):
        probe = {
            "id": "P1", "kind": "relation_path", "expected": "B_TO_A",
            "question": "Does winner point B to A?", "evidence_required": True,
        }
        class HallucinatingReader:
            def complete_json(self, system, user, **kwargs):
                return {"answers": [{
                    "id": "P1", "answer": "B_TO_A",
                    "evidence": "The first group is the winner of the 1896 Summer Olympics.",
                    "subject": "The first group", "object": "the 1896 Summer Olympics",
                    "subject_role": "B", "object_role": "A",
                }]}
        result = probe_candidate_semantics(
            HallucinatingReader(), "PRIVATE PUBLIC REFERENCE", CANDIDATE_11, [probe],
        )[0]
        self.assertEqual(result["raw_answer"], "B_TO_A")
        self.assertEqual(result["answer"], "UNCLEAR")
        self.assertEqual(result["evidence_status"], "evidence_invalid")
        self.assertEqual(result["evidence_reason"], "evidence_not_verbatim")

    def test_evidence_repair_receives_own_previous_answer_never_gold(self):
        candidate = "Events whose recorded 'winner' points to a person."
        probe = {
            "id": "P1", "kind": "relation_path", "expected": "B_TO_A",
            "question": "Read the winner hop", "evidence_required": True,
        }
        class Reader:
            def complete_json(self, system, user, **kwargs):
                payload = json.loads(user)
                self.assert_payload(payload, system)
                return {"answers": [{
                    "id": "P1", "answer": "B_TO_A",
                    "evidence": candidate, "subject": "Events",
                    "object": "a person", "subject_role": "B", "object_role": "A",
                }]}
            def assert_payload(self, payload, system):
                assert "reference_question" not in payload
                assert "expected" not in json.dumps(payload)
                assert payload["probes"][0]["initial_model_answer"] == "B_TO_A"
                assert "EVIDENCE-ONLY REPAIR" in system
        observed = probe_candidate_semantics(
            Reader(), "DO NOT TRANSMIT REFERENCE", candidate, [probe],
            focused=True, claimed_answers={"P1": "B_TO_A"},
        )[0]
        self.assertEqual(observed["answer"], "B_TO_A")
        self.assertEqual(observed["evidence_status"], "verified")

    def test_case29_owner_quote_conflicts_with_same_edge_role(self):
        edge = "edge-1"
        path = {
            "id": "P1", "kind": "relation_path", "expected": "A_TO_B",
            "edge_id": edge, "evidence_required": True,
        }
        owner = {
            "id": "P2", "kind": "condition_owner", "expected": "A",
            "edge_id": edge, "value_type": "string",
            "canonical_value": "crimson mountain emblem",
            "evidence_required": True,
        }
        output = {
            "id": "P3", "kind": "answer_type", "expected": "VERIFY",
            "evidence_required": True,
        }
        path_record = record(
            path, evidence=("organizations that are targets of the 'published on' "
                            "relation from events with an end date greater than 1912"),
            subject="events with an end date greater than 1912",
            object="organizations", subject_role="A", object_role="B",
        )
        owner_record = record(
            owner, evidence=(
                "organizations that are targets of the 'published on' relation "
                "from events with an end date greater than 1912, have a logo "
                "of the crimson mountain emblem"
            ), owner="organizations",
        )
        output_record = record(
            output, evidence="For this first group, is any recorded address equal to 238 Maple Road?"
        )
        probes = [path, owner, output]
        records = [path_record, owner_record, output_record]
        self.assertEqual(records[1]["answer"], records[1]["expected"])
        conflicts = conflicting_endpoint_evidence(probes, records)
        self.assertEqual(
            conflicts, {"P2": "owner_quote_matches_opposite_relation_endpoint"},
        )
        with self.assertRaisesRegex(ValueError, "stage2_owner_relation_evidence_conflict"):
            validate_coverage_manifest(
                probes, records, coverage_manifest(probes, records),
                candidate=CANDIDATE_29,
            )
        correct_owner = copy.deepcopy(records)
        correct_owner[1]["owner"] = "events with an end date greater than 1912"
        correct_owner[1]["evidence"] = (
            "organizations that are targets of the 'published on' relation "
            "from events with an end date greater than 1912, have a logo "
            "of the crimson mountain emblem"
        )
        self.assertEqual(conflicting_endpoint_evidence(probes, correct_owner), {})

    def test_evidence_only_repair_does_not_rewrite_author_candidate(self):
        class OneFabricatedEvidence(FakeCritic):
            def complete_json(self, system, user, **kwargs):
                r = super().complete_json(system, user, **kwargs)
                if ("independent reader of candidate text only" in system
                        and "EVIDENCE-ONLY REPAIR" not in system):
                    for item in r["answers"]:
                        if item["answer"] == "A_TO_B":
                            item["evidence"] = "This sentence was fabricated by model."
                return r
        payload, world, capability, _ = Stage2SurfaceCertificationTests().data()
        author, reviewer = FakeAuthor(), OneFabricatedEvidence()
        ok, row, audit = _realize_one(
            payload, world, capability, author, reviewer, max_revisions=0,
        )
        self.assertTrue(ok, str((row, audit)))
        self.assertEqual(len(author.calls), 1)
        self.assertEqual(audit[-1]["candidate_probe_evidence_repair_calls"], 1)
        self.assertEqual(audit[-1]["candidate_probe_recheck_calls"], 0)
        self.assertTrue(all(r["evidence_status"] == "verified"
                            for r in audit[-1]["candidate_probe_answers"]))

    def test_unrepairable_quote_is_quarantined_not_sent_to_author(self):
        class AlwaysFabricates(FakeCritic):
            def complete_json(self, system, user, **kwargs):
                r = super().complete_json(system, user, **kwargs)
                if "independent reader of candidate text only" in system:
                    for item in r["answers"]:
                        if item["answer"] == "A_TO_B":
                            item["evidence"] = "This sentence was fabricated by model."
                return r
        payload, world, capability, _ = Stage2SurfaceCertificationTests().data()
        author = FakeAuthor()
        ok, row, audit = _realize_one(
            payload, world, capability, author, AlwaysFabricates(),
            max_revisions=2,
        )
        self.assertFalse(ok)
        self.assertEqual(row["reason"], "rejected_evidence_unverified")
        self.assertEqual(audit[-1]["failure"], "candidate_evidence_unverified")
        self.assertEqual(len(author.calls), 1)
        self.assertEqual(audit[-1]["candidate_probe_evidence_repair_calls"], 1)


if __name__ == "__main__":
    unittest.main()
