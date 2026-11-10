"""Opt-in grammar: sampled topology, grounded Worlds, strict replay and Stage 2."""
from __future__ import annotations

import copy
import unittest

from envir.generate_question.spec import make_spec, spec_issues
from envir.generate_question.question import reference_question, guard_question
from envir.generate_question.quality import compile_and_validate, validate_replay
from envir.generate_question.compiler import QuestionSession, oracle_schedule
from envir.generate_question.macro_contract import macro_graph_signature
from envir.generate_question.graph_grammar import ARCHETYPES
from envir.generate_question.world_synthesizer import mutate_evidence_witness
from envir.generate_question.program_ir import evaluate_program, answer_values
from envir.implement_subtask.realize import realize_scene


def graph_spec(i, difficulty):
    return make_spec(
        seed_record={"record_id": f"graph-{i}",
                     "topic": f"Fictional Collection {i}",
                     "domain_hint": "history",
                     "source_modalities": ["image", "text", "table"]},
        ordinal=i, run_seed=17, difficulty=difficulty,
        evidence_graph_share=100,
    )


class EvidenceGraphGrammarTests(unittest.TestCase):
    def test_legacy_is_exactly_unchanged_with_opt_in_disabled(self):
        seed = {"record_id": "old", "topic": "Legacy Collection"}
        old = make_spec(seed_record=seed, ordinal=9,
                        run_seed=17, difficulty="medium")
        explicit_off = make_spec(seed_record=seed, ordinal=9,
                                 run_seed=17, difficulty="medium",
                                 evidence_graph_share=0)
        self.assertEqual(old, explicit_off)
        self.assertNotIn("evidence_graph", old)
        with self.assertRaisesRegex(ValueError, "evidence_graph_share"):
            make_spec(seed_record=seed, ordinal=9, run_seed=17,
                      difficulty="medium", evidence_graph_share=101)
        easy = make_spec(seed_record=seed, ordinal=9, run_seed=17,
                         difficulty="easy", evidence_graph_share=100)
        self.assertNotEqual(easy["reasoning_pattern"], "evidence_graph")

    def test_64_seeded_graphs_are_executable_diverse_and_reproducible(self):
        archetypes, signatures = set(), set()
        target_labels = set()
        for i in range(64):
            difficulty = "hard" if i % 2 else "medium"
            spec = graph_spec(i, difficulty)
            self.assertEqual(spec_issues(spec), [])
            self.assertEqual(spec, graph_spec(i, difficulty))
            graph = spec["evidence_graph"]
            archetypes.add(graph["archetype"])
            question = reference_question(spec)
            self.assertEqual(guard_question(question, spec), [])
            scene = compile_and_validate(spec, question)
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertEqual(len(scene["gold_answer"]), 1)
            self.assertEqual(set(scene["gold_answer"][0]), {"accession_year"})
            self.assertLessEqual(scene["macro_graph"]["macro_task_count"], 9)
            self.assertTrue(scene["difficulty_audit"][
                "evidence_graph_audit"]["source_counterfactuals_pass"])
            refs = {ref.split(":")[0]
                    for row in scene["difficulty_audit"]["answer_provenance"]
                    for ref in row["source_fact_refs"]}
            self.assertEqual(refs, set(scene["world"]["sources"]))
            self.assertEqual(
                oracle_schedule(scene)["answer"], scene["gold_answer"]
            )
            signatures.add(macro_graph_signature(scene["macro_graph"]))
            if graph["archetype"] == "chain":
                feature = graph["visual_features"]
                rows = scene["world"]["sources"]["illustration"]["observations"]
                matches = [r for r in rows if all(
                    r["values"][k] == v for k, v in feature.items()
                )]
                self.assertEqual(len(matches), 1)
                target_labels.add(matches[0]["values"]["candidate_id"])
        self.assertEqual(archetypes, set(ARCHETYPES))
        self.assertGreaterEqual(len(signatures), 6)
        self.assertGreater(len(target_labels), 1)

    def test_bad_key_type_and_source_permissions_fail_closed(self):
        spec = graph_spec(7, "hard")
        modified = copy.deepcopy(spec)
        key_role = next(role for role in modified["sources"]
                        if role["access_mode"] == "keyed_lookup")
        key_role["key_field"] = "nonexistent"
        self.assertTrue(any("lookup" in x or "key" in x
                            for x in spec_issues(modified)))
        scene = compile_and_validate(spec, reference_question(spec))
        session = QuestionSession(scene)
        self.assertNotIn("world", session.public_view)
        for role in spec["sources"]:
            if role["access_mode"] == "keyed_lookup":
                with self.assertRaisesRegex(ValueError,
                                            "full_source_read_forbidden"):
                    session.read_source(role["source_id"])

    def test_all_sources_affect_answer(self):
        for i in range(16):
            spec = graph_spec(i, "hard")
            scene = compile_and_validate(spec, reference_question(spec))
            for sid in scene["world"]["sources"]:
                altered = mutate_evidence_witness(
                    scene["world"]["sources"], sid,
                    scene["difficulty_audit"]["answer_provenance"], spec
                )
                self.assertNotEqual(
                    answer_values(evaluate_program(
                        scene["gold_program"], sources=altered
                    )),
                    scene["gold_answer"],
                    f"Source not needed: {spec['evidence_graph']['archetype']} {sid}"
                )
        self.assertTrue(True)

    def test_program_mutation_is_rejected_even_if_types_still_match(self):
        spec = graph_spec(4, "medium")
        modified = copy.deepcopy(spec)
        # This is a valid typed filter operation, but changes the frozen
        # semantic program without going through the grammar sampler.
        modified["evidence_graph"]["nodes"][-2]["fields"] = ["accession_year"]
        modified["evidence_graph"]["nodes"].append({
            "id": "extra_no_op", "op": "unique",
            "inputs": [modified["evidence_graph"]["output"]],
        })
        self.assertIn("evidence_graph_program_not_from_grammar",
                      spec_issues(modified))

    def test_archive_lookup_key_claim_is_checked_against_executed_field(self):
        from envir.implement_subtask.reporting import _teacher_contract, _guard
        clue = make_spec(
            seed_record={"record_id": "key-check", "topic": "Fictional Museum"},
            ordinal=2, run_seed=17, difficulty="medium",
            clue_chain_share=100,
        )
        scene = compile_and_validate(clue, reference_question(clue))
        records = realize_scene(scene)["macro_tasks"]
        archive = next(row for row in records if row["authorized_source_ids"]
                       == ["archive"])
        contract = _teacher_contract(
            archive, question=scene["question"], is_terminal=False
        )
        contract["all_evidence_refs"] = sorted({
            ref for step in archive["trace_summary"]["steps"]
            for ref in step["supporting_fact_refs"]
        })
        lookup = next(step["lookup_keys"] for step in
                      contract["observed_operator_steps"]
                      if step["lookup_keys"] is not None)
        self.assertEqual(lookup["source_id"], "archive")
        self.assertEqual(lookup["key_field"], "accession_id")
        candidate = {
            "status": "done",
            "result": archive["answer"],
            "summary": (
                "I executed a keyed lookup using the upstream marker code "
                "M-55 against the archive and returned the matching record."
            ),
        }
        self.assertIn("summary_wrong_archive_lookup_key",
                      _guard(candidate, contract))

    def test_one_oracle_stage2_scene_from_each_archetype(self):
        found = {}
        for i in range(64):
            spec = graph_spec(i, "medium")
            mode = spec["evidence_graph"]["archetype"]
            if mode not in found:
                scene = compile_and_validate(spec, reference_question(spec))
                scene["question_authoring"] = {"status": "accepted"}
                realized = realize_scene(scene)
                self.assertEqual(realized["macro_task_count"],
                                 scene["macro_graph"]["macro_task_count"])
                terminal = next(row for row in realized["macro_tasks"]
                                if row["macro_task_id"] ==
                                realized["terminal_macro_task_id"])
                self.assertEqual(terminal["answer"], scene["gold_answer"])
                found[mode] = True
            if len(found) == len(ARCHETYPES):
                break
        self.assertEqual(set(found), set(ARCHETYPES))


if __name__ == "__main__":
    unittest.main()
