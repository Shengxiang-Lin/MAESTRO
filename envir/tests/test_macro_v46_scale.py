"""V4.6 scale, provenance and trace regression tests (no external corpora)."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from envir.macro.scene_catalog import parse_composition
from envir.macro.data_taskfirst import compose_seed_taskfirst
from envir.macro.v4_builder import build_one, build_batch
from envir.macro.v4_validation import validate_clean_scene
from envir.macro.source_contexts import load_supporting_contexts

KQA = [{
    "signature": "fixture-full-kopl-shape", "logical_depth": 5,
    "branch_nodes": 1, "merge_nodes": 1, "operator_count": 7,
    "program": {"nodes": [{"function": "Find", "dependencies": []}]},
}]


def _leaves(tree):
    if not tree.get("children"):
        return [tree["op"]]
    out = []
    for item in tree["children"]:
        out.extend(_leaves(item))
    return out


def fixture(index, kind):
    tree = parse_composition(kind)
    modalities = list(dict.fromkeys(
        next((m for m in ("table", "text", "image") if m in leaf.lower()), "text")
        for leaf in _leaves(tree)
    ))
    supporting = []
    for j, modality in enumerate(modalities):
        supporting.append({
            "modality": modality, "doc_id": f"{index}-{modality}-{j}",
            "provenance": "mmqa_supporting_context",
        })
    if len(supporting) < 2:
        supporting.append({
            "modality": supporting[0]["modality"], "doc_id": f"{index}-extra",
            "provenance": "mmqa_supporting_context",
        })
    if index % 3 == 0:
        supporting.append({
            "modality": supporting[-1]["modality"], "doc_id": f"{index}-extra2",
            "provenance": "mmqa_supporting_context",
        })
    family = ("comparison" if kind.startswith("Compare") else
              "intersection" if kind.startswith("Intersect") else
              "composition" if kind.startswith("Compose") else
              "lookup")
    return {
        "record_id": f"fixture-{index}",
        "question": f"Which candidate satisfies relation and attribute tests in case {index}?",
        "topic": f"candidate groups {index}",
        "entities": [f"Item {index}"],
        "domain_hint": ["sports", "culture", "science", "civic"][index % 4],
        "operation": family, "question_type": kind,
        "logical_pattern": family, "composition": tree,
        "source_modalities": modalities,
        "supporting_sources": supporting,
        "yago_matches": [],
    }


class V46ScaleTests(unittest.TestCase):
    def test_hundred_varied_compositions_have_real_structures_and_no_missing_sources(self):
        types = [
            "Compose(TableQ,ImageListQ)",
            "Compose(TextQ,TableQ)",
            "Intersect(ImageListQ,TextQ)",
            "Compare(Compose(TableQ,ImageQ),TableQ)",
            "Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ))",
            "TextQ", "ImageListQ",
        ]
        records = [fixture(i, types[i % len(types)]) for i in range(100)]
        scenes, stats = build_batch(
            kqa=KQA, scenario_seeds=records, target=100, seed=17)
        self.assertEqual(stats["generated"], 100)
        self.assertEqual(stats["missing_original_supporting_sources"], 0)
        self.assertGreaterEqual(stats["mean_macro_tasks_per_question"], 5.0)
        self.assertGreaterEqual(stats["mean_dependency_depth"], 3.0)
        self.assertGreaterEqual(stats["macro_graph_signatures"], 7)
        self.assertGreaterEqual(len(stats["macro_work_types"]), 8)
        self.assertTrue(all(
            scene["macro_graph"]["source_coverage"]["assigned"] >= 2
            for scene in scenes))
        self.assertTrue(all(not validate_clean_scene(scene)
                            for scene in scenes[:8]))

    def test_repeated_reference_leaf_is_shared_upstream_not_reinvestigated(self):
        record = fixture(7, "Compare(Compose(TableQ,ImageQ),TableQ)")
        record["supporting_sources"] = [
            {"modality": "table", "doc_id": "table1",
             "provenance": "mmqa_supporting_context"},
            {"modality": "image", "doc_id": "image1",
             "provenance": "mmqa_supporting_context"},
        ]
        scene = build_one(
            ordinal=7, seed=17, kqa=KQA, seed_record=record)
        roots = [t for t in scene["macro_graph"]["tasks"] if not t["depends_on"]]
        self.assertEqual(len(roots), 2)
        self.assertEqual(len(scene["world"]["sources"]), 2)
        self.assertGreaterEqual(scene["macro_graph"]["complexity"]["fanouts"], 1)
        self.assertEqual([], validate_clean_scene(scene))

    def test_original_mmqa_text_bank_content_is_preserved_with_exact_doc_id(self):
        record = fixture(2, "Compose(TableQ,TextQ)")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "MMQA_texts.jsonl").write_text(json.dumps({
                "id": "2-text-1",
                "text": "The museum exhibition opened on 6 March 2018. "
                        "The program names Ada as curator.",
            }) + "\n", encoding="utf-8")
            (root / "MMQA_tables.jsonl").write_text(json.dumps({
                "id": "2-table-0",
                "table": {
                    "table_name": "Museum program",
                    "header": [{"column_name": "Curator"}],
                    "table_rows": [[{"text": "Ada"}]],
                },
            }) + "\n", encoding="utf-8")
            contexts, stats = load_supporting_contexts([record], root)
            self.assertEqual(stats["available_context_excerpts"], 2)
            self.assertIn(("text", "2-text-1"), contexts)
            scene = build_one(
                ordinal=2, seed=17, kqa=KQA,
                seed_record=record, contexts=contexts)
        actual = {
            (s["modality_hint"], s["original_doc_id"]): s
            for s in scene["world"]["sources"].values()
        }
        text_facts = {f["field"]: f for f in actual[("text", "2-text-1")]["facts"]}
        self.assertIn("museum exhibition opened", text_facts["original_context_excerpt"]["value"])
        self.assertEqual(text_facts["original_context_excerpt"]["verification"],
                         "original_mmqa_text_excerpt_not_pixels")
        self.assertEqual(actual[("table", "2-table-0")]["context_provenance"],
                         "original_mmqa_table_text")
        self.assertEqual([], validate_clean_scene(scene))

    def test_distinct_observations_and_internal_operation_results_are_replayed(self):
        scene = build_one(ordinal=13, seed=17, kqa=KQA,
                          seed_record=fixture(13, "ImageListQ"))
        self.assertEqual([], validate_clean_scene(scene))
        flags = []
        operator_types = set()
        for task_id, feedback in scene["feedback_bank"].items():
            trace = [row for row in feedback["summary_trace"]
                     if row["action"] == "ExecuteTypedEvidenceOperation"]
            self.assertEqual(len(trace), 1)
            self.assertEqual(trace[0]["result"]["simulation_only"], True)
            self.assertEqual(trace[0]["result"],
                             next(iter(feedback["answer"].values()))["operation_result"])
            operator_types.add(trace[0]["operation"])
            flags.extend(r["attribute_present"] for r in
                         next(iter(feedback["answer"].values()))["candidate_observations"])
        self.assertGreaterEqual(len(operator_types), 3)
        self.assertTrue(all(isinstance(x, bool) for x in flags))


if __name__ == "__main__":
    unittest.main()
