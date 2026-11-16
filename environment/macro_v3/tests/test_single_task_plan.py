"""Single Task Plan: direct witnessed YAGO grounding, no old-spec nesting."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from environment.generate_question.task_plan import (
    make_spec, validate_spec, CALIBRATED, BLIND_EVENTS, GOAL_QUESTIONS,
)
from environment.generate_question.task_plan_runtime import audit_information
from environment.generate_question.task_plan import (
    SEMANTIC_GOALS, _loaded_yago_unique, _semantic_recipe,
)
from environment.generate_question.question import guard_question
from environment.generate_question.knowledge_macro import YagoKnowledgeIndex
from environment.generate_question.world import compile_program, generate_world
from environment.generate_question.question import reference_question
from environment.generate_question.quality import compile_and_validate, validate_replay
from environment.implement_subtask.realize import realize_scene
from environment.macro_v3.yago import Edge, KnowledgeGraph

try:
    import pyoxigraph  # noqa: F401
    RDF_READY = True
except ImportError:
    RDF_READY = False


def graph_fixture(n=54):
    d, p = "https://example.org/", "http://schema.org/"
    edges = []
    for i in range(n):
        person, town, region, country, institute = (
            d+f"Person{i}", d+f"Town{i}", d+f"Region{i}",
            d+f"Country{i}", d+f"Institution{i}",
        )
        edges += [
            Edge(person, p+"birthPlace", town),
            Edge(town, p+"containedInPlace", region),
            Edge(region, p+"containedInPlace", country),
            Edge(person, p+"memberOf", institute),
        ]
    return KnowledgeGraph(
        edges=edges, input_file="fixture-yago-facts.ttl",
        scanned=len(edges),
    )


class DirectTaskPlanTests(unittest.TestCase):
    def test_one_plan_without_any_legacy_backbone(self):
        graph = graph_fixture()
        indexed = set(graph.edges)
        seen_modes, sizes = set(), set()
        from environment.generate_question.knowledge_macro import YagoKnowledgeIndex
        index = YagoKnowledgeIndex(graph)
        for ordinal in range(9):
            spec = make_spec(graph, seed=17, ordinal=ordinal,
                             difficulty="medium" if ordinal % 2 == 0 else "hard",
                             index=index)
            self.assertFalse(validate_spec(spec))
            self.assertEqual(spec["reasoning_pattern"], "task_plan")
            self.assertEqual(spec["family"], "task_plan")
            self.assertEqual(spec["macro_plan"]["difficulty_sampling_policy"],
                             CALIBRATED)
            self.assertEqual(spec["macro_plan"]["evidence_sampling_policy"],
                             BLIND_EVENTS)
            self.assertEqual(spec["macro_plan"]["question_sampling_policy"],
                             GOAL_QUESTIONS)
            self.assertTrue(not any(
                "backbone" in field for field in spec
            ))
            self.assertFalse(any(k in spec for k in (
                "task_first_v1_backbone", "hybrid_backbone_v3",
                "task_first_relation_sketch", "task_first_grounding",
            )))
            plan = spec["macro_plan"]
            seen_modes.add(plan["mode"])
            sizes.add(len(plan["macro_tasks"]))
            self.assertLessEqual(len(plan["macro_tasks"]), 9)
            self.assertGreaterEqual(len(plan["macro_tasks"]), 4)
            for facts in plan["world_binding"]["candidate_paths"].values():
                for fact in facts:
                    self.assertIn(Edge(
                        fact["subject"], fact["predicate"], fact["object"]
                    ), indexed)
            if plan["mode"] == "verify":
                fork = plan["world_binding"]["real_fork"]
                for fact in fork["candidate_forks"].values():
                    self.assertIn(Edge(
                        fact["subject"], fact["predicate"], fact["object"]
                    ), indexed)
            question = reference_question(spec)
            scene = compile_and_validate(spec, question)
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertEqual(compile_program(spec), scene["gold_program"])
            self.assertEqual(generate_world(spec), scene["world"])
            realized = realize_scene(scene, strict=True)
            self.assertEqual(len(realized["macro_tasks"]),
                             scene["macro_graph"]["macro_task_count"])
            self.assertEqual(
                scene["gold_answer"], [{
                    "answer": plan["world_binding"]["candidate_paths"][
                        plan["world_binding"]["target_case"]][-1][
                            "object_label"]
                }]
            )
            info = scene["difficulty_audit"]["macro_first_audit"][
                "candidate_information_necessity"
            ]
            self.assertTrue(info["passed"], info)
            self.assertTrue(info["non_target_field_substitution"]["passed"])
            self.assertNotIn("followed by", question.casefold())
            self.assertTrue(all(info["clue_necessary"].values()))
            self.assertTrue(all(info["discovery_necessary"].values()))
            if plan["condition"]:
                self.assertTrue(info["event_field_necessary"])
            if plan["mode"] == "verify":
                self.assertTrue(info["real_fork_necessary"])
                self.assertTrue(all(info["simulated_sources_necessary"].values()))
            else:
                self.assertIsNone(info["real_fork_necessary"])
        self.assertEqual(seen_modes, {"trace", "event", "verify"})
        self.assertGreaterEqual(len(sizes), 3)

    def test_cross_field_leakage_negative_control(self):
        # Replay the old naming shortcut in the NEW materialized World.
        # The original condition-removal test would still pass, but
        # replacing the condition with event_name must now catch it.
        kg = graph_fixture()
        for ordinal, difficulty in (
            (0, "medium"), (0, "hard"),
            (1, "hard"), (3, "hard"),
        ):
            spec = make_spec(
                kg, seed=17, ordinal=ordinal, difficulty=difficulty,
            )
            self.assertIsNotNone(spec["macro_plan"]["condition"])
            scene = compile_and_validate(
                spec, reference_question(spec)
            )
            audit = scene["difficulty_audit"]["macro_first_audit"][
                "candidate_information_necessity"
            ]
            self.assertTrue(audit["non_target_field_substitution"]["passed"])
            self.assertTrue(audit["event_field_necessary"])
            case = spec["macro_plan"]["world_binding"]["target_case"]
            from environment.generate_question.task_plan_runtime import _project
            pid = _project(spec["seed"], spec["ordinal"], case)
            target_records = [
                r["values"] for r in scene["world"]["sources"][
                    "event_ledger"]["observations"]
                if r["values"]["project_id"] == pid
            ]
            self.assertEqual(
                len({r["event_name"] for r in target_records}), 1
            )
            # No alternative descriptive field can replace the selected
            # event_status or event_year predicate for this project.
            attacked = deepcopy(scene)
            condition = spec["macro_plan"]["condition"]
            for obs in attacked["world"]["sources"][
                "event_ledger"]["observations"
            ]:
                r = obs["values"]
                eligible = (r["event_status"] == "completed"
                            if condition["field"] == "event_status"
                            else r["event_year"] >= 2025)
                r["event_name"] = (
                    "Qualified project session" if eligible
                    else "Unqualified project session"
                )
            bypass = audit_information(attacked)
            self.assertFalse(bypass["passed"])
            self.assertTrue(bypass["event_field_necessary"])
            self.assertFalse(bypass[
                "non_target_field_substitution"]["tested_fields"][
                    "event_name"]["passed"]
            )
            self.assertTrue(any(
                row["unique_shortcut"]
                for row in bypass["non_target_field_substitution"][
                    "tested_fields"]["event_name"]["attempts"].values()
            ))

    def test_structural_difficulty_stratification(self):
        kg = graph_fixture()
        plans = {"medium": [], "hard": []}
        modes = {"medium": set(), "hard": set()}
        for ordinal in range(12):
            for level in ("medium", "hard"):
                spec = make_spec(
                    kg, seed=17, ordinal=ordinal,
                    difficulty=level,
                )
                plan = spec["macro_plan"]
                plans[level].append(len(plan["macro_tasks"]))
                modes[level].add(plan["mode"])
        self.assertGreater(
            min(plans["hard"]), max(plans["medium"])
        )
        self.assertEqual(modes["medium"], {"trace", "event"})
        self.assertEqual(modes["hard"], {"event", "verify"})

    def test_semantic_goal_questions_are_witnessed_and_scoped(self):
        kg = graph_fixture()
        index = YagoKnowledgeIndex(kg)
        goals = set()
        for ordinal in range(8):
            level = "medium" if ordinal % 2 == 0 else "hard"
            spec = make_spec(
                kg, seed=17, ordinal=ordinal,
                difficulty=level, index=index,
            )
            plan = spec["macro_plan"]
            self.assertEqual(plan["semantic_grounding_policy"], SEMANTIC_GOALS)
            sem = plan["goal"]["semantic"]
            paths = plan["world_binding"]["candidate_paths"]
            target_path = paths[plan["world_binding"]["target_case"]]
            self.assertIsNotNone(_semantic_recipe(target_path))
            self.assertEqual(sem["relation_sequence"], [
                edge["predicate"] for edge in target_path
            ])
            self.assertEqual(sem["type_evidence"],
                             "predicate_signature_heuristic_not_rdf_type")
            self.assertTrue(plan["world_binding"]["semantic_audit"][
                "loaded_yago_singleton_witness"])
            self.assertFalse(plan["world_binding"]["semantic_audit"][
                "rdf_type_verification"])
            for candidate in paths.values():
                self.assertTrue(_loaded_yago_unique(
                    index, [Edge(
                        e["subject"], e["predicate"], e["object"]
                    ) for e in candidate]
                ))
            question = reference_question(spec)
            self.assertIn(sem["question"], question)
            self.assertNotIn("relation target", question)
            self.assertNotIn("followed by", question)
            self.assertFalse(guard_question(question, spec))
            changed = question.replace(
                sem["question"],
                "Which unrelated location is mentioned in these records?"
            )
            self.assertIn(
                "task_plan_semantic_goal_question_missing",
                guard_question(changed, spec),
            )
            self.assertFalse(validate_spec(spec))
            goals.add(sem["profile"])
        self.assertTrue(goals)

    def test_reject_uninterpretable_and_multivalued_yago(self):
        p = "http://schema.org/"
        graph = KnowledgeGraph(
            edges=[
                Edge("https://test.org/a", p+"location", "https://test.org/b"),
                Edge("https://test.org/b", p+"follows", "https://test.org/c"),
            ],
            input_file="bad-meaningful-route.ttl",
            scanned=2,
        )
        # An unsupported real path must become explicitly fictional, not
        # a fabricated "real YAGO" witness.
        spec = make_spec(graph, seed=17, ordinal=0, difficulty="medium")
        self.assertEqual(spec["macro_plan"]["knowledge_origin"], "synthetic")
        self.assertTrue(spec["knowledge_witness"]["fictional"])
        self.assertTrue(all(
            row["origin"] == "synthetic"
            for path in spec["macro_plan"]["world_binding"][
                "candidate_paths"].values()
            for row in path
        ))
        scene = compile_and_validate(spec, reference_question(spec))
        self.assertEqual(validate_replay(scene)["status"], "valid")
        self.assertTrue(scene["difficulty_audit"]["macro_first_audit"][
            "candidate_information_necessity"]["passed"])
        self.assertFalse(guard_question(reference_question(spec), spec))
        kg = graph_fixture()
        extra = Edge(
            "https://example.org/Person0",
            p+"birthPlace", "https://example.org/AlternativeTown",
        )
        kg.edges.append(extra)
        index = YagoKnowledgeIndex(kg)
        original = next(x for x in kg.edges
                        if x.subject == "https://example.org/Person0"
                        and x.predicate == p+"birthPlace"
                        and x.object != extra.object)
        self.assertFalse(_loaded_yago_unique(index, [original]))


    def test_sparse_world_fallback_across_modes_and_provenance(self):
        from environment.generate_question.intent import intent_contract
        kg = KnowledgeGraph(edges=[], input_file="empty-facts.ttl", scanned=0)
        seen_modes = set()
        for ordinal in range(16):
            difficulty = "medium" if ordinal % 2 == 0 else "hard"
            spec = make_spec(
                kg, seed=17, ordinal=ordinal, difficulty=difficulty,
            )
            plan = spec["macro_plan"]
            seen_modes.add(plan["mode"])
            self.assertEqual(plan["knowledge_origin"], "synthetic")
            self.assertEqual(spec["knowledge_witness"]["origin"], "synthetic")
            self.assertFalse(validate_spec(spec))
            question = reference_question(spec)
            self.assertNotIn("YAGO", question)
            self.assertNotIn("real-world", question)
            self.assertIn("do not", " ".join(
                intent_contract(spec)["nonnegotiable_conditions"]
            ))
            scene = compile_and_validate(spec, question)
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertTrue(scene["difficulty_audit"]["macro_first_audit"][
                "candidate_information_necessity"]["passed"])
            realized = realize_scene(scene, strict=True)
            self.assertEqual(
                len(realized["macro_tasks"]),
                scene["macro_graph"]["macro_task_count"],
            )
            self.assertEqual(
                scene["world"]["knowledge_origin"], "synthetic",
            )
            for sid, source in scene["world"]["sources"].items():
                if sid.startswith("relation_") or sid == "real_fork":
                    self.assertEqual(
                        source["evidence_kind"], "fictional_relation_facts",
                    )
                    self.assertTrue(all(
                        x["evidence_origin_id"].startswith("synthetic:")
                        for x in source["observations"]
                    ))
            if plan["mode"] == "verify":
                self.assertEqual(
                    plan["world_binding"]["real_fork"]["motif_kind"],
                    "fictional_fork",
                )
            altered = deepcopy(spec)
            case = altered["macro_plan"]["world_binding"]["target_case"]
            altered["macro_plan"]["world_binding"]["candidate_paths"][
                case][0]["origin"] = "yago"
            self.assertTrue(validate_spec(altered))
        self.assertEqual(seen_modes, {"trace", "event", "verify"})


    def test_composed_yago_motifs_form_a_canonical_typed_world(self):
        from environment.generate_question.world_motifs import (
            motif_features, validate_composed_world, COMPOSE_VERSION,
        )
        from environment.generate_question.compiler import QuestionSession
        graph = graph_fixture()
        index = YagoKnowledgeIndex(graph)
        features = motif_features(graph, index)
        self.assertEqual(features["edge_counts"]["birthplace"], 54)
        self.assertEqual(features["edge_counts"]["containedinplace"], 108)
        originals = {
            (f.subject, f.predicate, f.object) for f in graph.edges
        }
        profiles, modes = set(), set()
        for ordinal in range(18):
            difficulty = "medium" if ordinal % 2 == 0 else "hard"
            spec = make_spec(
                graph, seed=17, ordinal=ordinal, difficulty=difficulty,
                world_mode="compose", index=index,
            )
            plan = spec["macro_plan"]
            self.assertEqual(plan["knowledge_world_mode"], "compose")
            self.assertEqual(plan["knowledge_origin"], "synthetic")
            self.assertFalse(validate_spec(spec))
            binding = plan["world_binding"]
            world_model = binding["world_model"]
            self.assertEqual(world_model["version"], COMPOSE_VERSION)
            self.assertEqual(validate_composed_world(binding), [])
            self.assertGreater(world_model["profile"]["source_yago_edges"], 0)
            self.assertFalse(
                world_model["profile"]["actual_yago_assertions_reused"]
            )
            self.assertEqual(
                world_model["profile"]["goal_signature"],
                plan["goal"]["semantic"]["profile"].split("_"),
            )
            self.assertEqual(
                len(world_model["relations"]),
                len(binding["all_cases"]) * plan["hops"],
            )
            profiles.add(plan["goal"]["semantic"]["profile"])
            modes.add(plan["mode"])
            for row in world_model["relations"]:
                self.assertNotIn(
                    (row["subject"], row["predicate"], row["object"]),
                    originals,
                )
                self.assertEqual(
                    world_model["entities"][row["object"]]["label"],
                    row["object_label"],
                )
            question = reference_question(spec)
            self.assertFalse(guard_question(question, spec))
            for word in ("fictional", "simulated", "synthetic", "YAGO"):
                self.assertNotIn(word, question)
            scene = compile_and_validate(spec, question)
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertTrue(scene["difficulty_audit"]["macro_first_audit"][
                "candidate_information_necessity"]["passed"])
            self.assertEqual(
                scene["world"]["typed_knowledge_world"], world_model,
            )
            self.assertTrue(all(
                "urn:maestro:fictional:" not in value
                for source in scene["world"]["sources"].values()
                for obs in source["observations"]
                for value in obs["values"].values()
                if isinstance(value, str)
            ))
            self.assertEqual(scene["public"]["status"],
                             "ready_for_investigation")
            self.assertTrue(all(
                "representation" not in row
                for row in scene["public"]["available_source_descriptions"]
            ))
            session = QuestionSession(scene)
            sid = next(
                t["source_id"] for t in spec["sources"]
                if t["access_mode"] == "browse"
            )
            material = session.read_source(sid)
            self.assertNotIn("representation", material)
            self.assertNotIn("evidence_kind", material)
            self.assertNotIn(
                "evidence_origin_id", material["observations"][0],
            )
            altered = deepcopy(spec)
            first = next(iter(altered["macro_plan"]["world_binding"][
                "world_model"]["entities"].values()))
            first["type"] = "unsupported_role"
            self.assertTrue(validate_spec(altered))
        self.assertEqual(modes, {"trace", "event", "verify"})
        self.assertGreaterEqual(len(profiles), 5)


    def test_yago_field_labels_are_semantically_typed(self):
        from environment.generate_question.world_motifs import (
            motif_features, _academic_field_label,
        )
        graph = graph_fixture()
        p = "http://schema.org/"
        graph.edges += [
            Edge("https://example.org/Scholar1", p+"fieldOfWork",
                 "https://example.org/Manufacturer"),
            Edge("https://example.org/Scholar2", p+"fieldOfWork",
                 "https://example.org/Crystallography"),
        ]
        graph.labels.update({
            "https://example.org/Manufacturer": "Aerospace manufacturer",
            "https://example.org/Crystallography": "Crystallography",
        })
        index = YagoKnowledgeIndex(graph)
        features = motif_features(graph, index)
        self.assertNotIn(
            "Aerospace manufacturer", features["observed_field_labels"],
        )
        self.assertIn(
            "Crystallography", features["observed_field_labels"],
        )
        self.assertEqual(features["rejected_nonacademic_field_labels"], 1)
        self.assertFalse(_academic_field_label("Aerospace manufacturer"))
        self.assertTrue(_academic_field_label("comparative philology"))
        self.assertTrue(_academic_field_label("Crystallography"))

    def test_composed_shared_relation_is_required_in_existing_macro(self):
        from collections import Counter
        from environment.schema import digest
        from environment.generate_question.task_plan import SALT
        kg = graph_fixture()
        index = YagoKnowledgeIndex(kg)
        seen_leads = set()
        for ordinal in range(12):
            level = "medium" if ordinal % 2 == 0 else "hard"
            spec = make_spec(
                kg, seed=17, ordinal=ordinal, difficulty=level,
                index=index, world_mode="compose",
            )
            plan = spec["macro_plan"]
            binding = plan["world_binding"]
            model = binding["world_model"]
            self.assertEqual(model["context_version"],
                             "maestro.shared_context.v1")
            mapping = model["context_by_case"]
            required = model["required_context_relation"]
            sid = required["source_id"]
            self.assertEqual(sid, "M1_b")
            self.assertEqual(required["source_slot"], 1)
            self.assertEqual(
                plan["clues"][sid]["value"],
                required["target_anchor_label"],
            )
            refs = Counter(row["anchor_id"] for row in mapping.values())
            self.assertEqual(
                sorted(refs.values()),
                sorted([
                    len(required["excluded_cases"]),
                    len(binding["all_cases"])-len(required["excluded_cases"]),
                ]),
            )
            self.assertEqual(
                len([r for r in model["context_relations"]
                     if r["scope"] == "required_clue"]),
                len(binding["all_cases"]),
            )
            self.assertEqual(
                mapping[required["excluded_case"]]["anchor_label"],
                plan["clues"][sid]["alternative"],
            )
            for case in required["excluded_cases"]:
                self.assertEqual(
                    mapping[case]["anchor_id"], required["excluded_anchor_id"]
                )
            question = reference_question(spec)
            seen_leads.add(question.split(":", 1)[0].split(
                " with the following", 1
            )[0][:72])
            self.assertIn(plan["clues"][sid]["value"], question)
            scene = compile_and_validate(spec, question)
            self.assertEqual(validate_replay(scene)["status"], "valid")
            audit = scene["difficulty_audit"]["macro_first_audit"][
                "candidate_information_necessity"
            ]
            self.assertTrue(audit["passed"], audit)
            self.assertTrue(audit["required_shared_relation_necessary"])
            self.assertTrue(audit["clue_necessary"][sid])
            self.assertTrue(all(audit["clue_necessary"].values()))
            self.assertEqual(audit["required_shared_relation_source"], sid)
            source_values = {
                obs["values"]["case_id"]: obs["values"]["cue"]
                for obs in scene["world"]["sources"][sid]["observations"]
            }
            self.assertEqual(source_values, {
                case: row["anchor_label"] for case, row in mapping.items()
            })
            # The relation is readable ONLY from its authorized source.
            # Repeating its value in another record's document_context is
            # a hidden shortcut, even if Gold still has to join the sources.
            for other_sid, source in scene["world"]["sources"].items():
                for obs in source["observations"]:
                    self.assertNotIn("document_context", obs)
                    self.assertNotIn("record_excerpt", obs)
                    if other_sid != sid:
                        self.assertNotIn(
                            required["target_anchor_label"],
                            list(map(str, obs["values"].values())),
                        )
            realized = realize_scene(scene, strict=True)
            self.assertEqual(
                len(realized["macro_tasks"]),
                scene["macro_graph"]["macro_task_count"],
            )
            tampered = deepcopy(spec)
            target = binding["target_case"]
            tampered["macro_plan"]["world_binding"][
                "world_model"]["context_by_case"][target]["anchor_id"] = (
                    "ENT-UNREFERENCED"
                )
            tampered["knowledge_contract_hash"] = digest(
                tampered["macro_plan"], tampered["knowledge_witness"],
                tampered["seed"], tampered["ordinal"], SALT,
            )
            self.assertIn(
                "composed_world_bad_context_reference", validate_spec(tampered),
            )
            tampered = deepcopy(spec)
            tampered["macro_plan"]["world_binding"]["world_model"][
                "required_context_relation"]["excluded_case"] = target
            tampered["knowledge_contract_hash"] = digest(
                tampered["macro_plan"], tampered["knowledge_witness"],
                tampered["seed"], tampered["ordinal"], SALT,
            )
            self.assertIn(
                "composed_world_required_relation_not_unique_shadow",
                validate_spec(tampered),
            )
        self.assertGreaterEqual(len(seen_leads), 4)

    def test_compact_macro_goal_question_preserves_public_target_anchor(self):
        """Question is user-level intent; separate reference card is input.

        The card identifies exactly one dossier by intersecting the already
        necessary discovery observations; it is not a shortcut to Gold.
        """
        from environment.generate_question.question import guard_question
        from environment.generate_question.compiler import QuestionSession
        from environment.generate_question.task_plan import (
            MACRO_GOAL_QUESTION,
        )
        from environment.base_orchestration.o1_compat import (
            o1_initial_instruction,
        )

        kg = graph_fixture()
        index = YagoKnowledgeIndex(kg)
        old_question = None
        for ordinal in range(20):
            difficulty = "medium" if ordinal % 2 == 0 else "hard"
            spec = make_spec(
                kg, seed=17, ordinal=ordinal, difficulty=difficulty,
                index=index, world_mode="compose",
            )
            plan = spec["macro_plan"]
            binding = plan["world_binding"]
            self.assertEqual(
                plan["question_surface_policy"], MACRO_GOAL_QUESTION,
            )
            card = spec["task_input"]
            self.assertEqual(card["version"], MACRO_GOAL_QUESTION)
            self.assertEqual(
                len(card["observations"]), len(plan["clues"]),
            )
            self.assertNotIn("case_id", json.dumps(card))
            self.assertNotIn("project_id", json.dumps(card))
            self.assertNotIn("event_id", json.dumps(card))
            self.assertNotIn("answer", json.dumps(card))
            question = reference_question(spec)
            self.assertLess(len(question), 355, question)
            self.assertFalse(guard_question(question, spec), question)
            self.assertIn(spec["scenario"]["topic"], question)
            self.assertIn(plan["goal"]["semantic"]["question"][
                0:15].lower(), question.lower())
            for cue in plan["clues"].values():
                self.assertNotIn(
                    cue["value"].casefold(), question.casefold(),
                )
            scene = compile_and_validate(spec, question)
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertEqual(scene["public"]["task_input"], card)
            self.assertEqual(
                QuestionSession(scene).public_view["task_input"], card,
            )
            target = binding["target_case"]
            candidate_sets = []
            for sid, cue in sorted(plan["clues"].items()):
                candidate_sets.append({
                    obs["values"]["case_id"]
                    for obs in scene["world"]["sources"][sid]["observations"]
                    if obs["values"]["cue"] == cue["value"]
                })
            self.assertEqual(set.intersection(*candidate_sets), {target})
            rendered = o1_initial_instruction(scene)
            self.assertTrue(rendered.startswith(question + "\n\n"))
            self.assertIn("Provided reference materials", rendered)
            for cue in plan["clues"].values():
                self.assertIn(cue["value"], rendered)
            tampered = deepcopy(spec)
            tampered["task_input"]["observations"][0][
                "observed_value"] = "UNLICENSED-OBSERVATION"
            self.assertIn(
                "task_plan_reference_card_not_frozen",
                validate_spec(tampered),
            )
            missing_pointer = question.replace(
                "supplied reference materials", "records"
            ).replace("provided reference materials", "records").replace(
                "attached reference materials", "records"
            )
            if missing_pointer != question:
                self.assertIn(
                    "task_plan_reference_material_not_identified",
                    guard_question(missing_pointer, spec),
                )
            old_question = question
        self.assertIsNotNone(old_question)

    def test_world_event_links_require_a_real_upstream_key(self):
        """A wrong but valid project key must retrieve another event chain.

        This is not a missing-source/blocked-program necessity proxy:
        the counterfactual returns a nonempty *wrong* answer with provenance.
        """
        from environment.generate_question.world_motifs import (
            KEYED_EVENT_VERSION,
        )
        from environment.generate_question.program_ir import (
            ProgramError, MissingSourceError, evaluate_program,
        )
        from environment.generate_question.compiler import QuestionSession
        from environment.generate_question.task_plan import SALT
        from environment.schema import digest

        kg = graph_fixture()
        index = YagoKnowledgeIndex(kg)
        seen = 0
        for ordinal in range(24):
            difficulty = "medium" if ordinal % 2 == 0 else "hard"
            spec = make_spec(
                kg, seed=17, ordinal=ordinal, difficulty=difficulty,
                index=index, world_mode="compose",
            )
            if spec["macro_plan"]["mode"] != "event":
                continue
            seen += 1
            world_model = spec["macro_plan"]["world_binding"]["world_model"]
            self.assertEqual(
                world_model["keyed_event_relation_version"],
                KEYED_EVENT_VERSION,
            )
            links = world_model["event_links"]
            self.assertEqual(
                len(links),
                2*len(spec["macro_plan"]["world_binding"]["all_cases"]),
            )
            self.assertFalse(validate_spec(spec))
            scene = compile_and_validate(spec, reference_question(spec))
            self.assertEqual(validate_replay(scene)["status"], "valid")
            audit = scene["difficulty_audit"]["macro_first_audit"][
                "candidate_information_necessity"
            ]
            self.assertTrue(audit["passed"], audit)
            key_audit = audit["keyed_event_dependency"]
            self.assertTrue(key_audit["passed"], key_audit)
            self.assertTrue(key_audit["world_links_match"])
            self.assertTrue(key_audit["wrong_upstream_key_changes_answer"])
            self.assertTrue(key_audit["wrong_upstream_key_result"])
            self.assertNotEqual(
                key_audit["wrong_upstream_key_result"], scene["gold_answer"]
            )
            ledger = scene["world"]["sources"]["event_ledger"]
            self.assertEqual(ledger["access_policy"], {
                "mode": "keyed_lookup",
                "key_field": "project_id",
                "key_cardinality": "many",
            })
            with self.assertRaises(MissingSourceError):
                QuestionSession(scene).read_source("event_ledger")
            node_ops = [
                node for node in scene["gold_program"]["nodes"]
                if node.get("source_id") == "event_ledger"
            ]
            self.assertEqual(len(node_ops), 1)
            self.assertEqual(node_ops[0]["op"], "lookup_by_key")
            self.assertEqual(node_ops[0]["input_key_field"], "project_id")
            source_keys = [
                x["values"]["project_id"] for x in ledger["observations"]
            ]
            self.assertTrue(all(source_keys.count(pid) == 2
                                for pid in set(source_keys)))
            event_values = {
                (r["values"]["project_id"], r["values"]["event_id"]):
                    r["values"] for r in ledger["observations"]
            }
            references = {
                r["values"]["event_id"]: r["values"]["entity_0"]
                for r in scene["world"]["sources"]["event_ref"]["observations"]
            }
            for link in links:
                row = event_values[link["project_id"], link["event_id"]]
                self.assertEqual(row["event_status"], link["event_status"])
                self.assertEqual(row["event_year"], link["event_year"])
                self.assertEqual(
                    references[link["event_id"]], link["entity_0"],
                )
            # The old strict singleton policy must not be weakened for
            # other keyed registers or when the explicit flag is removed.
            blocked = deepcopy(scene["world"]["sources"])
            del blocked["event_ledger"]["access_policy"]["key_cardinality"]
            with self.assertRaisesRegex(
                ProgramError, "ambiguous_lookup_source_key"
            ):
                evaluate_program(scene["gold_program"], sources=blocked)
            realized = realize_scene(scene, strict=True)
            self.assertEqual(
                len(realized["macro_tasks"]),
                scene["macro_graph"]["macro_task_count"],
            )
            self.assertTrue(any(
                step["operation"] == "lookup_by_key" and
                step.get("lookup_keys", {}).get("source_id") == "event_ledger"
                for item in realized["macro_tasks"]
                for step in item["trace_summary"]["steps"]
                if step["lookup_keys"] is not None
            ))
            tampered = deepcopy(spec)
            tampered["macro_plan"]["world_binding"]["world_model"][
                "event_links"][0]["entity_0"] = "ENT-NONEXISTENT"
            tampered["knowledge_contract_hash"] = digest(
                tampered["macro_plan"], tampered["knowledge_witness"],
                tampered["seed"], tampered["ordinal"], SALT,
            )
            self.assertIn(
                "composed_world_event_subject_missing",
                validate_spec(tampered),
            )
        self.assertGreaterEqual(seen, 6)

    def test_spec_tamper_and_replay_fail_closed(self):
        kg = graph_fixture()
        spec = make_spec(kg, seed=17, ordinal=1, difficulty="medium")
        broken = deepcopy(spec)
        broken["macro_plan"]["macro_tasks"][-1]["depends_on"] = []
        self.assertTrue(validate_spec(broken))
        broken = deepcopy(spec)
        case = broken["macro_plan"]["world_binding"]["target_case"]
        broken["macro_plan"]["world_binding"]["candidate_paths"][case][0][
            "object"] = "urn:fake"
        self.assertTrue(validate_spec(broken))
        scene = compile_and_validate(spec, reference_question(spec))
        altered = deepcopy(scene)
        one = next(iter(altered["world"]["sources"]))
        altered["world"]["sources"][one]["observations"][0][
            "values"]["cue"] = "unmatched"
        self.assertEqual(validate_replay(altered)["status"], "invalid")


@unittest.skipUnless(RDF_READY, "requires pyoxigraph")
class TaskPlanCanonicalCliTests(unittest.TestCase):
    def test_20_scene_cli_with_sparse_yago(self):
        from environment.config import load_config, args_from_config
        from environment.generate_question.pipeline import generate_questions
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ttl = root / "sparse.ttl"
            ttl.write_text(
                "@prefix x: <https://example.org/> .\n"
                "@prefix s: <http://schema.org/> .\n"
                "x:A s:location x:B .\n"
                "x:B s:follows x:C .\n", encoding="utf-8",
            )
            cfg = load_config("environment/configs/questions_task_plan.yaml")
            cfg["generation"]["knowledge_yago"] = str(ttl)
            cfg["generation"]["output_dir"] = str(root / "output")
            cfg["generation"]["target"] = 20
            args = args_from_config(cfg)
            args.no_progress = True
            result = generate_questions(args)
            self.assertEqual(result["status"], "done", result)
            self.assertEqual(result["generated"], 20)
            report = json.loads(
                (args.output_dir / "quality_report.json").read_text()
            )
            stats = report["knowledge_world_grounding"]["single_task_plan"]
            self.assertEqual(stats["scenes"], 20)
            self.assertEqual(stats["necessity_passed"], 20)
            self.assertEqual(stats["semantic_grounding"]["fictional_worlds"], 20)
            self.assertEqual(stats["semantic_grounding"]["composed_motif_worlds"], 20)
            self.assertEqual(stats["semantic_grounding"]["shared_context_worlds"], 20)
            self.assertEqual(
                stats["semantic_grounding"]["required_shared_relation_worlds"],
                20,
            )
            self.assertEqual(
                stats["semantic_grounding"][
                    "required_shared_relation_necessity_passed"], 20,
            )
            self.assertEqual(
                stats["semantic_grounding"]["source_records_with_context"], 0,
            )
            self.assertGreater(
                stats["semantic_grounding"]["keyed_event_worlds"], 0
            )
            self.assertEqual(
                stats["semantic_grounding"]["keyed_event_worlds"],
                stats["semantic_grounding"][
                    "keyed_event_wrong_key_audit_passed"],
            )
            self.assertEqual(report["replay_pass"], 20)

    def test_full_canonical_cli_and_quality_report(self):
        from environment.config import load_config, args_from_config
        from environment.generate_question.pipeline import generate_questions
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ttl = root/"yago-fixture.ttl"
            lines = [
                "@prefix x: <https://example.org/> .",
                "@prefix s: <http://schema.org/> .",
            ]
            for i in range(54):
                lines.extend([
                    f"x:Person{i} s:birthPlace x:Town{i} .",
                    f"x:Town{i} s:containedInPlace x:Region{i} .",
                    f"x:Region{i} s:containedInPlace x:Country{i} .",
                    f"x:Person{i} s:memberOf x:Institution{i} .",
                ])
            ttl.write_text("\n".join(lines), encoding="utf-8")
            conf = load_config("environment/configs/questions_task_plan.yaml")
            conf["generation"]["knowledge_yago"] = str(ttl)
            conf["generation"]["output_dir"] = str(root/"output")
            conf["generation"]["target"] = 6
            args = args_from_config(conf)
            args.no_progress = True
            result = generate_questions(args)
            self.assertEqual(result["status"], "done", result)
            self.assertEqual(result["generated"], 6)
            report = json.loads(
                (args.output_dir/"quality_report.json").read_text()
            )
            self.assertEqual(report["replay_pass"], 6)
            stats = report["knowledge_world_grounding"]["single_task_plan"]
            self.assertEqual(stats["scenes"], 6)
            self.assertEqual(stats["semantic_grounding"]["scenes"], 6)
            self.assertEqual(
                stats["semantic_grounding"]["loaded_graph_uniqueness_verified"], 0
            )
            self.assertEqual(
                stats["semantic_grounding"]["fictional_worlds"], 6
            )
            self.assertEqual(
                stats["semantic_grounding"]["composed_motif_worlds"], 6
            )
            self.assertGreaterEqual(
                stats["semantic_grounding"]["sampling_attempts"], 6
            )
            self.assertTrue(
                stats["semantic_grounding"]["goal_profile_distribution"]
            )
            self.assertEqual(stats["necessity_passed"], 6)
            self.assertEqual(len(stats["mode_distribution"]), 3)
            self.assertGreaterEqual(stats["macro_task_count_range"][1],
                                    stats["macro_task_count_range"][0])
            scenes = [
                json.loads(x) for x in (
                    args.output_dir/"constructed"/"scenes.jsonl"
                ).read_text().splitlines()
            ]
            self.assertEqual(len(scenes), 6)
            for scene in scenes:
                self.assertEqual(validate_replay(scene)["status"], "valid")


if __name__ == "__main__":
    unittest.main()
