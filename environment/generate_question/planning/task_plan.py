"""Canonical Stage 1 Task Plan, evidence binding and frozen validation.

An output-first Goal determines the typed evidence to acquire. The Macro
DAG is the only authoritative orchestration structure; World, Gold and
Question all depend on its frozen contracts.
"""
from __future__ import annotations

import copy
import random
from collections import Counter

from environment.schema import digest
from environment.generate_question.planning.knowledge_index import YagoKnowledgeIndex, predicate_label
from environment.generate_question.planning.goal_plan import COMPOSER as OUTPUT_FIRST_COMPOSER

PATTERN = "task_plan"
VERSION = "maestro.task_plan.v1"
POLICY = "single_task_plan_counterfactual_v1"
SALT = "maestro-single-task-plan-frozen"
MODES = ("trace", "event", "verify")
# Sampling policy flags only: do not create another complete generator or
# invalidate frozen task_plan scenes that omitted these fields.
CALIBRATED = "calibrated_structural_difficulty_v1"
BLIND_EVENTS = "eligibility_blind_event_names_v1"
GOAL_QUESTIONS = "grounded_relation_goal_question_v1"
MACRO_GOAL_QUESTION = "macro_goal_reference_card_v2"
SEMANTIC_GOALS = "goal_constrained_yago_grounding_v1"
SYNTHETIC_WORLD = "maestro.fictional_semantic_world.v1"
SYNTHETIC_WITNESS = "maestro.task_plan.fictional_witness.v1"
SYNTHETIC_FORK = "maestro.fictional_fork_grounding.v1"
# Local predicate combinations with interpretable user goals. These are
# *predicate-signature hypotheses*, NOT RDF-type attestations. The bounded
# YAGO loader intentionally excludes rdf:type triples.
_SEMANTIC_RECIPES = (
    (("birthplace","containedinplace"), "biographical", "person", "administrative area", "Which administrative area contains the referenced person's birthplace?"),
    (("birthplace","location"), "biographical", "person", "location", "Where is the referenced person's birthplace located?"),
    (("deathplace","location"), "biographical", "person", "location", "What location is recorded for the referenced person's place of death?"),
    (("doctoraladvisor","fieldofwork"), "academic", "scholar", "research field", "What research field is recorded for the referenced scholar's doctoral advisor?"),
    (("doctoraladvisor","birthplace"), "academic", "scholar", "birthplace", "Where was the referenced scholar's doctoral advisor born?"),
    (("spouse","birthplace"), "biographical", "person", "birthplace", "Where was the referenced person's spouse born?"),
    (("memberof","location"), "institutional", "entity", "location", "Where is the organization affiliated with the referenced entity located?"),
    (("location","containedinplace"), "geographical", "entity", "administrative area", "Which administrative area contains the referenced entity's recorded location?"),
    (("containedinplace","containedinplace"), "geographical", "place", "larger area", "Which larger area contains the administrative area of the referenced place?"),
    (("neighbor","location"), "geographical", "place", "location", "Where is the neighboring place of the referenced place located?"),
    (("location","neighbor"), "geographical", "entity", "neighboring place", "Which place borders the referenced entity's recorded location?"),
    (("birthplace","containedinplace","containedinplace"), "biographical", "person", "larger administrative area", "Which larger administrative area contains the region of the referenced person's birthplace?"),
    (("birthplace","location","containedinplace"), "biographical", "person", "administrative area", "Which administrative area contains the location recorded for the referenced person's birthplace?"),
    (("doctoraladvisor","birthplace","location"), "academic", "scholar", "location", "Where is the birthplace of the referenced scholar's doctoral advisor located?"),
    (("doctoraladvisor","birthplace","containedinplace"), "academic", "scholar", "administrative area", "Which administrative area contains the birthplace of the referenced scholar's doctoral advisor?"),
    (("memberof","location","containedinplace"), "institutional", "entity", "administrative area", "Which administrative area contains the location of the referenced entity's affiliated organization?"),
    (("location","containedinplace","containedinplace"), "geographical", "entity", "larger administrative area", "Which larger administrative area contains the region of the referenced entity's recorded location?"),
    (("spouse","birthplace","containedinplace"), "biographical", "person", "administrative area", "Which administrative area contains the birthplace of the referenced person's spouse?"),
    (("deathplace","location","containedinplace"), "biographical", "person", "administrative area", "Which administrative area contains the location recorded for the referenced person's place of death?"),
    (("nationality","capital","location"), "biographical", "person", "location", "Where is the capital of the referenced person's recorded nationality located?"),
    (("location","neighbor","location"), "geographical", "entity", "location", "Which location is recorded for the place neighboring the referenced entity's location?"),
    (("deathplace","location","neighbor"), "biographical", "person", "neighboring area", "Which area neighbors the location recorded for the referenced person's place of death?"),
    (("location","capital","twinnedwith"), "geographical", "entity", "twin municipality", "Which municipality is twinned with the capital of the area containing the referenced entity?"),
)


def _predicate_key(iri: str) -> str:
    return "".join(ch for ch in predicate_label(iri).casefold()
                   if ch.isalnum())


def _semantic_recipe(path: list) -> tuple | None:
    keys = tuple(_predicate_key(
        item["predicate"] if isinstance(item, dict) else item.predicate
    ) for item in path)
    return next((entry for entry in _SEMANTIC_RECIPES
                 if entry[0] == keys), None)


def _semantic_goal(path: list, count: int) -> dict:
    recipe = _semantic_recipe(path)
    if recipe is None:
        raise ValueError("uninterpretable_yago_relation_composition")
    keys, domain, subject_hint, answer_hint, question = recipe
    fictional = path[0].get("origin") == "synthetic"
    return {
        "profile": "_".join(keys),
        "relation_sequence": [
            item["predicate"] if isinstance(item, dict) else item.predicate
            for item in path
        ],
        "domain": domain, "subject_type_hint": subject_hint,
        "answer_type_hint": answer_hint, "question": question,
        "type_evidence": (
            "fictional_relation_type_contract" if fictional
            else "predicate_signature_heuristic_not_rdf_type"
        ),
        "answer_scope": (
            "one_result_per_candidate_in_simulated_world" if fictional
            else "one_result_per_candidate_in_loaded_yago_graph"
        ),
        "candidate_count": count,
    }


def _fictional_fork(paths: dict, *, source_style: str = "legacy") -> dict:
    # A distinct simulated registry is an independent *world source*, not
    # independent historical corroboration.
    predicate = "urn:maestro:fictional:predicate:cataloguedIn"
    forks = {}
    for case, path in paths.items():
        subject = path[0]["subject"]
        object_id = (
            "REG-" + digest(case, subject)[:12].upper()
            if source_style == "archive" else
            "urn:maestro:fictional:register:" + digest(case, subject)[:20]
        )
        forks[case] = {
            "subject": subject, "predicate": predicate, "object": object_id,
            "subject_label": subject.rsplit(":", 1)[-1],
            "object_label": "Registry card " + digest(case)[:8].upper(),
            "origin": "synthetic",
            "fact_id": "synthetic:" + digest(subject, predicate, object_id)[:20],
        }
    return {
        "version": SYNTHETIC_FORK, "motif_kind": "fictional_fork",
        "predicate": predicate, "candidate_forks": forks,
        "source_lineage": SYNTHETIC_WORLD,
        "causal_use": "required_fictional_fork_macro_evidence",
    }


def _validate_fictional_fork(fork: dict, paths: dict) -> list[str]:
    try:
        if (fork["version"] != SYNTHETIC_FORK
                or fork["motif_kind"] != "fictional_fork"
                or fork["source_lineage"] != SYNTHETIC_WORLD
                or fork["causal_use"] != "required_fictional_fork_macro_evidence"
                or fork["predicate"] != "urn:maestro:fictional:predicate:cataloguedIn"
                or set(fork["candidate_forks"]) != set(paths)):
            return ["task_plan_fictional_fork_wrong_contract"]
        seen = set()
        for case, path in paths.items():
            fact = fork["candidate_forks"][case]
            if (fact["subject"] != path[0]["subject"]
                    or fact["predicate"] != fork["predicate"]
                    or fact["origin"] != "synthetic"
                    or not fact["object"].startswith((
                        "urn:maestro:fictional:register:", "REG-"
                    ))
                    or fact["fact_id"] != "synthetic:" + digest(
                        fact["subject"], fact["predicate"], fact["object"]
                    )[:20] or fact["object"] in seen):
                return ["task_plan_fictional_fork_bad_witness"]
            seen.add(fact["object"])
        return []
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return ["task_plan_fictional_fork_invalid:" + str(exc)[:140]]


def _compose_ground(kg, index, plan, *, seed: int, ordinal: int):
    """Compose a task-driven fictional world from *observed YAGO motifs*.

    YAGO supplies relation vocabulary, observed predicate prevalence and
    generic field words; no original entity-to-entity assertion is reassigned
    to any factual person. The new canonical World owns all target/shadow
    paths, and the same original Macro executor consumes its evidence views.
    """
    from environment.generate_question.grounding.world_motifs import (
        motif_features, compose_relation_goal, compose_typed_world,
    )
    rng = random.Random(int(digest(
        seed, ordinal, "task-plan-world-composition"
    )[:15], 16))
    recipes = tuple(r for r in _SEMANTIC_RECIPES
                    if len(r[0]) == plan["hops"])
    features = motif_features(kg, index, recipes)
    chosen_keys = plan.get("goal_recipe_keys")
    if chosen_keys is not None:
        recipe = next((item for item in recipes
                       if list(item[0]) == chosen_keys), None)
        if recipe is None:
            raise ValueError("composed_goal_recipe_not_supported")
    else:
        recipe = compose_relation_goal(features, recipes, rng=rng)
    n_clues = 2 * sum(
        t["operation"] == "match_clues" for t in plan["macro_tasks"]
    )
    n_cases = n_clues + 4
    target = "Dossier-" + digest(seed, ordinal, "target")[:10]
    cases = [target] + [
        "Dossier-" + digest(seed, ordinal, i, "case")[:10]
        for i in range(n_cases-1)
    ]
    rng.shuffle(cases)
    shadow_slots = {
        case: i if i < n_clues else None
        for i, case in enumerate(c for c in cases if c != target)
    }
    # Reuse an EXISTING discovery source, not a new Macro or evidence bank.
    # Its associated shadow is the only case without the required relation.
    discovery = next(t for t in plan["macro_tasks"]
                     if t["operation"] == "match_clues")
    source_id = discovery["source_scope"][1]
    source_slot = 1
    excluded = next(case for case, slot in shadow_slots.items()
                    if slot == source_slot)
    paths, world_model = compose_typed_world(
        kg, index, recipe=recipe, cases=cases, seed=seed, ordinal=ordinal,
        required_excluded_cases=[
            case for case, slot in shadow_slots.items()
            if slot is None or slot == source_slot
        ],
    )
    subject_role = world_model["entities"][paths[target][0]["subject"]]["type"]
    world_model["required_context_relation"] = {
        "source_id": source_id, "source_slot": source_slot,
        "excluded_case": excluded,
        "excluded_cases": [
            case for case, slot in shadow_slots.items()
            if slot is None or slot == source_slot
        ],
        "field": "cue",
        "predicate": ("http://schema.org/affiliation"
                      if subject_role == "person" else
                      "http://schema.org/subjectOf"),
        "target_anchor_id": world_model["context_by_case"][target]["anchor_id"],
        "target_anchor_label": world_model["context_by_case"][target]["anchor_label"],
        "excluded_anchor_id": world_model["context_by_case"][excluded]["anchor_id"],
        "excluded_anchor_label": world_model["context_by_case"][excluded]["anchor_label"],
    }
    fork = (_fictional_fork(paths, source_style="archive")
            if plan["mode"] == "verify" else None)
    world_model["verification_facts"] = (
        {case: copy.deepcopy(row) for case, row
         in fork["candidate_forks"].items()} if fork else {}
    )
    if plan["mode"] == "event":
        # Freeze project -> event -> subject links BEFORE source records.
        # Two events per project force an actual keyed one-to-many lookup
        # followed by the existing eligibility condition. Neither source
        # invents a separate copy of the relationship.
        from environment.generate_question.grounding.world_motifs import (
            composed_project_id, composed_event_id, KEYED_EVENT_VERSION,
        )
        from environment.generate_question.execution.task_plan_runtime import _condition_rows
        qualified, other = _condition_rows(plan)
        links = []
        for i, case in enumerate(cases):
            pid = composed_project_id(seed, ordinal, case)
            world_model["entities"][pid] = {
                "entity_id": pid, "type": "project",
                "label": "Project " + case.split("-")[-1].upper(),
            }
            for role, details, subject_case in (
                ("qualified", qualified, case),
                ("other", other, cases[(i+1) % len(cases)]),
            ):
                eid = composed_event_id(pid, role)
                world_model["entities"][eid] = {
                    "entity_id": eid, "type": "event",
                    "label": "Archive session " + eid[-6:],
                }
                links.append({
                    "case_id": case, "project_id": pid,
                    "event_id": eid, "event_role": role,
                    "entity_0": paths[subject_case][0]["subject"],
                    "event_status": details["event_status"],
                    "event_year": details["event_year"],
                    "project_event_predicate": "http://schema.org/hasPart",
                    "event_subject_predicate": "http://schema.org/about",
                    "origin": "synthetic",
                })
        world_model["event_links"] = links
        world_model["keyed_event_relation_version"] = KEYED_EVENT_VERSION
    return {
        "target_case": target, "all_cases": cases,
        "candidate_paths": paths, "real_fork": fork,
        "world_model": world_model,
        "case_shadow_slots": shadow_slots,
        "origin_file": SYNTHETIC_WORLD, "sample_triples_scanned": 0,
        "semantic_audit": {
            "policy": SEMANTIC_GOALS, "attempts": 1, "rejections": {},
            "grounding_origin": "synthetic",
            "fallback_from_yago": False, "composed_from_yago_motifs": True,
            "synthetic_consistency_checked": True,
            "loaded_yago_singleton_witness": False,
            "complete_candidate_count": n_cases,
            "rdf_type_verification": False,
        },
    }


def _bind_sources(plan, grounding, *, seed: int, ordinal: int):
    rng = random.Random(int(digest(seed, ordinal, "task-plan-evidence")[:15], 16))
    clues, catalog, schemas = {}, [], {}
    def source(sid, modality, title, purpose, schema, *, key=None):
        catalog.append({
            "source_id": sid, "modality": modality, "title": title,
            "purpose": purpose, "access_mode": "keyed_lookup" if key else "browse",
            **({"key_field": key} if key else {}),
        })
        schemas[sid] = dict(schema)
    disc = [t for t in plan["macro_tasks"]
            if t["operation"] == "match_clues"]
    contextual_titles = {
        "academic": "research correspondence",
        "biographical": "biographical records",
        "institutional": "institutional registry",
        "geographical": "regional fieldwork",
    }
    semantic_domain = (
        _semantic_goal(
            grounding["candidate_paths"][grounding["target_case"]],
            len(grounding["all_cases"]),
        )["domain"]
        if plan.get("semantic_grounding_policy") == SEMANTIC_GOALS
        else "institutional"
    )
    context = contextual_titles[semantic_domain]
    from environment.generate_question.grounding.scene_semantics import clue_profiles
    profiles = clue_profiles(semantic_domain)
    for i, task in enumerate(disc):
        for j, (modality, title, attr, choices) in enumerate(profiles[i]):
            if context is not None:
                title = f"{context} — {title}"
            sid = task["source_scope"][j]
            chosen = rng.choice(choices)
            alt = rng.choice([x for x in choices if x != chosen])
            required = grounding.get("world_model", {}).get(
                "required_context_relation"
            )
            if required and sid == required["source_id"]:
                chosen = required["target_anchor_label"]
                alt = required["excluded_anchor_label"]
                is_affiliation = required["predicate"].endswith("affiliation")
                attr = (
                    "institutional affiliation" if is_affiliation
                    else "survey-volume association"
                )
                title = (
                    "Research affiliation register" if is_affiliation
                    else "Regional survey-volume register"
                )
            clues[sid] = {
                "value": chosen, "alternative": alt,
                "attribute": attr, "source_slot": 2*i+j,
                "title": title,
            }
            source(sid, modality, title,
                   "Find dossier candidates by an observed clue",
                   {"case_id": "str", "cue": "str"},
                   key="case_id" if task["depends_on"] else None)
    source("entity_index", "table",
           "Dossier identity crosswalk" if context else "Project identity crosswalk",
           "Map the identified dossier to one named project",
           {"case_id": "str", "project_id": "str", "project_name": "str"},
           key="case_id")
    mode = plan["mode"]
    fictional = grounding["candidate_paths"][grounding["target_case"]][0][
        "origin"] == "synthetic"
    if mode == "trace":
        source("project_ref", "text",
               "Project subject reference register",
               "Resolve the project's documented entity",
               {"project_id": "str", "entity_0": "str"}, key="project_id")
    if mode in {"event", "verify"}:
        schema = {"project_id": "str", "event_id": "str",
                  "event_status": "str", "event_year": "int",
                  "event_name": "str"}
        if mode == "verify":
            schema["motif_object"] = "str"
        keyed_event = grounding.get("world_model", {}).get(
            "keyed_event_relation_version"
        )
        source("event_ledger", "table", "Project event ledger",
               "Inspect events subject to an explicit eligibility rule",
               schema, key="project_id" if keyed_event else None)
    if mode == "event":
        source("event_ref", "text",
               "Event-specific subject reference",
               "Follow the selected event's independently indexed entity",
               {"event_id": "str", "entity_0": "str"}, key="event_id")
    if mode == "verify":
        for sid in ("link_a", "link_b"):
            source(sid, "text", f"Project reference report {sid[-1].upper()}",
                   "Investigate competing source claims",
                   {"project_id": "str", "route_key": "str", "entity_0": "str"})
        source("real_fork", "text",
               "Secondary registry relationship",
               "Check the candidate against the independent registry record",
               {"entity_0": "str", "motif_object": "str"}, key="entity_0")
    facts = grounding["candidate_paths"][grounding["target_case"]]
    typed_goal = plan.get("goal_spec", {})
    typed_kind = typed_goal.get(
        "source_profile", typed_goal.get("kind", "entity_resolution")
    )
    for i, fact in enumerate(facts, 1):
        input_key = f"entity_{i-1}"
        output_key = (
            "entity_final" if i == plan["hops"] and
            typed_kind != "entity_resolution" else
            "answer" if i == plan["hops"] else f"entity_{i}"
        )
        source(f"relation_{i}", "text",
               f"{predicate_label(fact['predicate'])} relationship index",
               "Look up the documented relationship required by the goal",
               {input_key: "str", output_key: "str"}, key=input_key)
    # These are executable media-derived facts, not decorative clue labels.
    # Media sources are browsed, then joined against the resolved entity;
    # text/table sources retain keyed lookup and explicit handoff semantics.
    from environment.generate_question.grounding.scene_semantics import measurement_source_profile
    domain = (_semantic_goal(
        grounding["candidate_paths"][grounding["target_case"]],
        len(grounding["all_cases"]),
    )["domain"] if plan.get("semantic_grounding_policy") == SEMANTIC_GOALS
        else "institutional")
    for sid, fields in typed_goal["source_fields"].items():
        profile = measurement_source_profile(
            sid, domain=domain, capabilities=typed_goal["capabilities"],
        )
        source(
            sid, profile["modality"], profile["title"],
            profile["purpose"], fields, key=profile["key_field"],
        )
    expected = Counter(
        sid for t in plan["macro_tasks"] for sid in t["source_scope"]
    )
    if expected != Counter(schemas.keys()):
        raise ValueError("task_plan_source_not_bound_exactly_once")
    return clues, catalog, schemas


def _reference_observation_text(modality: str, attribute: str,
                                value: str) -> str:
    """Render ONLY an already frozen target observation as a media note.

    No extra entity, date, provenance or inferred fact is introduced.
    Unlike a raw table dump, these are user-provided *description* snippets;
    media decoding is not claimed.
    """
    quoted = "'" + str(value) + "'"
    templates = {
        "seal motif": "The photographed folio carries a seal with a {value} motif.",
        "drawer code": "The attached catalog slip gives the drawer marker as {value}.",
        "institutional affiliation": (
            "The accompanying register associates the record with {value}."
        ),
        "survey-volume association": (
            "An attached catalog note places the record in {value}."
        ),
        "shelf code": "The recording includes the spoken shelf marker {value}.",
        "document series": (
            "A cross-reference note identifies the document series as {value}."
        ),
        "reel motif": (
            "The supplied reel observation describes a sequence marked {value}."
        ),
        "record class": (
            "A separate register note classifies the item as {value}."
        ),
    }
    if attribute in templates:
        return templates[attribute].format(value=quoted)
    return (
        "The supplied " + modality + " observation records " +
        attribute + " as " + quoted + "."
    )


def _reference_card(plan: dict, sources: list[dict], *,
                    seed: int, ordinal: int) -> dict:
    """Public *starting material*, not a second evidence or World planner.

    It names the observations on the provided target reference without
    disclosing the matching case_id, project_id, event key or downstream
    answer. The Agent still has to acquire the candidate source records.
    Existing clue values already identify the target in the old question;
    moving them into a separate task input changes presentation only.
    """
    modality = {source["source_id"]: source["modality"]
                for source in sources}
    if plan.get("question_surface_policy") != MACRO_GOAL_QUESTION:
        raise ValueError("reference_card_unknown_surface_version")
    observations = [
        {
            "modality": modality[sid],
            "content": _reference_observation_text(
                modality[sid], info["attribute"], info["value"]
            ),
        }
        for sid, info in sorted(plan["clues"].items())
    ]
    return {
        "version": MACRO_GOAL_QUESTION,
        "reference_id": "REF-" + digest(
            seed, ordinal, "task-plan-provided-reference"
        )[:12].upper(),
        "kind": "provided_reference_materials",
        "observations": observations,
    }


def make_spec(kg, *, seed: int, ordinal: int, difficulty: str,
              min_hops: int = 2, max_hops: int = 3,
              index: YagoKnowledgeIndex | None = None) -> dict:
    """Construct the canonical output-first Goal, World and Task Plan."""
    if difficulty not in ("medium", "hard"):
        raise ValueError("task_plan_requires_medium_or_hard")
    if not 2 <= min_hops <= max_hops <= 4:
        raise ValueError("task_plan_invalid_hops")
    from environment.generate_question.grounding.world_motifs import motif_features, compose_relation_goal
    from environment.generate_question.planning.goal_plan import sample_output, compose_goal_plan
    hops = min_hops if difficulty == "medium" else max_hops
    yago_index = index or YagoKnowledgeIndex(kg)
    choices = tuple(
        row for row in _SEMANTIC_RECIPES if len(row[0]) == hops
    )
    output_fields = set(
        sample_output(seed, ordinal)["bindings"].values()
    )
    if "distance_km" in output_fields:
        choices = tuple(
            row for row in choices if row[1] == "geographical"
        )
    if not choices:
        raise ValueError("no_semantically_compatible_goal_recipe")
    features = motif_features(kg, yago_index, choices)
    goal_rng = random.Random(int(digest(
        seed, ordinal, "goal-conditioned-semantic-goal"
    )[:15], 16))
    goal_recipe = compose_relation_goal(features, choices, rng=goal_rng)
    plan = compose_goal_plan(
        seed=seed, ordinal=ordinal, difficulty=difficulty,
        hops=hops, goal_keys=goal_recipe[0],
    )
    plan["semantic_grounding_policy"] = SEMANTIC_GOALS
    plan["knowledge_world_mode"] = "compose"
    plan["question_surface_policy"] = MACRO_GOAL_QUESTION
    grounding = _compose_ground(
        kg, yago_index, plan, seed=seed, ordinal=ordinal,
    )
    from environment.generate_question.planning.goal_plan import build_measurements
    measurements = build_measurements(
        plan=plan, binding=grounding, seed=seed, ordinal=ordinal,
    )
    if measurements is not None:
        grounding["world_model"]["goal_measurements"] = measurements
    fictional = grounding["candidate_paths"][grounding["target_case"]][0][
        "origin"] == "synthetic"
    clues, sources, schemas = _bind_sources(
        plan, grounding, seed=seed, ordinal=ordinal,
    )
    plan.update({
        "clues": clues, "source_schemas": schemas,
        "world_binding": grounding,
    })
    plan["goal"]["semantic"] = _semantic_goal(
        grounding["candidate_paths"][grounding["target_case"]],
        len(grounding["all_cases"]),
    )
    if fictional:
        plan["knowledge_origin"] = "synthetic"
    if plan["goal_spec"]["source_profile"] == "entity_resolution":
        plan["goal"]["meaning"] = (
            "Determine the entity at the end of the documented relation chain"
        )
    # Semantic scenarios are selected *after* the Goal/World is frozen.
    # Varying the working context must never alter the executable Gold.
    from environment.generate_question.grounding.scene_semantics import scene_profile
    scenario = scene_profile(
        plan["goal"]["semantic"]["domain"], ordinal=ordinal,
        goal_label=plan["goal"]["public_label"],
        output_kind=plan["goal_spec"]["output_spec"]["kind"],
    )
    witness = {
        "version": (SYNTHETIC_WITNESS if fictional else
                    "maestro.task_plan.yago_witness.v1"),
        "origin_file": grounding["origin_file"],
        **({"origin": "synthetic", "fictional": True} if fictional else {}),
        "candidate_facts": {
            case: [
                {k: fact[k] for k in ("subject", "predicate", "object")}
                for fact in paths
            ] for case, paths in grounding["candidate_paths"].items()
        },
        "real_fork": copy.deepcopy(grounding["real_fork"]),
    }
    spec = {
        "version": "maestro.problem_spec.v4",
        "task_id": f"question-tp-{ordinal:06d}-"+digest(
            seed, ordinal, plan["mode"], hops
        )[:8],
        "scene_id": f"task-plan-{ordinal:06d}",
        "source_anchor": {
            "dataset": ("MAESTRO-fictional-semantic-world" if fictional
                        else "YAGO-4.6-plus-fictional-project"),
            "source_record_id": str(ordinal),
            "original_question_reused": False,
            "original_answer_reused": False,
            "original_media_used": False,
        },
        "seed": seed, "ordinal": ordinal,
        "difficulty_target": difficulty,
        "reasoning_pattern": PATTERN, "family": PATTERN,
        "scenario": scenario,
        "sources": sources,
        "requirements": {
            "answer_kind": "macro_value",
            "must_resolve_aliases": False,
            "must_match_independent_reviews": False,
            "must_check_independent_clearance": False,
            "must_preserve_source_provenance": True,
            "simulation_only": True,
        },
        "macro_plan": plan,
        **({"task_input": _reference_card(
            plan, sources, seed=seed, ordinal=ordinal
        )} if plan.get("question_surface_policy") ==
            MACRO_GOAL_QUESTION else {}),
        "knowledge_witness": witness,
        "macro_information_necessity_version": POLICY,
    }
    spec["knowledge_contract_hash"] = digest(
        plan, witness, seed, ordinal, SALT,
    )
    problems = validate_spec(spec)
    if problems:
        raise ValueError("task_plan_invalid_constructed_spec:" +
                         ",".join(problems))
    return spec


def validate_spec(spec: dict) -> list[str]:
    """Fail closed on the *single* plan and independent real fact witness."""
    try:
        plan = spec["macro_plan"]
        binding = plan["world_binding"]
        witness = spec["knowledge_witness"]
        if (spec["family"] != PATTERN or spec["reasoning_pattern"] != PATTERN
                or plan["version"] != VERSION
                or plan["mode"] not in MODES
                or witness["version"] not in (
                    "maestro.task_plan.yago_witness.v1", SYNTHETIC_WITNESS)
                or plan["information_necessity_policy"] != POLICY):
            return ["task_plan_wrong_version"]
        if spec["knowledge_contract_hash"] != digest(
            plan, witness, spec["seed"], spec["ordinal"], SALT,
        ):
            return ["task_plan_frozen_hash_mismatch"]
        question_policy = plan.get("question_surface_policy")
        if question_policy == MACRO_GOAL_QUESTION:
            if (plan.get("knowledge_world_mode") != "compose"
                    or spec.get("task_input") != _reference_card(
                        plan, spec["sources"], seed=spec["seed"],
                        ordinal=spec["ordinal"],
                    )):
                return ["task_plan_reference_card_not_frozen"]
        elif "task_input" in spec:
            return ["task_plan_unversioned_reference_card"]
        if any(k in spec for k in ("task_first_v1_backbone",
                                   "hybrid_backbone_v3",
                                   "task_first_v2_relation_sketch",
                                   "task_first_grounding")):
            return ["task_plan_nested_legacy_spec"]
        target, cases = binding["target_case"], binding["all_cases"]
        paths = binding["candidate_paths"]
        if (set(paths) != set(cases) or target not in cases
                or len(cases) != 2*sum(t["operation"] == "match_clues"
                                    for t in plan["macro_tasks"])+4):
            return ["task_plan_invalid_competing_paths"]
        fictional = paths[target][0]["origin"] == "synthetic"
        world_mode = plan.get("knowledge_world_mode", "witnessed")
        if world_mode not in {"witnessed", "compose"}:
            return ["task_plan_unknown_world_mode"]
        if world_mode == "compose":
            from environment.generate_question.grounding.world_motifs import (
                validate_composed_world, KEYED_EVENT_VERSION,
            )
            issues = validate_composed_world(binding)
            if issues:
                return issues
            event_links = binding["world_model"].get("event_links")
            event_policy = binding["world_model"].get(
                "keyed_event_relation_version"
            )
            if event_policy:
                if (plan["mode"] != "event"
                        or event_policy != KEYED_EVENT_VERSION):
                    return ["task_plan_invalid_keyed_event_policy"]
                event_source = next(
                    (row for row in spec["sources"]
                     if row["source_id"] == "event_ledger"), None
                )
                if (event_source is None
                        or event_source["access_mode"] != "keyed_lookup"
                        or event_source["key_field"] != "project_id"):
                    return ["task_plan_event_source_not_keyed"]
                from environment.generate_question.execution.task_plan_runtime import _condition_rows
                qualified, other = _condition_rows(plan)
                if any(
                    link["event_status"] !=
                        (qualified if link["event_role"] == "qualified"
                         else other)["event_status"]
                    or link["event_year"] !=
                        (qualified if link["event_role"] == "qualified"
                         else other)["event_year"]
                    for link in event_links
                ):
                    return ["task_plan_event_eligibility_not_from_world"]
            elif event_links is not None:
                return ["task_plan_unversioned_event_links"]
            required = binding["world_model"].get("required_context_relation")
            if required:
                if (required["source_id"] not in plan["clues"]
                        or plan["clues"][required["source_id"]][
                            "source_slot"] != required["source_slot"]
                        or plan["clues"][required["source_id"]]["value"] !=
                            required["target_anchor_label"]
                        or plan["clues"][required["source_id"]][
                            "alternative"] != required["excluded_anchor_label"]
                        or binding["case_shadow_slots"].get(
                            required["excluded_case"]
                        ) != required["source_slot"]):
                    return ["task_plan_required_shared_clue_mismatch"]
            expected_fork = (
                binding["real_fork"]["candidate_forks"]
                if binding["real_fork"] else {}
            )
            if binding["world_model"]["verification_facts"] != expected_fork:
                return ["composed_world_verification_not_authoritative"]
        elif "world_model" in binding:
            return ["task_plan_unexpected_composed_world"]
        if ((fictional and (
                    witness["version"] != SYNTHETIC_WITNESS
                    or witness.get("origin") != "synthetic"
                    or witness.get("fictional") is not True
                    or plan.get("knowledge_origin") != "synthetic"
                    or binding["origin_file"] != SYNTHETIC_WORLD
                    or spec["source_anchor"]["dataset"] !=
                        "MAESTRO-fictional-semantic-world"
                )) or (not fictional and witness["version"] !=
                        "maestro.task_plan.yago_witness.v1")):
            return ["task_plan_knowledge_provenance_mismatch"]
        predicates = [f["predicate"] for f in paths[target]]
        if len(predicates) != plan["hops"]:
            return ["task_plan_wrong_real_hops"]
        labels = set()
        each_hop_subjects = [set() for _ in predicates]
        for case, facts in paths.items():
            if len(facts) != len(predicates):
                return ["task_plan_incomplete_real_path"]
            for i, fact in enumerate(facts):
                if (fact["origin"] != ("synthetic" if fictional else "yago")
                        or fact["predicate"] != predicates[i]):
                    return ["task_plan_mixed_knowledge_predicate"]
                if fictional and (
                    not fact["subject"].startswith(
                        "ENT-" if world_mode == "compose"
                        else "urn:maestro:fictional:entity:"
                    )
                    or not fact["object"].startswith(
                        "ENT-" if world_mode == "compose"
                        else "urn:maestro:fictional:entity:"
                    )
                    or _semantic_recipe(facts) is None
                ):
                    return ["task_plan_invalid_fictional_relation"]
                if fact["subject"] in each_hop_subjects[i]:
                    return ["task_plan_non_unique_real_lookup_keys"]
                each_hop_subjects[i].add(fact["subject"])
                if i and facts[i-1]["object"] != fact["subject"]:
                    return ["task_plan_disconnected_real_path"]
                if {k: fact[k] for k in ("subject", "predicate", "object")} != \
                        witness["candidate_facts"][case][i]:
                    return ["task_plan_real_witness_mismatch"]
            if facts[-1]["object_label"] in labels:
                return ["task_plan_duplicate_real_answer"]
            labels.add(facts[-1]["object_label"])
        if plan["mode"] == "verify":
            if fictional:
                problems = _validate_fictional_fork(binding["real_fork"], paths)
            else:
                from environment.generate_question.grounding.real_motif_grounding import validate_frozen_required_fork
                problems = validate_frozen_required_fork(
                    binding["real_fork"], paths)
            if problems:
                return problems
        elif binding["real_fork"] is not None:
            return ["task_plan_unneeded_fork"]
        if witness["real_fork"] != binding["real_fork"]:
            return ["task_plan_fork_witness_mismatch"]
        semantic_policy = plan.get("semantic_grounding_policy")
        if semantic_policy not in (None, SEMANTIC_GOALS):
            return ["task_plan_unknown_semantic_policy"]
        if semantic_policy == SEMANTIC_GOALS:
            if plan["goal"].get("semantic") != _semantic_goal(
                paths[target], len(cases)
            ):
                return ["task_plan_semantic_goal_mismatched"]
            audit = binding.get("semantic_audit", {})
            if (audit.get("policy") != SEMANTIC_GOALS
                    or audit.get("loaded_yago_singleton_witness") is not
                    (not fictional)
                    or (fictional and (
                        audit.get("grounding_origin") != "synthetic"
                        or audit.get("synthetic_consistency_checked") is not True
                        or audit.get("fallback_from_yago") is not
                            (world_mode != "compose")
                        or (world_mode == "compose" and
                            audit.get("composed_from_yago_motifs") is not True)
                    ))
                    or audit.get("complete_candidate_count") != len(cases)
                    or audit.get("attempts", 0) < 1
                    or audit.get("rdf_type_verification") is not False):
                return ["task_plan_semantic_scope_not_verified"]
        elif "semantic" in plan["goal"] or "semantic_audit" in binding:
            return ["task_plan_unversioned_semantic_goal"]
        ndisc = sum(t["operation"] == "match_clues" for t in plan["macro_tasks"])
        if (set(plan["clues"]) != {
                sid for t in plan["macro_tasks"]
                if t["operation"] == "match_clues"
                for sid in t["source_scope"]}
                or len(plan["clues"]) != 2*ndisc):
            return ["task_plan_wrong_discovery_clues"]
        if set(binding["case_shadow_slots"]) != set(cases)-{target}:
            return ["task_plan_missing_shadow_cases"]
        if sorted(x for x in binding["case_shadow_slots"].values()
                  if x is not None) != list(range(2*ndisc)):
            return ["task_plan_bad_shadow_mapping"]
        sampling = plan.get("difficulty_sampling_policy")
        if sampling not in (None, CALIBRATED):
            return ["task_plan_unknown_sampling_policy"]
        if sampling == CALIBRATED and (
            plan.get("evidence_sampling_policy") != BLIND_EVENTS
            or plan.get("question_sampling_policy") != GOAL_QUESTIONS
        ):
            return ["task_plan_missing_blind_evidence_contract"]
        if sampling is None and any(
            key in plan for key in ("evidence_sampling_policy",
                                   "question_sampling_policy")
        ):
            return ["task_plan_unpaired_sampling_policy"]
        composer = plan.get("composer_policy")
        if composer != OUTPUT_FIRST_COMPOSER:
            return ["task_plan_unsupported_composer"]
        from environment.generate_question.planning.goal_plan import compose_goal_plan, validate_measurements
        expected = compose_goal_plan(
            seed=spec["seed"], ordinal=spec["ordinal"],
            difficulty=spec["difficulty_target"], hops=plan["hops"],
            goal_keys=tuple(plan["goal_recipe_keys"]),
        )
        if (plan["goal_recipe_keys"] != [
                _predicate_key(fact["predicate"])
                for fact in paths[target]]
                or any(plan[key] != expected[key] for key in (
                    "mode", "condition", "macro_tasks",
                    "terminal_macro_id", "goal_recipe_keys",
                    "information_requirements",
                ))):
            return ["task_plan_evidence_plan_mismatch"]
        if plan.get("goal_spec") != expected["goal_spec"]:
            return ["task_plan_goal_not_frozen"]
        if any(plan["goal"].get(key) != expected["goal"][key]
               for key in ("answer_field", "answer_type",
                           "public_label", "cardinality", "meaning")):
            return ["task_plan_goal_output_not_frozen"]
        issues = validate_measurements(
            plan, binding, seed=spec["seed"],
            ordinal=spec["ordinal"],
        )
        if issues:
            return issues
        discovery = [
            task for task in plan["macro_tasks"]
            if task["operation"] == "match_clues"
        ]
        access = plan["information_requirements"]["source_access"]
        if (len(discovery) != len(access)
                or any(bool(task["depends_on"]) != (
                    mode == "keyed_lookup"
                ) for task, mode in zip(discovery, access))):
            return ["task_plan_source_affordance_mismatch"]
        by_id, scopes = {}, []
        for task in plan["macro_tasks"]:
            if (task["input_contract"] != {
                    pid: by_id[pid]["output_schema"]
                    for pid in task["depends_on"]
                }):
                return ["task_plan_input_contract_mismatch"]
            scopes += task["source_scope"]
            by_id[task["macro_id"]] = task
        budget = 11
        if (len(by_id) > budget
                or plan["terminal_macro_id"] != plan["macro_tasks"][-1]["macro_id"]
                or plan["macro_tasks"][-1]["output_schema"] != {
                    "answer": plan["goal"]["answer_type"]
                }
                or Counter(scopes) != Counter(plan["source_schemas"].keys())
                or set(plan["source_schemas"]) != {
                    s["source_id"] for s in spec["sources"]}):
            return ["task_plan_source_or_terminal_mismatch"]
        if (plan["mode"] == "trace" and plan["condition"] is not None
                or plan["mode"] != "trace" and plan["condition"] not in (
                    {"field": "event_status", "op": "filter_eq",
                     "value": "completed"},
                    {"field": "event_year", "op": "filter_gte",
                     "value": 2025},
                )):
            return ["task_plan_condition_contract_mismatch"]
        return []
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return ["task_plan_invalid_spec:" + str(exc)[:160]]
