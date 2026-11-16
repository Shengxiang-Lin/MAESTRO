"""Single Task Plan for NEW MAESTRO Stage-1 generation.

Decisions: typed information tasks + their relation needs -> actual YAGO
witnesses and fictional observations. No historical make_spec is called.
Old typed_knowledge / graph_first versions remain replay-only.

macro_plan.macro_tasks is the *only* authoritative task graph. There is
no nested legacy Spec, duplicate Relation Sketch, or duplicate DAG.
"""
from __future__ import annotations

import copy
import random
from collections import Counter

from environment.schema import digest
from .knowledge_macro import YagoKnowledgeIndex, CUES, predicate_label
from .knowledge_macro_v2 import _alternatives, _serialized_path
from .real_motif_grounding import mine_required_fork

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


def _sample_task_plan(*, seed: int, ordinal: int, difficulty: str,
                      hops: int, sampling_policy: str | None = None,
                      world_mode: str = "witnessed") -> dict:
    """Choose typed information work before consulting World facts.

    Legacy samples retain their exact old deterministic choices. The new
    sampling policy gives Hard samples more independent information work,
    without treating this proxy as measured live-Agent hardness.
    """
    if sampling_policy == CALIBRATED:
        if difficulty not in {"medium", "hard"}:
            raise ValueError("unsupported_task_plan_difficulty")
        # Block-wise balancing avoids accidental coupling to the
        # parity of the Medium/Hard schedule (the old pilot inverted it).
        selector = (seed + ordinal // 2) % 2
        mode = (
            ("trace", "event") if difficulty == "medium"
            else ("event", "verify")
        )[selector]
        discoveries = 2 if difficulty == "medium" or mode == "verify" else 3
    elif sampling_policy is None:
        mode = MODES[(seed + ordinal) % len(MODES)]
        discoveries = 2 if mode == "verify" else 2 + (ordinal % 2)
    else:
        raise ValueError("unsupported_task_plan_sampling_policy")
    condition = (
        {"field": "event_status", "op": "filter_eq", "value": "completed"}
        if ordinal % 2 == 0 else
        {"field": "event_year", "op": "filter_gte", "value": 2025}
    ) if mode != "trace" else None
    tasks = []

    def add(role, op, dependencies, output, sources, relation_need):
        mid = f"M{len(tasks)+1}"
        previous = {t["macro_id"]: t for t in tasks}
        if any(pid not in previous for pid in dependencies):
            raise ValueError("task_plan_non_topological_dependency")
        tasks.append({
            "macro_id": mid, "role": role, "operation": op,
            "depends_on": list(dependencies),
            "input_contract": {
                pid: copy.deepcopy(previous[pid]["output_schema"])
                for pid in dependencies
            },
            "output_schema": dict(output),
            "source_scope": list(sources),
            "solver_template": op,
            "relation_need": relation_need,
            "evidence_plan": {
                "description": relation_need,
                "source_observation_targets": {
                    sid: 1 for sid in sources
                },
            },
        })
        return mid

    ds = []
    for i in range(discoveries):
        j = len(tasks) + 1
        ds.append(add(
            "discovery", "match_clues", [],
            {"case_id": "str"}, [f"M{j}_a", f"M{j}_b"],
            "Find candidate dossiers through two independent simulated clues",
        ))
    identified = add(
        "identification", "identify_project", ds,
        {"project_id": "str"}, ["entity_index"],
        "Intersect dossier candidates and resolve the fictional project",
    )
    if mode == "trace":
        last = add(
            "investigation", "reference", [identified],
            {"entity_0": "str"}, ["project_ref"],
            "Resolve the project's real YAGO reference",
        )
    elif mode == "event":
        event = add(
            "investigation", "qualifying_event", [identified],
            {"event_id": "str"}, ["event_ledger"],
            "Select the project's event satisfying the status or time condition",
        )
        last = add(
            "association", "event_reference", [event],
            {"entity_0": "str"}, ["event_ref"],
            "Use the qualified event as a key for the real YAGO reference",
        )
    else:
        event = add(
            "investigation", "event_eligibility", [identified],
            {"motif_object": "str"}, ["event_ledger"],
            "Determine eligible relation objects from the qualified event",
        )
        a = add(
            "investigation", "directory_a", [identified],
            {"route_key": "str", "entity_0": "str"}, ["link_a"],
            "Investigate reference hypotheses in simulated source A",
        )
        b = add(
            "investigation", "directory_b", [identified],
            {"route_key": "str", "entity_0": "str"}, ["link_b"],
            "Independently investigate reference hypotheses in source B",
        )
        assoc = add(
            "association", "intersect_references", [a, b],
            {"entity_0": "str"}, [],
            "Associate references supported by both competing sources",
        )
        last = add(
            "verification", "verify_real_fork", [assoc, event],
            {"entity_0": "str"}, ["real_fork"],
            "Cross-check fictional eligibility against a genuine YAGO relation",
        )

    groups = [[i for i in range(1, hops + 1)]]
    if mode != "verify" and hops > 2 and (
        sampling_policy == CALIBRATED or ordinal % 2
    ):
        groups = [[1], list(range(2, hops+1))]
    for group in groups:
        tail = group[-1]
        out = "answer" if tail == hops else f"entity_{tail}"
        last = add(
            "synthesis" if tail == hops else "investigation",
            "follow_yago", [last], {out: "str"},
            [f"relation_{i}" for i in group],
            "Follow the documented genuine YAGO relationship sequence",
        )

    if len(tasks) > 9:
        raise ValueError("task_plan_macro_budget_exceeded")
    if world_mode == "compose":
        for task in tasks:
            task["relation_need"] = task["relation_need"].replace(
                "real YAGO", "documented").replace(
                "genuine YAGO", "documented").replace(
                "fictional", "recorded").replace(
                "simulated", "independent")
            task["evidence_plan"]["description"] = task["relation_need"]
    elif world_mode != "witnessed":
        raise ValueError("unknown_task_plan_world_mode")
    return {
        "version": VERSION,
        "mode": mode, "condition": condition,
        **({"difficulty_sampling_policy": CALIBRATED,
            "evidence_sampling_policy": BLIND_EVENTS,
            "question_sampling_policy": GOAL_QUESTIONS}
           if sampling_policy == CALIBRATED else {}),
        "hops": hops, "macro_tasks": tasks,
        "terminal_macro_id": last,
        "goal": {
            "answer_field": "answer", "answer_type": "str",
            "public_label": "related entity", "cardinality": "exactly_one",
            "meaning": "Name the entity at the end of the stated real YAGO relation sequence",
        },
        "information_necessity_policy": POLICY,
    }


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


def _loaded_yago_unique(index: YagoKnowledgeIndex, path: list) -> bool:
    """A single selected RDF edge must not hide other same-key outputs."""
    return all(
        {edge.object for edge in index.outgoing.get(fact.subject, ())
         if edge.predicate == fact.predicate} == {fact.object}
        for fact in path
    )


def _semantic_ground(kg, index, plan, *, seed: int, ordinal: int):
    """Choose a human-interpretable relation goal before YAGO facts.

    Both the target and all COMPLETE shadow paths must have exactly one
    output for each required predicate within the LOADED YAGO sample.
    This does not claim global uniqueness in the entire public YAGO.
    """
    rng = random.Random(int(digest(
        seed, ordinal, "task-plan-semantic-ground"
    )[:15], 16))
    n_clues = 2 * sum(
        t["operation"] == "match_clues" for t in plan["macro_tasks"]
    )
    n_cases = n_clues + 4
    # Shared index cache: do not rescan the RDF for every Scene.
    by_name = getattr(index, "_task_plan_semantic_index", None)
    if by_name is None:
        by_name = {}
        for predicate, edges in index.by_predicate.items():
            by_name.setdefault(_predicate_key(predicate), []).extend(edges)
        index._task_plan_semantic_index = by_name
    profiles = [
        recipe for recipe in _SEMANTIC_RECIPES
        if len(recipe[0]) == plan["hops"] and by_name.get(recipe[0][0])
    ]
    if not profiles:
        raise ValueError("no_interpretable_yago_goal_in_loaded_sample")
    rng.shuffle(profiles)
    rejects = {"missing_join": 0, "multivalued": 0,
               "incomplete_shadows": 0, "missing_fork": 0}
    for attempt in range(128):
        keys = profiles[attempt % len(profiles)][0]
        start = rng.choice(by_name[keys[0]])
        path, seen = [start], {start.subject, start.object}
        for key in keys[1:]:
            options = [
                e for e in index.outgoing.get(path[-1].object, ())
                if _predicate_key(e.predicate) == key
                and e.object not in seen and e.subject != e.object
            ]
            if not options:
                break
            step = rng.choice(options)
            path.append(step)
            seen.add(step.object)
        if len(path) != len(keys):
            rejects["missing_join"] += 1
            continue
        if not _loaded_yago_unique(index, path):
            rejects["multivalued"] += 1
            continue
        alternatives = _alternatives(
            kg, index, path, count=n_cases-1, rng=rng,
            candidate_filter=lambda row: _loaded_yago_unique(index, row),
        )
        if not alternatives:
            rejects["incomplete_shadows"] += 1
            continue
        if not all(_loaded_yago_unique(index, row)
                   for row in alternatives):
            rejects["multivalued"] += 1
            continue
        target = "Dossier-" + digest(seed, ordinal, "target")[:10]
        cases = [target] + [
            "Dossier-" + digest(seed, ordinal, i, "case")[:10]
            for i in range(n_cases-1)
        ]
        rng.shuffle(cases)
        paths = {
            target: _serialized_path(kg, path),
            **{case: _serialized_path(kg, row)
               for case, row in zip(
                   [c for c in cases if c != target], alternatives)},
        }
        fork = None
        if plan["mode"] == "verify":
            try:
                fork = mine_required_fork(
                    kg, index, paths, seed=seed, ordinal=ordinal,
                )
            except ValueError:
                rejects["missing_fork"] += 1
                continue
        return {
            "target_case": target, "all_cases": cases,
            "candidate_paths": paths, "real_fork": fork,
            "case_shadow_slots": {
                case: i if i < n_clues else None
                for i, case in enumerate(c for c in cases if c != target)
            },
            "origin_file": str(kg.input_file),
            "sample_triples_scanned": kg.scanned,
            "semantic_audit": {
                "policy": SEMANTIC_GOALS,
                "attempts": attempt + 1,
                "rejections": rejects,
                "loaded_yago_singleton_witness": True,
                "complete_candidate_count": n_cases,
                "rdf_type_verification": False,
            },
        }
    raise ValueError(
        "no_goal_expressible_singleton_yago_with_complete_shadows:"
        + str(rejects)
    )



# Fictional relation worlds reuse approved interpretable relation signatures,
# but no simulated edge is ever represented as a witnessed YAGO fact.
_FICTIONAL_NAMES = (
    "Alder", "Briar", "Cedar", "Dunmere", "Elden", "Farrow",
    "Glenwick", "Hawthorn", "Iver", "Juniper", "Kestrel", "Linden",
)
_FICTIONAL_FIELDS = (
    "river ecology", "historical cartography", "computational linguistics",
    "maritime archaeology", "urban hydrology", "cultural astronomy",
    "digital paleography", "environmental history", "heritage informatics",
    "landscape modeling", "comparative philology", "materials conservation",
)
_PREDICATE_NAMES = {
    "birthplace": "birthPlace", "deathplace": "deathPlace",
    "containedinplace": "containedInPlace", "location": "location",
    "doctoraladvisor": "doctoralAdvisor", "fieldofwork": "fieldOfWork",
    "spouse": "spouse", "memberof": "memberOf",
    "neighbor": "neighbor", "nationality": "nationality",
    "capital": "capital", "twinnedwith": "twinnedWith",
}


def _fictional_label(predicate: str, name: str, case_index: int,
                     hop: int) -> str:
    if predicate in {"doctoraladvisor", "spouse"}:
        return "Dr. " + name + " Merrow"
    if predicate == "fieldofwork":
        return _FICTIONAL_FIELDS[case_index % len(_FICTIONAL_FIELDS)]
    if predicate == "memberof":
        return name + " Research Institute"
    if predicate == "nationality":
        return "Republic of " + name
    if predicate == "capital":
        return name + " City"
    if predicate == "twinnedwith":
        return "City of " + name + "haven"
    if predicate == "containedinplace":
        return ("Province of " if hop > 1 else "County of ") + name
    if predicate in {"birthplace", "deathplace", "location", "neighbor"}:
        return name + (" District" if hop > 1 else " Township")
    return name + " Region"


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


def _fictional_ground(plan, *, seed: int, ordinal: int,
                      real_failure: str) -> dict:
    """Make a coherent synthetic world with complete alternative paths."""
    rng = random.Random(int(digest(
        seed, ordinal, "task-plan-fictional-semantic-world"
    )[:15], 16))
    recipes = [r for r in _SEMANTIC_RECIPES if len(r[0]) == plan["hops"]]
    if not recipes:
        raise ValueError("no_interpretable_fictional_relation_goal")
    keys = rng.choice(recipes)[0]
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
    paths = {}
    for i, case in enumerate(cases):
        name = _FICTIONAL_NAMES[i % len(_FICTIONAL_NAMES)]
        nodes = [
            "urn:maestro:fictional:entity:" + digest(
                seed, ordinal, case, j, "semantic-entity"
            )[:24] for j in range(len(keys)+1)
        ]
        paths[case] = [{
            "subject": nodes[j],
            "predicate": "http://schema.org/" + _PREDICATE_NAMES[predicate],
            "object": nodes[j+1],
            "object_label": _fictional_label(predicate, name, i, j),
            "origin": "synthetic",
        } for j, predicate in enumerate(keys)]
    fork = _fictional_fork(paths) if plan["mode"] == "verify" else None
    return {
        "target_case": target, "all_cases": cases,
        "candidate_paths": paths, "real_fork": fork,
        "case_shadow_slots": {
            case: i if i < n_clues else None
            for i, case in enumerate(c for c in cases if c != target)
        },
        "origin_file": SYNTHETIC_WORLD, "sample_triples_scanned": 0,
        "semantic_audit": {
            "policy": SEMANTIC_GOALS, "attempts": 1, "rejections": {},
            "grounding_origin": "synthetic", "fallback_from_yago": True,
            "fallback_reason": real_failure[:240],
            "loaded_yago_singleton_witness": False,
            "synthetic_consistency_checked": True,
            "complete_candidate_count": n_cases,
            "rdf_type_verification": False,
        },
    }



def _compose_ground(kg, index, plan, *, seed: int, ordinal: int):
    """Compose a task-driven fictional world from *observed YAGO motifs*.

    YAGO supplies relation vocabulary, observed predicate prevalence and
    generic field words; no original entity-to-entity assertion is reassigned
    to any factual person. The new canonical World owns all target/shadow
    paths, and the same original Macro executor consumes its evidence views.
    """
    from .world_motifs import (
        motif_features, compose_relation_goal, compose_typed_world,
    )
    rng = random.Random(int(digest(
        seed, ordinal, "task-plan-world-composition"
    )[:15], 16))
    recipes = tuple(r for r in _SEMANTIC_RECIPES
                    if len(r[0]) == plan["hops"])
    features = motif_features(kg, index, recipes)
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
    paths, world_model = compose_typed_world(
        kg, index, recipe=recipe, cases=cases, seed=seed, ordinal=ordinal,
    )
    fork = (_fictional_fork(paths, source_style="archive")
            if plan["mode"] == "verify" else None)
    world_model["verification_facts"] = (
        {case: copy.deepcopy(row) for case, row
         in fork["candidate_forks"].items()} if fork else {}
    )
    return {
        "target_case": target, "all_cases": cases,
        "candidate_paths": paths, "real_fork": fork,
        "world_model": world_model,
        "case_shadow_slots": {
            case: i if i < n_clues else None
            for i, case in enumerate(c for c in cases if c != target)
        },
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


def _ground(kg, index, plan, *, seed: int, ordinal: int,
            world_mode: str = "witnessed"):
    """Compose new Worlds or preserve historical real-YAGO-first semantics."""
    if world_mode == "compose":
        return _compose_ground(
            kg, index, plan, seed=seed, ordinal=ordinal,
        )
    if world_mode != "witnessed":
        raise ValueError("unknown_task_plan_world_mode")
    if plan.get("semantic_grounding_policy") == SEMANTIC_GOALS:
        try:
            return _semantic_ground(kg, index, plan, seed=seed, ordinal=ordinal)
        except ValueError as exc:
            # Fall back only for legitimate RDF coverage shortfalls. Do not
            # conceal programming / contract errors as fictional Worlds.
            if not str(exc).startswith((
                "no_interpretable_yago_goal_in_loaded_sample",
                "no_goal_expressible_singleton_yago_with_complete_shadows",
            )):
                raise
            return _fictional_ground(
                plan, seed=seed, ordinal=ordinal, real_failure=str(exc),
            )
    rng = random.Random(int(digest(seed, ordinal, "task-plan-ground")[:15], 16))
    n_clues = 2 * sum(
        t["operation"] == "match_clues" for t in plan["macro_tasks"]
    )
    n_cases = n_clues + 4
    for _ in range(110):
        path = index.connected_path(
            rng, min_hops=plan["hops"], max_hops=plan["hops"],
        )
        others = _alternatives(
            kg, index, path, count=n_cases-1, rng=rng,
        )
        if not others:
            continue
        target = "Dossier-" + digest(seed, ordinal, "target")[:10]
        cases = [target] + [
            "Dossier-" + digest(seed, ordinal, i, "case")[:10]
            for i in range(n_cases-1)
        ]
        rng.shuffle(cases)
        candidate_paths = {
            target: _serialized_path(kg, path),
            **{case: _serialized_path(kg, p)
               for case, p in zip(
                   [x for x in cases if x != target], others)}
        }
        fork = None
        if plan["mode"] == "verify":
            try:
                fork = mine_required_fork(
                    kg, index, candidate_paths,
                    seed=seed, ordinal=ordinal,
                )
            except ValueError:
                continue
        return {
            "target_case": target,
            "all_cases": cases,
            "candidate_paths": candidate_paths,
            "real_fork": fork,
            "case_shadow_slots": {
                case: i if i < n_clues else None
                for i, case in enumerate(x for x in cases if x != target)
            },
            "origin_file": str(kg.input_file),
            "sample_triples_scanned": kg.scanned,
        }
    raise ValueError("task_plan_no_complete_real_yago_world")


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
    context = (
        contextual_titles[_semantic_goal(
            grounding["candidate_paths"][grounding["target_case"]],
            len(grounding["all_cases"]),
        )["domain"]]
        if plan.get("semantic_grounding_policy") == SEMANTIC_GOALS
        else None
    )
    for i, task in enumerate(disc):
        for j, (modality, title, attr, choices) in enumerate(CUES[i]):
            if context is not None:
                title = f"{context} — {title}"
            sid = task["source_scope"][j]
            chosen = rng.choice(choices)
            alt = rng.choice([x for x in choices if x != chosen])
            clues[sid] = {
                "value": chosen, "alternative": alt,
                "attribute": attr, "source_slot": 2*i+j,
                "title": title,
            }
            source(sid, modality, title,
                   "Find dossier candidates by an observed clue",
                   {"case_id": "str", "cue": "str"})
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
        source("event_ledger", "table", "Project event ledger",
               "Inspect events subject to an explicit eligibility rule",
               schema)
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
    for i, fact in enumerate(facts, 1):
        input_key = f"entity_{i-1}"
        output_key = "answer" if i == plan["hops"] else f"entity_{i}"
        source(f"relation_{i}", "text",
               f"{predicate_label(fact['predicate'])} relationship index",
               "Look up the documented relationship required by the goal",
               {input_key: "str", output_key: "str"}, key=input_key)
    expected = Counter(
        sid for t in plan["macro_tasks"] for sid in t["source_scope"]
    )
    if expected != Counter(schemas.keys()):
        raise ValueError("task_plan_source_not_bound_exactly_once")
    return clues, catalog, schemas


def make_spec(kg, *, seed: int, ordinal: int, difficulty: str,
              min_hops: int = 2, max_hops: int = 3,
              index: YagoKnowledgeIndex | None = None,
              world_mode: str = "witnessed") -> dict:
    """One contract, one direct World grounding, no nested frozen Specs."""
    if difficulty not in ("medium", "hard"):
        raise ValueError("task_plan_requires_medium_or_hard")
    if not 2 <= min_hops <= max_hops <= 4:
        raise ValueError("task_plan_invalid_hops")
    # Structural proxy calibration: Medium samples the shorter admissible
    # YAGO chain, Hard the longer; modes/independent clues are also
    # conditional on difficulty. This is NOT a learned-agent difficulty score.
    hops = min_hops if difficulty == "medium" else max_hops
    plan = _sample_task_plan(
        seed=seed, ordinal=ordinal, difficulty=difficulty, hops=hops,
        sampling_policy=CALIBRATED, world_mode=world_mode,
    )
    plan["semantic_grounding_policy"] = SEMANTIC_GOALS
    if world_mode == "compose":
        plan["knowledge_world_mode"] = "compose"
    grounding = _ground(
        kg, index or YagoKnowledgeIndex(kg),
        plan, seed=seed, ordinal=ordinal, world_mode=world_mode,
    )
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
    if world_mode == "compose":
        plan["goal"]["meaning"] = (
            "Determine the entity at the end of the documented relation chain"
        )
    condition = plan["condition"]
    clause = (
        "the completed event" if condition and
        condition["field"] == "event_status" else
        "the event held in or after 2025" if condition else
        "the documented project reference"
    )
    if plan.get("semantic_grounding_policy") == SEMANTIC_GOALS:
        topics = {
            "biographical": "biographical collection",
            "academic": "academic correspondence archive",
            "institutional": "institutional records collection",
            "geographical": "regional survey collection",
        }
        semantic = plan["goal"]["semantic"]
        scenario = {
            "topic": topics[semantic["domain"]],
            "setting": (
                "a research archive with personnel files, field records "
                "and independently maintained reference registers"
            ),
            "user_role": "archive researcher",
            "purpose": ("identify " + clause + " and determine the "
                        + semantic["answer_type_hint"]
                        + " specified by the recorded relationships"),
            "motivation": (
                "resolve inconsistent or incomplete references across "
                "separate archival records"
            ),
            "entity_kind": "archive dossier",
        }
    else:
        # Select a grounded *archive domain*, not a fixed template topic.
        # These are fictional collections; the only real historical claims
        # remain the witnessed YAGO relation edges. Do not infer entity type.
        route = [
            predicate_label(fact["predicate"]).lower()
            for fact in grounding["candidate_paths"][grounding["target_case"]]
        ]
        if any(
            key in name for name in route
            for key in ("neighbor", "location", "capital", "contained in", "twinned")
        ):
            topic = "regional research archive"
        elif any(
            key in name for name in route
            for key in ("birth", "student", "spouse", "influenced")
        ):
            topic = "biographical research archive"
        elif any(
            key in name for name in route
            for key in ("member", "institution", "works at", "organization")
        ):
            topic = "institutional research archive"
        else:
            topic = "cross-reference project archive"
        scenario = {
            "topic": topic,
            "setting": "a fictional project archive cross-referenced with real YAGO facts",
            "user_role": "records researcher",
            "purpose": "identify " + clause + " and trace its real-world relationship",
            "motivation": "resolve independently located observations before reporting a verified related entity",
            "entity_kind": "fictional project",
        }
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
            from .world_motifs import validate_composed_world
            issues = validate_composed_world(binding)
            if issues:
                return issues
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
                from .real_motif_grounding import validate_frozen_required_fork
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
        if plan["macro_tasks"] != _sample_task_plan(
            seed=spec["seed"], ordinal=spec["ordinal"],
            difficulty=spec["difficulty_target"], hops=plan["hops"],
            sampling_policy=sampling, world_mode=world_mode,
        )["macro_tasks"]:
            return ["task_plan_graph_not_from_information_needs"]
        by_id, scopes = {}, []
        for task in plan["macro_tasks"]:
            if (task["input_contract"] != {
                    pid: by_id[pid]["output_schema"]
                    for pid in task["depends_on"]
                }):
                return ["task_plan_input_contract_mismatch"]
            scopes += task["source_scope"]
            by_id[task["macro_id"]] = task
        if (len(by_id) > 9
                or plan["terminal_macro_id"] != plan["macro_tasks"][-1]["macro_id"]
                or plan["macro_tasks"][-1]["output_schema"] != {"answer": "str"}
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
