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


def _sample_task_plan(*, seed: int, ordinal: int, difficulty: str,
                      hops: int, sampling_policy: str | None = None) -> dict:
    """Choose typed information work before consulting World facts.

    Legacy samples retain their exact old deterministic choices. The new
    sampling policy gives Hard samples more independent information work,
    without treating this proxy as measured live-Agent hardness.
    """
    if sampling_policy == CALIBRATED:
        if difficulty not in {"medium", "hard"}:
            raise ValueError("unsupported_task_plan_difficulty")
        mode = (
            ("trace", "event") if difficulty == "medium"
            else ("event", "verify")
        )[(seed + ordinal) % 2]
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


def _ground(kg, index, plan, *, seed: int, ordinal: int):
    """Use ONLY pure, validated YAGO samplers, never historical Spec writers."""
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
    for i, task in enumerate(disc):
        for j, (modality, title, attr, choices) in enumerate(CUES[i]):
            sid = task["source_scope"][j]
            chosen = rng.choice(choices)
            alt = rng.choice([x for x in choices if x != chosen])
            clues[sid] = {
                "value": chosen, "alternative": alt,
                "attribute": attr, "source_slot": 2*i+j,
                "title": title,
            }
            source(sid, modality, title+" (simulated)",
                   "Find fictional dossier candidates by an observed clue",
                   {"case_id": "str", "cue": "str"})
    source("entity_index", "table", "Project identity crosswalk",
           "Map the identified fictional dossier to one named project",
           {"case_id": "str", "project_id": "str", "project_name": "str"},
           key="case_id")
    mode = plan["mode"]
    if mode == "trace":
        source("project_ref", "text", "Project YAGO reference register",
               "Resolve the project's documented real entity",
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
        source("event_ref", "text", "Event-specific YAGO reference",
               "Follow the selected event's independently indexed entity",
               {"event_id": "str", "entity_0": "str"}, key="event_id")
    if mode == "verify":
        for sid in ("link_a", "link_b"):
            source(sid, "text", f"Simulated project reference hypotheses {sid[-1].upper()}",
                   "Investigate competing fictional source claims",
                   {"project_id": "str", "route_key": "str", "entity_0": "str"})
        source("real_fork", "text", "Actual YAGO secondary relation",
               "Verify the candidate entity's genuine secondary YAGO fact",
               {"entity_0": "str", "motif_object": "str"}, key="entity_0")
    facts = grounding["candidate_paths"][grounding["target_case"]]
    for i, fact in enumerate(facts, 1):
        input_key = f"entity_{i-1}"
        output_key = "answer" if i == plan["hops"] else f"entity_{i}"
        source(f"relation_{i}", "text",
               f"YAGO {predicate_label(fact['predicate'])} relation",
               "Look up the real YAGO relation required by the goal",
               {input_key: "str", output_key: "str"}, key=input_key)
    expected = Counter(
        sid for t in plan["macro_tasks"] for sid in t["source_scope"]
    )
    if expected != Counter(schemas.keys()):
        raise ValueError("task_plan_source_not_bound_exactly_once")
    return clues, catalog, schemas


def make_spec(kg, *, seed: int, ordinal: int, difficulty: str,
              min_hops: int = 2, max_hops: int = 3,
              index: YagoKnowledgeIndex | None = None) -> dict:
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
        sampling_policy=CALIBRATED,
    )
    grounding = _ground(
        kg, index or YagoKnowledgeIndex(kg),
        plan, seed=seed, ordinal=ordinal,
    )
    clues, sources, schemas = _bind_sources(
        plan, grounding, seed=seed, ordinal=ordinal,
    )
    plan.update({
        "clues": clues, "source_schemas": schemas,
        "world_binding": grounding,
    })
    condition = plan["condition"]
    clause = (
        "the completed event" if condition and
        condition["field"] == "event_status" else
        "the event held in or after 2025" if condition else
        "the documented project reference"
    )
    scenario = {
        "topic": "project reference investigation",
        "setting": "a fictional project archive cross-referenced with real YAGO facts",
        "user_role": "records researcher",
        "purpose": "identify " + clause + " and trace its real-world relationship",
        "motivation": "resolve independently located observations before reporting a verified related entity",
        "entity_kind": "fictional project",
    }
    witness = {
        "version": "maestro.task_plan.yago_witness.v1",
        "origin_file": grounding["origin_file"],
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
            "dataset": "YAGO-4.6-plus-fictional-project",
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
                or witness["version"] != "maestro.task_plan.yago_witness.v1"
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
        predicates = [f["predicate"] for f in paths[target]]
        if len(predicates) != plan["hops"]:
            return ["task_plan_wrong_real_hops"]
        labels = set()
        each_hop_subjects = [set() for _ in predicates]
        for case, facts in paths.items():
            if len(facts) != len(predicates):
                return ["task_plan_incomplete_real_path"]
            for i, fact in enumerate(facts):
                if fact["origin"] != "yago" or fact["predicate"] != predicates[i]:
                    return ["task_plan_mixed_real_predicate"]
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
            from .real_motif_grounding import validate_frozen_required_fork
            problems = validate_frozen_required_fork(binding["real_fork"], paths)
            if problems:
                return problems
        elif binding["real_fork"] is not None:
            return ["task_plan_unneeded_fork"]
        if witness["real_fork"] != binding["real_fork"]:
            return ["task_plan_fork_witness_mismatch"]
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
            sampling_policy=sampling,
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
