"""Versioned YAGO-knowledge Macro plans compiled by the ORIGINAL Stage-1 engine.

This is the first *supported family* of the flexible Macro direction:
independently grounded discovery investigations -> identification -> 2/3-hop
knowledge investigation (with variable Macro coarsening). It preserves
the old relational Program IR, source permissions, FactRefs, handoff tests,
and Stage-2 WorkProduct format. Other typed role combinations are not
silently claimed as supported.

A frozen source_spec contains only explicitly witnessed YAGO facts; it is
replayable without reading local YAGO again, but origin authenticity must
be checked against the user-provided RDF at initial construction.
"""
from __future__ import annotations

import copy
import random
import re
from collections import defaultdict

from environment.schema import digest
from .typed_role_grammar import OUTPUTS

class YagoKnowledgeIndex:
    """Build once for a large YAGO dump; share indexes across all scenes.

    Neither the index nor the full YAGO corpus enters a frozen ProblemSpec.
    Each scene records only its actual entity-to-entity factual witnesses.
    """

    def __init__(self, graph):
        self.graph = graph
        self.outgoing = graph.outgoing()
        self.by_predicate = defaultdict(list)
        for fact in graph.edges:
            self.by_predicate[fact.predicate].append(fact)
        self.joinable = [
            fact for fact in graph.edges
            if any(
                next_fact.object != fact.subject
                and next_fact.object != fact.object
                for next_fact in self.outgoing.get(fact.object, ())
            )
        ]

    def connected_path(self, rng, *, min_hops: int, max_hops: int):
        if not 1 <= min_hops <= max_hops <= 5:
            raise ValueError("invalid_knowledge_hop_limits")
        lengths = list(range(min_hops, max_hops + 1))
        rng.shuffle(lengths)
        lengths.sort(key=lambda n: n > 1, reverse=True)
        for target in lengths:
            choices = self.joinable if target > 1 else self.graph.edges
            if not choices:
                continue
            for _ in range(120):
                start = rng.choice(choices)
                path, visited = [start], {start.subject, start.object}
                while len(path) < target:
                    following = [
                        e for e in self.outgoing.get(path[-1].object, ())
                        if e.object not in visited and e.subject != e.object
                    ]
                    if not following:
                        break
                    fact = rng.choice(following)
                    path.append(fact)
                    visited.add(fact.object)
                if len(path) == target:
                    return path
        raise ValueError("no_witnessed_yago_path_for_requested_depth")


VERSION = "maestro.macro_plan.typed_knowledge.v1"
KNOWLEDGE_VERSION = "maestro.yago_witness.v1"
PATTERN = "typed_knowledge_v1"
CUES = (
    (
        ("image", "photographed archive folio", "seal motif",
         ("spiral", "star", "wave", "triangle", "crescent")),
        ("table", "reference drawer index", "drawer code",
         ("cedar", "marble", "cobalt", "amber", "elm")),
    ),
    (
        ("audio", "spoken catalog record", "shelf code",
         ("delta", "north", "juniper", "harbor", "ridge")),
        ("text", "curatorial cross-reference", "document series",
         ("atlas", "heritage", "survey", "ledger", "fieldwork")),
    ),
    (
        ("video", "archival reel observation", "reel motif",
         ("opening", "handoff", "arrival", "departure", "closing")),
        ("text", "independent register notes", "record class",
         ("accession", "inscription", "receipt", "memo", "catalog")),
    ),
)


def predicate_label(iri: str) -> str:
    tail = iri.rstrip("/#").rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", tail).replace("_", " ").strip()


def make_spec(kg, *, seed: int, ordinal: int, difficulty: str,
              min_hops: int = 2, max_hops: int = 3,
              index: YagoKnowledgeIndex | None = None) -> dict:
    """Ground a versioned Macro graph on witnessed, connected YAGO facts."""
    if difficulty not in {"medium", "hard"}:
        raise ValueError("knowledge_v1_supports_medium_or_hard")
    rng = random.Random(int(digest(seed, ordinal, "knowledge-canonical-v1")[:15], 16))
    index = index or YagoKnowledgeIndex(kg)
    path = index.connected_path(rng, min_hops=min_hops, max_hops=max_hops)
    n_discovery = rng.randint(2, 3)
    total_clues = 2 * n_discovery
    target = "Ref-" + str(rng.randrange(100000, 999999))
    cases = [target] + [f"Ref-{rng.randrange(100000, 999999)}-D{i}"
                        for i in range(total_clues + 3)]
    if len(set(cases)) != len(cases):
        raise ValueError("duplicate_fictional_dossier_ids")
    rng.shuffle(cases)
    clues = {}
    source_roles = []
    schemas = {}
    tasks = []

    def source(sid, modality, schema, title, purpose, *, key=None):
        meta = {
            "source_id": sid, "modality": modality,
            "title": title, "purpose": purpose,
            "access_mode": "keyed_lookup" if key else "browse",
        }
        if key:
            meta["key_field"] = key
        source_roles.append(meta)
        schemas[sid] = schema

    def macro(role, parents, output, scoped, *, detail):
        mid = "M" + str(len(tasks) + 1)
        tasks.append({
            "macro_id": mid, "role": role, "depends_on": list(parents),
            "input_contract": {
                pid: next(task["output_schema"] for task in tasks
                          if task["macro_id"] == pid)
                for pid in parents
            },
            "output_schema": dict(output),
            "source_scope": list(scoped),
            "solver_template": role,
            "evidence_plan": {
                "description": detail,
                "source_observation_targets": {
                    sid: len(cases) if sid.endswith("_a") or sid.endswith("_b")
                    else 1 for sid in scoped
                },
            },
        })
        return mid

    for i in range(n_discovery):
        mid = "M" + str(i + 1)
        ids = []
        for j, (modality, title, attribute, choices) in enumerate(CUES[i]):
            sid = f"{mid}_{'a' if j == 0 else 'b'}"
            selected = rng.choice(choices)
            alternative = rng.choice([v for v in choices if v != selected])
            clues[sid] = {
                "value": selected, "alternative": alternative,
                "attribute": attribute, "source_slot": i * 2 + j,
                "title": title,
            }
            source(sid, modality, {"case_id": "str", "cue": "str"},
                   f"{title} ({'simulated' if modality in ('image','video','audio') else 'fictional'} source)",
                   "compare the archival dossier identifiers using the recorded cue")
            ids.append(sid)
        macro("discovery", [], {"case_id": "str"}, ids,
              detail="combine two separately scoped dossier observations")

    source("entity_index", "table", {"case_id": "str", "entity_0": "str"},
           "fictional dossier to YAGO entity crosswalk",
           "map a uniquely identified fictional dossier to its linked reference entity",
           key="case_id")
    parent_ids = [t["macro_id"] for t in tasks]
    last = macro("identification", parent_ids, {"entity_0": "str"},
                 ["entity_index"], detail="merge all candidate investigations")

    # The same real path is shared by all task variations. Coarse Macro
    # boundaries group 1-2 real relationships instead of forcing one
    # remote relation query to equal one dispatchable Macro.
    hops = len(path)
    relation_groups = [[i for i in range(1, hops + 1)]]
    if hops >= 3 and rng.random() < 0.55:
        cut = rng.randint(1, hops - 1)
        relation_groups = [list(range(1, cut + 1)),
                           list(range(cut + 1, hops + 1))]
    facts = []
    sources = {}
    for i, edge in enumerate(path, 1):
        sid = f"relation_{i}"
        prior_key = f"entity_{i-1}"
        dest_key = "answer" if i == hops else f"entity_{i}"
        source(sid, "text", {prior_key: "str", dest_key: "str"},
               f"YAGO relation evidence {i}: {predicate_label(edge.predicate)}",
               "retrieve the related entity from a documented YAGO assertion",
               key=prior_key)
        # Grounded witness is a genuine RDF triple, with no invented
        # second publisher/source for the same YAGO assertion.
        facts.append({
            "subject": edge.subject, "predicate": edge.predicate,
            "object": edge.object,
            "subject_label": kg.label(edge.subject),
            "object_label": kg.label(edge.object),
            "source_id": sid,
            "origin": "yago",
        })
        # Every decoy row also originates from an existing YAGO triple.
        alternatives = [e for e in index.by_predicate[edge.predicate]
                        if e.subject != edge.subject
                        and e.object != edge.object]
        rng.shuffle(alternatives) if len(alternatives) < 150 else None
        picked = set()
        extra = []
        attempts = 0
        while len(extra) < 3 and attempts < min(500, len(alternatives)):
            attempts += 1
            other = (alternatives.pop() if len(alternatives) < 150
                     else rng.choice(alternatives))
            if other.subject in picked:
                continue
            picked.add(other.subject)
            extra.append({
                "subject": other.subject, "predicate": other.predicate,
                "object": other.object,
                "object_label": kg.label(other.object),
                "origin": "yago",
            })
        sources[sid] = extra
    for group in relation_groups:
        scoped = [f"relation_{i}" for i in group]
        last_step = group[-1]
        goal_key = "answer" if last_step == hops else f"entity_{last_step}"
        last = macro("investigation", [last], {goal_key: "str"},
                     scoped, detail="trace a connected sequence of YAGO facts")

    plan = {
        "version": VERSION, "kind": "knowledge_relation_trace",
        "macro_tasks": tasks, "terminal_macro_id": last,
        "source_schemas": schemas, "clues": clues,
        "target_case": target, "all_cases": cases,
        "entity_anchor": path[0].subject,
        "real_facts": facts, "decoy_yago_facts": sources,
        "relation_groups": relation_groups,
        "goal": {
            "answer_field": "answer", "answer_type": "str",
            "public_label": "related entity", "cardinality": "exactly_one",
            "meaning": "name of the entity reached through the documented relation chain",
        },
    }
    snapshot = {
        "version": KNOWLEDGE_VERSION,
        "origin_file": str(kg.input_file),
        "sample_triples_scanned": kg.scanned,
        "facts": [
            {"subject": x["subject"], "predicate": x["predicate"],
             "object": x["object"]}
            for x in facts
        ],
    }
    spec = {
        "version": "maestro.problem_spec.v4",
        "task_id": f"question-kn-{ordinal:06d}-{digest(seed, ordinal, target)[:8]}",
        "scene_id": f"yago-knowledge-{ordinal:06d}",
        "source_anchor": {
            "dataset": "YAGO-4.6-facts-with-synthetic-dossier",
            "source_record_id": str(ordinal),
            "original_question_reused": False,
            "original_answer_reused": False,
            "original_media_used": False,
        },
        "difficulty_target": difficulty,
        "family": PATTERN, "reasoning_pattern": PATTERN,
        "scenario": {
            "topic": "archival reference collection",
            "setting": "a reference office investigating independently cataloged dossiers",
            "user_role": "archive researcher",
            "purpose": "identify the documented entity connected to an otherwise unknown reference",
            "motivation": "combine dispersed evidence before consulting a verified relation trail",
            "entity_kind": "archival dossier",
        },
        "sources": source_roles,
        "requirements": {
            "answer_kind": "macro_value",
            "must_resolve_aliases": False,
            "must_match_independent_reviews": False,
            "must_check_independent_clearance": False,
            "must_preserve_source_provenance": True,
            "simulation_only": True,
        },
        "seed": seed, "ordinal": ordinal,
        "macro_plan": plan, "knowledge_witness": snapshot,
    }
    spec["knowledge_contract_hash"] = digest(
        plan, snapshot, seed, ordinal, "canonical-knowledge-frozen-v1"
    )
    problems = validate_spec(spec)
    if problems:
        raise ValueError("invalid_knowledge_spec:" + ",".join(problems))
    return spec


def validate_spec(spec: dict) -> list[str]:
    """Fail closed on malformed, disconnected or modified frozen plans."""
    try:
        plan = spec["macro_plan"]
        if plan["version"] != VERSION or spec["reasoning_pattern"] != PATTERN:
            return ["wrong_knowledge_plan_version"]
        witness = spec["knowledge_witness"]
        if witness["version"] != KNOWLEDGE_VERSION:
            return ["wrong_knowledge_witness_version"]
        if spec["knowledge_contract_hash"] != digest(
            plan, witness, spec["seed"], spec["ordinal"],
            "canonical-knowledge-frozen-v1"
        ):
            return ["knowledge_plan_hash_mismatch"]
        tasks = plan["macro_tasks"]
        if not 2 <= len([t for t in tasks if t["role"] == "discovery"]) <= 3:
            return ["knowledge_discovery_budget_invalid"]
        if tasks[-1]["macro_id"] != plan["terminal_macro_id"]:
            return ["knowledge_terminal_changed"]
        if set(plan["source_schemas"]) != {s["source_id"] for s in spec["sources"]}:
            return ["knowledge_source_schema_mismatch"]
        seen, scopes = {}, []
        for t in tasks:
            if t["macro_id"] in seen or any(p not in seen for p in t["depends_on"]):
                return ["knowledge_non_topological_macro"]
            expected = {p: seen[p]["output_schema"] for p in t["depends_on"]}
            if t["input_contract"] != expected:
                return ["knowledge_handoff_schema_mismatch"]
            if any(sid not in plan["source_schemas"] for sid in t["source_scope"]):
                return ["knowledge_unknown_source"]
            scopes += t["source_scope"]
            seen[t["macro_id"]] = t
        if sorted(scopes) != sorted(plan["source_schemas"]):
            return ["knowledge_unused_or_repeated_source"]
        facts = plan["real_facts"]
        if (len(facts) < 2 or len(facts) != len(witness["facts"])
                or any(
                    {k: fact[k] for k in ("subject", "predicate", "object")}
                    != expected
                    for fact, expected in zip(facts, witness["facts"])
                )):
            return ["knowledge_provenance_witness_mismatch"]
        if any(facts[i]["object"] != facts[i+1]["subject"]
               for i in range(len(facts)-1)):
            return ["knowledge_disconnected_yago_path"]
        if plan["entity_anchor"] != facts[0]["subject"]:
            return ["knowledge_incorrect_start_entity"]
        groups = plan["relation_groups"]
        if [i for group in groups for i in group] != list(range(1, len(facts)+1)):
            return ["knowledge_relation_partition_invalid"]
        if len([t for t in tasks if t["role"] == "investigation"]) != len(groups):
            return ["knowledge_relation_macro_partition_changed"]
        if tasks[-1]["output_schema"] != {"answer": "str"}:
            return ["knowledge_goal_output_changed"]
        if len(plan["all_cases"]) != len(set(plan["all_cases"])):
            return ["knowledge_duplicate_case"]
        if plan["target_case"] not in plan["all_cases"]:
            return ["knowledge_missing_target_case"]
        return []
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return ["invalid_knowledge_contract:" + str(exc)[:130]]


def compile_macro_solvers(spec: dict) -> dict:
    """Lower coarse Macro plans to the LEGACY relation-based primitive IR."""
    problems = validate_spec(spec)
    if problems:
        raise ValueError("invalid_knowledge_plan:" + ",".join(problems))
    plan = spec["macro_plan"]
    nodes, groups, output_ids = [], [], {}

    for task in plan["macro_tasks"]:
        local = []
        mid, role = task["macro_id"], task["role"]
        parent_nodes = [output_ids[p] for p in task["depends_on"]]
        def add(op, *inputs, **kwargs):
            node_id = f"{mid.lower()}_op{len(local)+1:02d}"
            local.append({"id": node_id, "op": op,
                          "inputs": list(inputs), **kwargs})
            return node_id
        if role == "discovery":
            local_matches = []
            for sid in task["source_scope"]:
                acquired = add("scan", source_id=sid)
                filt = add("filter_eq", acquired, field="cue",
                           value=plan["clues"][sid]["value"])
                local_matches.append(add("project", filt, fields=["case_id"]))
            join = add("join", local_matches[0], local_matches[1],
                       left_on="case_id", right_on="case_id")
            last = add("unique", join)
        elif role == "identification":
            if len(parent_nodes) < 2:
                raise ValueError("knowledge_identity_requires_independent_branches")
            joined = parent_nodes[0]
            for next_node in parent_nodes[1:]:
                joined = add("join", joined, next_node,
                             left_on="case_id", right_on="case_id")
            distinct = add("unique", joined)
            acquired = add("lookup_by_key", distinct, source_id="entity_index",
                           input_key_field="case_id", source_key_field="case_id")
            last = add("project", acquired, fields=["entity_0"])
        elif role == "investigation":
            if len(parent_nodes) != 1:
                raise ValueError("knowledge_investigation_requires_one_subject")
            last = parent_nodes[0]
            for sid in task["source_scope"]:
                idx = int(sid.split("_")[-1])
                key = f"entity_{idx-1}"
                last = add("lookup_by_key", last, source_id=sid,
                           input_key_field=key, source_key_field=key)
            last = add("project", last, fields=list(task["output_schema"]))
        else:
            raise ValueError("unsupported_knowledge_role:" + role)
        nodes.extend(local)
        groups.append({
            "macro_id": mid,
            "node_ids": [row["id"] for row in local],
            "output_node_id": last,
            "external_input_nodes": parent_nodes,
        })
        output_ids[mid] = last
    return {
        "version": "maestro.reasoning_program.v4",
        "semantic_operator_pattern": PATTERN,
        "nodes": nodes,
        "output": output_ids[plan["terminal_macro_id"]],
        "authorized_source_ids": [s["source_id"] for s in spec["sources"]],
        "frozen_problem_spec_id": spec["task_id"],
        "macro_solver_groups": groups,
        "macro_plan_version": VERSION,
    }


def generate_world(spec: dict) -> dict:
    """Regenerate synthetic evidence deterministically from frozen RDF facts."""
    from .world import _rows, WORLD_VERSION
    problems = validate_spec(spec)
    if problems:
        raise ValueError("invalid_knowledge_world:" + ",".join(problems))
    plan = spec["macro_plan"]
    rng = random.Random(int(digest(
        spec["seed"], spec["task_id"], "knowledge-world-v1")[:15], 16))
    cases = list(plan["all_cases"])
    target = plan["target_case"]
    n_clues = len(plan["clues"])
    records = {}
    for sid, clue in plan["clues"].items():
        slot = clue["source_slot"]
        value, other = clue["value"], clue["alternative"]
        observations = []
        for case in cases:
            if case == target:
                ok = True
            elif case.endswith("-D" + str(slot)):
                ok = False
            elif "-D" in case:
                ok = True
            else:
                ok = False
            observations.append({
                "case_id": case, "cue": value if ok else other,
            })
        rng.shuffle(observations)
        records[sid] = observations
    index_rows = [
        {"case_id": case,
         "entity_0": plan["entity_anchor"] if case == target
         else "urn:maestro:synthetic:decoy:" + case}
        for case in cases
    ]
    rng.shuffle(index_rows)
    records["entity_index"] = index_rows
    for i, fact in enumerate(plan["real_facts"], 1):
        sid = f"relation_{i}"
        key = f"entity_{i-1}"
        value_field = "answer" if i == len(plan["real_facts"]) else f"entity_{i}"
        rows = [{
            key: fact["subject"],
            value_field: fact["object_label"] if value_field == "answer"
                         else fact["object"],
        }]
        for alternate in plan["decoy_yago_facts"].get(sid, []):
            rows.append({
                key: alternate["subject"],
                value_field: alternate["object_label"] if value_field == "answer"
                             else alternate["object"],
            })
        rng.shuffle(rows)
        records[sid] = rows
    sources = {}
    for role in spec["sources"]:
        sid = role["source_id"]
        source = _rows(
            sid, role["modality"], role["title"], role["purpose"],
            records[sid],
        )
        source["access_policy"] = {
            "mode": role["access_mode"],
            **({"key_field": role["key_field"]}
               if role["access_mode"] == "keyed_lookup" else {}),
        }
        if sid.startswith("relation_"):
            original = plan["real_facts"][int(sid.split("_")[-1])-1]
            source["evidence_origin_id"] = "YAGO-4.6:" + digest(
                original["subject"], original["predicate"], original["object"]
            )[:20]
            source["knowledge_fact"] = {
                key: original[key] for key in ("subject", "predicate", "object")
            }
        sources[sid] = source
    return {
        "version": WORLD_VERSION, "simulated_only": True,
        "sources": sources,
        "entities": [{
            "hidden_entity_id": target,
            "identity_source": "frozen_yago_reference_witness",
        }],
        "generation_seed": spec["seed"],
        "knowledge_witness": copy.deepcopy(spec["knowledge_witness"]),
        "value_sampling_contract": "frozen_yago_facts_plus_simulated_evidence_v1",
    }


def mutate_witness(sources: dict, sid: str, answer_rows: list[dict],
                   spec: dict) -> dict:
    """Controlled single-source modification for the original quality gate."""
    revised = copy.deepcopy(sources)
    refs = {ref for row in answer_rows for ref in row["source_fact_refs"]
            if ref.startswith(sid + ":")}
    if not refs:
        raise ValueError("knowledge_missing_source_witness:" + sid)
    selected = next((o for o in revised[sid]["observations"]
                     if set(o["fact_refs"]) & refs), None)
    if selected is None:
        raise ValueError("knowledge_mutation_no_cited_fact:" + sid)
    values = selected["values"]
    if sid in spec["macro_plan"]["clues"]:
        values["cue"] = "__missing_archival_clue__"
    elif sid == "entity_index":
        values["entity_0"] = "urn:maestro:synthetic:wrong-entity"
    elif sid.startswith("relation_"):
        idx = int(sid.split("_")[-1])
        last = idx == len(spec["macro_plan"]["real_facts"])
        key = "answer" if last else f"entity_{idx}"
        values[key] = (
            values[key] + " (counterfactual)" if last
            else "urn:maestro:synthetic:broken-relation"
        )
    else:
        raise ValueError("unsupported_knowledge_mutation:" + sid)
    return revised
