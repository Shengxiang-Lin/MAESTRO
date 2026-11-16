"""Hybrid real + synthetic World, compiled by canonical Stage-1 (v3).

Real:
  - A typed chain of genuinely observed YAGO facts for every candidate.
  - A separately mined real YAGO chain/fork/diamond motif (context only).
Synthetic:
  - Fictional projects/activities and *fictional* project->real-entity links.
  - Two simulated source directories with competing candidate claims.
  - Neither individual directory uniquely identifies the correct YAGO anchor.

The Macro graph can be serial investigation->verification OR two parallel
investigations->verification/join, followed by real YAGO relation investigation.
This is an incremental supported subgrammar, not arbitrary six-role DAGs.
All v1/v2 frozen replay and original quality/Stage2 interfaces are retained.
"""
from __future__ import annotations

import copy
import random
from collections import Counter

from environment.schema import digest
from . import knowledge_macro as v1
from . import knowledge_macro_v2 as v2
from .world_motifs import mine_motif, validate_frozen_motif

PATTERN = "typed_knowledge_v3"
VERSION = "maestro.macro_plan.typed_knowledge.v3"
KNOWLEDGE_VERSION = "maestro.yago_witness.v3"
HASH_SALT = "canonical-hybrid-world-motif-v3"
NECESSITY_POLICY = "hybrid_world_replanning_v3"

PROJECT_TYPES = (
    ("fictional research initiatives", "research_project", "Research study"),
    ("fictional exhibition programs", "exhibition", "Exhibition"),
    ("fictional field surveys", "field_survey", "Field survey"),
    ("fictional heritage projects", "heritage_project", "Heritage project"),
)
ADJECTIVES = (
    "Amber", "Lantern", "Mosaic", "Willow", "Violet", "Cedar",
    "Harbor", "Compass", "Cobalt", "Aurora", "Meridian", "Atlas",
)


def _source(sid: str, *, modality: str, title: str, purpose: str,
            key: str | None = None) -> dict:
    source = {
        "source_id": sid, "modality": modality,
        "title": title, "purpose": purpose,
        "access_mode": "keyed_lookup" if key else "browse",
    }
    if key is not None:
        source["key_field"] = key
    return source


def _route_key(project: str, real_uri: str) -> str:
    # Both sources must have the SAME project AND real-entity candidate.
    return "route:" + digest(project, real_uri, "hybrid-cross-source-v3")[:24]


def _macro(tasks: list, role: str, parents: list[str],
           output: dict[str, str], scope: list[str], detail: str) -> str:
    mid = f"M{len(tasks)+1}"
    prior = {task["macro_id"]: task for task in tasks}
    if any(parent not in prior for parent in parents):
        raise ValueError("hybrid_macro_invalid_parent")
    tasks.append({
        "macro_id": mid, "role": role,
        "depends_on": list(parents),
        "input_contract": {
            parent: copy.deepcopy(prior[parent]["output_schema"])
            for parent in parents
        },
        "output_schema": dict(output),
        "source_scope": list(scope),
        "solver_template": role,
        "evidence_plan": {
            "description": detail,
            "source_observation_targets": {
                sid: 1 for sid in scope
            },
        },
    })
    return mid


def make_spec(kg, *, seed: int, ordinal: int, difficulty: str,
              min_hops: int = 2, max_hops: int = 3,
              index: v1.YagoKnowledgeIndex | None = None) -> dict:
    """Freeze a real motif and a self-consistent fictional extension.

    Uses the v2 counterexample-safe selection of full YAGO candidate
    paths; no extra YAGO relation is invented or misattributed.
    """
    index = index or v1.YagoKnowledgeIndex(kg)
    backbone = v2.make_spec(
        kg, seed=seed, ordinal=ordinal, difficulty=difficulty,
        min_hops=min_hops, max_hops=max_hops, index=index,
    )
    spec = copy.deepcopy(backbone)
    spec["hybrid_backbone_v2"] = copy.deepcopy(backbone)
    rng = random.Random(f"maestro-hybrid-world-v3:{seed}:{ordinal}")
    topic, kind, prefix = PROJECT_TYPES[rng.randrange(len(PROJECT_TYPES))]
    shape = "parallel_fork_join" if rng.random() < 0.5 else "serial_refinement"
    plan = spec["macro_plan"]
    n_disc = sum(task["role"] == "discovery" for task in plan["macro_tasks"])
    candidates = list(plan["all_cases"])
    projects = {}
    for i, case in enumerate(candidates):
        project_id = (
            "urn:maestro:fictional:" + kind + ":" +
            digest(seed, ordinal, case, "project-v3")[:20]
        )
        projects[case] = {
            "entity_id": project_id,
            "entity_type": kind,
            "display_name": f"{prefix} {ADJECTIVES[(i+ordinal) % len(ADJECTIVES)]} {i+1}",
            "origin": "synthetic",
            "fictional": True,
        }
    # Each fictional project actually refers to one real YAGO subject.
    # The two simulated records additionally contain DIFFERENT false
    # candidate associations. Only their intersection proves which is
    # the intended reference. False records are claims, not World facts.
    claims = {"link_a": [], "link_b": []}
    truth = []
    anchors = {
        case: plan["candidate_paths"][case][0]["subject"]
        for case in candidates
    }
    for i, case in enumerate(candidates):
        project_id = projects[case]["entity_id"]
        correct = anchors[case]
        if len(candidates) < 4:
            raise ValueError("hybrid_not_enough_competing_candidates")
        alternates = [
            anchors[candidates[(i+1) % len(candidates)]],
            anchors[candidates[(i+2) % len(candidates)]],
        ]
        if len({correct, *alternates}) != 3:
            raise ValueError("hybrid_competing_anchors_not_distinct")
        truth.append({
            "subject": project_id,
            "predicate": "urn:maestro:synthetic:documentsYagoReference",
            "object": correct,
            "origin": "synthetic", "truth_scope": "fictional_project_world",
        })
        for sid, misleading in (("link_a", alternates[0]),
                                ("link_b", alternates[1])):
            for real in (correct, misleading):
                claims[sid].append({
                    "project_id": project_id,
                    "entity_0": real,
                    "route_key": _route_key(project_id, real),
                })
    rng.shuffle(claims["link_a"])
    rng.shuffle(claims["link_b"])
    motif = mine_motif(
        kg, index, plan["real_facts"], seed=seed, ordinal=ordinal,
    )
    old_tasks = plan["macro_tasks"]
    tasks = copy.deepcopy(old_tasks[:n_disc])
    identity = _macro(
        tasks, "identification",
        [t["macro_id"] for t in tasks],
        {"project_id": "str"}, ["entity_index"],
        "Join simulated clue investigations and resolve one fictional project",
    )
    a = _macro(
        tasks, "investigation", [identity],
        {"route_key": "str", "entity_0": "str"}, ["link_a"],
        "Read project reference candidates in the first simulated directory",
    )
    if shape == "parallel_fork_join":
        b = _macro(
            tasks, "investigation", [identity],
            {"route_key": "str", "entity_0": "str"}, ["link_b"],
            "Independently read candidate references from the second directory",
        )
        anchor_mid = _macro(
            tasks, "verification", [a, b],
            {"entity_0": "str"}, [],
            "Crosscheck both directory candidate sets and retain their unique shared YAGO entity",
        )
    else:
        anchor_mid = _macro(
            tasks, "verification", [a],
            {"entity_0": "str"}, ["link_b"],
            "Use the second simulated source to disambiguate the first source candidates",
        )
    for group in plan["relation_groups"]:
        scope = [f"relation_{i}" for i in group]
        out = "answer" if group[-1] == len(plan["real_facts"]) \
            else f"entity_{group[-1]}"
        anchor_mid = _macro(
            tasks, "investigation", [anchor_mid],
            {out: "str"}, scope,
            "Traverse frozen real YAGO relations using the confirmed entity reference",
        )

    # Every source is still actually consumed by the frozen Macro Solver.
    schemas = copy.deepcopy(plan["source_schemas"])
    schemas["entity_index"] = {"case_id": "str", "project_id": "str"}
    schemas["link_a"] = {"project_id": "str", "entity_0": "str",
                         "route_key": "str"}
    schemas["link_b"] = copy.deepcopy(schemas["link_a"])
    spec["sources"] = copy.deepcopy(backbone["sources"])
    for entry in spec["sources"]:
        if entry["source_id"] == "entity_index":
            entry["title"] = "Fictional project identity crosswalk"
            entry["purpose"] = "Resolve the project associated with the jointly identified case"
    spec["sources"].extend([
        _source("link_a", modality="table",
                title="Simulated project-to-entity reference index A",
                purpose="Find possible links between fictional projects and YAGO references"),
        _source("link_b", modality="text",
                title="Independent simulated project reference notes B",
                purpose="Cross-check project references with a separate simulated record"),
    ])
    plan.update({
        "version": VERSION,
        "kind": "hybrid_world_reference_investigation",
        "macro_tasks": tasks,
        "terminal_macro_id": tasks[-1]["macro_id"],
        "source_schemas": schemas,
        "synthetic_entities": projects,
        "synthetic_truth": truth,
        "synthetic_candidate_claims": claims,
        "hybrid_shape": shape,
        "information_necessity_policy": NECESSITY_POLICY,
        "real_motif": motif,
        "goal": {
            **plan["goal"],
            "meaning": "name of the real YAGO entity reached after resolving a fictional project's reference",
        },
    })
    spec.update({
        "family": PATTERN, "reasoning_pattern": PATTERN,
        "ordinal": ordinal,
        "task_id": f"question-hyb-{ordinal:06d}-" +
                   digest(seed, ordinal, shape, plan["target_case"])[:8],
        "scene_id": f"hybrid-world-{ordinal:06d}",
        "macro_information_necessity_version": NECESSITY_POLICY,
        "scenario": {
            "topic": topic,
            "setting": "simulated records describing a fictional " +
                       kind.replace("_", " ") + " and its background references",
            "user_role": "records investigator",
            "purpose": "identify the fictional project and trace a documented real-world entity",
            "motivation": "reconcile independent source claims before using a knowledge graph",
            "entity_kind": kind,
        },
    })
    spec["knowledge_witness"]["version"] = KNOWLEDGE_VERSION
    spec["knowledge_witness"]["real_motif"] = copy.deepcopy(motif)
    spec["knowledge_witness"]["synthetic_true_links"] = copy.deepcopy(truth)
    spec["knowledge_contract_hash"] = digest(
        plan, spec["knowledge_witness"], spec["hybrid_backbone_v2"],
        seed, ordinal, HASH_SALT,
    )
    problems = validate_spec(spec)
    if problems:
        raise ValueError("invalid_hybrid_spec:" + ",".join(problems))
    return spec


def validate_spec(spec: dict) -> list[str]:
    """Fail closed on altered synthetic truth, real motifs, or Macro DAG."""
    try:
        plan = spec["macro_plan"]
        snapshot = spec["knowledge_witness"]
        base = spec["hybrid_backbone_v2"]
        if (spec["family"] != PATTERN or spec["reasoning_pattern"] != PATTERN
                or plan["version"] != VERSION
                or snapshot["version"] != KNOWLEDGE_VERSION):
            return ["hybrid_bad_version"]
        if spec["knowledge_contract_hash"] != digest(
            plan, snapshot, base, spec["seed"], spec["ordinal"], HASH_SALT,
        ):
            return ["hybrid_frozen_hash_mismatch"]
        problems = v2.validate_spec(base)
        if problems:
            return ["hybrid_invalid_yago_backbone:" + ",".join(problems)]
        backbone = base["macro_plan"]
        for key in ("candidate_paths", "real_facts", "clues", "all_cases",
                    "target_case", "case_shadow_slots", "relation_groups",
                    "entity_anchor"):
            if plan[key] != backbone[key]:
                return ["hybrid_backbone_rewritten:" + key]
        if snapshot.get("candidate_facts") != base["knowledge_witness"].get(
            "candidate_facts"
        ):
            return ["hybrid_candidate_yago_witness_changed"]
        if snapshot.get("facts") != base["knowledge_witness"].get("facts"):
            return ["hybrid_real_yago_witness_changed"]
        if snapshot["synthetic_true_links"] != plan["synthetic_truth"]:
            return ["hybrid_false_synthetic_truth_witness"]
        problems = validate_frozen_motif(
            plan["real_motif"], plan["real_facts"],
        )
        if problems or snapshot["real_motif"] != plan["real_motif"]:
            return problems or ["hybrid_real_motif_witness_mismatch"]
        if plan["hybrid_shape"] not in {"parallel_fork_join",
                                       "serial_refinement"}:
            return ["hybrid_unsupported_shape"]
        cases = plan["all_cases"]
        projects = plan["synthetic_entities"]
        claims = plan["synthetic_candidate_claims"]
        if set(projects) != set(cases) or set(claims) != {"link_a", "link_b"}:
            return ["hybrid_missing_synthetic_entities"]
        ids = [projects[case]["entity_id"] for case in cases]
        if len(set(ids)) != len(ids):
            return ["hybrid_duplicate_synthetic_uri"]
        truth = {
            row["subject"]: row["object"]
            for row in plan["synthetic_truth"]
        }
        if set(truth) != set(ids) or len(plan["synthetic_truth"]) != len(cases):
            return ["hybrid_missing_synthetic_truth"]
        for case in cases:
            project = projects[case]
            if (project["origin"] != "synthetic"
                    or not project["fictional"]
                    or not project["entity_id"].startswith("urn:maestro:fictional:")
                    or truth[project["entity_id"]] !=
                    plan["candidate_paths"][case][0]["subject"]):
                return ["hybrid_inconsistent_synthetic_entity_truth"]
        for source, rows in claims.items():
            if len(rows) != 2 * len(cases):
                return ["hybrid_wrong_claim_count"]
            observed = {}
            for item in rows:
                pid, uri, route_key = (
                    item["project_id"], item["entity_0"], item["route_key"])
                if pid not in truth or route_key != _route_key(pid, uri):
                    return ["hybrid_bad_candidate_claim"]
                observed.setdefault(pid, set()).add(uri)
            if (set(observed) != set(ids) or
                    any(len(v) != 2 or truth[pid] not in v
                        for pid, v in observed.items())):
                return ["hybrid_missing_two_way_ambiguity"]
        for pid in ids:
            a = {r["entity_0"] for r in claims["link_a"]
                 if r["project_id"] == pid}
            b = {r["entity_0"] for r in claims["link_b"]
                 if r["project_id"] == pid}
            if a & b != {truth[pid]}:
                return ["hybrid_candidate_sources_not_disambiguating"]
        seen, scopes = {}, []
        for task in plan["macro_tasks"]:
            if task["macro_id"] in seen or any(
                parent not in seen for parent in task["depends_on"]
            ):
                return ["hybrid_invalid_partial_order"]
            if task["input_contract"] != {
                p: seen[p]["output_schema"] for p in task["depends_on"]
            }:
                return ["hybrid_invalid_typed_handoff"]
            scopes.extend(task["source_scope"])
            seen[task["macro_id"]] = task
        if (set(plan["source_schemas"]) !=
                {r["source_id"] for r in spec["sources"]}
                or Counter(scopes) != Counter(plan["source_schemas"].keys())):
            return ["hybrid_unused_or_duplicate_source"]
        if (len(plan["macro_tasks"]) > 9 or
                plan["terminal_macro_id"] != plan["macro_tasks"][-1]["macro_id"]
                or plan["macro_tasks"][-1]["output_schema"] != {"answer": "str"}):
            return ["hybrid_macro_budget_or_terminal"]
        if plan["synthetic_truth"] != snapshot["synthetic_true_links"]:
            return ["hybrid_ledger_not_replayable"]
        return []
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return ["hybrid_invalid_spec:" + str(exc)[:160]]


def compile_macro_solvers(spec: dict) -> dict:
    """Compile every causal port via the unmodified legacy relational IR."""
    issues = validate_spec(spec)
    if issues:
        raise ValueError("invalid_hybrid_compilation:" + ",".join(issues))
    plan = spec["macro_plan"]
    nodes, groups, outputs = [], [], {}
    for task in plan["macro_tasks"]:
        mid, role = task["macro_id"], task["role"]
        local = []
        parents = [outputs[pid] for pid in task["depends_on"]]

        def add(op, *incoming, **kwargs):
            node_id = f"{mid.lower()}_op{len(local)+1:02d}"
            local.append({
                "id": node_id, "op": op,
                "inputs": list(incoming), **kwargs,
            })
            return node_id

        if role == "discovery":
            matched = []
            for sid in task["source_scope"]:
                source = add("scan", source_id=sid)
                subset = add("filter_eq", source, field="cue",
                             value=plan["clues"][sid]["value"])
                matched.append(add("project", subset, fields=["case_id"]))
            joined = add("join", matched[0], matched[1],
                         left_on="case_id", right_on="case_id")
            terminal = add("unique", joined)
        elif role == "identification":
            joined = parents[0]
            for parent in parents[1:]:
                joined = add("join", joined, parent,
                             left_on="case_id", right_on="case_id")
            unique = add("unique", joined)
            lookup = add("lookup_by_key", unique, source_id="entity_index",
                         input_key_field="case_id", source_key_field="case_id")
            terminal = add("project", lookup, fields=["project_id"])
        elif role == "investigation" and task["source_scope"] in (
                ["link_a"], ["link_b"]):
            sid = task["source_scope"][0]
            scanned = add("scan", source_id=sid)
            linked = add("join", parents[0], scanned,
                         left_on="project_id", right_on="project_id")
            terminal = add("project", linked,
                           fields=["route_key", "entity_0"])
        elif role == "verification" and not task["source_scope"]:
            joined = add("join", parents[0], parents[1],
                         left_on="route_key", right_on="route_key")
            terminal = add("project", joined, fields=["entity_0"])
        elif role == "verification" and task["source_scope"] == ["link_b"]:
            scanned = add("scan", source_id="link_b")
            joined = add("join", parents[0], scanned,
                         left_on="route_key", right_on="route_key")
            terminal = add("project", joined, fields=["entity_0"])
        elif role == "investigation":
            if len(parents) != 1:
                raise ValueError("hybrid_real_relation_needs_one_subject")
            previous = parents[0]
            for sid in task["source_scope"]:
                i = int(sid.rsplit("_", 1)[-1])
                field = f"entity_{i-1}"
                previous = add("lookup_by_key", previous, source_id=sid,
                               input_key_field=field, source_key_field=field)
            terminal = add("project", previous,
                           fields=list(task["output_schema"]))
        else:
            raise ValueError("hybrid_unknown_macro_operation:" + role)
        nodes.extend(local)
        groups.append({
            "macro_id": mid,
            "node_ids": [n["id"] for n in local],
            "output_node_id": terminal,
            "external_input_nodes": parents,
        })
        outputs[mid] = terminal
    return {
        "version": "maestro.reasoning_program.v4",
        "semantic_operator_pattern": PATTERN,
        "nodes": nodes,
        "output": outputs[plan["terminal_macro_id"]],
        "authorized_source_ids": [r["source_id"] for r in spec["sources"]],
        "frozen_problem_spec_id": spec["task_id"],
        "macro_solver_groups": groups,
        "macro_plan_version": VERSION,
    }


def generate_world(spec: dict) -> dict:
    """Generate the same real source rows as v2 plus fictional associations."""
    from .world import _rows
    issues = validate_spec(spec)
    if issues:
        raise ValueError("invalid_hybrid_world:" + ",".join(issues))
    plan = spec["macro_plan"]
    world = v2.generate_world(spec["hybrid_backbone_v2"])
    bank = world["sources"]
    rng = random.Random(f"maestro-hybrid-observations-v3:{spec['seed']}:{spec['ordinal']}")
    project_map = plan["synthetic_entities"]
    rows = [{
        "case_id": case, "project_id": project_map[case]["entity_id"]
    } for case in plan["all_cases"]]
    rng.shuffle(rows)
    original = bank["entity_index"]
    bank["entity_index"] = _rows(
        "entity_index", "table",
        "Fictional project identity crosswalk",
        "Connect a uniquely established case with a fictional project record",
        rows,
    )
    bank["entity_index"]["access_policy"] = copy.deepcopy(
        original["access_policy"]
    )
    for sid in ("link_a", "link_b"):
        scope = next(r for r in spec["sources"] if r["source_id"] == sid)
        candidates = copy.deepcopy(plan["synthetic_candidate_claims"][sid])
        # These are candidate assertions from simulated documents, never
        # two independent historical YAGO witnesses.
        source = _rows(
            sid, scope["modality"], scope["title"], scope["purpose"],
            candidates,
        )
        source["access_policy"] = {"mode": "browse"}
        source["evidence_kind"] = "fictional_source_candidate_assertions"
        source["independence_note"] = (
            "Independently simulated records inside one fictional world; "
            "neither is an externally authenticated real-world publisher."
        )
        for row in source["observations"]:
            row["claim_status"] = "unverified_candidate"
            row["evidence_origin_id"] = (
                "simulated:" + digest(sid, row["values"]["route_key"])[:20]
            )
        bank[sid] = source
    world["synthetic_world"] = {
        "version": "maestro.hybrid_fictional_world.v1",
        "entities": copy.deepcopy(project_map),
        "true_relations": copy.deepcopy(plan["synthetic_truth"]),
        "source_candidate_claims": {
            sid: [{"source_id": sid, **row}
                  for row in plan["synthetic_candidate_claims"][sid]]
            for sid in ("link_a", "link_b")
        },
        "fictional_entities_are_not_yago": True,
    }
    world["real_world_motif"] = copy.deepcopy(plan["real_motif"])
    world["knowledge_witness"] = copy.deepcopy(spec["knowledge_witness"])
    world["value_sampling_contract"] = (
        "real_yago_motifs_plus_fictional_projects_and_two_source_join_v3"
    )
    return world


def mutate_witness(sources: dict, sid: str,
                   answer_rows: list[dict], spec: dict) -> dict:
    """One-source counterfactual using cited records, never hidden answers."""
    if sid in spec["macro_plan"]["clues"] or sid.startswith("relation_"):
        return v2.mutate_witness(sources, sid, answer_rows, spec)
    revised = copy.deepcopy(sources)
    relevant = {
        ref for row in answer_rows for ref in row["source_fact_refs"]
        if ref.startswith(sid + ":")
    }
    selected = next((
        o for o in revised[sid]["observations"]
        if set(o["fact_refs"]) & relevant
    ), None)
    if selected is None:
        raise ValueError("hybrid_no_cited_source_fact:" + sid)
    if sid == "entity_index":
        selected["values"]["project_id"] = "urn:maestro:synthetic:missing-project"
    elif sid in {"link_a", "link_b"}:
        selected["values"]["route_key"] = "urn:maestro:synthetic:broken-route"
    else:
        raise ValueError("hybrid_unknown_counterfactual_source:" + sid)
    return revised


def audit_hybrid_information(scene: dict) -> dict:
    """Bounded executable re-planning + synthetic-source value checks.

    All single clue, whole Discovery, and total Discovery bypass probes must
    yield *multiple real answer rows*. Independently, making either synthetic
    association directory nondiscriminative must leave multiple answers.
    This does not prove that all possible global shortcuts are impossible.
    """
    from .program_ir import evaluate_program, answer_values

    plan = scene["problem_spec"]["macro_plan"]
    sources = scene["world"]["sources"]
    program = scene["gold_program"]
    expected = scene["gold_answer"][0]["answer"]
    identity = next(t for t in plan["macro_tasks"]
                    if t["role"] == "identification")
    report = next(t for t in scene["task_feedback"]
                  if t["task_id"] == "work_" + identity["macro_id"])
    full = {}
    for sid, clue in plan["clues"].items():
        full[sid] = {
            row["values"]["case_id"] for row in sources[sid]["observations"]
            if row["values"]["cue"] == clue["value"]
        }
    unique = (
        set.intersection(*full.values()) == {plan["target_case"]}
        and report["result_rows"] == 1
        and report["workproduct"][0]["values"]["project_id"] ==
            plan["synthetic_entities"][plan["target_case"]]["entity_id"]
    )
    node_map = {n["id"]: n for n in program["nodes"]}
    filters = {}
    for n in program["nodes"]:
        if n["op"] == "filter_eq" and n.get("field") == "cue":
            scan = node_map[n["inputs"][0]]
            filters[scan["source_id"]] = n["id"]
    if set(filters) != set(full):
        raise ValueError("hybrid_missing_compiled_clue_filters")

    def alternatives(rows):
        distinct = {item["answer"] for item in rows}
        return len(rows) > 1 and len(distinct) > 1 and expected in distinct

    def relax_clues(sids: set[str]):
        changed = copy.deepcopy(program)
        ids = {filters[sid] for sid in sids}
        for node in changed["nodes"]:
            if node["id"] in ids:
                identifier, parents = node["id"], node["inputs"]
                node.clear()
                node.update({
                    "id": identifier, "op": "project", "inputs": parents,
                    "fields": ["case_id", "cue"],
                })
        return answer_values(evaluate_program(changed, sources=sources))

    clue_results = {
        sid: alternatives(relax_clues({sid})) and len(
            set.intersection(*(x for other, x in full.items() if other != sid))
        ) > 1
        for sid in full
    }
    disc_results = {
        task["macro_id"]: alternatives(
            relax_clues(set(task["source_scope"]))
        )
        for task in plan["macro_tasks"] if task["role"] == "discovery"
    }
    all_relaxed = relax_clues(set(full))
    true_project = plan["synthetic_entities"][
        plan["target_case"]
    ]["entity_id"]
    synthetic_results = {}
    for bypassed, other in (("link_a", "link_b"), ("link_b", "link_a")):
        updated = copy.deepcopy(sources)
        target_wrong = next(
            r for r in plan["synthetic_candidate_claims"][other]
            if r["project_id"] == true_project and r["entity_0"] !=
                plan["candidate_paths"][plan["target_case"]][0]["subject"]
        )
        current = {
            r["values"]["route_key"] for r in updated[bypassed]["observations"]
            if r["values"]["project_id"] == true_project
        }
        if target_wrong["route_key"] in current:
            raise ValueError("hybrid_branch_already_contains_false_match")
        updated[bypassed]["observations"].append({
            "observation_id": bypassed + ".COUNTERFACTUAL",
            "fact_refs": [bypassed + ":F-COUNTERFACTUAL"],
            "values": copy.deepcopy(target_wrong),
            "claim_status": "unverified_candidate",
        })
        synthetic_results[bypassed] = alternatives(
            answer_values(evaluate_program(program, sources=updated))
        )
    passed = (
        unique and all(clue_results.values())
        and all(disc_results.values()) and alternatives(all_relaxed)
        and all(synthetic_results.values())
    )
    return {
        "policy": NECESSITY_POLICY,
        "passed": passed,
        "identification_unique": unique,
        "clue_relaxation_necessary": clue_results,
        "discovery_relaxation_necessary": disc_results,
        "all_discovery_filters_removed": {
            "possible_answers": len(all_relaxed),
            "ambiguous": alternatives(all_relaxed),
        },
        "synthetic_branch_necessity": synthetic_results,
        "real_world_motif_kind": plan["real_motif"]["kind"],
        "hybrid_macro_shape": plan["hybrid_shape"],
        "limitation": (
            "Bounded materialized-world re-planning interventions; "
            "not global solution minimality or proof of independent publishers."
        ),
    }
