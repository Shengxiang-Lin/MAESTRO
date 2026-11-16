"""Canonical Hybrid World v4: causal real RDF motif + fictional event network.

This opt-in family upgrades v3's contextual Fork/Diamond catalog into a
*required* YAGO Fork Investigation with one genuine assertion for every
competing entity. It adds a fictional project -> event -> organizer/venue ->
allowed-real-reference graph, visible to Sub-Agents through actual source
observations.

The source candidates are constructed deliberately:
  association A = {T,U,V}
  association B = {T,U,W}
  event allowed real fork objects = {fork(T), fork(V), fork(W)}
  actual real-fork lookup = fork(candidate)
  all constraints -> T only
omit A -> {T,W}; omit B -> {T,V}; omit event/motif -> {T,U}.

Every candidate also has a complete genuine YAGO 2–3 hop final answer
path (v2), so none of these re-planning interventions uses missing-path
shortcuts. All user-visible media and new entities are explicitly synthetic.

Do NOT rewrite v1/v2/v3 frozen specs. Reuse their canonical old pipeline.
"""
from __future__ import annotations

from collections import Counter
import copy
import random

from environment.schema import digest
from . import knowledge_macro as v1
from . import hybrid_world as v3
from .real_motif_grounding import (
    mine_required_fork, validate_frozen_required_fork,
)

PATTERN = "typed_knowledge_v4"
VERSION = "maestro.macro_plan.typed_knowledge.v4"
KNOWLEDGE_VERSION = "maestro.yago_witness.v4"
HASH_SALT = "maestro-causal-hybrid-v4"
NECESSITY_POLICY = "hybrid_world_causal_motif_v4"
GRAPH_VERSION = "maestro.synthetic_event_graph.v1"


def _id(seed, ordinal, label, case):
    return "urn:maestro:fictional:" + label + ":" + digest(
        seed, ordinal, label, case
    )[:20]


def _fact(subject, predicate, obj, *, record_type="synthetic_world_fact"):
    return {
        "subject": subject, "predicate": predicate, "object": obj,
        "origin": "synthetic", "truth_scope": "fictional_event_world",
        "record_type": record_type,
    }


def _synthetic_graph(base: dict, fork: dict, *, seed: int,
                     ordinal: int) -> dict:
    plan = base["macro_plan"]
    paths = plan["candidate_paths"]
    cases = list(plan["all_cases"])
    projects = copy.deepcopy(plan["synthetic_entities"])
    from_v3 = copy.deepcopy(plan["synthetic_truth"])
    events, people, venues, truth, event_rows = {}, {}, {}, [], []
    for i, case in enumerate(cases):
        p = projects[case]
        pid = p["entity_id"]
        event_id = _id(seed, ordinal, "event", case)
        person_id = _id(seed, ordinal, "organizer", case)
        venue_id = _id(seed, ordinal, "venue", case)
        event_name = f"{p['display_name']} — review session {i+1}"
        organiser = f"Coordinator {i+1} of {p['display_name']}"
        venue_name = f"Fictional Venue {i+1} / {p['display_name']}"
        status = ("scheduled", "completed", "revised")[i % 3]
        events[case] = {
            "entity_id": event_id, "entity_type": "fictional_event",
            "display_name": event_name,
            "status": status, "year": 2023 + i % 4,
            "origin": "synthetic", "fictional": True,
        }
        people[case] = {
            "entity_id": person_id, "entity_type": "fictional_organizer",
            "display_name": organiser,
            "origin": "synthetic", "fictional": True,
        }
        venues[case] = {
            "entity_id": venue_id, "entity_type": "fictional_venue",
            "display_name": venue_name,
            "origin": "synthetic", "fictional": True,
        }
        p["event_id"] = event_id
        p["organizer_id"] = person_id
        p["venue_id"] = venue_id
        truth.extend([
            _fact(pid, "urn:maestro:synthetic:hasEvent", event_id),
            _fact(event_id, "urn:maestro:synthetic:organizedBy", person_id),
            _fact(event_id, "urn:maestro:synthetic:heldAt", venue_id),
        ])
        # Cyclic case ordering is deterministic but not a global selector.
        # Real branches are supported by actual RDF, never fabricated.
        for offset in (0, 2, 3):
            other_case = cases[(i + offset) % len(cases)]
            real_object = fork["candidate_forks"][other_case]["object"]
            truth.append(_fact(
                event_id, "urn:maestro:synthetic:permitsYagoForkObject",
                real_object,
            ))
            event_rows.append({
                "project_id": pid,
                "project_name": p["display_name"],
                "event_id": event_id,
                "event_name": event_name,
                "event_status": status,
                "event_year": 2023 + i % 4,
                "organizer_id": person_id,
                "organizer_name": organiser,
                "venue_id": venue_id,
                "venue_name": venue_name,
                "motif_object": real_object,
            })
    return {
        "version": GRAPH_VERSION,
        "projects": projects,
        "events": events,
        "organizers": people,
        "venues": venues,
        "true_relations": from_v3 + truth,
        "event_rows": event_rows,
        "all_entities_fictional": True,
        "real_refs_are_yago_objects_not_synthetic_entities": True,
    }


def _claims_for_graph(base: dict) -> dict:
    plan = base["macro_plan"]
    cases = list(plan["all_cases"])
    roots = {
        case: plan["candidate_paths"][case][0]["subject"]
        for case in cases
    }
    result = {"link_a": [], "link_b": []}
    for i, case in enumerate(cases):
        pid = plan["synthetic_entities"][case]["entity_id"]
        candidates = {
            "link_a": (0, 1, 2),
            "link_b": (0, 1, 3),
        }
        for source, offsets in candidates.items():
            for shift in offsets:
                chosen = roots[cases[(i + shift) % len(cases)]]
                result[source].append({
                    "project_id": pid,
                    "entity_0": chosen,
                    "route_key": v3._route_key(pid, chosen),
                })
    return result


def _make_macro(tasks, role, parents, schema, source_scope, detail):
    return v3._macro(tasks, role, parents, schema, source_scope, detail)


def _plan_tasks(base: dict) -> list[dict]:
    plan = base["macro_plan"]
    n_discoveries = sum(
        task["role"] == "discovery" for task in plan["macro_tasks"]
    )
    tasks = copy.deepcopy(plan["macro_tasks"][:n_discoveries])
    identity = _make_macro(
        tasks, "identification",
        [x["macro_id"] for x in tasks],
        {"project_id": "str"}, ["entity_index"],
        "Identify a fictional project from independent simulated media records",
    )
    # Event network can be investigated in parallel with project references.
    event_mid = _make_macro(
        tasks, "investigation", [identity],
        {"event_id": "str", "motif_object": "str"}, ["event_ledger"],
        "Retrieve real-reference eligibility from the fictional project's event, organizer and venue roster",
    )
    first = _make_macro(
        tasks, "investigation", [identity],
        {"route_key": "str", "entity_0": "str"}, ["link_a"],
        "Read the first simulated project-to-YAGO candidate directory",
    )
    if plan["hybrid_shape"] == "parallel_fork_join":
        second = _make_macro(
            tasks, "investigation", [identity],
            {"route_key": "str", "entity_0": "str"}, ["link_b"],
            "Independently inspect a second simulated directory",
        )
        verified = _make_macro(
            tasks, "verification", [first, second],
            {"entity_0": "str"}, [],
            "Intersect both simulated directories; retain still-ambiguous candidates for real-world cross-check",
        )
    else:
        verified = _make_macro(
            tasks, "verification", [first],
            {"entity_0": "str"}, ["link_b"],
            "Refine the first directory candidates using a second simulated source",
        )
    selected = _make_macro(
        tasks, "verification", [verified, event_mid],
        {"entity_0": "str"}, ["real_fork"],
        "Resolve remaining candidates by checking a genuine YAGO fork against the fictional event's roster",
    )
    _make_macro(
        tasks, "investigation", [selected],
        {"answer": "str"},
        [f"relation_{i}" for i in range(1, len(plan["real_facts"])+1)],
        "Follow the selected real entity's witnessed YAGO relation path to the final answer",
    )
    if len(tasks) > 9:
        raise ValueError("hybrid_v4_macro_budget_exceeded")
    return tasks


def make_spec(kg, *, seed: int, ordinal: int, difficulty: str,
              min_hops: int = 2, max_hops: int = 3,
              index: v1.YagoKnowledgeIndex | None = None) -> dict:
    index = index or v1.YagoKnowledgeIndex(kg)
    # Search for a real fact motif whose SAME predicate is instantiated
    # for every competing YAGO candidate; do not manufacture missing RDF.
    for attempt in range(65):
        try:
            base = v3.make_spec(
                kg, seed=seed, ordinal=ordinal + attempt * 1009,
                difficulty=difficulty, min_hops=min_hops,
                max_hops=max_hops, index=index,
            )
            base_plan = base["macro_plan"]
            fork = mine_required_fork(
                kg, index, base_plan["candidate_paths"],
                seed=seed, ordinal=ordinal + attempt * 1009,
            )
        except ValueError:
            continue
        n_discovery = sum(x["role"] == "discovery"
                          for x in base_plan["macro_tasks"])
        if n_discovery == 3 and base_plan["hybrid_shape"] == "parallel_fork_join":
            # Avoid exceeding the existing hard cap of nine Macros.
            base = copy.deepcopy(base)
            base["macro_plan"]["hybrid_shape"] = "serial_refinement"
            # Construct a truly valid v3 base to retain its frozen replay.
            base["macro_plan"]["macro_tasks"] = _rebuild_v3_serial_tasks(
                base["macro_plan"]
            )
            base["macro_plan"]["terminal_macro_id"] = (
                base["macro_plan"]["macro_tasks"][-1]["macro_id"]
            )
            base["knowledge_contract_hash"] = digest(
                base["macro_plan"], base["knowledge_witness"],
                base["hybrid_backbone_v2"], base["seed"],
                base["ordinal"], v3.HASH_SALT,
            )
            if v3.validate_spec(base):
                continue
        spec = copy.deepcopy(base)
        spec["hybrid_backbone_v3"] = copy.deepcopy(base)
        plan = spec["macro_plan"]
        graph = _synthetic_graph(
            base, fork, seed=seed, ordinal=ordinal,
        )
        plan["version"] = VERSION
        plan["kind"] = "causal_yago_motif_plus_fictional_event_world"
        plan["macro_tasks"] = _plan_tasks(base)
        plan["terminal_macro_id"] = plan["macro_tasks"][-1]["macro_id"]
        plan["real_fork"] = fork
        plan["synthetic_graph"] = graph
        plan["synthetic_entities"] = graph["projects"]
        plan["synthetic_truth"] = graph["true_relations"]
        plan["synthetic_candidate_claims"] = _claims_for_graph(base)
        plan["information_necessity_policy"] = NECESSITY_POLICY
        plan["source_schemas"] = {
            **plan["source_schemas"],
            "entity_index": {
                "case_id": "str", "project_id": "str", "project_name": "str",
            },
            "link_a": {
                "project_id": "str", "entity_0": "str", "route_key": "str",
            },
            "link_b": {
                "project_id": "str", "entity_0": "str", "route_key": "str",
            },
            "event_ledger": {
                "project_id": "str", "project_name": "str",
                "event_id": "str", "event_name": "str",
                "event_status": "str", "event_year": "int",
                "organizer_id": "str", "organizer_name": "str",
                "venue_id": "str", "venue_name": "str",
                "motif_object": "str",
            },
            "real_fork": {
                "entity_0": "str", "motif_object": "str",
            },
        }
        spec["sources"] = [
            copy.deepcopy(r) for r in base["sources"]
        ] + [
            v3._source(
                "event_ledger", modality="table",
                title="Simulated event, organizer and venue relationship ledger",
                purpose="List the fictional project's event relationships and which real-fork objects its event permits",
            ),
            v3._source(
                "real_fork", modality="text",
                title="Grounded YAGO fork relationship assertions",
                purpose="Check the actual secondary real relation for each possible referenced YAGO entity",
                key="entity_0",
            ),
        ]
        spec.update({
            "family": PATTERN, "reasoning_pattern": PATTERN,
            "ordinal": ordinal,
            "task_id": f"question-hyb4-{ordinal:06d}-" + digest(
                seed, ordinal, attempt, fork["predicate"]
            )[:8],
            "scene_id": f"hybrid-causal-motif-{ordinal:06d}",
            "macro_information_necessity_version": NECESSITY_POLICY,
            "scenario": {
                **base["scenario"],
                "purpose": (
                    "identify the fictional project, trace its event/organizer "
                    "relations and cross-check actual YAGO fork evidence"
                ),
                "motivation": (
                    "both simulated reference directories and the project "
                    "event permit several candidates; the real YAGO fork "
                    "and the event together resolve the remaining ambiguity"
                ),
            },
        })
        witness = spec["knowledge_witness"]
        witness["version"] = KNOWLEDGE_VERSION
        witness["required_real_fork"] = copy.deepcopy(fork)
        witness["synthetic_event_truth"] = copy.deepcopy(graph["true_relations"])
        spec["knowledge_contract_hash"] = digest(
            plan, witness, spec["hybrid_backbone_v3"],
            seed, ordinal, HASH_SALT,
        )
        issues = validate_spec(spec)
        if issues:
            raise ValueError("causal_hybrid_construct_invalid:" +
                             ",".join(issues))
        return spec
    raise ValueError("cannot_ground_complete_real_fork_for_all_candidates")


def _rebuild_v3_serial_tasks(plan):
    """Maintain a well-typed frozen v3 base when shrinking the graph budget."""
    n = sum(t["role"] == "discovery" for t in plan["macro_tasks"])
    tasks = copy.deepcopy(plan["macro_tasks"][:n])
    identity = _make_macro(
        tasks, "identification",
        [t["macro_id"] for t in tasks],
        {"project_id": "str"}, ["entity_index"],
        "Join simulated clue investigations",
    )
    first = _make_macro(
        tasks, "investigation", [identity],
        {"route_key": "str", "entity_0": "str"}, ["link_a"],
        "Review first project reference directory",
    )
    final = _make_macro(
        tasks, "verification", [first],
        {"entity_0": "str"}, ["link_b"],
        "Review second project reference directory",
    )
    for group in plan["relation_groups"]:
        key = "answer" if group[-1] == len(plan["real_facts"]) \
            else f"entity_{group[-1]}"
        final = _make_macro(
            tasks, "investigation", [final], {key: "str"},
            [f"relation_{i}" for i in group],
            "Follow witnessed YAGO relation path",
        )
    return tasks


def validate_spec(spec: dict) -> list[str]:
    try:
        plan, witness = spec["macro_plan"], spec["knowledge_witness"]
        base = spec["hybrid_backbone_v3"]
        if (spec["reasoning_pattern"] != PATTERN or spec["family"] != PATTERN
                or plan["version"] != VERSION
                or witness["version"] != KNOWLEDGE_VERSION):
            return ["hybrid_v4_version_mismatch"]
        if spec["knowledge_contract_hash"] != digest(
            plan, witness, base, spec["seed"], spec["ordinal"], HASH_SALT,
        ):
            return ["hybrid_v4_frozen_hash_mismatch"]
        issues = v3.validate_spec(base)
        if issues:
            return ["hybrid_v4_invalid_frozen_v3_backbone:" + ",".join(issues)]
        original = base["macro_plan"]
        for key in ("candidate_paths", "real_facts", "clues", "all_cases",
                    "target_case", "entity_anchor", "case_shadow_slots"):
            if plan[key] != original[key]:
                return ["hybrid_v4_backbone_modified:" + key]
        fork = plan["real_fork"]
        issues = validate_frozen_required_fork(fork, plan["candidate_paths"])
        if issues or witness["required_real_fork"] != fork:
            return issues or ["hybrid_v4_real_fork_witness_mismatch"]
        graph = plan["synthetic_graph"]
        if (graph["version"] != GRAPH_VERSION
                or graph["projects"] != plan["synthetic_entities"]
                or graph["true_relations"] != plan["synthetic_truth"]
                or witness["synthetic_event_truth"] != plan["synthetic_truth"]
                or not graph["all_entities_fictional"]):
            return ["hybrid_v4_synthetic_graph_inconsistent"]
        cases = plan["all_cases"]
        if any(set(layer) != set(cases) for layer in (
                graph["projects"], graph["events"],
                graph["organizers"], graph["venues"])):
            return ["hybrid_v4_missing_typed_synthetic_entity"]
        allowed = {}
        for row in graph["event_rows"]:
            allowed.setdefault(row["project_id"], set()).add(
                row["motif_object"]
            )
        claims = plan["synthetic_candidate_claims"]
        roots = {case: plan["candidate_paths"][case][0]["subject"]
                 for case in cases}
        for i, case in enumerate(cases):
            pid = graph["projects"][case]["entity_id"]
            if graph["projects"][case]["origin"] != "synthetic":
                return ["hybrid_v4_fake_project_origin"]
            t, u, v, w = [
                roots[cases[(i + k) % len(cases)]]
                for k in range(4)
            ]
            a = {r["entity_0"] for r in claims["link_a"]
                 if r["project_id"] == pid}
            b = {r["entity_0"] for r in claims["link_b"]
                 if r["project_id"] == pid}
            permitted = {
                fork["candidate_forks"][cases[(i+k) % len(cases)]]["object"]
                for k in (0, 2, 3)
            }
            if (a != {t, u, v} or b != {t, u, w}
                    or allowed.get(pid) != permitted):
                return ["hybrid_v4_false_candidate_set"]
            if not all((r["route_key"] ==
                        v3._route_key(pid, r["entity_0"]))
                       for sid in ("link_a", "link_b")
                       for r in claims[sid] if r["project_id"] == pid):
                return ["hybrid_v4_candidate_route_key_mismatch"]
        truth = graph["true_relations"]
        if any(row["origin"] != "synthetic" for row in truth):
            return ["hybrid_v4_false_world_relation_origin"]
        if (len(truth) != len(original["all_cases"]) * (1+3+3)
                or Counter(row["predicate"] for row in truth) != Counter({
                    "urn:maestro:synthetic:documentsYagoReference": len(cases),
                    "urn:maestro:synthetic:hasEvent": len(cases),
                    "urn:maestro:synthetic:organizedBy": len(cases),
                    "urn:maestro:synthetic:heldAt": len(cases),
                    "urn:maestro:synthetic:permitsYagoForkObject": 3*len(cases),
                })):
            return ["hybrid_v4_synthetic_world_truth_mismatch"]
        expected_tasks = _plan_tasks(base)
        if plan["macro_tasks"] != expected_tasks:
            return ["hybrid_v4_macro_grammar_or_handoff_modified"]
        scopes = Counter(sid for task in expected_tasks
                         for sid in task["source_scope"])
        if (scopes != Counter(plan["source_schemas"].keys())
                or set(plan["source_schemas"]) !=
                {row["source_id"] for row in spec["sources"]}):
            return ["hybrid_v4_invalid_source_scope"]
        if (len(expected_tasks) > 9
                or expected_tasks[-1]["output_schema"] != {"answer": "str"}
                or plan["terminal_macro_id"] != expected_tasks[-1]["macro_id"]):
            return ["hybrid_v4_macro_budget_or_terminal"]
        return []
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return ["hybrid_v4_invalid_spec:" + str(exc)[:160]]


def compile_macro_solvers(spec: dict) -> dict:
    issues = validate_spec(spec)
    if issues:
        raise ValueError("hybrid_v4_compiler_contract:" + ",".join(issues))
    plan = spec["macro_plan"]
    nodes, groups, output_ids = [], [], {}
    for task in plan["macro_tasks"]:
        mid, role = task["macro_id"], task["role"]
        parents = [output_ids[parent] for parent in task["depends_on"]]
        local = []

        def add(op, *inputs, **kwargs):
            node_id = f"{mid.lower()}_op{len(local)+1:02d}"
            local.append({
                "id": node_id, "op": op, "inputs": list(inputs), **kwargs,
            })
            return node_id

        if role == "discovery":
            matching = []
            for sid in task["source_scope"]:
                scan = add("scan", source_id=sid)
                filtered = add(
                    "filter_eq", scan, field="cue",
                    value=plan["clues"][sid]["value"],
                )
                matching.append(add(
                    "project", filtered, fields=["case_id"],
                ))
            joined = add("join", matching[0], matching[1],
                         left_on="case_id", right_on="case_id")
            output = add("unique", joined)
        elif role == "identification":
            joined = parents[0]
            for source in parents[1:]:
                joined = add("join", joined, source,
                             left_on="case_id", right_on="case_id")
            narrowed = add("unique", joined)
            indexed = add(
                "lookup_by_key", narrowed, source_id="entity_index",
                input_key_field="case_id", source_key_field="case_id",
            )
            output = add("project", indexed, fields=["project_id"])
        elif role == "investigation" and task["source_scope"] == ["event_ledger"]:
            scanned = add("scan", source_id="event_ledger")
            joined = add(
                "join", parents[0], scanned,
                left_on="project_id", right_on="project_id",
            )
            output = add(
                "project", joined, fields=["event_id", "motif_object"],
            )
        elif role == "investigation" and task["source_scope"] in (
                ["link_a"], ["link_b"]):
            scanned = add("scan", source_id=task["source_scope"][0])
            joined = add("join", parents[0], scanned,
                         left_on="project_id", right_on="project_id")
            output = add(
                "project", joined, fields=["route_key", "entity_0"],
            )
        elif role == "verification" and not task["source_scope"]:
            joined = add("join", parents[0], parents[1],
                         left_on="route_key", right_on="route_key")
            output = add("project", joined, fields=["entity_0"])
        elif role == "verification" and task["source_scope"] == ["link_b"]:
            scanned = add("scan", source_id="link_b")
            joined = add("join", parents[0], scanned,
                         left_on="route_key", right_on="route_key")
            output = add("project", joined, fields=["entity_0"])
        elif role == "verification" and task["source_scope"] == ["real_fork"]:
            # Parent 0: after two synthetic sources, still {T, U}.
            # Parent 1: fictional event allows fork objects {T, V, W}.
            # Lookup actual YAGO fork per candidate, then intersect with
            # synthetic event permission. Each branch is needed.
            observed = add(
                "lookup_by_key", parents[0], source_id="real_fork",
                input_key_field="entity_0", source_key_field="entity_0",
            )
            joined = add(
                "join", observed, parents[1],
                left_on="motif_object", right_on="motif_object",
            )
            output = add("project", joined, fields=["entity_0"])
        elif role == "investigation" and all(
                source.startswith("relation_")
                for source in task["source_scope"]):
            previous = parents[0]
            for sid in task["source_scope"]:
                num = int(sid.rsplit("_", 1)[-1])
                previous = add(
                    "lookup_by_key", previous, source_id=sid,
                    input_key_field=f"entity_{num-1}",
                    source_key_field=f"entity_{num-1}",
                )
            output = add(
                "project", previous, fields=["answer"],
            )
        else:
            raise ValueError("hybrid_v4_unmapped_macro_operation:" + role)
        nodes.extend(local)
        groups.append({
            "macro_id": mid, "node_ids": [n["id"] for n in local],
            "output_node_id": output,
            "external_input_nodes": parents,
        })
        output_ids[mid] = output
    return {
        "version": "maestro.reasoning_program.v4",
        "semantic_operator_pattern": PATTERN,
        "nodes": nodes,
        "output": output_ids[plan["terminal_macro_id"]],
        "authorized_source_ids": [r["source_id"] for r in spec["sources"]],
        "frozen_problem_spec_id": spec["task_id"],
        "macro_solver_groups": groups,
        "macro_plan_version": VERSION,
    }


def generate_world(spec: dict) -> dict:
    from .world import _rows
    issues = validate_spec(spec)
    if issues:
        raise ValueError("hybrid_v4_world_contract:" + ",".join(issues))
    base = spec["hybrid_backbone_v3"]
    plan = spec["macro_plan"]
    world = v3.generate_world(base)
    sources = world["sources"]
    rng = random.Random(f"hybrid-causal-world:{spec['seed']}:{spec['ordinal']}")
    rows = [{
        "case_id": case, "project_id": plan["synthetic_entities"][case]["entity_id"],
        "project_name": plan["synthetic_entities"][case]["display_name"],
    } for case in plan["all_cases"]]
    rng.shuffle(rows)
    original = sources["entity_index"]
    sources["entity_index"] = _rows(
        "entity_index", "table", "Fictional project identity crosswalk",
        "Connect identified case to a named fictional project", rows,
    )
    sources["entity_index"]["access_policy"] = copy.deepcopy(
        original["access_policy"]
    )

    for sid in ("link_a", "link_b"):
        meta = next(r for r in spec["sources"] if r["source_id"] == sid)
        rows = copy.deepcopy(plan["synthetic_candidate_claims"][sid])
        rng.shuffle(rows)
        source = _rows(sid, meta["modality"], meta["title"],
                       meta["purpose"], rows)
        source["access_policy"] = {"mode": "browse"}
        source["evidence_kind"] = "unverified_fictional_reference_candidates"
        for observation in source["observations"]:
            observation["claim_status"] = "unverified_candidate"
            observation["evidence_origin_id"] = (
                "simulated:" + digest(sid, observation["values"]["route_key"])[:20]
            )
        sources[sid] = source
    events = copy.deepcopy(plan["synthetic_graph"]["event_rows"])
    rng.shuffle(events)
    event_source = _rows(
        "event_ledger", "table",
        "Simulated event, organizer and venue relationship ledger",
        "Look up fictional project events and eligible real YAGO relation references",
        events,
    )
    event_source["access_policy"] = {"mode": "browse"}
    event_source["evidence_kind"] = "fictional_event_and_eligibility_facts"
    for observation in event_source["observations"]:
        observation["evidence_origin_id"] = (
            "synthetic:" + digest(
                observation["values"]["event_id"],
                observation["values"]["motif_object"],
            )[:20]
        )
    sources["event_ledger"] = event_source

    real_fork = plan["real_fork"]
    rows = [{
        "entity_0": plan["candidate_paths"][case][0]["subject"],
        "motif_object": fact["object"],
    } for case, fact in real_fork["candidate_forks"].items()]
    rng.shuffle(rows)
    source = _rows(
        "real_fork", "text", "Grounded YAGO fork relationship assertions",
        "Find a real secondary YAGO relation for each candidate entity",
        rows,
    )
    source["access_policy"] = {
        "mode": "keyed_lookup", "key_field": "entity_0",
    }
    source["evidence_kind"] = "real_yago_fork_facts"
    by_subject = {
        fact["subject"]: fact
        for fact in real_fork["candidate_forks"].values()
    }
    for observation in source["observations"]:
        fact = by_subject[observation["values"]["entity_0"]]
        observation["knowledge_fact"] = {
            k: fact[k] for k in ("subject", "predicate", "object")
        }
        observation["evidence_origin_id"] = "YAGO-4.6:" + digest(
            fact["subject"], fact["predicate"], fact["object"]
        )[:20]
    sources["real_fork"] = source
    world["synthetic_world"] = {
        "version": GRAPH_VERSION,
        "entities": copy.deepcopy(plan["synthetic_graph"]),
        "true_relations": copy.deepcopy(plan["synthetic_truth"]),
        "candidate_claims": copy.deepcopy(plan["synthetic_candidate_claims"]),
        "fictional_entities_are_not_yago": True,
    }
    world["real_world_motif"] = copy.deepcopy(plan["real_fork"])
    world["knowledge_witness"] = copy.deepcopy(spec["knowledge_witness"])
    world["value_sampling_contract"] = (
        "causal_real_yago_fork_plus_fictional_event_relations_v4"
    )
    return world


def mutate_witness(sources: dict, sid: str,
                   answer_rows: list[dict], spec: dict) -> dict:
    plan = spec["macro_plan"]
    if sid in plan["clues"] or sid.startswith("relation_"):
        return v3.mutate_witness(sources, sid, answer_rows, spec)
    if sid in {"entity_index", "link_a", "link_b"}:
        return v3.mutate_witness(sources, sid, answer_rows, spec)
    modified = copy.deepcopy(sources)
    cited = {
        ref for row in answer_rows for ref in row["source_fact_refs"]
        if ref.startswith(sid + ":")
    }
    observation = next((
        row for row in modified[sid]["observations"]
        if set(row["fact_refs"]) & cited
    ), None)
    if observation is None:
        raise ValueError("hybrid_v4_missing_cited_source:" + sid)
    if sid in {"event_ledger", "real_fork"}:
        observation["values"]["motif_object"] = (
            "urn:maestro:synthetic:counterfactual:unmatched-fork"
        )
    else:
        raise ValueError("hybrid_v4_unknown_mutation_source:" + sid)
    return modified


def audit_hybrid_information(scene: dict) -> dict:
    """Probe actual executable shortcuts across both World layers.

    All probes must produce MULTIPLE reachable final answers (not empty
    results or an exception). A source modification used here is an
    *intervention*, not an assertion of a new true YAGO RDF fact.
    """
    from .program_ir import evaluate_program, answer_values

    plan, sources, program = (
        scene["problem_spec"]["macro_plan"],
        scene["world"]["sources"],
        scene["gold_program"],
    )
    target = plan["target_case"]
    expected = scene["gold_answer"][0]["answer"]
    identity = next(t for t in plan["macro_tasks"]
                    if t["role"] == "identification")
    report = next(t for t in scene["task_feedback"]
                  if t["task_id"] == "work_" + identity["macro_id"])
    unique = (
        report["result_rows"] == 1
        and report["workproduct"][0]["values"]["project_id"] ==
        plan["synthetic_entities"][target]["entity_id"]
    )
    nodes = {row["id"]: row for row in program["nodes"]}
    clue_filters = {}
    for row in program["nodes"]:
        if row["op"] == "filter_eq" and row.get("field") == "cue":
            source = nodes[row["inputs"][0]]
            clue_filters[source["source_id"]] = row["id"]
    if set(clue_filters) != set(plan["clues"]):
        raise ValueError("hybrid_v4_clue_filter_schema_mismatch")

    def multiple(rows):
        values = {r["answer"] for r in rows}
        return len(values) > 1 and expected in values

    def run(modified_sources=None, dropped_filters=()):
        rewritten = copy.deepcopy(program)
        selected = {clue_filters[sid] for sid in dropped_filters}
        for op in rewritten["nodes"]:
            if op["id"] in selected:
                oid, parents = op["id"], list(op["inputs"])
                op.clear()
                op.update({
                    "id": oid, "op": "project",
                    "inputs": parents, "fields": ["case_id", "cue"],
                })
        return answer_values(evaluate_program(
            rewritten, sources=modified_sources or sources,
        ))

    clue_checks = {
        sid: multiple(run(dropped_filters=(sid,)))
        for sid in plan["clues"]
    }
    discovery_checks = {
        t["macro_id"]: multiple(
            run(dropped_filters=t["source_scope"])
        )
        for t in plan["macro_tasks"] if t["role"] == "discovery"
    }
    all_relaxed = run(dropped_filters=plan["clues"])
    project_id = plan["synthetic_entities"][target]["entity_id"]
    cases = plan["all_cases"]
    i = cases.index(target)
    roots = {
        case: plan["candidate_paths"][case][0]["subject"]
        for case in cases
    }
    add_cases = {
        "link_a": cases[(i+3) % len(cases)], # W
        "link_b": cases[(i+2) % len(cases)], # V
    }
    link_checks = {}
    for sid, other_case in add_cases.items():
        modified = copy.deepcopy(sources)
        uri = roots[other_case]
        source = {
            "project_id": project_id, "entity_0": uri,
            "route_key": v3._route_key(project_id, uri),
        }
        modified[sid]["observations"].append({
            "observation_id": sid + ".CF", "fact_refs": [sid + ":F-CF"],
            "values": source, "claim_status": "counterfactual_candidate",
        })
        link_checks[sid] = multiple(run(modified))

    u = cases[(i+1) % len(cases)]
    u_object = plan["real_fork"]["candidate_forks"][u]["object"]
    modified = copy.deepcopy(sources)
    row = next(item for item in plan["synthetic_graph"]["event_rows"]
               if item["project_id"] == project_id)
    changed = copy.deepcopy(row)
    changed["motif_object"] = u_object
    modified["event_ledger"]["observations"].append({
        "observation_id": "event_ledger.CF",
        "fact_refs": ["event_ledger:F-CF"],
        "values": changed,
    })
    event_check = multiple(run(modified))

    # If the actual secondary YAGO relation for U were *falsely* reported
    # as a permitted real-fork object, it would survive as another answer.
    # This is a robustness/necessity probe, never recorded as true RDF.
    modified = copy.deepcopy(sources)
    real_fork_obj = plan["real_fork"]["candidate_forks"][target]["object"]
    u_root = roots[u]
    changed_row = next(
        record for record in modified["real_fork"]["observations"]
        if record["values"]["entity_0"] == u_root
    )
    changed_row["values"]["motif_object"] = real_fork_obj
    motif_check = multiple(run(modified))
    passed = (
        unique and all(clue_checks.values())
        and all(discovery_checks.values())
        and multiple(all_relaxed) and all(link_checks.values())
        and event_check and motif_check
    )
    return {
        "policy": NECESSITY_POLICY,
        "passed": passed,
        "identification_unique": unique,
        "single_clue_necessary": clue_checks,
        "discovery_macro_necessary": discovery_checks,
        "all_discovery_bypass_ambiguous": multiple(all_relaxed),
        "link_branch_necessary": link_checks,
        "synthetic_event_necessary": event_check,
        "grounded_real_fork_necessary": motif_check,
        "required_real_motif_kind": "real_fork",
        "real_source_lineage": "YAGO-4.6",
        "synthetic_event_count": len(plan["synthetic_graph"]["events"]),
        "limitation": (
            "Controlled source and filter re-planning, with hypothetical "
            "corruption of one YAGO observation for sensitivity. No global "
            "proof of solution optimality or independent historical sources."
        ),
    }
