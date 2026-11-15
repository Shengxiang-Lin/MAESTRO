"""Macro-first v3.1: coarse investigations grounded in real YAGO and fictional case files.

Distinct from the frozen v3 and v2 contracts. A Macro owns a *bundle* of
source-scoped evidence operations, not a single filter or KB hop.
All fictional dossiers and reports are explicitly synthetic.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import random

from .composer import _digest, _source
from .yago import Edge, KnowledgeGraph

VERSION = "maestro.macro_graph.v3.1"
SCENE_VERSION = "maestro.graph_grounded_scene.v3.1"

# Each group investigates TWO independent observations about the same
# fictional catalog case, not arbitrary color/status attributes of a real person.
BUNDLES = (
    (
        ("image", "photographed archival intake card", "seal motif",
         ("spiral", "star", "wave", "triangle", "crescent")),
        ("table", "collection finding-aid index", "drawer code",
         ("cedar", "marble", "cobalt", "amber", "elm")),
    ),
    (
        ("audio", "spoken catalog annotations", "announced shelf code",
         ("delta", "north", "juniper", "harbor", "ridge")),
        ("text", "curator's cross-reference notes", "series designation",
         ("atlas", "heritage", "survey", "ledger", "fieldwork")),
    ),
    (
        ("video", "digitized exhibition reel notes", "reel marker",
         ("opening", "handoff", "arrival", "departure", "handover")),
        ("text", "independent collection inventory", "document class",
         ("accession", "inscription", "receipt", "memo", "catalog")),
    ),
)
FRAMES = {
    "biographical": (
        ("historical archive", "historian", "reconcile the identity behind a recovered biographical dossier"),
        ("municipal records office", "research librarian", "identify the subject of a mislabeled historical file"),
        ("oral history collection", "archivist", "connect catalog fragments to a documented person"),
    ),
    "geographic": (
        ("regional archive", "geographic researcher", "identify the place referenced by several map records"),
        ("municipal documentation office", "records analyst", "reconcile place references in archived documents"),
        ("heritage mapping collection", "curator", "resolve an unidentified location dossier"),
    ),
    "institutional": (
        ("institutional archives", "records specialist", "trace an organization or membership reference"),
        ("public documentation center", "research analyst", "reconcile organizational records from different catalogs"),
        ("university reference archive", "catalog editor", "identify an institutional dossier and its associations"),
    ),
    "cultural": (
        ("cultural history collection", "museum registrar", "identify a work or contributor across archival sources"),
        ("broadcast research archive", "curator", "reconcile several records of a historical work"),
        ("literary collection", "research librarian", "identify an archived cultural reference"),
    ),
    "general": (
        ("special collections archive", "archivist", "identify the subject behind a fragmented reference packet"),
        ("digital preservation office", "research analyst", "reconcile a set of disconnected catalog entries"),
        ("public reference desk", "catalog researcher", "recover the correct reference from mixed archival materials"),
    ),
}
QUESTIONS = (
    "At the {institution}, a {occupation} is asked to {purpose}. {observations} "
    "Identify the shared archival case, then use its linked reference records to follow {route}. {request}",
    "A {occupation} at the {institution} receives several independently cataloged records. "
    "{observations} Which documented entity can be established after tracing {route} from the matching case? {request}",
    "The {institution} needs to {purpose}. The available clues are {observations} "
    "First reconcile the case identifiers; next investigate the reference trail {route}. {request}",
    "A misplaced dossier is being reconstructed at the {institution}. {observations} "
    "Determine which case the records describe and investigate the documented chain {route}. {request}",
    "During an audit of the {institution}, a {occupation} encounters inconsistent filing locations. "
    "{observations} Locate the common dossier and investigate its associated {route} records. {request}",
    "A {occupation} needs a source-backed answer rather than a guess: {observations} "
    "Reconcile the archived case before retrieving the entity reached through {route}. {request}",
    "For the {institution}'s reference project, the immediate goal is to {purpose}. "
    "{observations} Use the common file reference to investigate {route}. {request}",
    "Several archival sources point to a single yet unnamed case. {observations} "
    "As the {occupation} responsible for the {institution}, determine the entity reached by following {route}. {request}",
)


def _domain(path: list[Edge]) -> str:
    name = " ".join(e.predicate.rsplit("/", 1)[-1].lower() for e in path)
    if any(x in name for x in ("birth", "death", "spouse", "parent", "child", "gender")):
        return "biographical"
    if any(x in name for x in ("location", "place", "neighbor", "contained", "capital", "country")):
        return "geographic"
    if any(x in name for x in ("member", "affiliation", "employee", "organization", "founder", "headquarter")):
        return "institutional"
    if any(x in name for x in ("director", "author", "appear", "perform", "producer", "actor")):
        return "cultural"
    return "general"


def choose_real_path(graph: KnowledgeGraph, rng: random.Random,
                     *, min_hops: int = 1, max_hops: int = 3) -> list[Edge]:
    """Prefer TRUE multi-hop YAGO paths. Never shorten below min_hops."""
    if not (1 <= min_hops <= max_hops <= 5):
        raise ValueError("invalid_real_hop_range")
    adjacency = graph.outgoing()
    # Candidates have a real outgoing relation on the destination.
    joinable = [edge for edge in graph.edges
                if any(n.object != edge.subject and n.object != edge.object
                       for n in adjacency.get(edge.object, ()))]
    target_lengths = list(range(min_hops, max_hops + 1))
    rng.shuffle(target_lengths)
    # Prefer genuine multi-hop paths, but vary 2-hop versus 3-hop tasks
    # rather than always taking the longest available chain.
    target_lengths.sort(key=lambda length: length > 1, reverse=True)
    for target in target_lengths:
        first_pool = joinable if target > 1 else graph.edges
        if not first_pool:
            continue
        for _ in range(120):
            chosen = rng.choice(first_pool)
            path = [chosen]
            seen = {chosen.subject, chosen.object}
            while len(path) < target:
                next_edges = [e for e in adjacency.get(path[-1].object, [])
                              if e.object not in seen and e.subject != e.object]
                if not next_edges:
                    break
                edge = rng.choice(next_edges)
                path.append(edge)
                seen.add(edge.object)
            if len(path) == target:
                return path
    raise ValueError("insufficient_connected_yago_path_for_required_hops")


def _macro(mid: str, *, role: str, objective: str, deps: list[str],
           source_scope: list[str], output_schema: dict[str, str],
           strategy: str, params: dict | None = None) -> dict:
    return {
        "macro_id": mid, "role": role, "objective": objective,
        "depends_on": deps, "source_scope": source_scope,
        "output_schema": output_schema,
        "solver_strategy": strategy, "solver_params": params or {},
        "evidence_requirements": ["typed_workproduct", "scoped_fact_refs"],
    }


def _structural_graph(rng: random.Random, n: int, review: bool) -> list[dict]:
    """First construct a typed partial order; World/entity binding happens later."""
    tasks = []
    for i in range(n):
        parent = (
            tasks[i - 1]["macro_id"]
            if i and rng.random() < 0.30 else None
        )
        task = _macro(
            f"M{i+1}", role=("identification", "investigation", "discovery")[i],
            objective="Investigate a catalog case by comparing two source records",
            deps=[parent] if parent else [],
            source_scope=[f"bundle_{i+1}_a", f"bundle_{i+1}_b"],
            output_schema={"candidate_ids": "list[str]"},
            strategy="investigate_bundle", params={"bundle_index": i},
        )
        tasks.append(task)
    # Only maximal investigation outputs are needed; transitive dependencies
    # should not be copied as fake direct edges.
    used = {p for t in tasks for p in t["depends_on"]}
    frontier = [t["macro_id"] for t in tasks if t["macro_id"] not in used]
    identity = f"M{len(tasks)+1}"
    tasks.append(_macro(
        identity, role="identification",
        objective="Reconcile the independent dossier investigations and recover its named reference entity",
        deps=frontier, source_scope=["case_crosswalk"],
        output_schema={"case_id": "str", "entity_uri": "str"},
        strategy="resolve_case",
    ))
    link = f"M{len(tasks)+1}"
    tasks.append(_macro(
        link, role="investigation",
        objective="Trace the archived research request through documented YAGO entity relations",
        deps=[identity], source_scope=["research_request"],
        output_schema={"linked_entity_uri": "str", "linked_entity": "str"},
        strategy="investigate_chain",
    ))
    if review:
        check = f"M{len(tasks)+1}"
        tasks.append(_macro(
            check, role="verification",
            objective="Resolve conflicting dossier assessments using a separately scoped adjudication record",
            deps=[identity],
            source_scope=["assessment_a", "assessment_b", "adjudication"],
            output_schema={"review_status": "str", "agreement": "str"},
            strategy="review_case",
        ))
        final = f"M{len(tasks)+1}"
        tasks.append(_macro(
            final, role="synthesis",
            objective="Combine the linked entity and the resolved dossier assessment",
            deps=[link, check], source_scope=[],
            output_schema={
                "answer": "str", "review_status": "str", "evidence_agreement": "str",
            },
            strategy="synthesize",
        ))
    for task in tasks:
        task["input_contract"] = {
            parent: dict(next(x["output_schema"] for x in tasks
                              if x["macro_id"] == parent))
            for parent in task["depends_on"]
        }
    return tasks


def _build_sources(graph: KnowledgeGraph, path: list[Edge], tasks: list[dict],
                   rng: random.Random) -> tuple[dict, dict]:
    n = sum(t["solver_strategy"] == "investigate_bundle" for t in tasks)
    # Exactly one distractor survives every source except source j; deleting
    # ANY one source makes the global dossier identity ambiguous.
    total_clues = n * 2
    cases = [f"Archive-{i+1:03d}" for i in range(total_clues + 1 + rng.randint(2, 4))]
    rng.shuffle(cases)
    target = cases[0]
    shadows = {cases[j+1]: j for j in range(total_clues)}
    background = cases[total_clues+1:]
    source_bank = {}
    clues = []
    for i in range(n):
        for j, (modality, title, attr, values) in enumerate(BUNDLES[i]):
            slot = i * 2 + j
            selected = rng.choice(values)
            alternative = rng.choice([x for x in values if x != selected])
            sid = f"bundle_{i+1}_{'a' if j == 0 else 'b'}"
            records = []
            for case in cases:
                if case == target:
                    ok = True
                elif case in shadows:
                    ok = shadows[case] != slot
                else:
                    ok = (slot not in (0, 1)) and rng.random() < 0.25
                records.append({"case_id": case, "cue": selected if ok else alternative})
            rng.shuffle(records)
            source_bank[sid] = _source(
                sid, modality, title, ["case_id", "cue"], records,
            )
            clues.append({
                "source_id": sid, "title": title, "attribute": attr,
                "value": selected, "group": i+1,
            })
        task = tasks[i]
        task["objective"] = (
            "Inspect the " + BUNDLES[i][0][1] + " and cross-check the "
            + BUNDLES[i][1][1] + " to identify matching archive cases"
        )
        task["solver_params"] = {
            "criteria": {
                f"bundle_{i+1}_a": clues[i*2]["value"],
                f"bundle_{i+1}_b": clues[i*2+1]["value"],
            }
        }

    others = []
    seen_entities = {path[0].subject}
    for _ in range(len(cases) * 50):
        if len(others) >= len(cases) - 1:
            break
        candidate = rng.choice(graph.edges).subject
        if candidate not in seen_entities:
            seen_entities.add(candidate)
            others.append(candidate)
    crosswalk = []
    for case in cases:
        entity = path[0].subject if case == target else (
            others.pop() if others else f"urn:maestro:fictional:unlinked:{case}"
        )
        crosswalk.append({"case_id": case, "entity_uri": entity})
    rng.shuffle(crosswalk)
    source_bank["case_crosswalk"] = _source(
        "case_crosswalk", "table", "catalog reference-to-entity crosswalk",
        ["case_id", "entity_uri"], crosswalk, mode="keyed_lookup",
    )
    path_rows = []
    for k, fact in enumerate(path, 1):
        sid = f"kb_fact_{k}"
        rows = [{
            "subject_uri": fact.subject,
            "predicate": fact.predicate,
            "object_uri": fact.object,
            "object_label": graph.label(fact.object),
        }]
        decoy_count = rng.randint(2, 5)
        decoys = set()
        for _ in range(decoy_count * 30):
            if len(decoys) >= decoy_count:
                break
            other = rng.choice(graph.edges)
            key = (other.subject, other.predicate, other.object)
            if (other.subject == fact.subject or other.object == fact.object
                    or key in decoys):
                continue
            decoys.add(key)
            rows.append({"subject_uri": other.subject,
                         "predicate": other.predicate,
                         "object_uri": other.object,
                         "object_label": graph.label(other.object)})
        rng.shuffle(rows)
        source_bank[sid] = _source(
            sid, "text", "YAGO relationship extract " + str(k),
            ["subject_uri", "predicate", "object_uri", "object_label"],
            rows, mode="keyed_lookup", origin="yago",
        )
        path_rows.append({
            "subject": fact.subject, "predicate": fact.predicate,
            "object": fact.object, "origin": "yago", "source_id": sid,
        })
    relation_task = next(t for t in tasks if t["solver_strategy"] == "investigate_chain")
    relation_task["source_scope"] += [f"kb_fact_{i+1}" for i in range(len(path))]
    route = [edge.predicate for edge in path]
    relation_task["solver_params"] = {"predicates": route}
    source_bank["research_request"] = _source(
        "research_request", "text", "curatorial research instruction",
        ["case_id", "route"], [
            {"case_id": target, "route": " | ".join(route)},
            *({"case_id": case, "route": " | ".join(reversed(route))}
              for case in cases[1:]),
        ], mode="keyed_lookup",
    )
    if any(t["solver_strategy"] == "review_case" for t in tasks):
        truth = rng.choice(("cleared", "conditional", "pending"))
        conflict = rng.random() < 0.55
        contrary = rng.choice([x for x in ("cleared", "conditional", "pending")
                               if x != truth])
        source_bank["assessment_a"] = _source(
            "assessment_a", "text", "first dossier assessment",
            ["case_id", "decision"],
            [{"case_id": target, "decision": truth}],
            mode="keyed_lookup",
        )
        source_bank["assessment_b"] = _source(
            "assessment_b", "table", "second assessor's independent register",
            ["case_id", "decision"],
            [{"case_id": target, "decision": contrary if conflict else truth}],
            mode="keyed_lookup",
        )
        source_bank["adjudication"] = _source(
            "adjudication", "text", "fictional archive adjudication memo",
            ["case_id", "decision"],
            [{"case_id": target, "decision": truth}], mode="keyed_lookup",
        )
    return {
        "version": "maestro.graph_world.v3.1",
        "simulated_only": True,
        "sources": source_bank,
        "real_fact_path": path_rows,
        "private_witness": {"case_id": target, "entity_uri": path[0].subject},
    }, {"clues": clues, "domain": _domain(path)}


def _resolve_one(source: dict, *, key: str, value: str) -> dict:
    rows = [r for r in source["observations"] if r["values"].get(key) == value]
    if len(rows) != 1:
        raise ValueError("nonunique_or_missing_source_record:" + source["source_id"])
    return rows[0]


def execute_macros(scene: dict) -> tuple[list[dict], dict]:
    """Gold execution sees only its own sources and declared ancestor products."""
    tasks = scene["macro_plan"]["macro_tasks"]
    sources = scene["world"]["sources"]
    products = {}
    feedback = []
    for task in tasks:
        mid, strategy = task["macro_id"], task["solver_strategy"]
        source_ids = task["source_scope"]
        scoped = {sid: sources[sid] for sid in source_ids}
        inputs = [products[p] for p in task["depends_on"]]
        refs: set[str] = set()
        steps = []
        if strategy == "investigate_bundle":
            upstream = set(inputs[0]["value"]["candidate_ids"]) if inputs else None
            candidate_sets = []
            witness = defaultdict(set)
            for sid in source_ids:
                source = scoped[sid]
                criterion = task["solver_params"]["criteria"][sid]
                matching = {}
                for row in source["observations"]:
                    values = row["values"]
                    if values["cue"] == criterion and (
                        upstream is None or values["case_id"] in upstream
                    ):
                        matching[values["case_id"]] = row
                candidate_sets.append(set(matching))
                for case_id, row in matching.items():
                    witness[case_id].update(row["fact_refs"])
                steps.append({
                    "tool": ("observe_simulated_media"
                             if source["modality"] in {"image", "video", "audio"}
                             else "search_and_read_records"),
                    "source_id": sid, "filter": criterion,
                    "candidate_count": len(matching),
                    "supporting_fact_refs": sorted(
                        ref for row in matching.values() for ref in row["fact_refs"]
                    ),
                })
            selected = set.intersection(*candidate_sets)
            if upstream is not None:
                selected &= upstream
            if not selected:
                raise ValueError("empty_bundle_workproduct")
            matched_witness = {
                case: sorted(set(witness[case]) |
                             (set(inputs[0]["witness"].get(case, [])) if inputs else set()))
                for case in selected
            }
            value = {"candidate_ids": sorted(selected)}
            refs.update(ref for row in matched_witness.values() for ref in row)
            steps.append({"tool": "compare_and_join_evidence", "source_id": None,
                          "candidate_count": len(selected),
                          "supporting_fact_refs": sorted(refs)})
            witness_out = matched_witness
        elif strategy == "resolve_case":
            if not inputs:
                raise ValueError("missing_candidate_inputs")
            cases = set(inputs[0]["value"]["candidate_ids"])
            for parent in inputs[1:]:
                cases &= set(parent["value"]["candidate_ids"])
            if len(cases) != 1:
                raise ValueError("dossier_identity_not_unique")
            case_id = next(iter(cases))
            obs = _resolve_one(scoped["case_crosswalk"], key="case_id", value=case_id)
            value = {"case_id": case_id, "entity_uri": obs["values"]["entity_uri"]}
            refs.update(obs["fact_refs"])
            for p in inputs:
                refs.update(p["witness"].get(case_id, []))
            steps.extend([
                {"tool": "reconcile_candidate_handoffs", "source_id": None,
                 "candidate_count": 1, "supporting_fact_refs": sorted(refs)},
                {"tool": "query_reference_directory", "source_id": "case_crosswalk",
                 "matched_record": case_id,
                 "supporting_fact_refs": obs["fact_refs"]},
            ])
            witness_out = {}
        elif strategy == "investigate_chain":
            upstream = inputs[0]["value"]
            request = _resolve_one(scoped["research_request"],
                                   key="case_id", value=upstream["case_id"])
            preds = request["values"]["route"].split(" | ")
            if preds != task["solver_params"]["predicates"]:
                raise ValueError("research_request_route_mismatch")
            refs.update(inputs[0]["refs"])
            refs.update(request["fact_refs"])
            current = upstream["entity_uri"]
            steps.append({"tool": "read_research_brief", "source_id": "research_request",
                          "supporting_fact_refs": request["fact_refs"]})
            label = ""
            for k, predicate in enumerate(preds, 1):
                sid = f"kb_fact_{k}"
                matches = [r for r in scoped[sid]["observations"]
                           if r["values"]["subject_uri"] == current
                           and r["values"]["predicate"] == predicate]
                if len(matches) != 1:
                    raise ValueError("nonunique_or_missing_yago_relation")
                obs = matches[0]
                current = obs["values"]["object_uri"]
                label = obs["values"]["object_label"]
                refs.update(obs["fact_refs"])
                steps.append({
                    "tool": "query_yago_relation", "source_id": sid,
                    "predicate": predicate, "result_entity_uri": current,
                    "supporting_fact_refs": obs["fact_refs"],
                })
            value = {"linked_entity_uri": current, "linked_entity": label}
            witness_out = {}
        elif strategy == "review_case":
            case_id = inputs[0]["value"]["case_id"]
            decisions = []
            for sid in ("assessment_a", "assessment_b", "adjudication"):
                obs = _resolve_one(scoped[sid], key="case_id", value=case_id)
                decisions.append(obs["values"]["decision"])
                refs.update(obs["fact_refs"])
                steps.append({"tool": "read_independent_assessment",
                              "source_id": sid, "decision": obs["values"]["decision"],
                              "supporting_fact_refs": obs["fact_refs"]})
            agreement = "corroborated" if decisions[0] == decisions[1] == decisions[2] else "discrepancy_resolved"
            value = {"review_status": decisions[2], "agreement": agreement}
            steps.append({"tool": "compare_and_adjudicate_claims",
                          "source_id": None, "agreement": agreement,
                          "supporting_fact_refs": sorted(refs)})
            witness_out = {}
        elif strategy == "synthesize":
            refs.update(inputs[0]["refs"])
            refs.update(inputs[1]["refs"])
            value = {"answer": inputs[0]["value"]["linked_entity"],
                     "review_status": inputs[1]["value"]["review_status"],
                     "evidence_agreement": inputs[1]["value"]["agreement"]}
            steps.append({"tool": "synthesize_source_backed_result",
                          "source_id": None, "supporting_fact_refs": sorted(refs)})
            witness_out = {}
        else:
            raise ValueError("unknown_macro_strategy:" + strategy)
        if set(value) != set(task["output_schema"]):
            raise ValueError("v31_macro_schema_mismatch:" + mid)
        products[mid] = {"value": value, "refs": sorted(refs), "witness": witness_out}
        feedback.append({
            "task_id": "work_" + mid, "macro_id": mid,
            "macro_role": task["role"], "objective": task["objective"],
            "used_source_ids": source_ids,
            "used_upstream_handoffs": ["work_" + p for p in task["depends_on"]],
            "output_schema": task["output_schema"],
            "result_rows": len(value["candidate_ids"]) if strategy == "investigate_bundle" else 1,
            "fact_refs": sorted(refs),
            "workproduct": (
                [{"values": {"case_id": case},
                  "source_fact_refs": products[mid]["witness"][case]}
                 for case in value["candidate_ids"]]
                if strategy == "investigate_bundle"
                else [{"values": deepcopy(value), "source_fact_refs": sorted(refs)}]
            ),
            "internal_trace": steps,
            "trace_grounding": "simulated_tools_executed_against_scoped_structured_sources",
        })
    return feedback, products[tasks[-1]["macro_id"]]["value"]


def _question(rng: random.Random, graph: KnowledgeGraph, path: list[Edge],
              clues: list[dict], review: bool) -> tuple[str, dict]:
    domain = _domain(path)
    institution, occupation, purpose = rng.choice(FRAMES[domain])
    clue_groups = defaultdict(list)
    for item in clues:
        clue_groups[item["group"]].append(
            f"{item['title']} notes {item['attribute']} '{item['value']}'"
        )
    observations = " ".join(
        "One evidence bundle says " + " and ".join(group) + "."
        for _, group in sorted(clue_groups.items())
    )
    route = " → ".join(graph.label(e.predicate) for e in path)
    request = (
        "Also report the final dossier review status and whether independent assessments agreed."
        if review else "Give the linked entity and its catalog reference URI as the final answer."
    )
    draft = rng.choice(QUESTIONS).format(
        institution=institution, occupation=occupation,
        purpose=purpose, observations=observations,
        route=route, request=request,
    )
    return draft, {
        "domain": domain, "institution": institution, "occupation": occupation,
        "purpose": purpose, "clue_source_ids": [x["source_id"] for x in clues],
        "clue_values": {x["source_id"]: x["value"] for x in clues},
        "predicates": [e.predicate for e in path],
        "review_required": review, "answer_fields": (
            ["answer", "review_status", "evidence_agreement"]
            if review else ["linked_entity_uri", "linked_entity"]
        ),
        "style": "deterministic_contract_draft_not_llm_reviewed",
    }


def validate_v31(scene: dict, graph: KnowledgeGraph | None = None) -> dict:
    """Audit graph typing, grounded real facts, all-source necessity and replay."""
    if scene.get("version") != SCENE_VERSION:
        raise ValueError("invalid_v31_version")
    tasks = scene["macro_plan"]["macro_tasks"]
    names = set()
    for task in tasks:
        if task["macro_id"] in names or any(p not in names for p in task["depends_on"]):
            raise ValueError("invalid_macro_partial_order")
        names.add(task["macro_id"])
        if task.get("input_contract") != {
            p: next(x["output_schema"] for x in tasks if x["macro_id"] == p)
            for p in task["depends_on"]
        }:
            raise ValueError("invalid_typed_handoff")
        if not set(task["source_scope"]) <= set(scene["world"]["sources"]):
            raise ValueError("unauthorized_macro_source")
        if not task["objective"]:
            raise ValueError("missing_macro_objective")
    clue_sources = [
        sid for task in tasks if task["solver_strategy"] == "investigate_bundle"
        for sid in task["source_scope"]
    ]
    if len(clue_sources) < 4 or len(set(clue_sources)) != len(clue_sources):
        raise ValueError("invalid_evidence_bundle_partition")
    all_case_ids = {
        row["values"]["case_id"]
        for row in scene["world"]["sources"][clue_sources[0]]["observations"]
    }
    valid_sets = []
    clue_contract = scene["question_contract"]["clue_values"]
    for sid in clue_sources:
        if sid not in clue_contract:
            raise ValueError("missing_question_clue")
        source = scene["world"]["sources"][sid]
        valid_sets.append({
            row["values"]["case_id"] for row in source["observations"]
            if row["values"]["cue"] == clue_contract[sid]
        })
        if {r["values"]["case_id"] for r in source["observations"]} != all_case_ids:
            raise ValueError("inconsistent_source_case_universe")
    full = set.intersection(*valid_sets)
    if len(full) != 1:
        raise ValueError("dossier_not_uniquely_identified")
    if next(iter(full)) != scene["world"]["private_witness"]["case_id"]:
        raise ValueError("dossier_witness_wrong")
    # Necessity is about evidence content, not only erroring on a missing input.
    for i in range(len(valid_sets)):
        if len(set.intersection(*[s for j, s in enumerate(valid_sets) if j != i])) <= 1:
            raise ValueError("redundant_evidence_source:" + clue_sources[i])
    for i, fact in enumerate(scene["world"]["real_fact_path"], 1):
        if fact["origin"] != "yago" or fact["source_id"] != f"kb_fact_{i}":
            raise ValueError("false_yago_provenance")
        expected = {
            "subject_uri": fact["subject"], "predicate": fact["predicate"],
            "object_uri": fact["object"],
        }
        observations = scene["world"]["sources"][fact["source_id"]]["observations"]
        if not any(all(o["values"].get(k) == v for k, v in expected.items())
                   and o["origin"] == "yago" for o in observations):
            raise ValueError("missing_original_yago_fact")
        if graph is not None and Edge(fact["subject"], fact["predicate"],
                                      fact["object"]) not in graph.edges:
            raise ValueError("real_yago_fact_not_in_graph")
    expected_feedback, expected_answer = execute_macros(scene)
    if expected_feedback != scene["task_feedback"] or [expected_answer] != scene["gold_answer"]:
        raise ValueError("v31_replay_mismatch")
    replay_hash = _digest({
        "macro_plan": scene["macro_plan"], "world": scene["world"],
        "gold_answer": scene["gold_answer"], "question": scene["question"],
    })
    if replay_hash != scene["replay_hash"]:
        raise ValueError("v31_content_hash_mismatch")
    return {
        "passed": True, "macro_count": len(tasks),
        "evidence_source_count": len(clue_sources),
        "all_clue_sources_necessary": True,
        "real_yago_hops": len(scene["world"]["real_fact_path"]),
        "uses_conflict_resolution": any(
            task["solver_strategy"] == "review_case" for task in tasks
        ),
        "roles": dict(Counter(task["role"] for task in tasks)),
        "macro_internal_source_counts": {
            t["macro_id"]: len(t["source_scope"]) for t in tasks
        },
    }


def compose_v31(graph: KnowledgeGraph, *, seed: int = 17, ordinal: int = 0,
                min_hops: int = 1, max_hops: int = 3,
                max_branches: int = 3) -> dict:
    if not 2 <= max_branches <= len(BUNDLES):
        raise ValueError("invalid_max_branches")
    rng = random.Random(int(_digest([seed, ordinal, "macro-v31"]), 16))
    path = choose_real_path(graph, rng, min_hops=min_hops, max_hops=max_hops)
    n = rng.randint(2, max_branches)
    review = rng.random() < 0.65
    tasks = _structural_graph(rng, n, review)
    world, semantic = _build_sources(graph, path, tasks, rng)
    question, contract = _question(rng, graph, path, semantic["clues"], review)
    task_id = f"question-v31-{ordinal:06d}-{_digest([seed, ordinal, path[0].subject])[:8]}"
    # Append invariant semantic clause to the written contract rather than
    # allowing an author to silently change the task's requested deliverables.
    feedback, answer = execute_macros({
        "macro_plan": {"macro_tasks": tasks}, "world": world,
    })
    pending = {t["macro_id"] for t in tasks}
    complete = set()
    waves = []
    while pending:
        next_wave = [t["macro_id"] for t in tasks
                     if t["macro_id"] in pending
                     and set(t["depends_on"]) <= complete]
        if not next_wave:
            raise ValueError("macro_graph_is_cyclic")
        waves.append(["work_" + mid for mid in next_wave])
        pending.difference_update(next_wave)
        complete.update(next_wave)
    scene = {
        "version": SCENE_VERSION, "task_id": task_id,
        "question": question, "question_contract": contract,
        "question_origin": "v31_deterministic_contract_draft",
        "question_authoring": {"status": "draft", "model": None},
        "problem_spec": {"version": VERSION, "seed": seed, "ordinal": ordinal,
                         "goal": {"output_fields": list(answer)}},
        "macro_plan": {"version": VERSION, "macro_tasks": tasks,
                       "terminal_macro_id": tasks[-1]["macro_id"]},
        "macro_graph": {
            "version": VERSION,
            "tasks": [{"task_id": "work_" + t["macro_id"],
                       "depends_on": ["work_" + p for p in t["depends_on"]],
                       "source_ids": t["source_scope"], "macro_role": t["role"],
                       "objective": t["objective"], "outputs": t["output_schema"]}
                      for t in tasks],
            "reference_waves": waves,
            "terminal_task_id": "work_" + tasks[-1]["macro_id"],
            "macro_task_count": len(tasks),
        },
        "world": world,
        "gold_program": {
            "version": VERSION,
            "macro_ops": [{"macro_id": t["macro_id"],
                           "strategy": t["solver_strategy"],
                           "params": t["solver_params"]}
                          for t in tasks],
        },
        "gold_answer": [answer],
        "gold_trace": feedback, "task_feedback": feedback,
        "public": {
            "task_id": task_id, "question": question,
            "available_source_descriptions": [
                {"source_id": sid, "title": source["title"],
                 "modality_hint": source["modality"]}
                for sid, source in world["sources"].items()
            ],
            "status": "ready_for_investigation",
            "max_concurrent_subagents": 10,
        },
        "execution_assumptions": {
            "simulated_media": True, "full_world_hidden_from_main": True,
            "scope_limited_macro_execution": True,
        },
    }
    scene["replay_hash"] = _digest({
        "macro_plan": scene["macro_plan"], "world": scene["world"],
        "gold_answer": scene["gold_answer"], "question": question,
    })
    scene["quality_report"] = validate_v31(scene, graph)
    return scene
