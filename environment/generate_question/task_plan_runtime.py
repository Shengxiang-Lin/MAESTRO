"""Interpret the SINGLE frozen Task Plan using the original Stage-1 IR.

No historical make_spec(), no duplicate DAG/Sketch/World Spec, and no
independent executor. The same IR is consumed by old Compiler, Quality,
Replay and Stage-2 Sub-Agent WorkProduct.
"""
from __future__ import annotations

import copy
import random

from environment.schema import digest
from .task_plan import PATTERN, VERSION, validate_spec


def _route_key(pid, entity):
    return "route:" + digest(pid, entity, "task-plan-source-link")[:20]


def _project(seed, ordinal, case):
    return "urn:maestro:fictional:project:" + digest(
        seed, ordinal, case, "project"
    )[:20]


def _event(pid, which):
    return "urn:maestro:fictional:event:" + digest(pid, which)[:20]


def _condition_rows(plan):
    """Only the chosen status OR year differentiates eligible events."""
    condition = plan["condition"]
    if condition["field"] == "event_status":
        return (
            {"event_status": "completed", "event_year": 2025},
            {"event_status": "scheduled", "event_year": 2025},
        )
    return (
        {"event_status": "recorded", "event_year": 2025},
        {"event_status": "recorded", "event_year": 2024},
    )


def generate_world(spec: dict) -> dict:
    from .world import _rows, WORLD_VERSION
    problems = validate_spec(spec)
    if problems:
        raise ValueError("invalid_task_plan_world:" + ",".join(problems))
    plan = spec["macro_plan"]
    b = plan["world_binding"]
    target, cases = b["target_case"], b["all_cases"]
    paths = b["candidate_paths"]
    composed = plan.get("knowledge_world_mode") == "compose"
    if composed:
        # Source views are materialized from the canonical typed World,
        # not separate ad-hoc path/fork tables. The private projection is
        # checked against the World by validate_spec above.
        model = b["world_model"]
        relation_by_case = {case: [] for case in cases}
        for edge in model["relations"]:
            relation_by_case[edge["case_id"]].append(edge)
        paths = {
            case: [{
                key: edge[key] for key in (
                    "subject", "predicate", "object", "object_label", "origin"
                )
            } for edge in sorted(relation_by_case[case],
                                  key=lambda row: row["hop"])]
            for case in cases
        }
    rng = random.Random(int(digest(
        spec["seed"], spec["ordinal"], "task-plan-rows"
    )[:15], 16))
    tables = {}
    required = (
        model.get("required_context_relation") if composed else None
    )
    from .world_motifs import (
        composed_project_id, composed_event_id, KEYED_EVENT_VERSION,
    )
    keyed_events = (
        composed and model.get("keyed_event_relation_version")
        == KEYED_EVENT_VERSION
    )
    for sid, cue in plan["clues"].items():
        rows = []
        for case in cases:
            shadow = b["case_shadow_slots"].get(case)
            ok = case == target or (shadow is not None
                                    and shadow != cue["source_slot"])
            value = (
                model["context_by_case"][case]["anchor_label"]
                if required and sid == required["source_id"]
                else cue["value"] if ok else cue["alternative"]
            )
            if required and sid == required["source_id"]:
                # Validate the evidence projection against the actual World
                # relation, not the chosen target answer or a copied value.
                if (value == cue["value"]) != ok:
                    raise ValueError("required_shared_world_evidence_not_necessary")
            rows.append({"case_id": case, "cue": value})
        rng.shuffle(rows)
        tables[sid] = rows
    projects = {
        case: (
            composed_project_id(
                spec["seed"], spec["ordinal"], case
            )
            if composed else _project(
                spec["seed"], spec["ordinal"], case
            )
        ) for case in cases
    }
    def event_key(pid, which):
        return (composed_event_id(pid, which)
                if composed else _event(pid, which))
    tables["entity_index"] = [{
        "case_id": case, "project_id": pid,
        "project_name": "Project " + case.split("-")[-1].upper(),
    } for case, pid in projects.items()]
    mode = plan["mode"]
    if mode == "trace":
        tables["project_ref"] = [{
            "project_id": projects[case],
            "entity_0": paths[case][0]["subject"],
        } for case in cases]
    if mode in {"event", "verify"}:
        # Task Plan's new evidence policy forbids expressing the answer
        # criterion twice. Recurring events deliberately share the SAME
        # neutral title inside each project; only the required status/year
        # field distinguishes eligibility. Historical frozen task_plan
        # scenes (without this policy) replay their original row names.
        from .task_plan import BLIND_EVENTS
        blind = plan.get("evidence_sampling_policy") == BLIND_EVENTS
        semantic = plan["goal"].get("semantic") if (
            plan.get("semantic_grounding_policy") ==
            "goal_constrained_yago_grounding_v1"
        ) else None
        neutral_names = {
            "academic": "Correspondence review session",
            "biographical": "Biography file review",
            "institutional": "Registry file review",
            "geographical": "Regional record review",
        }
        # This descriptor is shared by all events of the same dossier;
        # it cannot act as a shortcut for status/year eligibility.
        neutral_label = (
            neutral_names[semantic["domain"]] if semantic
            else "Project review meeting"
        )
        eligible, ineligible = _condition_rows(plan)
        event_rows = []
        if mode == "event":
            event_refs = []
            indexed_links = {
                (link["case_id"], link["event_role"]): link
                for link in model["event_links"]
            } if keyed_events else {}
        for i, case in enumerate(cases):
            pid = projects[case]
            good_event = event_key(pid, "qualified")
            other_event = event_key(pid, "other")
            if mode == "event":
                qualified_link = indexed_links.get((case, "qualified"))
                other_link = indexed_links.get((case, "other"))
                if keyed_events:
                    if (not qualified_link or not other_link
                            or qualified_link["project_id"] != pid
                            or other_link["project_id"] != pid
                            or qualified_link["event_id"] != good_event
                            or other_link["event_id"] != other_event):
                        raise ValueError("event_relation_world_projection_mismatch")
                    good_details = {
                        field: qualified_link[field]
                        for field in ("event_status", "event_year")
                    }
                    other_details = {
                        field: other_link[field]
                        for field in ("event_status", "event_year")
                    }
                    qualified_entity = qualified_link["entity_0"]
                    other_entity = other_link["entity_0"]
                else:
                    good_details, other_details = eligible, ineligible
                    qualified_entity = paths[case][0]["subject"]
                    other_entity = paths[cases[(i+1) % len(cases)]][0]["subject"]
                event_rows.extend([
                    {"project_id": pid, "event_id": good_event,
                     "event_name": ((neutral_label + " " + case[-5:]) if blind else
                                    ("Qualified project review " + case[-5:])),
                     **good_details},
                    {"project_id": pid, "event_id": other_event,
                     "event_name": ((neutral_label + " " + case[-5:]) if blind else
                                    ("Other project review " + case[-5:])),
                     **other_details},
                ])
                event_refs.extend([
                    {"event_id": good_event, "entity_0": qualified_entity},
                    {"event_id": other_event, "entity_0": other_entity},
                ])
            else:
                fork = (model["verification_facts"] if composed else
                        b["real_fork"]["candidate_forks"])
                for shift in (0, 2, 3):
                    related = cases[(i+shift) % len(cases)]
                    event_rows.append({
                        "project_id": pid, "event_id": good_event,
                        "event_name": ((neutral_label + " " + case[-5:]) if blind else
                                        ("Qualified project session " + case[-5:])),
                        **eligible,
                        "motif_object": fork[related]["object"],
                    })
                other = cases[(i+1) % len(cases)]
                event_rows.append({
                    "project_id": pid, "event_id": other_event,
                    "event_name": ((neutral_label + " " + case[-5:]) if blind else
                                    ("Unqualified project session " + case[-5:])),
                    **ineligible, "motif_object": fork[other]["object"],
                })
        rng.shuffle(event_rows)
        tables["event_ledger"] = event_rows
        if mode == "event":
            tables["event_ref"] = event_refs
    if mode == "verify":
        from .real_motif_grounding import validate_frozen_required_fork
        if b["candidate_paths"][b["target_case"]][0]["origin"] == "synthetic":
            from .task_plan import _validate_fictional_fork
            failures = _validate_fictional_fork(b["real_fork"], paths)
        else:
            failures = validate_frozen_required_fork(b["real_fork"], paths)
        if failures:
            raise ValueError("missing_required_relation_fork:" +
                             ",".join(failures))
        for sid, shifts in (
            ("link_a", (0, 1, 2)), ("link_b", (0, 1, 3)),
        ):
            tables[sid] = [
                {"project_id": projects[case],
                 "route_key": _route_key(
                     projects[case],
                     paths[cases[(i+shift)%len(cases)]][0]["subject"],
                 ),
                 "entity_0": paths[cases[(i+shift)%len(cases)]][0]["subject"]}
                for i, case in enumerate(cases)
                for shift in shifts
            ]
        tables["real_fork"] = [{
            "entity_0": paths[case][0]["subject"],
            "motif_object": fact["object"],
        } for case, fact in (
            model["verification_facts"] if composed else
            b["real_fork"]["candidate_forks"]
        ).items()]
    for hop in range(plan["hops"]):
        left = f"entity_{hop}"
        right = "answer" if hop == plan["hops"]-1 else f"entity_{hop+1}"
        tables[f"relation_{hop+1}"] = [{
            left: path[hop]["subject"],
            right: path[hop]["object_label"] if right == "answer"
                   else path[hop]["object"],
        } for path in paths.values()]
    bank = {}
    for role in spec["sources"]:
        sid = role["source_id"]
        rows = tables[sid]
        rng.shuffle(rows)
        src = _rows(
            sid, role["modality"], role["title"], role["purpose"], rows,
        )
        src["access_policy"] = {
            "mode": role["access_mode"],
            **({"key_field": role["key_field"]}
               if role["access_mode"] == "keyed_lookup" else {}),
            **({"key_cardinality": "many"}
               if sid == "event_ledger" and keyed_events else {}),
        }
        if sid.startswith("relation_") or sid == "real_fork":
            fictional = paths[target][0]["origin"] == "synthetic"
            if sid.startswith("relation_"):
                hop = int(sid.rsplit("_", 1)[-1]) - 1
                source_facts = {
                    path[hop]["subject"]: path[hop]
                    for path in paths.values()
                }
                key = f"entity_{hop}"
            else:
                source_facts = {
                    fact["subject"]: fact
                    for fact in (
                        model["verification_facts"].values() if composed else
                        b["real_fork"]["candidate_forks"].values()
                    )
                }
                key = "entity_0"
            src["evidence_kind"] = (
                "fictional_relation_facts" if fictional else "real_yago_facts"
            )
            src["independence_note"] = (
                "A simulated record within the fictional world, NOT "
                "independent real-world corroboration." if fictional
                else "Each true RDF fact originates in YAGO, not a second "
                     "independent historical publisher."
            )
            for obs in src["observations"]:
                row = source_facts[obs["values"][key]]
                fact = {field: row[field]
                        for field in ("subject", "predicate", "object")}
                obs["knowledge_fact"] = fact
                obs["evidence_origin_id"] = (
                    ("synthetic:" if fictional else "YAGO-4.6:")
                    + digest(*fact.values())[:20]
                )
        else:
            src["evidence_kind"] = (
                "unverified_fictional_candidate_claims"
                if sid in {"link_a", "link_b"} else
                "fictional_project_and_event_facts"
            )
            for obs in src["observations"]:
                obs["evidence_origin_id"] = "synthetic:" + digest(
                    sid, obs["values"]
                )[:20]
                obs["claim_status"] = (
                    "unverified_candidate"
                    if sid in {"link_a", "link_b"} else
                    "fictional_observation"
                )
        bank[sid] = src
    if composed and "context_version" in model and not required:
        # Old frozen composed Worlds retain their original source metadata.
        # New Worlds put the *required* affiliation only in its authorized
        # source. Repeating it as document_context in every observation
        # would leak the answer and invalidate the evidence-dependency task.
        case_by_project = {pid: case for case, pid in projects.items()}
        case_by_event = {
            event_key(pid, which): case
            for case, pid in projects.items()
            for which in ("qualified", "other")
        }
        start_case = {
            path[0]["subject"]: case for case, path in paths.items()
        }
        hop_case = [
            {path[hop]["subject"]: case for case, path in paths.items()}
            for hop in range(plan["hops"])
        ]
        for sid, src in bank.items():
            for obs in src["observations"]:
                values = obs["values"]
                if "case_id" in values:
                    case = values["case_id"]
                elif "project_id" in values:
                    case = case_by_project[values["project_id"]]
                elif "event_id" in values:
                    case = case_by_event[values["event_id"]]
                elif sid.startswith("relation_"):
                    hop = int(sid.rsplit("_", 1)[-1]) - 1
                    case = hop_case[hop][values[f"entity_{hop}"]]
                elif sid == "real_fork":
                    case = start_case[values["entity_0"]]
                else:
                    raise ValueError("unmapped_context_evidence:" + sid)
                context = model["context_by_case"][case]
                obs["document_context"] = {
                    "collection_id": context["anchor_id"],
                    "collection_title": context["anchor_label"],
                    "site_id": context["site_id"],
                    "site_title": context["site_label"],
                    "record_id": "DOC-" + digest(
                        spec["seed"], spec["ordinal"], case, sid,
                    )[:12].upper(),
                }
                # Readable text is a redundant rendering of the existing
                # executable fields, never another answer-defining predicate.
                # In particular it may not encode target/eligibility in a
                # descriptive field that bypasses a Macro's explicit filter.
                obs["record_excerpt"] = (
                    "The " + src["title"] + " lists " + ", ".join(
                        str(k).replace("_", " ") + ": " + str(v)
                        for k, v in values.items()
                    ) + "."
                )
    return {
        "version": WORLD_VERSION, "simulated_only": True,
        "sources": bank,
        "entities": [{
            "entity_id": pid, "entity_type": "fictional_project",
            "display_name": "Project " + case.split("-")[-1].upper(),
            "origin": "synthetic", "fictional": True,
        } for case, pid in projects.items()],
        "generation_seed": spec["seed"],
        "knowledge_witness": copy.deepcopy(spec["knowledge_witness"]),
        "value_sampling_contract": "direct_task_plan_grounding_v1",
        **({"typed_knowledge_world": copy.deepcopy(b["world_model"])}
           if composed else {}),
        **({"knowledge_origin": "synthetic"}
           if paths[target][0]["origin"] == "synthetic" else {}),
    }


def compile_macro_solvers(spec: dict) -> dict:
    issues = validate_spec(spec)
    if issues:
        raise ValueError("invalid_task_plan_program:" + ",".join(issues))
    plan = spec["macro_plan"]
    nodes, groups, outputs = [], [], {}
    for task in plan["macro_tasks"]:
        mid, op = task["macro_id"], task["operation"]
        parents = [outputs[p] for p in task["depends_on"]]
        local = []

        def add(name, *inputs, **kw):
            nid = f"{mid.lower()}_op{len(local)+1:02d}"
            local.append({
                "id": nid, "op": name, "inputs": list(inputs), **kw
            })
            return nid

        if op == "match_clues":
            matches = []
            for sid in task["source_scope"]:
                scanned = add("scan", source_id=sid)
                filtered = add(
                    "filter_eq", scanned, field="cue",
                    value=plan["clues"][sid]["value"],
                )
                matches.append(add(
                    "project", filtered, fields=["case_id"]
                ))
            joined = add(
                "join", matches[0], matches[1],
                left_on="case_id", right_on="case_id",
            )
            terminal = add("unique", joined)
        elif op == "identify_project":
            joined = parents[0]
            for parent in parents[1:]:
                joined = add(
                    "join", joined, parent,
                    left_on="case_id", right_on="case_id",
                )
            unique = add("unique", joined)
            indexed = add(
                "lookup_by_key", unique, source_id="entity_index",
                input_key_field="case_id", source_key_field="case_id",
            )
            terminal = add("project", indexed, fields=["project_id"])
        elif op == "reference":
            lookup = add(
                "lookup_by_key", parents[0], source_id="project_ref",
                input_key_field="project_id", source_key_field="project_id",
            )
            terminal = add("project", lookup, fields=["entity_0"])
        elif op in {"qualifying_event", "event_eligibility"}:
            # New composed event Worlds use a true dependent retrieval:
            # the project_id must arrive with provenance from the previous
            # identification Macro. Historical scenes still scan+join.
            keyed = (
                plan.get("knowledge_world_mode") == "compose"
                and plan["world_binding"]["world_model"].get(
                    "keyed_event_relation_version"
                ) is not None
            )
            if keyed:
                observed = add(
                    "lookup_by_key", parents[0], source_id="event_ledger",
                    input_key_field="project_id", source_key_field="project_id",
                )
            else:
                observed = add("scan", source_id="event_ledger")
            condition = plan["condition"]
            filtered = add(
                condition["op"], observed, field=condition["field"],
                value=condition["value"],
            )
            if keyed:
                linked = filtered
            else:
                linked = add(
                    "join", parents[0], filtered,
                    left_on="project_id", right_on="project_id",
                )
            fields = ["event_id"] if op == "qualifying_event" else [
                "motif_object"
            ]
            terminal = add("project", linked, fields=fields)
        elif op == "event_reference":
            lookup = add(
                "lookup_by_key", parents[0], source_id="event_ref",
                input_key_field="event_id", source_key_field="event_id",
            )
            terminal = add("project", lookup, fields=["entity_0"])
        elif op in {"directory_a", "directory_b"}:
            sid = task["source_scope"][0]
            scanned = add("scan", source_id=sid)
            joined = add(
                "join", parents[0], scanned,
                left_on="project_id", right_on="project_id",
            )
            terminal = add(
                "project", joined, fields=["route_key", "entity_0"]
            )
        elif op == "intersect_references":
            joined = add(
                "join", parents[0], parents[1],
                left_on="route_key", right_on="route_key",
            )
            projected = add(
                "project", joined, fields=["entity_0"]
            )
            terminal = add("unique", projected)
        elif op == "verify_real_fork":
            real = add(
                "lookup_by_key", parents[0], source_id="real_fork",
                input_key_field="entity_0", source_key_field="entity_0",
            )
            joined = add(
                "join", real, parents[1],
                left_on="motif_object", right_on="motif_object",
            )
            projected = add(
                "project", joined, fields=["entity_0"]
            )
            terminal = add("unique", projected)
        elif op == "follow_yago":
            last = parents[0]
            for sid in task["source_scope"]:
                hop = int(sid.rsplit("_", 1)[-1]) - 1
                key = f"entity_{hop}"
                last = add(
                    "lookup_by_key", last, source_id=sid,
                    input_key_field=key, source_key_field=key,
                )
            terminal = add(
                "project", last, fields=list(task["output_schema"])
            )
        else:
            raise ValueError("unsupported_task_plan_operation:" + op)
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
        "nodes": nodes, "output": outputs[plan["terminal_macro_id"]],
        "authorized_source_ids": [
            r["source_id"] for r in spec["sources"]
        ],
        "frozen_problem_spec_id": spec["task_id"],
        "macro_solver_groups": groups, "macro_plan_version": VERSION,
    }


def mutate_witness(sources: dict, sid: str,
                   answer_rows: list[dict], spec: dict) -> dict:
    """Make ONE cited source observation unusable, retaining old audit."""
    updated = copy.deepcopy(sources)
    refs = {ref for row in answer_rows
            for ref in row["source_fact_refs"] if ref.startswith(sid+":")}
    selected = next(
        (row for row in updated[sid]["observations"]
         if refs.intersection(row["fact_refs"])), None
    )
    if selected is None:
        raise ValueError("missing_cited_task_plan_source:" + sid)
    data = selected["values"]
    if sid in spec["macro_plan"]["clues"]:
        data["cue"] = "__counterfactual_clue__"
    elif sid == "entity_index":
        data["project_id"] = "urn:maestro:fictional:missing-project"
    elif sid in ("project_ref", "event_ref"):
        data["entity_0"] = "urn:maestro:counterfactual:missing-entity"
    elif sid in ("link_a", "link_b"):
        data["route_key"] = "urn:maestro:counterfactual:wrong-route"
    elif sid == "event_ledger":
        condition = spec["macro_plan"]["condition"]
        data[condition["field"]] = (
            "__other_event_status__" if condition["field"] ==
            "event_status" else -1
        )
    elif sid == "real_fork":
        data["motif_object"] = "urn:maestro:counterfactual:fork"
    elif sid.startswith("relation_"):
        key = [f for f in data if f !=
               f"entity_{int(sid.rsplit('_',1)[-1])-1}"][0]
        data[key] = "urn:maestro:counterfactual:broken-relation"
    else:
        raise ValueError("unexpected_task_plan_mutation_source:" + sid)
    return updated


def audit_information(scene: dict) -> dict:
    """One generic, bounded counterfactual audit over the executable plan.

    Never treat an empty or blocked execution as evidence of necessity.
    Only fields declared as answer-defining are counted; background names
    and descriptive context are not credited.
    """
    from .program_ir import evaluate_program, answer_values
    plan = scene["problem_spec"]["macro_plan"]
    program = scene["gold_program"]
    sources = scene["world"]["sources"]
    expected = scene["gold_answer"][0]["answer"]
    def project_for_case(case):
        # Resolve the frozen source-level identifier instead of guessing
        # it from a particular URI or archive-key presentation scheme.
        records = [
            obs["values"]["project_id"] for obs in
            sources["entity_index"]["observations"]
            if obs["values"]["case_id"] == case
        ]
        if len(records) != 1:
            raise ValueError("missing_or_duplicate_target_project")
        return records[0]
    nodes = {n["id"]: n for n in program["nodes"]}
    cue_filters, condition_filter = {}, None
    for node in program["nodes"]:
        if node["op"] == "filter_eq" and node.get("field") == "cue":
            source = nodes[node["inputs"][0]]
            cue_filters[source["source_id"]] = node["id"]
        if (plan["condition"] and node.get("field") ==
                plan["condition"]["field"] and
                node.get("op") == plan["condition"]["op"]):
            condition_filter = node["id"]

    if set(cue_filters) != set(plan["clues"]):
        raise ValueError("task_plan_clue_filter_mismatch")
    if bool(plan["condition"]) != bool(condition_filter):
        raise ValueError("task_plan_condition_filter_mismatch")

    def _run(*, drop=(), drop_condition=False, changed=None):
        p = copy.deepcopy(program)
        targets = {cue_filters[sid] for sid in drop}
        for node in p["nodes"]:
            if node["id"] in targets or (
                drop_condition and node["id"] == condition_filter
            ):
                ident, parents = node["id"], list(node["inputs"])
                field_names = ["case_id", "cue"] if node["id"] in targets \
                    else list(sources["event_ledger"]["schema"])
                node.clear()
                node.update({
                    "id": ident, "op": "project", "inputs": parents,
                    "fields": field_names,
                })
        return answer_values(evaluate_program(
            p, sources=sources if changed is None else changed,
        ))

    def ambiguous(rows):
        labels = {row["answer"] for row in rows}
        return len(labels) > 1 and expected in labels

    clue = {
        sid: ambiguous(_run(drop=[sid])) for sid in plan["clues"]
    }
    required = plan["world_binding"].get("world_model", {}).get(
        "required_context_relation"
    )
    required_necessary = None
    if required:
        sid = required["source_id"]
        mapping = plan["world_binding"]["world_model"]["context_by_case"]
        observed = {
            obs["values"]["case_id"]: obs["values"]["cue"]
            for obs in sources[sid]["observations"]
        }
        required_necessary = (
            clue.get(sid) is True
            and set(observed) == set(mapping)
            and all(observed[case] == item["anchor_label"]
                    for case, item in mapping.items())
            and sum(v == required["target_anchor_label"]
                    for v in observed.values()) == (
                        len(observed) - len(required["excluded_cases"])
                    )
        )
    discoveries = {
        t["macro_id"]: ambiguous(
            _run(drop=t["source_scope"])
        )
        for t in plan["macro_tasks"] if t["operation"] == "match_clues"
    }
    all_discovery = ambiguous(_run(drop=plan["clues"]))
    condition = ambiguous(
        _run(drop_condition=True)
    ) if plan["condition"] else None
    branches = {}
    fork_necessary = None
    if plan["mode"] == "verify":
        from .task_plan import validate_spec
        if validate_spec(scene["problem_spec"]):
            raise ValueError("bad_frozen_plan_at_audit")
        b = plan["world_binding"]
        cases = b["all_cases"]
        target = b["target_case"]
        i = cases.index(target)
        project = project_for_case(target)
        for sid, shift in (("link_a", 3), ("link_b", 2)):
            changed = copy.deepcopy(sources)
            uri = b["candidate_paths"][cases[(i+shift)%len(cases)]][0]["subject"]
            row = {
                "project_id": project, "entity_0": uri,
                "route_key": _route_key(project, uri),
            }
            changed[sid]["observations"].append({
                "observation_id": sid+".COUNTERFACTUAL",
                "fact_refs": [sid+":F-COUNTERFACTUAL"],
                "values": row, "claim_status": "hypothetical_candidate",
            })
            branches[sid] = ambiguous(_run(changed=changed))
        # A hypothetical corrupted real-source observation for U, NEVER
        # added as a true RDF witness.
        u = cases[(i+1)%len(cases)]
        u_uri = b["candidate_paths"][u][0]["subject"]
        changed = copy.deepcopy(sources)
        hit = next(r for r in changed["real_fork"]["observations"]
                   if r["values"]["entity_0"] == u_uri)
        hit["values"]["motif_object"] = b["real_fork"][
            "candidate_forks"][target]["object"]
        fork_necessary = ambiguous(_run(changed=changed))
    # Extend the OLD shadow-case / executable re-planning audit with
    # a direct cross-field SUBSTITUTION probe. Removing a condition alone
    # does not catch an alternate visible label like "Qualified review".
    # Attempt to execute each descriptive or inactive-criterion value
    # INSTEAD of the required filter; no alternative may recover the
    # canonical unique answer. Avoid treating entity IDs / foreign keys
    # as semantic eligibility labels: they are legitimate join ports.
    substitution = None
    if plan.get("evidence_sampling_policy") == "eligibility_blind_event_names_v1":
        substitution = {}
        if plan["condition"]:
            selected_field = plan["condition"]["field"]
            alternatives = ["event_name", (
                "event_year" if selected_field == "event_status"
                else "event_status"
            )]
            by_field = {}
            target_case = plan["world_binding"]["target_case"]
            project = project_for_case(target_case)
            records = [
                row["values"] for row in sources["event_ledger"]["observations"]
                if row["values"]["project_id"] == project
            ]
            for field in alternatives:
                vals = sorted({row[field] for row in records})
                attempts = {}
                for value in vals:
                    alternate = copy.deepcopy(program)
                    for node in alternate["nodes"]:
                        if node["id"] == condition_filter:
                            node.update({
                                "op": "filter_eq", "field": field,
                                "value": value,
                            })
                    answers = answer_values(evaluate_program(
                        alternate, sources=sources,
                    ))
                    attempts[str(value)] = {
                        "answer_count": len(answers),
                        "ambiguous_with_original": ambiguous(answers),
                        "unique_shortcut": answers == scene["gold_answer"],
                    }
                by_field[field] = {
                    "candidate_values": [str(v) for v in vals],
                    "attempts": attempts,
                    "passed": bool(attempts) and all(
                        row["ambiguous_with_original"]
                        and not row["unique_shortcut"]
                        for row in attempts.values()
                    ),
                }
            substitution = {
                "applicable": True, "tested_fields": by_field,
                "passed": all(row["passed"] for row in by_field.values()),
                "scope": (
                    "Executable replacement of eligibility predicate by "
                    "all values of non-target descriptive/other-condition "
                    "fields for the identified project; opaque ID and "
                    "relation-object keys are excluded."
                ),
            }
        else:
            substitution = {
                "applicable": False, "tested_fields": {}, "passed": True,
            }
    result = {
        "policy": "single_task_plan_counterfactual_v1",
        "passed": (
            all(clue.values()) and all(discoveries.values())
            and all_discovery and (condition is None or condition)
            and all(branches.values())
            and (fork_necessary is None or fork_necessary)
            and (required_necessary is None or required_necessary)
            and (substitution is None or substitution["passed"])
        ),
        "clue_necessary": clue,
        "discovery_necessary": discoveries,
        "all_discovery_ambiguous": all_discovery,
        "event_field_necessary": condition,
        "simulated_sources_necessary": branches,
        "real_fork_necessary": fork_necessary,
        **({"required_shared_relation_necessary": required_necessary,
            "required_shared_relation_source": required["source_id"]}
           if required else {}),
        "mode": plan["mode"],
        "limitations": (
            "Bounded executable filter relaxation and source interventions, "
            "not proof of globally optimal or unique possible plans."
        ),
    }
    if substitution is not None:
        result["non_target_field_substitution"] = substitution
    return result
