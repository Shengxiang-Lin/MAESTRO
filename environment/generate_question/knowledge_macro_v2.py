"""Causally-necessary YAGO world for canonical Stage-1 (versioned v2).

Old typed_knowledge_v1 scenes are never changed. V2 requires all plausible
dossier candidates to have REAL, COMPLETE, distinct YAGO relation trails.
A missing discovery clue must leave multiple possible final answers.
"""
from __future__ import annotations

import copy
import random

from environment.schema import digest
from . import knowledge_macro as old

PATTERN = "typed_knowledge_v2"
VERSION = "maestro.macro_plan.typed_knowledge.v2"
KNOWLEDGE_VERSION = "maestro.yago_witness.v2"
NECESSITY_POLICY = "knowledge_candidate_replanning_v2"
HASH_SALT = "canonical-knowledge-frozen-v2"


def _as_v1(spec: dict) -> dict:
    """Return a *copy* satisfying v1 compiler's unmodified frozen contract."""
    result = copy.deepcopy(spec)
    result["reasoning_pattern"] = old.PATTERN
    result["family"] = old.PATTERN
    result["macro_plan"]["version"] = old.VERSION
    result["knowledge_witness"]["version"] = old.KNOWLEDGE_VERSION
    result["knowledge_contract_hash"] = digest(
        result["macro_plan"], result["knowledge_witness"],
        result["seed"], result["ordinal"], "canonical-knowledge-frozen-v1",
    )
    return result


def _alternatives(graph, index, anchor: list, *, count: int,
                  rng: random.Random) -> list[list]:
    """Find *real* disjoint trajectories with the same relation sequence.

    The complete chain survives later keyed lookups, and every candidate
    ends with a different terminal label. No synthetic 'wrong entity' can
    be used to manufacture necessity. Strict disjointness also ensures
    each keyed relation source has exactly one row per upstream URI.
    """
    predicates = tuple(e.predicate for e in anchor)
    first = index.by_predicate.get(predicates[0], [])
    if len(first) < count + 1:
        return []
    forbidden = [
        {e.subject} for e in anchor
    ]
    forbidden.append({anchor[-1].object})
    terminal_labels = {graph.label(anchor[-1].object)}
    result = []
    samples = rng.sample(first, min(len(first), 18000))
    for start in samples:
        if len(result) == count:
            break
        if start.subject in forbidden[0] or start.object in forbidden[1]:
            continue
        if start.subject == start.object:
            continue
        chosen = [start]
        seen = {start.subject, start.object}
        for hop, predicate in enumerate(predicates[1:], 1):
            options = [
                e for e in index.outgoing.get(chosen[-1].object, ())
                if e.predicate == predicate
                and e.object not in forbidden[hop+1]
                and e.object not in seen
                and e.subject != e.object
            ]
            if not options:
                break
            nxt = rng.choice(options)
            chosen.append(nxt)
            seen.add(nxt.object)
        if len(chosen) != len(anchor):
            continue
        label = graph.label(chosen[-1].object)
        if label in terminal_labels:
            continue
        result.append(chosen)
        for hop, fact in enumerate(chosen):
            forbidden[hop].add(fact.subject)
        forbidden[-1].add(chosen[-1].object)
        terminal_labels.add(label)
    return result if len(result) == count else []


def _serialized_path(graph, path) -> list[dict]:
    return [{
        "subject": fact.subject,
        "predicate": fact.predicate,
        "object": fact.object,
        "object_label": graph.label(fact.object),
        "origin": "yago",
    } for fact in path]


def make_spec(kg, *, seed: int, ordinal: int, difficulty: str,
              min_hops: int = 2, max_hops: int = 3,
              index: old.YagoKnowledgeIndex | None = None) -> dict:
    """Select a YAGO motif only if *all* clue-counterfactuals are solvable."""
    index = index or old.YagoKnowledgeIndex(kg)
    # Use separate deterministic search attempts. Reject sparse motifs
    # instead of backing them with fake YAGO facts.
    for attempt in range(120):
        try:
            base = old.make_spec(
                kg, seed=seed, ordinal=ordinal + attempt * 100003,
                difficulty=difficulty, min_hops=min_hops,
                max_hops=max_hops, index=index,
            )
        except ValueError:
            continue
        plan = base["macro_plan"]
        n_clues = len(plan["clues"])
        cases = [case for case in plan["all_cases"]
                 if case != plan["target_case"]]
        anchor = [
            type(kg.edges[0])(f["subject"], f["predicate"], f["object"])
            for f in plan["real_facts"]
        ]
        rng = random.Random(int(digest(
            seed, ordinal, attempt, "knowledge-replan-motif-v2")[:15], 16))
        alternatives = _alternatives(
            kg, index, anchor, count=len(cases), rng=rng,
        )
        if len(alternatives) != len(cases):
            continue
        plan["candidate_paths"] = {
            plan["target_case"]: _serialized_path(kg, anchor),
            **{
                case: _serialized_path(kg, path)
                for case, path in zip(cases, alternatives)
            },
        }
        # For source slot k, only shadow k disagrees with the target;
        # every background distractor disagrees with ALL required clues.
        plan["case_shadow_slots"] = {
            case: (slot if slot < n_clues else None)
            for slot, case in enumerate(cases)
        }
        plan["decoy_yago_facts"] = {
            f"relation_{hop+1}": [
                path[hop] for case, path in plan["candidate_paths"].items()
                if case != plan["target_case"]
            ]
            for hop in range(len(anchor))
        }
        plan["information_necessity_policy"] = NECESSITY_POLICY
        plan["version"] = VERSION
        base["reasoning_pattern"] = PATTERN
        base["family"] = PATTERN
        base["ordinal"] = ordinal
        base["task_id"] = (
            f"question-kn2-{ordinal:06d}-"
            f"{digest(seed, ordinal, attempt, plan['target_case'])[:8]}"
        )
        base["scene_id"] = f"yago-knowledge-v2-{ordinal:06d}"
        base["macro_information_necessity_version"] = NECESSITY_POLICY
        snapshot = base["knowledge_witness"]
        snapshot["version"] = KNOWLEDGE_VERSION
        # Frozen hash covers every target and competing true RDF triple.
        snapshot["candidate_facts"] = {
            case: [{
                key: fact[key] for key in ("subject", "predicate", "object")
            } for fact in path]
            for case, path in plan["candidate_paths"].items()
        }
        base["knowledge_contract_hash"] = digest(
            plan, snapshot, seed, ordinal, HASH_SALT,
        )
        problems = validate_spec(base)
        if problems:
            raise ValueError("knowledge_v2_bad_constructed_spec:" +
                             ",".join(problems))
        return base
    raise ValueError(
        "no_yago_motif_with_sufficient_complete_distinct_candidate_paths"
    )


def validate_spec(spec: dict) -> list[str]:
    """Validate v2 on top of unchanged v1 macro and witness invariants."""
    try:
        plan = spec["macro_plan"]
        witness = spec["knowledge_witness"]
        if (spec["reasoning_pattern"] != PATTERN or
                spec["family"] != PATTERN or plan["version"] != VERSION or
                witness["version"] != KNOWLEDGE_VERSION):
            return ["knowledge_v2_wrong_version"]
        if spec["knowledge_contract_hash"] != digest(
            plan, witness, spec["seed"], spec["ordinal"], HASH_SALT,
        ):
            return ["knowledge_v2_hash_mismatch"]
        legacy = _as_v1(spec)
        errors = old.validate_spec(legacy)
        if errors:
            return ["knowledge_v2_legacy_contract:" + ",".join(errors)]
        cases = plan["all_cases"]
        target = plan["target_case"]
        paths = plan["candidate_paths"]
        if (set(paths) != set(cases) or len(cases) < 7 or
                plan.get("information_necessity_policy") != NECESSITY_POLICY):
            return ["knowledge_v2_missing_candidate_paths"]
        if set(plan["case_shadow_slots"]) != set(cases) - {target}:
            return ["knowledge_v2_shadow_case_contract"]
        positions = sorted(
            v for v in plan["case_shadow_slots"].values() if v is not None
        )
        if positions != list(range(len(plan["clues"]))):
            return ["knowledge_v2_each_clue_needs_one_shadow"]
        route = [f["predicate"] for f in plan["real_facts"]]
        starts = [set() for _ in range(len(route)+1)]
        labels = set()
        for case in cases:
            chain = paths[case]
            if len(chain) != len(route):
                return ["knowledge_v2_incomplete_candidate_path"]
            for hop, fact in enumerate(chain):
                if fact["predicate"] != route[hop] or fact["origin"] != "yago":
                    return ["knowledge_v2_candidate_predicate_mismatch"]
                if fact["subject"] in starts[hop]:
                    return ["knowledge_v2_ambiguous_keyed_lookup"]
                starts[hop].add(fact["subject"])
                if hop and chain[hop-1]["object"] != fact["subject"]:
                    return ["knowledge_v2_disconnected_candidate"]
                if hop == len(route)-1:
                    if fact["object_label"] in labels:
                        return ["knowledge_v2_duplicate_terminal_answer"]
                    labels.add(fact["object_label"])
                if hop == len(route)-1 and fact["object"] in starts[-1]:
                    return ["knowledge_v2_duplicate_terminal_entity"]
            starts[-1].add(chain[-1]["object"])
        if any(
            paths[target][hop][key] != plan["real_facts"][hop][key]
            for hop in range(len(route))
            for key in ("subject", "predicate", "object", "object_label")
        ):
            return ["knowledge_v2_anchor_not_target"]
        for case, chain in paths.items():
            for hop, fact in enumerate(chain):
                if {k: fact[k] for k in ("subject", "predicate", "object")} != \
                        witness["candidate_facts"][case][hop]:
                    return ["knowledge_v2_forged_candidate_witness"]
        return []
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return ["knowledge_v2_invalid_spec:" + str(exc)[:140]]


def compile_macro_solvers(spec: dict) -> dict:
    issues = validate_spec(spec)
    if issues:
        raise ValueError("invalid_knowledge_v2:" + ",".join(issues))
    program = old.compile_macro_solvers(_as_v1(spec))
    program["semantic_operator_pattern"] = PATTERN
    program["macro_plan_version"] = VERSION
    return program


def generate_world(spec: dict) -> dict:
    """Build all real-path alternatives; no missing path can erase distractors."""
    from .world import _rows, WORLD_VERSION
    errors = validate_spec(spec)
    if errors:
        raise ValueError("invalid_knowledge_v2_world:" + ",".join(errors))
    plan = spec["macro_plan"]
    rng = random.Random(int(digest(
        spec["seed"], spec["task_id"], "knowledge-world-v2")[:15], 16))
    target, cases = plan["target_case"], list(plan["all_cases"])
    records = {}
    for sid, clue in plan["clues"].items():
        slot = clue["source_slot"]
        desired, alternate = clue["value"], clue["alternative"]
        rows = []
        for case in cases:
            shadow = plan["case_shadow_slots"].get(case)
            match = (
                case == target or
                (shadow is not None and shadow != slot)
            )
            rows.append({
                "case_id": case,
                "cue": desired if match else alternate,
            })
        rng.shuffle(rows)
        records[sid] = rows
    records["entity_index"] = [{
        "case_id": case,
        "entity_0": plan["candidate_paths"][case][0]["subject"],
    } for case in cases]
    rng.shuffle(records["entity_index"])
    for hop in range(len(plan["real_facts"])):
        key = f"entity_{hop}"
        value = "answer" if hop == len(plan["real_facts"])-1 \
            else f"entity_{hop+1}"
        sid = f"relation_{hop+1}"
        rows = [{
            key: chain[hop]["subject"],
            value: chain[hop]["object_label"] if value == "answer"
                   else chain[hop]["object"],
        } for chain in plan["candidate_paths"].values()]
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
            hop = int(sid.split("_")[-1])-1
            source["evidence_kind"] = "real_yago_fact_excerpts"
            source["independence_note"] = (
                "All facts originate from YAGO; separate snippets are NOT "
                "independent historical publishers."
            )
            triples = {}
            for chain in plan["candidate_paths"].values():
                fact = chain[hop]
                triples[fact["subject"]] = {
                    k: fact[k] for k in ("subject", "predicate", "object")
                }
            for observation in source["observations"]:
                subject = observation["values"][f"entity_{hop}"]
                fact = triples[subject]
                observation["knowledge_fact"] = fact
                observation["evidence_origin_id"] = (
                    "YAGO-4.6:" + digest(
                        fact["subject"], fact["predicate"], fact["object"],
                    )[:20]
                )
        sources[sid] = source
    return {
        "version": WORLD_VERSION,
        "simulated_only": True,
        "sources": sources,
        "entities": [{
            "hidden_entity_id": target,
            "identity_source": "frozen_yago_reference_witness",
        }],
        "generation_seed": spec["seed"],
        "knowledge_witness": copy.deepcopy(spec["knowledge_witness"]),
        "value_sampling_contract": (
            "frozen_complete_yago_candidate_paths_and_necessary_clues_v2"
        ),
    }


def mutate_witness(sources, sid, answer_rows, spec):
    return old.mutate_witness(sources, sid, answer_rows, spec)
