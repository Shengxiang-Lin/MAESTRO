"""Mine real YAGO relation motifs for hybrid-world task grounding.

The miner labels only RDF edges from the provided KnowledgeGraph as real.
Extra motif edges are optional *world context*, not automatically claimed
as necessary evidence or independent publishers. The generated task's
mandatory path is supplied by the causally checked knowledge v2 backbone.
"""
from __future__ import annotations

from collections import defaultdict
import random

from environment.schema import digest

VERSION = "maestro.real_world_motif.v1"


def _fact(edge, graph) -> dict:
    return {
        "subject": edge.subject,
        "predicate": edge.predicate,
        "object": edge.object,
        "subject_label": graph.label(edge.subject),
        "object_label": graph.label(edge.object),
        "origin": "yago",
        "fact_id": "yago:" + digest(
            edge.subject, edge.predicate, edge.object,
        )[:20],
    }


def mine_motif(graph, index, anchored_path: list[dict],
               *, seed: int, ordinal: int) -> dict:
    """Classify an observed chain and optionally a true fork/diamond.

    Diamond: another 2-hop route from A to the already observed B, with
    a different intermediate entity. Fork: an actual non-path outgoing
    relation from a node on the mandatory chain. Never create an RDF edge.
    """
    rng = random.Random(f"maestro-yago-motifs-v1:{seed}:{ordinal}")
    triples = {(e.subject, e.predicate, e.object): e for e in graph.edges}
    chain = []
    for row in anchored_path:
        key = row["subject"], row["predicate"], row["object"]
        if key not in triples:
            raise ValueError("knowledge_motif_missing_yago_anchor:" + str(key))
        chain.append(triples[key])
    if not chain:
        raise ValueError("knowledge_motif_empty_anchor")

    chain_set = {(e.subject, e.predicate, e.object) for e in chain}
    # A genuine alternate route to an existing anchor destination is
    # especially useful for future verification Macros.
    for i in range(len(chain)-1):
        start, end = chain[i].subject, chain[i+1].object
        direct_middle = chain[i].object
        proposals = list(index.outgoing.get(start, ()))
        rng.shuffle(proposals)
        for left in proposals[:120]:
            if left.object == direct_middle or left.object == start:
                continue
            for right in index.outgoing.get(left.object, ())[:250]:
                if right.object == end and right.subject == left.object:
                    return {
                        "version": VERSION, "kind": "diamond",
                        "anchor_facts": [_fact(e, graph) for e in chain],
                        "extra_facts": [_fact(left, graph), _fact(right, graph)],
                        "source_lineage": "YAGO-4.6",
                        "causal_use": "context_only_not_macro_evidence",
                    }

    possible = []
    chain_nodes = {e.subject for e in chain} | {chain[-1].object}
    for i, item in enumerate(chain):
        for edge in index.outgoing.get(item.subject, ())[:160]:
            key = edge.subject, edge.predicate, edge.object
            if key not in chain_set and edge.object not in chain_nodes:
                possible.append((i, edge))
    if possible:
        i, sibling = rng.choice(possible)
        return {
            "version": VERSION, "kind": "fork",
            "fork_at_chain_position": i,
            "anchor_facts": [_fact(e, graph) for e in chain],
            "extra_facts": [_fact(sibling, graph)],
            "source_lineage": "YAGO-4.6",
            "causal_use": "context_only_not_macro_evidence",
        }
    return {
        "version": VERSION, "kind": "chain",
        "anchor_facts": [_fact(e, graph) for e in chain],
        "extra_facts": [],
        "source_lineage": "YAGO-4.6",
        "causal_use": "context_only_not_macro_evidence",
    }


def validate_frozen_motif(motif: dict, anchored_path: list[dict]) -> list[str]:
    """Self-contained replay checks; never assert external RDF re-verification."""
    if motif.get("version") != VERSION or motif.get("kind") not in {
        "chain", "fork", "diamond",
    }:
        return ["hybrid_bad_real_motif_version"]
    anchor = motif.get("anchor_facts")
    if not isinstance(anchor, list) or len(anchor) != len(anchored_path):
        return ["hybrid_bad_real_motif_anchor_count"]
    for frozen, expected in zip(anchor, anchored_path):
        if any(frozen.get(k) != expected.get(k)
               for k in ("subject", "predicate", "object", "object_label")):
            return ["hybrid_real_motif_anchor_mismatch"]
        if frozen.get("origin") != "yago":
            return ["hybrid_real_motif_wrong_origin"]
    extra = motif.get("extra_facts", [])
    if len(extra) != {"chain": 0, "fork": 1, "diamond": 2}[motif["kind"]]:
        return ["hybrid_bad_motif_extra_fact_count"]
    for row in anchor + extra:
        if (row.get("origin") != "yago"
                or row.get("fact_id") != "yago:" + digest(
                    row["subject"], row["predicate"], row["object"],
                )[:20]):
            return ["hybrid_bad_motif_fact_id"]
    if motif.get("causal_use") != "context_only_not_macro_evidence":
        return ["hybrid_motif_unjustified_causal_claim"]
    return []


# --- Task Plan motif composition ------------------------------------------------
# Real knowledge is a distributional/semantic prior, not a constraint that
# imaginary entities must already have co-occurred in a particular RDF dump.
# This module does not alter the historical witnessed YAGO motif contracts.

COMPOSE_VERSION = "maestro.composed_typed_world.v1"

_GIVEN = (
    "Mara", "Evelyn", "Talia", "Jonas", "Soren", "Lina",
    "Nolan", "Iris", "Elena", "Silas", "Amara", "Theo",
    "Celia", "Adrian", "Leona", "Felix", "Mina", "Julian",
)
_SURNAMES = (
    "Hart", "Vale", "Merrow", "Chen", "Rowan", "Ellis",
    "Bennett", "Solano", "Arden", "Novak", "Quinn", "Keene",
    "Navarro", "Reed", "Hale", "Foster", "Marin", "Hayes",
)
_PLACES = (
    "Alderwick", "Northbridge", "Bellwater", "Cedarport",
    "Dunhaven", "Lakehurst", "Westmere", "Ravenfield",
    "Eastford", "Harrowgate", "Oakridge", "Riverton",
    "Greenvale", "Ashbourne", "Stonehaven", "Foxmere",
)
_FIELDS = (
    "historical cartography", "computational linguistics", "river ecology",
    "urban hydrology", "maritime archaeology", "digital paleography",
    "materials conservation", "environmental history",
    "landscape modeling", "cultural astronomy", "heritage informatics",
    "comparative philology",
)


# Conservative lexical type check for generic fields inherited from YAGO.
# A 'fieldOfWork' object in RDF is not necessarily an academic discipline.
# Do not treat arbitrary entity labels (e.g. "Aerospace manufacturer") as
# valid research-field values in a newly composed university world.
_ACADEMIC_TERMS = frozenset((
    "science", "studies", "history", "physics", "chemistry", "biology",
    "geography", "ecology", "geology", "philosophy", "literature",
    "linguistics", "mathematics", "engineering", "astronomy",
    "archaeology", "anthropology", "cartography", "hydrology",
    "paleography", "philology", "informatics", "crystallography",
    "conservation", "modeling", "economics", "medicine", "music",
    "psychology", "sociology", "architecture", "education",
    "environmental", "statistics", "computer", "computational",
))
_NON_FIELD_TERMS = frozenset((
    "manufacturer", "corporation", "company", "business", "association",
    "foundation", "school", "university", "institute", "college",
    "person", "politician", "musician", "athlete", "artist", "city",
    "county", "town", "village", "country", "organisation", "organization",
    "football", "airline", "band", "museum", "newspaper",
))


def _academic_field_label(label: str) -> bool:
    """Conservative type gate for vocabulary borrowed from YAGO labels."""
    words = label.casefold().replace("-", " ").split()
    if not 1 <= len(words) <= 5 or any(
        not word.isalpha() for word in words
    ):
        return False
    if any(word in _NON_FIELD_TERMS for word in words):
        return False
    return any(
        word in _ACADEMIC_TERMS
        or word.endswith(("ology", "ography", "istics", "onomy"))
        for word in words
    )


_ROLE_OUTPUT = {
    "birthplace": "place", "deathplace": "place",
    "containedinplace": "region", "location": "place",
    "doctoraladvisor": "person", "fieldofwork": "field",
    "spouse": "person", "memberof": "organization",
    "neighbor": "place", "nationality": "country",
    "capital": "city", "twinnedwith": "city",
}
_ROLE_INPUT = {
    "birthplace": "person", "deathplace": "person",
    "containedinplace": "place", "location": "any_locatable",
    "doctoraladvisor": "person", "fieldofwork": "person",
    "spouse": "person", "memberof": "person",
    "neighbor": "place", "nationality": "person",
    "capital": "country", "twinnedwith": "city",
}


def motif_features(graph, index, recipes=None) -> dict:
    """One cached, bounded inventory from *observed* YAGO relation edges.

    This is not an RDF type inference claim. Predicate role assignments
    belong to the explicit composer grammar, and observed statistics are
    not passed off as original relations of the constructed entities.
    """
    cache = getattr(index, "_task_plan_motif_features", None)
    if cache is not None:
        if recipes is not None and "join_support" not in cache:
            cache["join_support"] = _observed_join_support(
                cache["samples_by_predicate"], index, recipes,
            )
        return cache
    counts = defaultdict(int)
    predicates = {}
    observed_fields = []
    field_labels_rejected = 0
    by_name = defaultdict(list)
    for edge in graph.edges:
        name = "".join(c for c in edge.predicate.rsplit("/", 1)[-1]
                       .rsplit("#", 1)[-1].casefold() if c.isalnum())
        counts[name] += 1
        predicates.setdefault(name, edge.predicate)
        if len(by_name[name]) < 96:
            by_name[name].append(edge)
        if name == "fieldofwork" and len(observed_fields) < 96:
            label = graph.label(edge.object)
            if (4 <= len(label) <= 48 and
                    _academic_field_label(label)):
                observed_fields.append(label)
            else:
                field_labels_rejected += 1
    fanouts = [len(x) for x in index.outgoing.values()]
    cache = {
        "edge_counts": dict(counts),
        "predicate_uris": predicates,
        "observed_field_labels": list(dict.fromkeys(observed_fields)),
        "rejected_nonacademic_field_labels": field_labels_rejected,
        "samples_by_predicate": dict(by_name),
        "total_scanned_edges": len(graph.edges),
        "observed_fork_fraction": (
            round(sum(x >= 2 for x in fanouts) / len(fanouts), 4)
            if fanouts else 0.0
        ),
    }
    if recipes is not None:
        cache["join_support"] = _observed_join_support(
            cache["samples_by_predicate"], index, recipes,
        )
    index._task_plan_motif_features = cache
    return cache



def _observed_join_support(by_name: dict, index, recipes: tuple) -> dict:
    """Sample real YAGO contiguous predicate motifs. No invented join is
    credited as observed, and incomplete RDF samples are not failure cases."""
    def key_for(edge):
        return "".join(c for c in edge.predicate.rsplit("/", 1)[-1]
                       .rsplit("#", 1)[-1].casefold() if c.isalnum())
    support = {}
    for recipe in recipes:
        keys = recipe[0]
        matches = 0
        for start in by_name.get(keys[0], ())[:64]:
            cur, seen = start.object, {start.subject, start.object}
            for key in keys[1:]:
                options = [
                    edge for edge in index.outgoing.get(cur, ())
                    if key_for(edge) == key and edge.object not in seen
                ]
                if not options:
                    break
                edge = options[0]
                cur = edge.object
                seen.add(cur)
            else:
                matches += 1
        support["_".join(keys)] = matches
    return support


def compose_relation_goal(features: dict, recipes: tuple,
                          *, rng) -> tuple:
    """Choose an interpretable goal, favoring observed motifs but retaining
    novel combinations of individually observed relation types."""
    if not recipes:
        raise ValueError("composer_has_no_semantic_recipes")
    if rng.random() < 0.50:
        return rng.choice(recipes)
    counts = features["edge_counts"]
    # A weak coverage preference; the corpus distribution must not collapse
    # all generated worlds to a handful of extremely popular predicates.
    joins = features.get("join_support", {})
    weights = [
        1 + sum(counts.get(key, 0) > 0 for key in recipe[0])
        + min(3, joins.get("_".join(recipe[0]), 0))
        for recipe in recipes
    ]
    return rng.choices(recipes, weights=weights, k=1)[0]


def _typed_roles(keys: tuple, domain: str) -> list[str]:
    first = "person" if domain in {"academic", "biographical"} else (
        "place" if domain == "geographical" else "person"
    )
    roles = [first]
    for i, key in enumerate(keys):
        if key not in _ROLE_OUTPUT:
            raise ValueError("unsupported_composed_relation:" + key)
        output = _ROLE_OUTPUT[key]
        if i+1 < len(keys):
            following = _ROLE_INPUT[keys[i+1]]
            if following == "any_locatable":
                pass
            elif following == "country" and output in {"place", "region"}:
                output = "country"
            elif following == "city" and output == "place":
                output = "city"
            elif following == "place" and output in {"region", "city", "country"}:
                # Regions/cities/countries are all a kind of place.
                pass
            elif following != output and not (
                following == "place" and output in {"region", "city", "country"}
            ):
                raise ValueError(
                    "incompatible_composed_relation_types:" + key + ":" + keys[i+1]
                )
        roles.append(output)
    return roles


def _entity_name(role: str, case_index: int, node_index: int,
                 rng, features: dict) -> str:
    k = case_index * 11 + node_index * 7
    town = _PLACES[k % len(_PLACES)]
    if role == "person":
        return _GIVEN[(k + rng.randrange(len(_GIVEN))) % len(_GIVEN)] + " " + (
            _SURNAMES[(k + rng.randrange(len(_SURNAMES))) % len(_SURNAMES)]
        )
    if role == "organization":
        return town + rng.choice((
            " Research Institute", " University", " Academy",
            " Field Station", " Science Centre",
        ))
    if role == "field":
        labels = features["observed_field_labels"]
        # Generic public field labels are reusable vocabulary, never
        # invented biographical claims about their original holders.
        return rng.choice(labels if labels and rng.random() < 0.6 else _FIELDS)
    if role == "country":
        return "Republic of " + town
    if role == "region":
        return town + rng.choice((" Region", " Province", " County"))
    if role == "city":
        return "City of " + town
    return town + rng.choice((" Township", " District", " Valley"))


def compose_typed_world(graph, index, *, recipe: tuple,
                        cases: list[str], seed: int, ordinal: int,
                        required_excluded_cases: list[str] | None = None) -> tuple:
    """Rewire real relation *types* into a new, coherent fictional KG.

    All source entities and assertions in the new graph are distinct from
    original RDF IDs. The new graph, not the independent source tables, owns
    the answer-defining edges. Candidate paths are a backward-compatible
    projection; downstream evidence views consume the canonical relations.
    """
    from environment.schema import digest
    import random
    rng = random.Random(int(digest(
        seed, ordinal, "task-plan-motif-composition"
    )[:15], 16))
    keys, domain = recipe[0], recipe[1]
    roles = _typed_roles(keys, domain)
    features = motif_features(graph, index)
    predicates = features["predicate_uris"]
    entity_rows, relation_rows, paths = {}, [], {}
    used_final = set()
    for case_index, case in enumerate(cases):
        nodes = [
            "ENT-" + digest(
                seed, ordinal, case, i, "recomposed-motif"
            )[:12].upper() for i in range(len(keys)+1)
        ]
        for j, (nid, role) in enumerate(zip(nodes, roles)):
            label = _entity_name(role, case_index, j, rng, features)
            if j == len(keys):
                if role == "field":
                    # Differentiating two legitimate research disciplines
                    # must not create non-discipline labels such as Physics 3.
                    original = label
                    for prefix in (
                        "", "applied ", "comparative ", "experimental ",
                        "computational ", "historical ", "theoretical ",
                        "interdisciplinary ", "regional ", "advanced ",
                        "environmental ", "digital ",
                    ):
                        label = prefix + original
                        if label.casefold() not in used_final:
                            break
                else:
                    while label.casefold() in used_final:
                        label = label + " " + str(case_index+1)
                if label.casefold() in used_final:
                    raise ValueError("composed_world_non_unique_answer_label")
                used_final.add(label.casefold())
            entity_rows[nid] = {
                "entity_id": nid, "type": role, "label": label,
            }
        path = []
        for i, key in enumerate(keys):
            predicate = predicates.get(
                key, "http://schema.org/" + "".join(
                    word.capitalize() if n else word
                    for n, word in enumerate(key.split("_"))
                )
            )
            # Use canonical schema.org spellings when a predicate is absent.
            if key not in predicates:
                canonical = {
                    "birthplace": "birthPlace", "deathplace": "deathPlace",
                    "containedinplace": "containedInPlace",
                    "doctoraladvisor": "doctoralAdvisor",
                    "fieldofwork": "fieldOfWork",
                    "memberof": "memberOf", "twinnedwith": "twinnedWith",
                }.get(key, key)
                predicate = "http://schema.org/" + canonical
            row = {
                "subject": nodes[i], "predicate": predicate,
                "object": nodes[i+1],
                "object_label": entity_rows[nodes[i+1]]["label"],
                "origin": "synthetic",
            }
            path.append(row)
            relation_rows.append({
                "case_id": case, "hop": i+1, **row,
                "subject_type": roles[i], "object_type": roles[i+1],
            })
        paths[case] = path
    # One existing discovery clue can be the *actual association* in this
    # World, rather than a free-standing string. Exactly one shadow dossier
    # belongs to another institution/archive volume; other candidates share
    # the same entity. This preserves counterfactual necessity for ALL other
    # clue sources without inventing a new Macro or a new executor.
    exclusions = set(required_excluded_cases or ())
    if (required_excluded_cases is not None
            and (not exclusions or exclusions == set(cases)
                 or not exclusions <= set(cases))):
        raise ValueError("invalid_required_relation_exclusion_set")
    group_count = (
        2 if required_excluded_cases is not None
        else max(2, (len(cases) + 2) // 3)
    )
    groups = {}
    context_by_case = {}
    context_relations = []
    for group in range(group_count):
        place = _PLACES[(ordinal * 3 + group * 5) % len(_PLACES)]
        anchor_id = "ENT-" + digest(
            seed, ordinal, "shared-anchor", group,
        )[:12].upper()
        site_id = "ENT-" + digest(
            seed, ordinal, "shared-site", group // 2,
        )[:12].upper()
        anchor_type = "organization" if roles[0] == "person" else "archive_record"
        anchor_label = (
            place + " Research Institute" if roles[0] == "person"
            else place + " Regional Survey Gazetteer"
        )
        site_label = (
            "City of " + _PLACES[
                (ordinal * 3 + (group // 2) * 7 + 4) % len(_PLACES)
            ]
        )
        entity_rows[anchor_id] = {
            "entity_id": anchor_id, "type": anchor_type,
            "label": anchor_label,
        }
        if site_id not in entity_rows:
            entity_rows[site_id] = {
                "entity_id": site_id, "type": "city", "label": site_label,
            }
        groups[group] = (anchor_id, site_id)
        context_relations.append({
            "subject": anchor_id,
            "predicate": (
                "http://schema.org/location"
                if roles[0] == "person" else
                "http://schema.org/spatialCoverage"
            ),
            "object": site_id,
            "subject_type": anchor_type, "object_type": "city",
            "origin": "synthetic", "scope": "context_only",
        })
    for case_index, case in enumerate(cases):
        group = (
            (1 if case in exclusions else 0)
            if required_excluded_cases is not None
            else case_index % group_count
        )
        anchor_id, site_id = groups[group]
        subject = paths[case][0]["subject"]
        context_relations.append({
            "subject": subject,
            "predicate": (
                "http://schema.org/affiliation"
                if roles[0] == "person" else
                "http://schema.org/subjectOf"
            ),
            "object": anchor_id,
            "subject_type": roles[0],
            "object_type": entity_rows[anchor_id]["type"],
            "origin": "synthetic",
            "scope": ("required_clue"
                      if required_excluded_cases is not None
                      else "context_only"),
        })
        context_by_case[case] = {
            "anchor_id": anchor_id,
            "anchor_label": entity_rows[anchor_id]["label"],
            "site_id": site_id,
            "site_label": entity_rows[site_id]["label"],
        }
    profile = {
        "policy": COMPOSE_VERSION,
        "goal_signature": list(keys),
        "source_yago_edges": features["total_scanned_edges"],
        "observed_join_examples": features.get(
            "join_support", {}).get("_".join(keys), 0),
        "observed_fork_fraction": features["observed_fork_fraction"],
        "observed_predicate_counts": {
            key: features["edge_counts"].get(key, 0) for key in keys
        },
        "reused_public_field_vocabulary": bool(
            "fieldofwork" in keys and features["observed_field_labels"]
        ),
        "rejected_nonacademic_field_labels": features[
            "rejected_nonacademic_field_labels"
        ],
        "shared_context_group_count": group_count,
        "shared_context_case_count": len(context_by_case),
        "required_shared_association": required_excluded_cases is not None,
        "composition": "typed_semantic_motifs_rewired_to_new_entities",
        "actual_yago_assertions_reused": False,
    }
    return paths, {
        "version": COMPOSE_VERSION,
        "entities": entity_rows,
        "relations": relation_rows,
        "context_relations": context_relations,
        "context_by_case": context_by_case,
        "context_version": "maestro.shared_context.v1",
        "profile": profile,
    }


def validate_composed_world(binding: dict) -> list[str]:
    """Ensure canonical private world and derived candidate paths agree."""
    try:
        world = binding["world_model"]
        if world["version"] != COMPOSE_VERSION:
            return ["composed_world_wrong_version"]
        entities, facts = world["entities"], world["relations"]
        paths = binding["candidate_paths"]
        if set(paths) != set(binding["all_cases"]):
            return ["composed_world_wrong_cases"]
        expected = {
            (case, i+1, fact["subject"], fact["predicate"], fact["object"],
             fact["object_label"], fact["origin"])
            for case, path in paths.items() for i, fact in enumerate(path)
        }
        actual = {
            (fact["case_id"], fact["hop"], fact["subject"],
             fact["predicate"], fact["object"], fact["object_label"],
             fact["origin"]) for fact in facts
        }
        if expected != actual or len(expected) != len(facts):
            return ["composed_world_facts_not_authoritative"]
        role_by_id = {eid: row["type"] for eid, row in entities.items()
                      if row["entity_id"] == eid}
        if len(role_by_id) != len(entities):
            return ["composed_world_duplicate_entity"]
        for fact in facts:
            if (fact["origin"] != "synthetic"
                    or not fact["subject"].startswith("ENT-")
                    or not fact["object"].startswith("ENT-")
                    or role_by_id[fact["subject"]] != fact["subject_type"]
                    or role_by_id[fact["object"]] != fact["object_type"]
                    or entities[fact["object"]]["label"] != fact["object_label"]):
                return ["composed_world_type_or_identity_mismatch"]
        # Historical composed scenes without the shared-context extension
        # retain their exact frozen World/Replay semantics.
        if "context_version" in world:
            if world["context_version"] != "maestro.shared_context.v1":
                return ["composed_world_wrong_context_version"]
            mapping = world["context_by_case"]
            context = world["context_relations"]
            if set(mapping) != set(paths):
                return ["composed_world_context_cases_mismatch"]
            group_uses = defaultdict(int)
            expected_links = set()
            for case, item in mapping.items():
                if (item["anchor_id"] not in entities
                        or item["site_id"] not in entities
                        or entities[item["anchor_id"]]["label"] !=
                            item["anchor_label"]
                        or entities[item["site_id"]]["label"] !=
                            item["site_label"]):
                    return ["composed_world_bad_context_reference"]
                group_uses[item["anchor_id"]] += 1
                src_role = entities[paths[case][0]["subject"]]["type"]
                expected_links.add((
                    paths[case][0]["subject"], item["anchor_id"],
                    "http://schema.org/affiliation" if src_role == "person"
                    else "http://schema.org/subjectOf",
                ))
                expected_links.add((
                    item["anchor_id"], item["site_id"],
                    "http://schema.org/location" if src_role == "person"
                    else "http://schema.org/spatialCoverage",
                ))
            required = world.get("required_context_relation")
            if required:
                target, excluded = (
                    binding["target_case"], required["excluded_case"]
                )
                excluded_set = set(required.get("excluded_cases", ()))
                expected_exclusions = {
                    case for case, slot in binding["case_shadow_slots"].items()
                    if slot is None or slot == required["source_slot"]
                }
                if (excluded not in excluded_set
                        or target in excluded_set
                        or excluded_set != expected_exclusions
                        or required["field"] != "cue"
                        or required["target_anchor_id"] !=
                            mapping[target]["anchor_id"]
                        or required["target_anchor_label"] !=
                            mapping[target]["anchor_label"]
                        or required["excluded_anchor_id"] !=
                            mapping[excluded]["anchor_id"]
                        or required["excluded_anchor_label"] !=
                            mapping[excluded]["anchor_label"]
                        or mapping[excluded]["anchor_id"] ==
                            mapping[target]["anchor_id"]
                        or any(
                            item["anchor_id"] != (
                                mapping[excluded]["anchor_id"]
                                if case in excluded_set
                                else mapping[target]["anchor_id"]
                            )
                            for case, item in mapping.items()
                        )):
                    return ["composed_world_required_relation_not_unique_shadow"]
                if (len(group_uses) != 2 or
                        group_uses[mapping[target]["anchor_id"]] !=
                        len(paths)-len(excluded_set)):
                    return ["composed_world_required_relation_bad_sharing"]
            elif len(group_uses) < 2 or min(group_uses.values()) < 2:
                return ["composed_world_context_not_shared"]
            actual_links = {
                (r["subject"], r["object"], r["predicate"])
                for r in context
            }
            if expected_links != actual_links:
                return ["composed_world_missing_or_extra_shared_links"]
            if len(actual_links) != len(context):
                return ["composed_world_duplicate_context_link"]
            for r in context:
                is_required = bool(required) and (
                    r["predicate"] in {
                        "http://schema.org/affiliation",
                        "http://schema.org/subjectOf",
                    }
                )
                if (r["scope"] != (
                        "required_clue" if is_required else "context_only"
                    )
                        or r["origin"] != "synthetic"
                        or r["subject"] not in entities
                        or r["object"] not in entities
                        or r["subject_type"] !=
                            entities[r["subject"]]["type"]
                        or r["object_type"] !=
                            entities[r["object"]]["type"]):
                    return ["composed_world_invalid_context_relation"]
            if world["profile"]["shared_context_group_count"] != len(
                group_uses
            ):
                return ["composed_world_context_stats_mismatch"]
            if required:
                source_id = required.get("source_id", "")
                if (not source_id or
                        required.get("predicate") != (
                            "http://schema.org/affiliation"
                            if entities[paths[binding["target_case"]][0][
                                "subject"]]["type"] == "person"
                            else "http://schema.org/subjectOf"
                        )):
                    return ["composed_world_invalid_required_relation"]
            for row in entities.values():
                if row["type"] == "field" and not _academic_field_label(
                    row["label"]
                ):
                    return ["composed_world_invalid_research_field"]
        profile = world["profile"]
        if (profile["policy"] != COMPOSE_VERSION
                or profile["actual_yago_assertions_reused"] is not False):
            return ["composed_world_provenance_invalid"]
        return []
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        return ["composed_world_invalid:" + str(exc)[:140]]
