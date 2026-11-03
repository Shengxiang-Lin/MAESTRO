"""Subject-centric YAGO profile prior for coherent synthetic object semantics.

Predicate reservoirs answer "can this predicate apply to this type?" but lose the
fact that several attributes belonged to the *same* real subject. This adapter
revisits only a bounded set of subjects already sampled into the predicate catalog
and records their joint literal/predicate neighborhood. Real identifiers stay
private provenance; downstream code should use profiles as archetypes, never as
synthetic entity identities.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

from .yago import METADATA_PREDICATES, TYPE_PREDICATES, iter_turtle_triples

_LITERAL_RE = re.compile(r'^"((?:\\.|[^"\\])*)"(?:@[^\s]+|\^\^(\S+))?$')
_SINGLE_LITERAL_RE = re.compile(r"^'((?:\\.|[^'\\])*)'(?:@[^\s]+|\^\^(\S+))?$")
_NUMBER_TYPES = {"xsd:decimal", "xsd:double", "xsd:float", "xsd:integer", "xsd:int", "xsd:long", "xsd:nonNegativeInteger", "xsd:positiveInteger"}
_YEAR_TYPES = {"xsd:gYear"}
_DATE_TYPES = {"xsd:date", "xsd:dateTime"}
_BOOLEAN_TYPES = {"xsd:boolean"}
_LABEL_PREDICATES = {"rdfs:label", "schema:name", "skos:prefLabel"}
_IMAGE_PREDICATES = {"schema:image"}

def _decode_quoted(text: str) -> str:
    try:
        return json.loads('"' + text + '"')
    except (json.JSONDecodeError, UnicodeDecodeError):
        return text.replace('\\"', '"').replace('\\n', ' ').replace('\\t', ' ')

def parse_yago_literal(raw: str) -> dict[str, Any] | None:
    token = str(raw).strip()
    match = _LITERAL_RE.match(token) or _SINGLE_LITERAL_RE.match(token)
    if not match:
        return None
    lexical = _decode_quoted(match.group(1)); datatype = str(match.group(2) or ""); kind = "string"; value: Any = lexical
    try:
        if datatype in _YEAR_TYPES or (not datatype and re.fullmatch(r"\d{4}", lexical)):
            kind = "year"; value = int(lexical[:4])
        elif datatype in _DATE_TYPES:
            kind = "date"; value = lexical[:10]
        elif datatype in _NUMBER_TYPES:
            number = float(lexical); value = int(number) if number.is_integer() else number; kind = "number"
        elif datatype in _BOOLEAN_TYPES:
            kind = "bool"; value = lexical.strip().lower() in {"true", "1"}
    except (TypeError, ValueError, OverflowError):
        return None
    return {"value": value, "value_kind": kind, "datatype": datatype or None}

def _candidate_subjects(catalog: Mapping[str, Any]) -> set[str]:
    subjects: set[str] = set(); predicates = catalog.get("predicates", {})
    if not isinstance(predicates, Mapping):
        return subjects
    for predicate, info in predicates.items():
        if predicate in METADATA_PREDICATES | TYPE_PREDICATES or not isinstance(info, Mapping):
            continue
        for sample in info.get("samples", ()):
            if isinstance(sample, Mapping) and sample.get("subject"):
                subjects.add(str(sample["subject"]))
    return subjects

def build_entity_profile_index(facts_path: str | Path, catalog_path: str | Path, output_path: str | Path, *, max_profiles: int = 4096, max_literal_facts_per_profile: int = 24, max_values_per_predicate: int = 2, seed: int = 17, progress: bool = False) -> dict[str, Any]:
    if max_profiles < 1:
        raise ValueError("max_profiles must be >= 1")
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8")); candidates = sorted(_candidate_subjects(catalog)); rng = random.Random(seed)
    if len(candidates) > max_profiles:
        candidates = sorted(rng.sample(candidates, max_profiles))
    selected = set(candidates)
    types: dict[str, Counter[str]] = defaultdict(Counter); labels: dict[str, list[str]] = defaultdict(list); images: dict[str, list[str]] = defaultdict(list); predicates: dict[str, Counter[str]] = defaultdict(Counter); literals: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list)); literal_totals: Counter[str] = Counter()
    triples = iter_turtle_triples(facts_path, progress=progress, description="YAGO entity profiles")
    scanned = 0
    for triple in triples:
        scanned += 1
        if triple.subject not in selected:
            continue
        if triple.predicate in TYPE_PREDICATES:
            types[triple.subject][triple.object] += 1; continue
        if triple.predicate in _LABEL_PREDICATES and triple.object_is_literal:
            parsed = parse_yago_literal(triple.object)
            if parsed and len(labels[triple.subject]) < 3:
                labels[triple.subject].append(str(parsed["value"]))
            continue
        if triple.predicate in _IMAGE_PREDICATES:
            if len(images[triple.subject]) < 2:
                images[triple.subject].append(str(triple.object))
            continue
        if triple.predicate in METADATA_PREDICATES:
            continue
        predicates[triple.subject][triple.predicate] += 1
        if not triple.object_is_literal or literal_totals[triple.subject] >= max_literal_facts_per_profile:
            continue
        values = literals[triple.subject][triple.predicate]
        if len(values) >= max_values_per_predicate:
            continue
        parsed = parse_yago_literal(triple.object)
        if parsed is None:
            continue
        values.append(parsed); literal_totals[triple.subject] += 1
    profile_rows: list[dict[str, Any]] = []; type_profile_counts: Counter[str] = Counter(); type_pair_counts: dict[str, Counter[str]] = defaultdict(Counter); literal_kind_counts: Counter[str] = Counter()
    for subject in candidates:
        attrs = literals.get(subject, {}); observed_predicates = sorted(predicates.get(subject, {}))
        if not observed_predicates and not attrs:
            continue
        subject_types = [item for item, _ in types.get(subject, {}).most_common()]
        for type_name in subject_types or ["*"]:
            type_profile_counts[type_name] += 1; attribute_predicates = sorted(attrs)
            for left, right in combinations(attribute_predicates[:16], 2):
                type_pair_counts[type_name][left + "|" + right] += 1
        literal_payload = {predicate: list(values) for predicate, values in sorted(attrs.items()) if values}
        for values in literal_payload.values():
            literal_kind_counts.update(str(item.get("value_kind", "unknown")) for item in values)
        profile_rows.append({"source_subject": subject, "entity_types": subject_types, "labels": labels.get(subject, ()), "image_links": images.get(subject, ()), "observed_predicates": observed_predicates, "literal_attributes": literal_payload})
    index = {"version": 1, "policy": "bounded_subject_centric_yago_archetypes", "source": str(facts_path), "catalog": str(catalog_path), "candidate_subject_count": len(selected), "profile_count": len(profile_rows), "max_profiles": max_profiles, "max_literal_facts_per_profile": max_literal_facts_per_profile, "profiles": profile_rows, "type_profile_counts": dict(type_profile_counts.most_common()), "literal_kind_counts": dict(literal_kind_counts.most_common()), "joint_attribute_support": {type_name: dict(counter.most_common(256)) for type_name, counter in type_pair_counts.items() if counter}, "identity_policy": "real_subject_ids_private_profile_provenance_only"}
    target = Path(output_path); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: value for key, value in index.items() if key not in {"profiles", "joint_attribute_support"}} | {"output": str(target), "joint_attribute_type_count": len(index["joint_attribute_support"])}

def load_entity_profile_index(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("unsupported YAGO entity profile index")
    return payload

__all__ = ["build_entity_profile_index", "load_entity_profile_index", "parse_yago_literal"]
