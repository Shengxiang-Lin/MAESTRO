"""Streaming YAGO utilities for building a compact predicate/type catalog."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Tuple

from ..utils.progress import iter_file_lines


TRIPLE_RE = re.compile(r"^\s*(\S+)\s+(\S+)\s+(.+?)\s*\.\s*$")
QNAME_RE = r"[A-Za-z_][\w.-]*:[A-Za-z0-9_][\w.-]*"

METADATA_PREDICATES = {
    "rdf:type", "owl:sameAs", "rdfs:label", "rdfs:comment", "schema:name",
    "schema:alternateName", "schema:givenName", "schema:familyName", "schema:url",
    "schema:sameAs", "schema:image", "yago:siteLinks", "skos:prefLabel", "skos:altLabel",
}
TYPE_PREDICATES = {"rdf:type", "a"}
EMPTY_SCHEMA_SIGNATURE = {"domain": [], "range": [], "max_count": None, "min_count": None}

@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str
    @property
    def object_is_literal(self) -> bool:
        value = self.object.lstrip()
        return value.startswith('"') or value.startswith("'")

def iter_turtle_triples(path: str | Path, *, progress: bool = False, description: str | None = None) -> Iterator[Triple]:
    with_lines = iter_file_lines(path, progress=progress, description=description, errors="replace")
    try:
        for line in with_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "@prefix", "PREFIX")):
                continue
            match = TRIPLE_RE.match(line)
            if match:
                yield Triple(match.group(1), match.group(2), match.group(3))
    finally:
        with_lines.close()

def _iter_turtle_statements(path: str | Path) -> Iterator[str]:
    buffer: List[str] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "@prefix", "PREFIX")):
                continue
            buffer.append(stripped)
            if stripped.endswith("."):
                yield " ".join(buffer)
                buffer = []
    if buffer:
        yield " ".join(buffer)

def load_schema_signatures(path: str | Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    property_domains: Dict[str, set[str]] = defaultdict(set)
    property_shapes: Dict[str, Dict[str, Any]] = {}
    property_ref_re = re.compile(r"\bsh:property\s+(.+?)(?:\s*;\s*|\s*\.\s*$)")
    path_re = re.compile(rf"\bsh:path\s+({QNAME_RE})")
    range_re = re.compile(rf"\bsh:(?:node|class|datatype)\s+({QNAME_RE})")
    max_count_re = re.compile(r'\bsh:maxCount\s+"?(\d+)"?')
    min_count_re = re.compile(r'\bsh:minCount\s+"?(\d+)"?')
    for statement in _iter_turtle_statements(path):
        tokens = statement.split()
        if not tokens:
            continue
        subject = tokens[0]
        property_match = property_ref_re.search(statement)
        if property_match:
            for property_shape in re.findall(QNAME_RE, property_match.group(1)):
                property_domains[property_shape].add(subject)
        path_match = path_re.search(statement)
        if path_match:
            ranges = sorted(set(range_re.findall(statement)))
            max_match = max_count_re.search(statement)
            min_match = min_count_re.search(statement)
            property_shapes[subject] = {"path": path_match.group(1), "range": ranges,
                "max_count": int(max_match.group(1)) if max_match else None,
                "min_count": int(min_match.group(1)) if min_match else None}
    signatures: Dict[str, Dict[str, Any]] = {}
    for property_shape, shape in property_shapes.items():
        predicate = shape["path"]
        target = signatures.setdefault(predicate, {"domain": [], "range": [], "max_count": None, "min_count": None})
        target["domain"] = sorted(set(target["domain"]) | property_domains.get(property_shape, set()))
        target["range"] = sorted(set(target["range"]) | set(shape["range"]))
        for field in ("max_count", "min_count"):
            value = shape[field]
            if value is None:
                continue
            if target[field] is None:
                target[field] = value
            elif target[field] != value:
                target[field] = None
    return signatures

def _reservoir_add(reservoir: List[Tuple[str, str]], item: Tuple[str, str], seen: int, limit: int, rng: random.Random) -> None:
    if limit <= 0:
        return
    if len(reservoir) < limit:
        reservoir.append(item)
        return
    index = rng.randrange(seen)
    if index < limit:
        reservoir[index] = item

def _predicate_kind(predicate: str, count: int, literal_count: int) -> str:
    if predicate in METADATA_PREDICATES or predicate in TYPE_PREDICATES:
        return "metadata"
    return "attribute" if (literal_count / count if count else 0.0) >= 0.8 else "relation"

def _refresh_catalog_payload(catalog: Dict[str, Any], *, schema_path: str | Path | None) -> Dict[str, Any]:
    schema = load_schema_signatures(schema_path)
    for predicate, info in catalog.get("predicates", {}).items():
        count = int(info.get("count", 0)); literal_count = int(info.get("literal_count", 0))
        info["kind"] = _predicate_kind(predicate, count, literal_count)
        info["schema"] = schema.get(predicate, dict(EMPTY_SCHEMA_SIGNATURE))
    catalog["schema_source"] = str(schema_path) if schema_path else None
    catalog["schema_signature_count"] = len(schema)
    return catalog

def refresh_predicate_catalog(catalog_path: str | Path, output_path: str | Path | None = None, *, schema_path: str | Path | None = None) -> Dict[str, Any]:
    source = Path(catalog_path)
    catalog = json.loads(source.read_text(encoding="utf-8"))
    resolved_schema = schema_path or catalog.get("schema_source")
    if not resolved_schema:
        raise ValueError("schema_path is required when the catalog has no schema_source")
    result = _refresh_catalog_payload(catalog, schema_path=resolved_schema)
    destination = Path(output_path) if output_path else source
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

def build_predicate_catalog(facts_path: str | Path, output_path: str | Path, *, schema_path: str | Path | None = None, samples_per_predicate: int = 128, seed: int = 17, progress: bool = False) -> Dict[str, Any]:
    rng = random.Random(seed); counts: Counter[str] = Counter(); literal_counts: Counter[str] = Counter(); samples: Dict[str, List[Tuple[str, str]]] = defaultdict(list); total = 0
    for triple in iter_turtle_triples(facts_path, progress=progress, description="YAGO pass 1/2: catalog"):
        total += 1; predicate = triple.predicate; counts[predicate] += 1
        if triple.object_is_literal: literal_counts[predicate] += 1
        _reservoir_add(samples[predicate], (triple.subject, triple.object), counts[predicate], samples_per_predicate, rng)
    sampled_entities: set[str] = set(); sampled_centers: set[str] = set()
    for predicate, edges in samples.items():
        if predicate in TYPE_PREDICATES: continue
        for subject, obj in edges:
            sampled_entities.add(subject)
            if predicate not in METADATA_PREDICATES: sampled_centers.add(subject)
            if not (obj.startswith('"') or obj.startswith("'")): sampled_entities.add(obj)
    entity_types: Dict[str, Counter[str]] = defaultdict(Counter); incident_ports: Dict[str, set[str]] = defaultdict(set)
    if sampled_entities:
        for triple in iter_turtle_triples(facts_path, progress=progress, description="YAGO pass 2/2: sampled types"):
            if triple.predicate in TYPE_PREDICATES and triple.subject in sampled_entities:
                entity_types[triple.subject][triple.object] += 1
            elif triple.predicate not in METADATA_PREDICATES | TYPE_PREDICATES:
                if triple.subject in sampled_centers: incident_ports[triple.subject].add("out|" + triple.predicate)
                if not triple.object_is_literal and triple.object in sampled_centers: incident_ports[triple.object].add("in|" + triple.predicate)
    predicates: Dict[str, Any] = {}
    for predicate, count in counts.most_common():
        subject_type_counts: Counter[str] = Counter(); object_type_counts: Counter[str] = Counter(); edge_payload = []
        for subject, obj in samples[predicate]:
            subject_type_counts.update(entity_types.get(subject, {}))
            if not (obj.startswith('"') or obj.startswith("'")): object_type_counts.update(entity_types.get(obj, {}))
            edge_payload.append({"subject": subject, "object": obj})
        literal_count = literal_counts[predicate]
        predicates[predicate] = {"count": count, "literal_count": literal_count, "literal_ratio": literal_count / count if count else 0.0,
            "kind": _predicate_kind(predicate, count, literal_count), "schema": dict(EMPTY_SCHEMA_SIGNATURE),
            "empirical_subject_types": subject_type_counts.most_common(12), "empirical_object_types": object_type_counts.most_common(12), "samples": edge_payload}
    result = {"source": str(facts_path), "schema_source": str(schema_path) if schema_path else None, "total_triples": total,
        "predicate_count": len(predicates), "samples_per_predicate": samples_per_predicate, "predicates": predicates}
    result = _refresh_catalog_payload(result, schema_path=schema_path)
    from .observed import compile_observed_support
    compile_observed_support(result, entity_types, incident_ports, facts_path=facts_path)
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

def load_predicate_catalog(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
def compatible_predicates(catalog: Mapping[str, Any], *, kind: str | None = None, subject_type: str | None = None, object_type: str | None = None, min_count: int = 1) -> List[str]:
    result: List[str] = []
    for predicate, info in catalog.get("predicates", {}).items():
        if info.get("count", 0) < min_count: continue
        if kind and info.get("kind") != kind: continue
        if subject_type:
            schema_domain = set(info.get("schema", {}).get("domain", [])); empirical = {item[0] for item in info.get("empirical_subject_types", [])}
            if subject_type not in schema_domain and subject_type not in empirical: continue
        if object_type:
            schema_range = set(info.get("schema", {}).get("range", [])); empirical = {item[0] for item in info.get("empirical_object_types", [])}
            if object_type not in schema_range and object_type not in empirical: continue
        result.append(predicate)
    return result
