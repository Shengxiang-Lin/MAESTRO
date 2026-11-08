"""Dataset-driven scenario seeds for the V4.5 Task-First pipeline.

Only the three configured corpora are used. MultiModalQA supplies the user's
information need and source/composition metadata. YAGO supplies optional,
strictly matched entity context. KQA programs remain internal reasoning priors.
Neither MMQA dataset answers nor fabricated media perceptions are public.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from envir.schema import display, normalize_label
from envir.scenarios.mmqa_motifs import extract_motif
from envir.sources.mmqa import _contexts, _modality, _rows


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*|[(),]")


def parse_composition(expression: str) -> dict:
    """Parse nested MMQA types, preserving operation order and branch structure.

    Examples: Compose(TableQ,ImageQ), Compare(Compose(TableQ,TextQ),ImageQ).
    Unknown/malformed expressions stay as an opaque leaf; they never become
    invented dependencies.
    """
    tokens = _TOKEN.findall(expression or "")
    if not tokens:
        return {"op": "Unknown", "children": []}
    pos = 0

    def read(depth: int = 0) -> dict:
        nonlocal pos
        if depth > 16 or pos >= len(tokens) or tokens[pos] in {",", "(", ")"}:
            raise ValueError("invalid MMQA composition syntax")
        op = tokens[pos]
        pos += 1
        children = []
        if pos < len(tokens) and tokens[pos] == "(":
            pos += 1
            children.append(read(depth + 1))
            while pos < len(tokens) and tokens[pos] == ",":
                pos += 1
                children.append(read(depth + 1))
            if pos >= len(tokens) or tokens[pos] != ")":
                raise ValueError("unclosed MMQA composition")
            pos += 1
        return {"op": op, "children": children}

    try:
        result = read()
        if pos != len(tokens):
            raise ValueError("unconsumed tokens")
        return result
    except ValueError:
        return {"op": expression or "Unknown", "children": []}


def _entities(record: dict) -> list[str]:
    meta = record.get("metadata") or {}
    raw = record.get("wiki_entities_in_question") or meta.get("wiki_entities_in_question") or []
    if isinstance(raw, str):
        raw = [raw]
    result = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, str):
            label = item
        elif isinstance(item, dict):
            label = str(item.get("name") or item.get("text") or item.get("title") or "")
        else:
            continue
        label = label.strip()
        if label and normalize_label(label) not in {normalize_label(x) for x in result}:
            result.append(label[:140])
    return result[:12]


def _sources(record: dict, motif: dict) -> list[dict]:
    sources = []
    seen = set()
    for context in _contexts(record):
        doc_id = context.get("doc_id") or context.get("source_id") or context.get("id")
        if doc_id is None:
            continue
        modality = _modality(context.get("doc_part") or context.get("part")
                             or context.get("modality") or context.get("source_type"))
        key = (modality, str(doc_id))
        if key in seen:
            continue
        seen.add(key)
        sources.append({"doc_id": str(doc_id), "modality": modality,
                        "provenance": "mmqa_supporting_context"})
    # Some official records encode modalities in their question type but do
    # not provide complete context IDs. These are only simulated source roles.
    if not sources:
        for modality in motif["source_modalities"]:
            if modality in {"table", "text", "image"}:
                sources.append({"doc_id": None, "modality": modality,
                                "provenance": "mmqa_modality_role_only"})
    return sources


SCENE_STOPWORDS = {
    "Which", "What", "Who", "Where", "When", "How", "Why", "Is", "Are",
    "Was", "Were", "The", "A", "An", "Do", "Does", "Did", "In", "At",
    "Of", "On", "For", "From", "Can", "Have", "Has", "List", "Name",
}


def _scene_topic(motif: dict, entities: list[str], record: dict) -> tuple[str, str]:
    """Extract a subject/entity label, never the original question as a goal.

    Original MMQA question text may help identify proper nouns *only*.
    Query operators, target properties, and entire interrogative sentences
    must not become a newly generated Question or terminal work goal.
    """
    if entities:
        return entities[0], "original_mmqa_entity_metadata"
    meta = record.get("metadata") or {}
    for field in ("topic", "entity_title", "page_title", "subject_title"):
        title = meta.get(field) or record.get(field)
        if isinstance(title, str):
            title = " ".join(title.split())[:100]
            if title and "?" not in title and not re.match(
                r"^(which|what|who|when|where|why|how)\b", title, re.I
            ):
                return title, "original_mmqa_scene_metadata"
    raw_question = motif["question_hint"]
    # Only capitalized proper-noun spans. Never extract a whole clause or
    # use the dataset question verb/answer criterion as a new task.
    spans = re.findall(
        r"\b[A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*){0,4}\b",
        raw_question,
    )
    clean = []
    for span in spans:
        words = [word for word in span.split() if word not in SCENE_STOPWORDS]
        label = " ".join(words).strip(" .,:;!?")
        if not label or len(label) < 3 or "?" in label:
            continue
        clean.append(label)
    if clean:
        # Favor subject-specific named phrases over one-letter/source verbs.
        return max(clean, key=lambda value: (len(value.split()), len(value))), (
            "named_scene_entity_from_mmqa_question")
    group = motif["domain_hint"].replace("_", " ")
    return f"{group} reference collection {motif['record_id'][:7]}", (
        "original_mmqa_source_bundle_fallback")


def _question_operation(motif: dict, question: str) -> str:
    pattern = motif["logical_pattern"]
    lower = question.lower()
    if pattern == "aggregation" or re.search(r"\b(how many|number of|count)\b", lower):
        return "aggregation"
    if pattern == "comparison":
        return "comparison"
    if pattern == "intersection":
        return "intersection"
    if pattern == "composition":
        return "composition"
    if re.search(r"\b(when|what year|what date|before|after)\b", lower):
        return "temporal"
    return "lookup"


def load_scenario_seeds(path: str | Path, *, limit: int = 30000) -> tuple[list[dict], dict]:
    """Load *real* MMQA questions as need/scene anchors, without copying answers."""
    if limit < 1:
        raise ValueError("limit must be positive")
    seeds = []
    counters = Counter()
    dedup = set()
    for index, record in enumerate(_rows(Path(path))):
        if index >= limit:
            break
        counters["scanned"] += 1
        motif = extract_motif(record, index)
        if motif is None:
            counters["missing_question"] += 1
            continue
        question = motif["question_hint"].strip()
        key = normalize_label(question)
        if not key or key in dedup:
            counters["duplicate_question"] += 1
            continue
        dedup.add(key)
        sources = _sources(record, motif)
        tree = parse_composition(motif["question_type"])
        entities = _entities(record)
        topic, topic_policy = _scene_topic(motif, entities, record)
        original_op = _question_operation(motif, question)
        seed = {
            "record_id": motif["record_id"],
            "question": question,
            "topic": topic,
            "entities": entities,
            "domain_hint": motif["domain_hint"],
            "original_question_operation": original_op,
            "scene_topic_policy": topic_policy,
            "composition": tree,
            "question_type": motif["question_type"],
            "logical_pattern": motif["logical_pattern"],
            "source_modalities": motif["source_modalities"],
            "supporting_sources": sources,
            "intermediate_pattern": motif["intermediate_pattern"],
            "pseudo_language_question": motif["pseudo_language_question"],
            "provenance": "original_mmqa_question_and_supporting_metadata",
            "yago_matches": [],
        }
        seeds.append(seed)
        counters["original_question_operation:" + original_op] += 1
        counters["domain:" + motif["domain_hint"]] += 1
        for source in sources:
            counters["source_modality:" + source["modality"]] += 1
    counters["unique_scenario_seeds"] = len(seeds)
    return seeds, dict(counters)


def enrich_with_yago(seeds: list[dict], path: str | Path | None, *,
                     max_triples: int = 500000) -> dict:
    """Optional bounded YAGO lookup by unique exact entity label.

    No cross-dataset identity is asserted for ambiguous aliases. We retain
    original RDF subjects and observed predicate/value summaries, never make
    up an entity match. A zero-match run is valid and explicitly reported.
    """
    if path is None:
        return {"policy": "not_requested", "matched_seeds": 0, "scanned": 0}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    from environment.task_construction.data_sources.yago import iter_turtle_triples

    wanted = {normalize_label(e) for seed in seeds for e in seed["entities"]}
    wanted.discard("")
    matches: dict[str, dict[str, list]] = {}
    seen_uris: dict[str, set[str]] = {}
    scanned = 0
    for triple in iter_turtle_triples(path):
        scanned += 1
        if scanned > max_triples:
            break
        label = normalize_label(display(triple.subject))
        if label not in wanted:
            continue
        seen_uris.setdefault(label, set()).add(triple.subject)
        entry = matches.setdefault(label, {}).setdefault(triple.subject, [])
        if len(entry) < 8:
            entry.append({
                "predicate": display(triple.predicate),
                "object": display(triple.object) if not triple.object_is_literal
                          else str(triple.object)[:180],
                "raw_subject": triple.subject,
                "raw_object": triple.object,
            })
    for seed in seeds:
        for name in seed["entities"]:
            label = normalize_label(name)
            uris = seen_uris.get(label, set())
            if len(uris) == 1:
                subject = next(iter(uris))
                seed["yago_matches"].append({
                    "label": name, "entity_id": subject,
                    "facts": matches[label][subject],
                    "verification": "exact_unique_label_in_scanned_yago_prefix",
                })
    return {
        "policy": "bounded_exact_unique_label_not_entity_equivalence_proof",
        "scanned": min(scanned, max_triples),
        "candidate_labels": len(wanted),
        "matched_seeds": sum(bool(s["yago_matches"]) for s in seeds),
        "ambiguous_labels": sum(len(v) > 1 for v in seen_uris.values()),
    }
