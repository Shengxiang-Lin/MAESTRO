"""Deterministic task-local web-search simulation with no Gold-witness ranking.

The index contains *all* frozen public source documents for one task, including
non-answer distractors. The retrieval policy sees only Main-visible question,
delegated instruction, prior public evidence, search hits and resource
acquisition prerequisites. It NEVER receives the target answer, a Gold source
list, or a fragment's support-resource IDs when ranking results.

Search hits reveal only a title/short snippet/resource handle; factual claims
are exposed separately by the assigned inspection tool. This is not an
Internet API and cannot demonstrate real-web discoverability of synthetic
entity aliases.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

_INITIAL = frozenset(("attached", "provided", "initial", "visible", "provided_initial"))
_STOP = frozenset((
    "all", "also", "and", "any", "are", "based", "been", "between", "can",
    "candidate", "candidates", "check", "complete", "constraints", "data",
    "determine", "each", "entities", "entity", "evidence", "find", "for",
    "from", "have", "identify", "identified", "into", "its", "matching",
    "names", "only", "other", "provided", "record", "records", "reference",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth",
    "references", "related", "relevant", "required", "result", "results",
    "return", "search", "set", "source", "sources", "specified", "subtask",
    "supply", "supplied", "task", "that", "the", "their", "them", "these",
    "this", "those", "through", "using", "value", "values", "which", "with",
))


def _tokens(text: Any) -> set[str]:
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, sort_keys=True)
    # Split camel-case predicates, e.g. schema:malePopulation.
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return {
        token for token in re.findall(r"[a-z][a-z0-9]{2,}", text.lower())
        if token not in _STOP and not token.startswith("source")
    }


def _document(source: Mapping[str, Any]) -> str:
    # Only information that a *public* search index could index. No
    # fragment result, Gold witnesses, program node labels or hidden facts.
    types = " ".join(str(x) for x in source.get("entity_types", {}).values())
    return (str(source["title"]) + " " +
            json.dumps(source.get("content", ""), ensure_ascii=False) + " " +
            types)


def _snippet(source: Mapping[str, Any]) -> str:
    content = source.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return " ".join(content.split())[:180]


class SimulatedSearchIndex:
    """Answer-blind public-document retrieval over a small task-local corpus."""

    def __init__(self, sources: Sequence[Mapping[str, Any]]):
        by_id = {}
        for source in sources:
            sid = source["resource_id"]
            if sid in by_id:
                raise ValueError("duplicate_search_source:" + sid)
            by_id[sid] = source
        self.sources = by_id
        self.terms = {sid: _tokens(_document(row)) for sid, row in by_id.items()}

    def search(
        self, query: str, *, seen: set[str] | None = None,
        per_page: int = 4, page: int = 0,
    ) -> list[dict[str, Any]]:
        if per_page < 1 or page < 0:
            raise ValueError("invalid_search_pagination")
        words = _tokens(query)
        if not words:
            return []
        discovered = set(seen or ())
        scored = []
        for sid, source in self.sources.items():
            if source.get("visibility") in _INITIAL or sid in discovered:
                continue
            # Prerequisites are source handles already discovered by a
            # previous search/inspection, never oracle-furnished handles.
            parents = set(source.get("anchor_dependency_resource_ids", ()))
            if parents and not parents & discovered:
                continue
            overlap = words & self.terms[sid]
            if not overlap:
                continue
            title_terms = _tokens(str(source["title"]))
            type_terms = _tokens(" ".join(
                str(v) for v in source.get("entity_types", {}).values()
            ))
            score = len(overlap) + 2*len(overlap & title_terms) + len(overlap & type_terms)
            scored.append((-score, sid, source))
        scored.sort(key=lambda row:(row[0], row[1]))
        return [{
            "resource_id": source["resource_id"],
            "title": source["title"],
            "source_type": source["source_type"],
            "snippet": _snippet(source),
            "score": -score,
        } for score, _, source in scored[page*per_page:(page+1)*per_page]]


def simulate_search(
    catalog: Sequence[Mapping[str, Any]], *,
    public_question: str,
    public_task: str,
    already_known_ids: Sequence[str] = (),
    per_page: int = 4,
    max_passes: int = 4,
) -> dict[str, Any]:
    """Simulate query -> ranked hits -> query expansion, without Gold answers.

    The first query comes from the current Main public question, the second
    from its delegated instruction. Later queries use only *retrieved search
    snippets*, so an entity alias discovered in a hit may unlock downstream
    links. Searches are paginated so a longer result list is not implicitly
    replaced by a Gold witness set.
    """
    index = SimulatedSearchIndex(catalog)
    known = set(already_known_ids)
    observed_hits: dict[str, dict[str, Any]] = {}
    calls = []
    seeds = [public_task, public_question]
    queries = []
    for seed in seeds:
        clean = " ".join(str(seed).split())
        if clean and clean not in queries:
            queries.append(clean)
    for passage in range(max_passes):
        if passage >= len(queries):
            break
        query = queries[passage]
        hits = []
        for page in range(0, max(1, (len(index.sources)+per_page-1)//per_page)):
            batch = index.search(query, seen=known, per_page=per_page, page=page)
            if not batch:
                break
            hits.extend(batch)
            if len(batch) < per_page:
                break
        calls.append({"tool":"GoogleSearchAction", "query":query,
                      "hits":hits, "query_origin":
                      "public_task" if passage==0 else
                      "public_question" if passage==1 else "search_snippet"})
        new = [hit for hit in hits if hit["resource_id"] not in observed_hits]
        observed_hits.update({h["resource_id"]: h for h in hits})
        # Result snippets are now observable, but source claims are not read
        # until the inspection tool is invoked in the synthetic trajectory.
        known.update(h["resource_id"] for h in hits)
        if new and len(queries) < max_passes:
            terms = " ".join(h["snippet"][:100] for h in new[:2])
            if terms.strip() and terms not in queries:
                queries.append(terms)
    return {
        "policy": "task_local_ranked_public_index_v1",
        "ranking_uses_gold_witness": False,
        "live_tool_calls": False,
        "calls": calls,
        "discovered_source_ids": sorted(observed_hits),
    }


def verify_retrieved_witnesses(
    audit: Mapping[str, Any], required: Sequence[Mapping[str, Any]],
) -> None:
    discovered = set(audit["discovered_source_ids"])
    # Initial attachments are NOT supposed to be searched; they are
    # inspected directly with their modality-specific tools.
    missing = sorted({
        item["resource_id"] for item in required
        if item.get("visibility", "provided") not in _INITIAL
        and item["resource_id"] not in discovered
    })
    if missing:
        raise ValueError("public_search_could_not_discover_required_source:" +
                         ",".join(missing))


__all__ = ["SimulatedSearchIndex", "simulate_search", "verify_retrieved_witnesses"]
