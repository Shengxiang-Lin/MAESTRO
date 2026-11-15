"""Bounded, streaming reader for the YAGO 4.6 Turtle facts dump.

Only real RDF entity-to-entity relations are retained. The full RDF dump is
never materialized; --max-triples bounds parsing and the in-memory graph.
Requires pyoxigraph (not rdflib.Graph.parse on the multi-GB dump).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote
import gzip
import re
import zipfile


@dataclass(frozen=True)
class Edge:
    subject: str
    predicate: str
    object: str
    origin: str = "yago"


@dataclass
class KnowledgeGraph:
    edges: list[Edge] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    scanned: int = 0
    input_file: str = ""

    def outgoing(self) -> dict[str, list[Edge]]:
        index: dict[str, list[Edge]] = defaultdict(list)
        for edge in self.edges:
            index[edge.subject].append(edge)
        return dict(index)

    def label(self, iri: str) -> str:
        if iri in self.labels:
            return self.labels[iri]
        name = unquote(re.split(r"[/#]", iri.rstrip("/#"))[-1])
        name = re.sub(r"[_-]+", " ", name).strip()
        return name or iri

    def choose_walk(self, rng, *, min_hops: int = 1, max_hops: int = 3) -> list[Edge]:
        adj = self.outgoing()
        if not adj:
            raise ValueError("YAGO sample contains no entity-to-entity relations")
        # Build from an existing valid edge; never fabricate an alleged YAGO fact.
        for _ in range(300):
            first = rng.choice(self.edges)
            path = [first]
            seen = {first.subject, first.object}
            wanted = rng.randint(min_hops, max_hops)
            while len(path) < wanted:
                options = [e for e in adj.get(path[-1].object, [])
                           if e.object not in seen and e.predicate != path[-1].predicate]
                if not options:
                    break
                step = rng.choice(options)
                path.append(step)
                seen.add(step.object)
            if len(path) >= min_hops:
                return path
        raise ValueError(
            "No walk of the requested depth in the sampled YAGO graph. "
            "Increase --max-triples or reduce --min-hops."
        )


def _files(directory: Path) -> list[Path]:
    if directory.is_file():
        return [directory]
    if not directory.is_dir():
        raise FileNotFoundError(f"YAGO directory is missing: {directory}")
    candidates = [
        p for p in directory.rglob("*") if p.is_file()
        and p.name.lower().endswith((".ttl", ".ttl.gz", ".nt", ".nt.gz", ".zip"))
    ]
    # Fact dumps before tiny samples, never accidentally parse meta/RDF-star logs.
    candidates = [p for p in candidates if not any(
        x in p.name.lower() for x in ("fact-log", "range-log", "taxonomy-log", "meta")
    )]
    facts = [p for p in candidates if "facts" in p.name.lower()
             and "beyond" not in p.name.lower()]
    tiny = [p for p in candidates if "tiny" in p.name.lower()]
    if facts:
        return sorted(facts, key=lambda p: (len(p.parts), p.name))
    if tiny:
        return sorted(tiny)
    return sorted(p for p in candidates if
                  not any(s in p.name.lower() for s in ("labels", "taxonomy", "schema")))


def _parse_file(path: Path):
    try:
        from pyoxigraph import RdfFormat, parse
    except ImportError as exc:
        raise RuntimeError(
            "Reading YAGO 4.6 requires pyoxigraph; install with "
            "pip install 'pyoxigraph>=0.5,<0.6'"
        ) from exc
    is_nt = ".nt" in path.name.lower()
    fmt = RdfFormat.N_TRIPLES if is_nt else RdfFormat.TURTLE
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist()
                       if n.lower().endswith((".ttl", ".nt")) and not n.endswith("/")]
            if not members:
                raise ValueError(f"No Turtle/N-Triples member found in {path}")
            name = sorted(members)[0]
            inner_fmt = RdfFormat.N_TRIPLES if name.lower().endswith(".nt") else RdfFormat.TURTLE
            with archive.open(name) as stream:
                yield from parse(input=stream, format=inner_fmt)
    elif path.name.lower().endswith(".gz"):
        with gzip.open(path, "rb") as stream:
            yield from parse(input=stream, format=fmt)
    else:
        yield from parse(path=path, format=fmt)


def load_yago(directory: str | Path, *, max_triples: int = 250_000,
              max_edges: int = 70_000) -> KnowledgeGraph:
    """Read a bounded real-fact sample. No network or artificial YAGO assertions."""
    if max_triples < 1 or max_edges < 1:
        raise ValueError("max_triples and max_edges must be positive")
    paths = _files(Path(directory))
    if not paths:
        raise FileNotFoundError(f"No YAGO facts/tiny Turtle files under {directory}")
    graph = KnowledgeGraph(input_file=str(paths[0]))
    from_rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    from_rdfs = "http://www.w3.org/2000/01/rdf-schema#"
    ignored = {from_rdf + "type", from_rdfs + "subClassOf",
               "http://www.w3.org/2002/07/owl#sameAs",
               "http://schema.org/sameAs"}
    seen = set()
    for row in _parse_file(paths[0]):
        graph.scanned += 1
        if graph.scanned > max_triples:
            graph.scanned = max_triples
            break
        subj, pred, obj = row.subject, row.predicate, row.object
        if subj.__class__.__name__ != "NamedNode":
            continue
        s, p = subj.value, pred.value
        if obj.__class__.__name__ == "Literal":
            if p in {from_rdfs + "label", "http://schema.org/name"}:
                value = obj.value
                if value and len(value) <= 120:
                    graph.labels.setdefault(s, value)
            continue
        if obj.__class__.__name__ != "NamedNode" or p in ignored:
            continue
        o = obj.value
        if s == o or (s, p, o) in seen:
            continue
        seen.add((s, p, o))
        graph.edges.append(Edge(s, p, o))
        if len(graph.edges) >= max_edges:
            break
    if not graph.edges:
        raise ValueError(
            f"No usable entity relations in {graph.input_file} "
            f"after {graph.scanned} triples; try a facts dump or higher limit"
        )
    return graph
