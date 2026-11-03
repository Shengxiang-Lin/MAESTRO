"""Source-type-specific deterministic renderers for heterogeneous evidence."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List

from ..world import EvidenceSource, Fact, SyntheticWorld
from .naming import EntityAliasBook, predicate_label
from .schema import RenderedSource


def _fact_text(fact: Fact, aliases: EntityAliasBook) -> str:
    subject = aliases.entity(fact.subject)
    predicate = predicate_label(fact.predicate)
    value = aliases.value(fact.object, fact.object_kind, predicate=fact.predicate)
    return f"{subject} — {predicate}: {value}"


def _source_facts(source: EvidenceSource, world: SyntheticWorld) -> List[Fact]:
    fact_map = {fact.fact_id: fact for fact in world.facts}
    return [fact_map[fact_id] for fact_id in source.fact_ids if fact_id in fact_map]


def _render_text(source: EvidenceSource, facts: List[Fact], aliases: EntityAliasBook) -> Any:
    sentences = [_fact_text(fact, aliases) + "." for fact in facts]
    return " ".join(sentences)


def _render_table(source: EvidenceSource, facts: List[Fact], aliases: EntityAliasBook) -> Any:
    rows = []
    for fact in facts:
        rows.append({
            "entity": aliases.entity(fact.subject),
            "field": predicate_label(fact.predicate),
            "value": aliases.value(fact.object, fact.object_kind, predicate=fact.predicate),
        })
    return {"columns": ["entity", "field", "value"], "rows": rows}


def _render_image_report(source: EvidenceSource, facts: List[Fact], aliases: EntityAliasBook) -> Any:
    lines = ["Visual inspection notes:"]
    for fact in facts:
        lines.append("- " + _fact_text(fact, aliases))
    return "\n".join(lines)


def _render_video_report(source: EvidenceSource, facts: List[Fact], aliases: EntityAliasBook) -> Any:
    lines = ["Temporal inspection log:"]
    for index, fact in enumerate(facts):
        lines.append(f"- Segment {index + 1}: {_fact_text(fact, aliases)}")
    return "\n".join(lines)


def _render_audio_transcript(source: EvidenceSource, facts: List[Fact], aliases: EntityAliasBook) -> Any:
    lines = ["Transcript excerpts:"]
    speakers = ("Speaker A", "Speaker B", "Narrator")
    for index, fact in enumerate(facts):
        lines.append(f"- {speakers[index % len(speakers)]}: {_fact_text(fact, aliases)}")
    return "\n".join(lines)


def _render_database(source: EvidenceSource, facts: List[Fact], aliases: EntityAliasBook) -> Any:
    records: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for fact in facts:
        records[aliases.entity(fact.subject)][predicate_label(fact.predicate)] = aliases.value(
            fact.object, fact.object_kind, predicate=fact.predicate
        )
    return [{"record": entity, **fields} for entity, fields in sorted(records.items())]


_RENDERERS = {
    "text": _render_text,
    "table": _render_table,
    "image": _render_image_report,
    "image_report": _render_image_report,
    "video": _render_video_report,
    "video_report": _render_video_report,
    "audio": _render_audio_transcript,
    "audio_transcript": _render_audio_transcript,
    "database": _render_database,
}


def render_source(source: EvidenceSource, world: SyntheticWorld, aliases: EntityAliasBook) -> RenderedSource:
    facts = _source_facts(source, world)
    renderer = _RENDERERS.get(source.source_type, _render_text)
    content = renderer(source, facts, aliases)
    title = str(source.metadata.get("public_title") or f"Reference {int(source.metadata.get('layout_position', 0)) + 1}")
    hidden = {
        "role": source.metadata.get("role"),
        "proof_relevant": source.metadata.get("proof_relevant"),
        "indispensable": source.metadata.get("indispensable"),
        "layout_position": source.metadata.get("layout_position"),
        "source_type_origin": source.metadata.get("source_type_origin"),
        "bundle_size": source.metadata.get("bundle_size"),
        "material_role": source.metadata.get("material_role"),
        "surface_style": source.metadata.get("surface_style"),
    }
    return RenderedSource(
        source_id=source.source_id,
        source_type=source.source_type,
        title=title,
        content=content,
        fact_ids=tuple(source.fact_ids),
        hidden_metadata={key: value for key, value in hidden.items() if value is not None},
    )


__all__ = ["render_source"]
