"""Deterministic rendering pipeline from symbolic worlds to agent observations."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from ..executor import SymbolicExecutor
from ..ir import ProgramIR, ProgramNode
from ..validation import normalize_answer
from ..world import EvidenceSource, Fact, SyntheticWorld
from .audit import RenderingAudit
from .naming import EntityAliasBook, predicate_label
from .literal_context import collect_literal_surfaces
from .query_grounding import build_public_query_plan
from .questions import QuestionRenderer
from .schema import RenderedTask
from .sources import render_source


class RenderingError(RuntimeError):
    pass


def program_from_dict(payload: Mapping[str, Any]) -> ProgramIR:
    nodes = [
        ProgramNode(
            node_id=int(node["node_id"]),
            op=str(node["op"]),
            dependencies=tuple(int(dep) for dep in node.get("dependencies", ())),
            inputs=tuple(str(value) for value in node.get("inputs", ())),
            raw_function=str(node["raw_function"]) if node.get("raw_function") is not None else None,
        )
        for node in payload.get("nodes", ())
    ]
    return ProgramIR(nodes=nodes, metadata=dict(payload.get("metadata", {})))


def world_from_dict(payload: Mapping[str, Any]) -> SyntheticWorld:
    world = SyntheticWorld(
        entities={str(key): str(value) for key, value in payload.get("entities", {}).items()},
        metadata=dict(payload.get("metadata", {})),
    )
    for item in payload.get("facts", ()):
        world.add_fact(Fact(
            fact_id=str(item["fact_id"]), subject=str(item["subject"]),
            predicate=str(item["predicate"]), object=item.get("object"),
            object_kind=str(item.get("object_kind", "entity")),
            source_id=str(item["source_id"]) if item.get("source_id") is not None else None,
        ))
    sources = payload.get("sources", {})
    iterable = sources.values() if isinstance(sources, Mapping) else sources
    for item in iterable:
        world.add_source(EvidenceSource(
            source_id=str(item["source_id"]), source_type=str(item.get("source_type", "text")),
            fact_ids=tuple(str(value) for value in item.get("fact_ids", ())),
            metadata=dict(item.get("metadata", {})),
        ))
    return world


def _literal_contexts(world: SyntheticWorld) -> dict[str, tuple[tuple[str, str], ...]]:
    contexts: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for fact in world.facts:
        if fact.object_kind == "entity": continue
        key = str(fact.object); context = (fact.object_kind, fact.predicate)
        if context not in contexts[key]: contexts[key].append(context)
    return {key: tuple(values) for key, values in contexts.items()}


def _literal_profiles(world: SyntheticWorld) -> Mapping[str, Mapping[str, Any]] | None:
    raw = world.metadata.get("attribute_semantics", {})
    if not isinstance(raw, Mapping): return None
    profiles = {str(predicate): dict(profile) for predicate, profile in raw.items() if isinstance(profile, Mapping)}
    return profiles or None


def _display_answer(value: Any, aliases: EntityAliasBook, program: ProgramIR) -> Any:
    final = program.node_map()[program.topological_order()[-1]]
    answer_predicate = final.inputs[0] if final.op == "QUERY_ATTR" and final.inputs else None
    def render(item: Any) -> Any:
        if isinstance(item, list): return [render(value) for value in item]
        if isinstance(item, tuple): return [render(value) for value in item]
        if isinstance(item, set): return sorted(render(value) for value in item)
        if isinstance(item, dict): return {str(key): render(value) for key, value in item.items()}
        if isinstance(item, str):
            if aliases.is_entity(item): return aliases.entity(item)
            if final.op == "QUERY_RELATION": return predicate_label(item)
            return aliases.literal(item, predicate=answer_predicate)
        if isinstance(item, (int, float)) and final.op == "QUERY_ATTR":
            return aliases.literal(item, object_kind="number", predicate=answer_predicate)
        return item
    return render(value)


def _answer_entity_ids(value: Any, world: SyntheticWorld) -> set[str]:
    result: set[str] = set()
    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for value in item.values(): visit(value)
            return
        if isinstance(item, (list, tuple, set)):
            for value in item: visit(value)
            return
        token = str(item)
        if token in world.entities: result.add(token)
    visit(value); return result


def _contains_alias(text: str, alias: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(str(alias))}(?![A-Za-z0-9])", str(text), flags=re.IGNORECASE))


def _public_query_metadata(plan, *, question: str, aliases: EntityAliasBook, raw_answer: Any, world: SyntheticWorld) -> dict[str, Any]:
    payload = plan.to_dict(); proof_entities = set(plan.proof_entity_ids); answer_entities = _answer_entity_ids(raw_answer, world)
    exposed_proof = {entity_id for entity_id in proof_entities if _contains_alias(question, aliases.entity(entity_id))}
    latent_proof = proof_entities - exposed_proof; latent_intermediate = latent_proof - answer_entities
    payload.update({
        "answer_entity_ids": sorted(answer_entities), "exposed_proof_entity_ids": sorted(exposed_proof),
        "latent_proof_entity_ids": sorted(latent_proof), "latent_intermediate_entity_ids": sorted(latent_intermediate),
        "latent_intermediate_count": len(latent_intermediate),
        "exposed_proof_entity_ratio": len(exposed_proof) / len(proof_entities) if proof_entities else 0.0,
        "grounding_clue_supported_count": sum(bool(item.support_fact_ids) for item in plan.latent_entities),
    })
    return payload


def _task_id(record: Mapping[str, Any], program: ProgramIR) -> str:
    basis = str(record.get("source_program_signature") or program.canonical_signature())
    return f"world-{hashlib.sha1(basis.encode('utf-8')).hexdigest()[:12]}"


def _validate_public_boundary(task: RenderedTask) -> None:
    serialized = json.dumps(task.training_dict(include_hidden=False), ensure_ascii=False); lowered = serialized.lower()
    forbidden = ('"proof_relevant"','"indispensable"','"critical_source_ids"','"nonproof_source_ids"','"source_roles"','"minimal_proof"','"fact_ids"')
    leaked = [token for token in forbidden if token in lowered]
    if leaked: raise RenderingError(f"hidden evidence labels leaked into public payload: {leaked}")
    placeholder = re.search(r"\b(?:value|val|string|text|label|literal|token|item|name)[_-]?\d+(?:[_-]?(?:alt|variant|distractor)\d*)?\b", serialized, flags=re.IGNORECASE)
    alt_marker = re.search(r"(?:_alt\b|\balt[_-])", serialized, flags=re.IGNORECASE)
    if placeholder or alt_marker: raise RenderingError("synthetic literal marker leaked into public payload")
    legacy_entity_alias = re.search(r"\b(?:Person|Organization|Administrative Area|Event|Award|Fictional Entity|Taxon|Creative Work|TVSeries|TV Series|Airline|Thing)\s+[A-Z]\b", serialized)
    if legacy_entity_alias: raise RenderingError(f"legacy letter entity alias leaked into public payload: {legacy_entity_alias.group(0)}")


def render_task_record(record: Mapping[str, Any], *, seed: int = 17) -> RenderedTask:
    program = program_from_dict(record["program"]); world = world_from_dict(record["world"])
    aliases = EntityAliasBook(world.entities, literal_contexts=_literal_contexts(world), literal_profiles=_literal_profiles(world), seed=seed)
    executor_answer = SymbolicExecutor().execute(program, world)
    if "answer" in record and normalize_answer(executor_answer) != normalize_answer(record["answer"]):
        raise RenderingError("recorded answer disagrees with symbolic executor")
    raw_answer = record.get("answer", executor_answer); query_plan = build_public_query_plan(record, program, world)
    question_renderer = QuestionRenderer(aliases, seed=seed, grounding_plan=query_plan); question = question_renderer.render(program)
    sources = tuple(render_source(source, world, aliases) for source in sorted(world.sources.values(), key=EvidenceSource.sort_key))
    if not sources: raise RenderingError("rendered task has no evidence sources")
    hidden = {
        "entity_aliases": aliases.to_dict(), "literal_surfaces": aliases.literal_surfaces(),
        "literal_surfaces_by_predicate": collect_literal_surfaces(program, world, aliases),
        "question_template_id": question_renderer.template_id, "reasoning_motif": record.get("reasoning_motif"),
        "program_signature": program.canonical_signature(), "evidence": dict(record.get("evidence", {})),
        "candidate_elimination": dict(record.get("candidate_elimination", {})),
        "public_query_plan": _public_query_metadata(query_plan, question=question, aliases=aliases, raw_answer=raw_answer, world=world),
    }
    task = RenderedTask(task_id=_task_id(record, program), question=question, sources=sources,
                        answer=_display_answer(raw_answer, aliases, program), raw_answer=raw_answer, hidden_metadata=hidden)
    _validate_public_boundary(task); return task


def render_jsonl(input_path: str | Path, output_path: str | Path, *, include_hidden: bool = False,
                 max_records: int | None = None, seed: int = 17) -> dict[str, Any]:
    source = Path(input_path); target = Path(output_path); target.parent.mkdir(parents=True, exist_ok=True)
    rendered = 0; source_types: Counter[str] = Counter(); answer_types: Counter[str] = Counter(); source_count_histogram: Counter[int] = Counter(); audit = RenderingAudit()
    with source.open("r", encoding="utf-8") as reader, target.open("w", encoding="utf-8") as writer:
        for index, line in enumerate(reader):
            if max_records is not None and rendered >= max_records: break
            if not line.strip(): continue
            record = json.loads(line); task = render_task_record(record, seed=seed + index); audit.add(task)
            writer.write(json.dumps(task.training_dict(include_hidden=include_hidden), ensure_ascii=False) + "\n")
            rendered += 1; source_types.update(source.source_type for source in task.sources); source_count_histogram[len(task.sources)] += 1; answer_types[type(task.answer).__name__] += 1
    return {"rendered": rendered, "input": str(source), "output": str(target), "include_hidden": include_hidden,
            "source_type_usage": dict(source_types.most_common()),
            "source_count_histogram": {str(key): value for key, value in sorted(source_count_histogram.items())},
            "answer_types": dict(answer_types.most_common()), "rendering_audit": audit.summary()}
