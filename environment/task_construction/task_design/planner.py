"""Construct a frozen TaskDesign from one executable world record."""
from __future__ import annotations
import hashlib
from typing import Any,Iterable,Mapping
from ..ir import ProgramIR,ProgramNode
from ..world import EvidenceSource,Fact,SyntheticWorld
from .fuzzification import LatentEntityPlan,build_public_query_plan
from .motifs import infer_answer_type,infer_task_motif,motif_family
from .schema import ClueGraph,ClueGraphEdge,ClueGraphNode,ResourceVisibility,TaskDesign
_MEDIA_SOURCE_TYPES={"image","image_report","video","video_report","audio","audio_transcript"};_VALUE_FILTER_OPS={"FILTER_STRING","FILTER_NUMBER","FILTER_YEAR","FILTER_DATE"};_SELECTION_OPS={"SELECT_BETWEEN","SELECT_AMONG"}
def _program_from_dict(payload):return ProgramIR([ProgramNode(int(item["node_id"]),str(item["op"]),tuple(int(v) for v in item.get("dependencies",())),tuple(str(v) for v in item.get("inputs",())),None if item.get("raw_function") is None else str(item["raw_function"])) for item in payload.get("nodes",())],metadata=dict(payload.get("metadata",{})))
def _world_from_dict(payload):
    world=SyntheticWorld(entities={str(k):str(v) for k,v in payload.get("entities",{}).items()},metadata=dict(payload.get("metadata",{})))
    for item in payload.get("facts",()):world.add_fact(Fact(str(item["fact_id"]),str(item["subject"]),str(item["predicate"]),item.get("object"),str(item.get("object_kind","entity")),str(item["source_id"]) if item.get("source_id") is not None else None))
    raw=payload.get("sources",{});sources=raw.values() if isinstance(raw,Mapping) else raw
    for item in sources:
        if isinstance(item,Mapping):world.add_source(EvidenceSource(str(item["source_id"]),str(item.get("source_type","text")),tuple(str(v) for v in item.get("fact_ids",())),dict(item.get("metadata",{}))))
    return world
def _task_id(record,program):basis=str(record.get("source_program_signature") or program.canonical_signature());return f"world-{hashlib.sha1(basis.encode()).hexdigest()[:12]}"
def _subgraph_nodes(program,root_ids):
    nodes=program.node_map();seen=set();stack=[int(v) for v in root_ids]
    while stack:
        node_id=stack.pop()
        if node_id in seen or node_id not in nodes:continue
        seen.add(node_id);stack.extend(int(dep) for dep in nodes[node_id].dependencies)
    return [node_id for node_id in program.topological_order() if node_id in seen]
def _mapping(value):return value if isinstance(value,Mapping) else {}
def _node_type(binding,node_id):
    raw=_mapping(binding.get("node_types"));value=raw.get(str(node_id),raw.get(node_id));return str(value) if value else None
def _binding_for_node(binding,key,node_id):
    table=_mapping(binding.get(key));return _mapping(table.get(str(node_id),table.get(node_id)))
def _strip_type_token(value):return None if not value else value.split("type:",1)[1] if value.startswith("type:") else value
def _clue_graph(program,latent,binding):
    nodes=program.node_map();node_ids=_subgraph_nodes(program,latent.identifying_node_ids);public_ids={node_id:f"c{index}" for index,node_id in enumerate(node_ids)};graph_nodes=[];graph_edges=[]
    for node_id in node_ids:
        node=nodes[node_id];output_type=_node_type(binding,node_id)
        if node.op in {"FIND","FIND_ALL"}:kind="anchor";semantics={"entity_type":output_type}
        elif node.op in _VALUE_FILTER_OPS and node.inputs:
            kind="attribute_constraint";attribute_binding=_binding_for_node(binding,"attribute_bindings",node_id);dependency_type=_node_type(binding,node.dependencies[0]) if node.dependencies else None;schema_owner_type=str(attribute_binding.get("entity_type") or "") or None;owner_type=dependency_type or output_type or schema_owner_type;semantics={"owner_type":owner_type,"schema_owner_type":schema_owner_type,"predicate":str(node.inputs[0]),"comparator":str(node.inputs[1]) if len(node.inputs)>1 else "=","value":str(node.inputs[2]) if len(node.inputs)>2 else None}
            for source_key,public_key in (("value_kind","value_kind"),("semantic_family","semantic_family"),("canonical_unit","canonical_unit")):
                value=attribute_binding.get(source_key)
                if value is not None:semantics[public_key]=value
        elif node.op=="FILTER_TYPE":kind="type_constraint";semantics={"entity_type":_strip_type_token(str(node.inputs[0])) if node.inputs else output_type}
        elif node.op=="RELATE" and node.inputs:
            kind="relation_traversal";relation_binding=_binding_for_node(binding,"relation_bindings",node_id);direction=str(node.inputs[1]) if len(node.inputs)>1 else str(relation_binding.get("direction","forward"));subject_type=relation_binding.get("subject_type");object_type=relation_binding.get("object_type");schema_from_type,schema_to_type=(object_type,subject_type) if direction=="backward" else (subject_type,object_type);dependency_type=_node_type(binding,node.dependencies[0]) if node.dependencies else None;from_type=dependency_type or schema_from_type;to_type=output_type or schema_to_type;semantics={"predicate":str(node.inputs[0]),"direction":direction,"from_type":str(from_type) if from_type else None,"to_type":str(to_type) if to_type else None,"schema_from_type":str(schema_from_type) if schema_from_type else None,"schema_to_type":str(schema_to_type) if schema_to_type else None}
        elif node.op in {"INTERSECT","UNION"}:kind="composition";semantics={"operator":"AND" if node.op=="INTERSECT" else "OR"}
        elif node.op in _SELECTION_OPS:kind="comparison";semantics={"operation":node.op,"predicate":str(node.inputs[0]) if node.inputs else None,"comparator":str(node.inputs[1]) if len(node.inputs)>1 else None}
        else:kind="operation";semantics={"operation":node.op}
        graph_nodes.append(ClueGraphNode(public_ids[node_id],kind,output_type,semantics))
        for dependency in node.dependencies:
            if dependency in public_ids:graph_edges.append(ClueGraphEdge(public_ids[dependency],public_ids[node_id]))
    return ClueGraph(tuple(graph_nodes),tuple(graph_edges),tuple(public_ids[node_id] for node_id in latent.identifying_node_ids if node_id in public_ids))
def _public_clues(program,latent_entities,binding):return tuple({"kind":"latent_entity","entity_type":latent.entity_type,"recovery_rule":latent.rule,"clue_graph":_clue_graph(program,latent,binding).to_dict(),"entity_id":latent.entity_id,"node_ids":list(latent.identifying_node_ids),"fact_ids":list(latent.support_fact_ids)} for latent in latent_entities)
def _source_ids_for_facts(world,fact_ids):
    result=set()
    for source_id,source in world.sources.items():
        if fact_ids.intersection(source.fact_ids):result.add(source_id)
    for fact in world.facts:
        if fact.fact_id in fact_ids and fact.source_id:result.add(str(fact.source_id))
    return result
_WEB_SOURCE_TYPES = {"text", "search", "table", "database"}
_RETRIEVAL_POLICY = "task_local_ranked_public_index_v1"


def _source_scenario(world):
    kinds = {source.source_type for source in world.sources.values()}
    if kinds and kinds <= _WEB_SOURCE_TYPES:
        return "web_search"
    if kinds and kinds <= _MEDIA_SOURCE_TYPES:
        return "provided_media"
    return "hybrid"


def _visibility_plan(world, latent_entities):
    ordered = sorted(world.sources.values(), key=EvidenceSource.sort_key)
    if not ordered:
        return ()
    scenario = _source_scenario(world)
    if scenario == "web_search":
        # A text/structured Wiki/YAGO task begins with a SEARCH QUERY, not
        # with an arbitrary 'provided file' containing its first candidates.
        # Discoverability is later tested against a separate public index.
        return tuple(ResourceVisibility(
            source.source_id, source.source_type,
            "discoverable", "web_result",
            "obtain through simulated GoogleSearchAction",
        ) for source in ordered)
    clue_fact_ids = {
        fid for latent in latent_entities for fid in latent.support_fact_ids
    }
    clue_sources = _source_ids_for_facts(world, clue_fact_ids)
    preferred = [s for s in ordered if s.source_id in clue_sources]
    preferred_media = [s for s in preferred if s.source_type in _MEDIA_SOURCE_TYPES]
    any_media = [s for s in ordered if s.source_type in _MEDIA_SOURCE_TYPES]
    entry = (preferred_media[0] if preferred_media else
             preferred[0] if preferred else
             any_media[0] if any_media else ordered[0])
    result = []
    for source in ordered:
        if source.source_id == entry.source_id:
            attached = source.source_type in _MEDIA_SOURCE_TYPES
            result.append(ResourceVisibility(
                source.source_id, source.source_type,
                "attached" if attached else "provided",
                "entry_anchor", "provided multimedia or reference input",
            ))
        else:
            result.append(ResourceVisibility(
                source.source_id, source.source_type,
                "discoverable", "discovery",
                "downstream evidence to locate through search",
            ))
    return tuple(result)


def plan_task_design(record,*,program=None,world=None,max_latent_entities=2):
    if program is None:
        raw=record.get("program",{});program=_program_from_dict(raw) if isinstance(raw,Mapping) else (_ for _ in ()).throw(ValueError("task record is missing program"))
    if world is None:
        raw=record.get("world",{});world=_world_from_dict(raw) if isinstance(raw,Mapping) else (_ for _ in ()).throw(ValueError("task record is missing world"))
    binding=_mapping(record.get("binding"));query_plan=build_public_query_plan(record,program,world,max_latent_entities=max_latent_entities);minimum_sources=query_plan.minimum_required_source_count;motif=infer_task_motif(program,query_plan,minimum_required_source_count=minimum_sources);visibility=_visibility_plan(world,query_plan.latent_entities);final_node=program.node_map()[program.topological_order()[-1]];latent_find_nodes=tuple(sorted({node_id for latent in query_plan.latent_entities for node_id in latent.find_node_ids}));entry_strategy=_source_scenario(world);return TaskDesign(_task_id(record,program),motif,final_node.op,infer_answer_type(program,binding),entry_strategy,tuple(query_plan.latent_entity_ids),latent_find_nodes,_public_clues(program,query_plan.latent_entities,binding),tuple(program.topological_order()),visibility,minimum_sources,{"motif_family":motif_family(motif),"proof_entity_count":len(query_plan.proof_entity_ids),"latent_entity_count":len(query_plan.latent_entities),"existing_source_count":len(world.sources),"adds_synthetic_resources":False,"changes_symbolic_truth":False,"clue_graph_version":2,"retrieval_policy":_RETRIEVAL_POLICY,"evidence_scenario":entry_strategy})
