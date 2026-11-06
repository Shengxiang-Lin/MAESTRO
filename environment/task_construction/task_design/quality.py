"""Deterministic quality checks for frozen task designs."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any,Mapping,Sequence
from ..data_sources.taxonomy import TaxonomyIndex
from .schema import TaskDesign
@dataclass(frozen=True)
class TaskDesignQuality:
    accepted:bool;reasons:tuple[str,...];metrics:dict[str,Any]
    def to_dict(self):return {"accepted":self.accepted,"reasons":list(self.reasons),"metrics":dict(self.metrics)}
def _as_mapping(value):return value if isinstance(value,Mapping) else {}
def _as_sequence(value):return value if isinstance(value,Sequence) and not isinstance(value,(str,bytes)) else ()
def _schema_type_valid(actual_type,schema_type,taxonomy):return True if taxonomy is None or not actual_type or not schema_type else taxonomy.is_subtype(str(actual_type),str(schema_type))
def _validate_clue_graphs(design,*,taxonomy=None):
    reasons=[];graph_count=attribute_count=relation_count=type_checks=schema_checks=0
    for clue_index,raw_clue in enumerate(design.public_clues):
        clue=_as_mapping(raw_clue)
        if clue.get("kind")!="latent_entity":continue
        if "atoms" in clue:reasons.append(f"legacy_flat_clue_atoms:{clue_index}")
        graph=_as_mapping(clue.get("clue_graph"))
        if not graph:reasons.append(f"latent_clue_missing_graph:{clue_index}");continue
        graph_count+=1;nodes=[_as_mapping(item) for item in _as_sequence(graph.get("nodes")) if isinstance(item,Mapping)];node_ids=[str(item.get("clue_id","")) for item in nodes];valid_ids={value for value in node_ids if value};node_by_id={str(item.get("clue_id")):item for item in nodes if item.get("clue_id")}
        if not nodes:reasons.append(f"clue_graph_empty:{clue_index}");continue
        if len(valid_ids)!=len(nodes):reasons.append(f"clue_graph_duplicate_or_missing_node_id:{clue_index}")
        for node in nodes:
            kind=str(node.get("kind",""))
            if kind=="attribute_constraint":attribute_count+=1;reasons.extend([f"clue_graph_missing_attribute_owner:{clue_index}"] if not node.get("owner_type") else []);reasons.extend([f"clue_graph_missing_attribute_predicate:{clue_index}"] if not node.get("predicate") else [])
            elif kind=="relation_traversal":relation_count+=1;reasons.extend([f"clue_graph_missing_relation_endpoint:{clue_index}"] if not node.get("from_type") or not node.get("to_type") else []);reasons.extend([f"clue_graph_missing_relation_predicate:{clue_index}"] if not node.get("predicate") else [])
        edges=[_as_mapping(item) for item in _as_sequence(graph.get("edges")) if isinstance(item,Mapping)];incoming={node_id:set() for node_id in valid_ids};outgoing={node_id:set() for node_id in valid_ids}
        for edge in edges:
            source,target=str(edge.get("source","")),str(edge.get("target",""))
            if source not in valid_ids or target not in valid_ids:reasons.append(f"clue_graph_dangling_edge:{clue_index}");continue
            outgoing[source].add(target);incoming[target].add(source)
        for clue_id,node in node_by_id.items():
            kind=str(node.get("kind",""));parents=incoming.get(clue_id,set())
            if kind not in {"attribute_constraint","relation_traversal"}:continue
            if len(parents)!=1:reasons.append(f"clue_graph_non_unary_typed_step:{clue_index}");continue
            parent=node_by_id.get(next(iter(parents)),{});parent_type=parent.get("output_type") if isinstance(parent,Mapping) else None
            if kind=="attribute_constraint":
                owner_type=node.get("owner_type")
                if parent_type and owner_type:type_checks+=1;reasons.extend([f"clue_graph_attribute_owner_mismatch:{clue_index}"] if str(parent_type)!=str(owner_type) else [])
                schema_owner=node.get("schema_owner_type")
                if owner_type and schema_owner and taxonomy is not None:schema_checks+=1;reasons.extend([f"clue_graph_attribute_schema_incompatible:{clue_index}"] if not _schema_type_valid(owner_type,schema_owner,taxonomy) else [])
            else:
                from_type,to_type,output_type=node.get("from_type"),node.get("to_type"),node.get("output_type")
                if parent_type and from_type:type_checks+=1;reasons.extend([f"clue_graph_relation_from_mismatch:{clue_index}"] if str(parent_type)!=str(from_type) else [])
                if output_type and to_type:type_checks+=1;reasons.extend([f"clue_graph_relation_to_mismatch:{clue_index}"] if str(output_type)!=str(to_type) else [])
                for actual,schema,label in ((from_type,node.get("schema_from_type"),"from"),(to_type,node.get("schema_to_type"),"to")):
                    if actual and schema and taxonomy is not None:schema_checks+=1;reasons.extend([f"clue_graph_relation_{label}_schema_incompatible:{clue_index}"] if not _schema_type_valid(actual,schema,taxonomy) else [])
        output_ids=[str(value) for value in _as_sequence(graph.get("output_ids"))]
        if not output_ids or any(value not in valid_ids for value in output_ids):reasons.append(f"clue_graph_invalid_output:{clue_index}");continue
        target_type=clue.get("entity_type")
        if target_type:
            for output_id in output_ids:
                output_type=node_by_id.get(output_id,{}).get("output_type")
                if output_type:type_checks+=1;reasons.extend([f"clue_graph_output_type_mismatch:{clue_index}"] if str(output_type)!=str(target_type) else [])
        roots=[node_id for node_id in valid_ids if not incoming[node_id]];reachable=set(roots);frontier=list(roots)
        while frontier:
            current=frontier.pop()
            for child in outgoing[current]:
                if child not in reachable:reachable.add(child);frontier.append(child)
        if any(value not in reachable for value in output_ids):reasons.append(f"clue_graph_disconnected_output:{clue_index}")
    return reasons,{"clue_graph_count":graph_count,"clue_attribute_constraint_count":attribute_count,"clue_relation_traversal_count":relation_count,"clue_type_continuity_checks":type_checks,"clue_schema_compatibility_checks":schema_checks}
def assess_task_design(design,*,require_multisource=True,taxonomy=None):
    reasons=[];visibility=design.resource_visibility;initial=[item for item in visibility if item.visibility in {"attached","provided"}];discoverable=[item for item in visibility if item.visibility=="discoverable"]
    if not visibility:reasons.append("no_existing_resources")
    if visibility and not initial and not (
        design.entry_strategy == "web_search"
        and len(discoverable) == len(visibility)
        and design.metadata.get("retrieval_policy")
            == "task_local_ranked_public_index_v1"
    ):
        reasons.append("no_public_entry_resource")
    if design.entry_strategy == "web_search" and initial:
        reasons.append("web_search_has_presupplied_source")
    if require_multisource and (design.minimum_required_source_count or 0)<=1:reasons.append("single_source_shortcut")
    if len(visibility)>1 and not discoverable:reasons.append("all_resources_initial_no_acquisition")
    if bool(design.metadata.get("adds_synthetic_resources")):reasons.append("task_design_added_synthetic_resource")
    if bool(design.metadata.get("changes_symbolic_truth")):reasons.append("task_design_changed_symbolic_truth")
    if design.motif=="compliance" and design.entry_strategy!="compliance_standard":reasons.append("implicit_compliance_motif")
    clue_reasons,clue_metrics=_validate_clue_graphs(design,taxonomy=taxonomy);reasons.extend(clue_reasons);metrics={"resource_count":len(visibility),"initial_resource_count":len(initial),"discoverable_resource_count":len(discoverable),"latent_entity_count":len(design.latent_entity_ids),"reasoning_node_count":len(design.required_reasoning_node_ids),"minimum_required_source_count":design.minimum_required_source_count,"has_attached_media":any(item.visibility=="attached" for item in visibility),"acquisition_scenario":design.entry_strategy,"search_query_entry":design.entry_strategy=="web_search" and not initial,**clue_metrics};return TaskDesignQuality(not reasons,tuple(dict.fromkeys(reasons)),metrics)
