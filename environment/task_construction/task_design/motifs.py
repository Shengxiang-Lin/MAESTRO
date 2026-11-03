"""Reasoning-motif selection for frozen task design."""
from __future__ import annotations
from typing import Any,Mapping
from ..ir import ProgramIR
from .fuzzification import PublicQueryPlan
_TEMPORAL_OPS={"FILTER_YEAR","FILTER_DATE","VERIFY_YEAR","VERIFY_DATE"};_BRANCH_OPS={"INTERSECT","UNION"};_SELECTION_OPS={"SELECT_BETWEEN","SELECT_AMONG"}
def infer_answer_type(program:ProgramIR,binding:Mapping[str,Any]|None=None)->str:
    final=program.node_map()[program.topological_order()[-1]]
    if final.op=="COUNT":return "number"
    if final.op=="QUERY_NAME":return "entity"
    if final.op=="QUERY_RELATION":return "relation"
    if final.op.startswith("VERIFY_"):return "boolean"
    if final.op=="QUERY_ATTR":
        bindings=binding.get("attribute_bindings",{}) if isinstance(binding,Mapping) else {};item=bindings.get(str(final.node_id),{}) if isinstance(bindings,Mapping) else {}
        return str(item["value_kind"]) if isinstance(item,Mapping) and item.get("value_kind") else "attribute_value"
    return "value"
def infer_task_motif(program,query_plan,*,minimum_required_source_count):
    nodes=program.node_map();order=program.topological_order();final=nodes[order[-1]];ops=[nodes[node_id].op for node_id in order]
    if final.op.startswith("VERIFY_"):return "verification"
    if any(op in _SELECTION_OPS for op in ops):return "comparison_ranking"
    if sum(op in _TEMPORAL_OPS for op in ops)>=2:return "temporal_reasoning"
    if any(op in _BRANCH_OPS for op in ops) or any(len(nodes[node_id].dependencies)>1 for node_id in order):return "branch_join"
    if query_plan.latent_entities:return "cross_source_disambiguation" if (minimum_required_source_count or 0)>=3 else "reverse_identification"
    if sum(op=="RELATE" for op in ops)>=2:return "relation_chain"
    return "evidence_retrieval"
def motif_family(motif):
    if motif in {"reverse_identification","cross_source_disambiguation"}:return "identification"
    if motif in {"comparison_ranking","temporal_reasoning"}:return "derived_reasoning"
    if motif=="branch_join":return "composition"
    if motif=="verification":return "verification"
    if motif=="relation_chain":return "relational"
    if motif=="compliance":return "compliance"
    return "retrieval"
