"""Bind abstract ProgramIR relation/attribute/type slots to YAGO-compatible types."""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any,Dict,List,Mapping
from ..ir import ProgramIR,ProgramNode
from ..data_sources.taxonomy import TaxonomyIndex
from .diversity_guidance import current_binding_diversity_guidance
from .predicate_binder import AttributeBinding,PredicateBinder,RelationBinding
from .semantic_support import MIN_EMPIRICAL_TYPE_MASS,assess_type_refinement
from .type_solver import TypeConstraintSolver
UNSUPPORTED_OPS={"FILTER_QUALIFIER_STRING","FILTER_QUALIFIER_NUMBER","FILTER_QUALIFIER_YEAR","FILTER_QUALIFIER_DATE","QUERY_ATTR_CONDITIONAL","QUERY_ATTR_QUALIFIER","QUERY_RELATION_QUALIFIER"}
ENTITY_OUTPUT_OPS={"FIND","FIND_ALL","RELATE","FILTER_TYPE","FILTER_STRING","FILTER_NUMBER","FILTER_YEAR","FILTER_DATE","INTERSECT","UNION","SELECT_BETWEEN","SELECT_AMONG"}
ATTRIBUTE_OP_KIND={"FILTER_STRING":"string","FILTER_NUMBER":"number","FILTER_YEAR":"year","FILTER_DATE":"date","SELECT_BETWEEN":"number","SELECT_AMONG":"number"};VERIFY_KIND={"VERIFY_STRING":"string","VERIFY_NUMBER":"number","VERIFY_YEAR":"year","VERIFY_DATE":"date"}
class BindingError(RuntimeError):pass
@dataclass
class BindingPlan:
    program:ProgramIR;node_types:Dict[int,str];relation_bindings:Dict[int,RelationBinding]=field(default_factory=dict);attribute_bindings:Dict[int,AttributeBinding]=field(default_factory=dict)
    def to_dict(self):return {"program":self.program.to_dict(),"node_types":{str(k):v for k,v in sorted(self.node_types.items())},"relation_bindings":{str(k):b.__dict__ for k,b in sorted(self.relation_bindings.items())},"attribute_bindings":{str(k):b.__dict__ for k,b in sorted(self.attribute_bindings.items())}}
def _is_deep_dependency_program(program):
    deep=program.metadata.get("deep_dependency",{});return isinstance(deep,Mapping) and bool(deep.get("composition_links"))
class TypedProgramBinder:
    def __init__(self,catalog:Mapping[str,Any],taxonomy:TaxonomyIndex,*,seed:int=17,min_predicate_count:int=25,max_attempts:int=64)->None:
        self.catalog=catalog;self.taxonomy=taxonomy;self.seed=seed;self.min_predicate_count=min_predicate_count;self.max_attempts=max_attempts
        from ..quality.observed_support import ObservedSupport
        self.observed_support=ObservedSupport(catalog,taxonomy)
    def bind(self,program):
        unsupported=sorted({node.op for node in program.nodes if node.op in UNSUPPORTED_OPS})
        if unsupported:raise BindingError(f"qualifier/conditional operators not enabled yet: {unsupported}")
        last_error="no feasible binding"
        for attempt in range(self.max_attempts):
            try:return self._bind_once(program,seed=self.seed+attempt*9973)
            except BindingError as exc:last_error=str(exc)
        raise BindingError(f"failed after {self.max_attempts} attempts: {last_error}")
    def _semantic_support(self,*,predicate,side,actual_type,schema_type):return assess_type_refinement(self.catalog.get("predicates",{}).get(predicate,{}),self.taxonomy,side=side,actual_type=actual_type,schema_type=schema_type,min_empirical_mass=MIN_EMPIRICAL_TYPE_MASS)
    @staticmethod
    def _record_support(audit,assessment):audit["checks"]=audit.get("checks",0)+1;audit[str(assessment.decision)]=audit.get(str(assessment.decision),0)+1
    @staticmethod
    def _raise_unsupported(predicate,assessment):raise BindingError(f"semantic_domain_unsupported predicate={predicate} side={assessment.side} actual={assessment.actual_type} schema={assessment.schema_type} decision={assessment.decision} support={assessment.supporting_mass}/{assessment.empirical_mass}")
    def _check_final_consistency(self,program,node_types,relation_bindings,attribute_bindings):
        nodes=program.node_map();audit={}
        for node_id,binding in relation_bindings.items():
            node=nodes[node_id];dep=node.dependencies[0];direction=node.inputs[1] if len(node.inputs)>1 else binding.direction;input_type=node_types.get(dep);output_type=node_types.get(node_id);expected_input=binding.subject_type if direction=="forward" else binding.object_type;expected_output=binding.object_type if direction=="forward" else binding.subject_type
            if not self.taxonomy.compatible(input_type,expected_input):raise BindingError(f"final input type {input_type} conflicts with {binding.predicate} domain {expected_input}")
            if not self.taxonomy.compatible(output_type,expected_output):raise BindingError(f"final output type {output_type} conflicts with {binding.predicate} range {expected_output}")
            actual_subject=input_type if direction=="forward" else output_type;actual_object=output_type if direction=="forward" else input_type;subject_support=self._semantic_support(predicate=binding.predicate,side="subject",actual_type=actual_subject,schema_type=binding.subject_type);object_support=self._semantic_support(predicate=binding.predicate,side="object",actual_type=actual_object,schema_type=binding.object_type);self._record_support(audit,subject_support);self._record_support(audit,object_support)
            if not subject_support.accepted:self._raise_unsupported(binding.predicate,subject_support)
            if not object_support.accepted:self._raise_unsupported(binding.predicate,object_support)
            joint=self.observed_support.edge(binding.predicate,actual_subject,actual_object);audit["observed_edge_"+joint["status"]]=audit.get("observed_edge_"+joint["status"],0)+1
            if not joint["accepted"]:raise BindingError(f"observed_edge_unsupported predicate={binding.predicate} subject={actual_subject} object={actual_object}")
        for node_id,binding in attribute_bindings.items():
            node=nodes[node_id];entity_node=node.dependencies[0] if node.op=="QUERY_ATTR" else node_id;final_type=node_types.get(entity_node)
            if not self.taxonomy.compatible(final_type,binding.entity_type):raise BindingError(f"final entity type {final_type} conflicts with {binding.predicate} domain {binding.entity_type}")
            support=self._semantic_support(predicate=binding.predicate,side="subject",actual_type=final_type,schema_type=binding.entity_type);self._record_support(audit,support)
            if not support.accepted:self._raise_unsupported(binding.predicate,support)
            actual_types=[node_types.get(dep) for dep in node.dependencies] if node.op=="SELECT_BETWEEN" else [final_type]
            for actual in actual_types:
                info=self.catalog.get("predicates",{}).get(binding.predicate,{})
                if binding.value_kind=="number" and info.get("typed_literal_support",{}).get("version")==1:
                    from ..semantic.profile import compile_attribute_profile
                    from ..semantic.typed_profile import binding_profile_matches
                    profile=compile_attribute_profile(binding.predicate,info,entity_type=actual,value_kind="number")
                    if not binding_profile_matches(binding,profile):raise BindingError("typed_attribute_profile_drift "+binding.predicate+" "+profile.evidence)
                observed=self.observed_support.edge(binding.predicate,actual);audit["observed_attribute_"+observed["status"]]=audit.get("observed_attribute_"+observed["status"],0)+1
                if not observed["accepted"]:raise BindingError(f"observed_edge_unsupported predicate={binding.predicate} subject={actual}")
        units={}
        for binding in attribute_bindings.values():
            if not binding.profile_evidence.startswith("typed-observed:"):continue
            prior=units.setdefault(binding.predicate,binding.canonical_unit)
            if prior!=binding.canonical_unit:raise BindingError("typed_attribute_unit_conflict "+binding.predicate)
        return audit
    def _bind_once(self,program,*,seed):
        solver=TypeConstraintSolver.for_program(program,self.taxonomy);guidance=current_binding_diversity_guidance() if _is_deep_dependency_program(program) else None;binder=PredicateBinder(self.catalog,self.taxonomy,seed=seed,min_count=self.min_predicate_count,diversity_guidance=guidance);relation_bindings={};attribute_bindings={};nodes=program.node_map()
        for node_id in program.topological_order():
            node=nodes[node_id]
            if node.op!="RELATE":continue
            if len(node.dependencies)!=1:raise BindingError(f"RELATE node {node_id} must have one dependency")
            direction=node.inputs[1] if len(node.inputs)>1 else "forward";binding=binder.choose_relation(solver,node.dependencies[0],node_id,direction=direction)
            if binding is None:raise BindingError(f"no typed relation candidate for RELATE node {node_id}")
            relation_bindings[node_id]=binding
        children={node.node_id:[] for node in program.nodes}
        for node in program.nodes:
            for dep in node.dependencies:children[dep].append(node)
        for node_id in program.topological_order():
            node=nodes[node_id];desired_kind=ATTRIBUTE_OP_KIND.get(node.op);entity_node=node_id if desired_kind is not None else None
            if node.op=="QUERY_ATTR":entity_node=node.dependencies[0] if node.dependencies else None;verify_kinds=[VERIFY_KIND[c.op] for c in children[node_id] if c.op in VERIFY_KIND];desired_kind=verify_kinds[0] if verify_kinds else None
            elif desired_kind is None:continue
            if entity_node is None:raise BindingError(f"attribute node {node_id} has no entity dependency")
            binding=binder.choose_attribute(solver,entity_node,value_kind=desired_kind)
            if binding is None:raise BindingError(f"no attribute candidate for node {node_id} kind={desired_kind or 'any'}")
            attribute_bindings[node_id]=binding
        for node in program.nodes:
            if node.op not in ENTITY_OUTPUT_OPS:continue
            if solver.get(node.node_id) is None:
                fallback=binder.choose_common_entity_type()
                if fallback is None or not solver.constrain(node.node_id,fallback):raise BindingError(f"cannot assign fallback type to node {node.node_id}")
        node_types=solver.materialize(node.node_id for node in program.nodes)
        from ..semantic.profile import compile_attribute_profile
        from ..semantic.typed_profile import update_binding_profile
        for node_id,binding in list(attribute_bindings.items()):
            info=self.catalog.get("predicates",{}).get(binding.predicate,{})
            if binding.value_kind!="number" or info.get("typed_literal_support",{}).get("version")!=1:continue
            node=nodes[node_id];entity_node=node.dependencies[0] if node.op=="QUERY_ATTR" else node_id;profile=compile_attribute_profile(binding.predicate,info,entity_type=node_types[entity_node],value_kind="number")
            if not profile.safe:raise BindingError("typed_attribute_profile_unsupported "+binding.predicate+" "+profile.evidence)
            attribute_bindings[node_id]=update_binding_profile(binding,profile)
        audit=self._check_final_consistency(program,node_types,relation_bindings,attribute_bindings);bound_nodes=[]
        for node in program.nodes:
            inputs=list(node.inputs)
            if node.op=="RELATE":binding=relation_bindings[node.node_id];direction=inputs[1] if len(inputs)>1 else binding.direction;inputs=[binding.predicate,direction]
            elif node.node_id in attribute_bindings:binding=attribute_bindings[node.node_id];inputs=[binding.predicate]+inputs[1:] if inputs else [binding.predicate]
            elif node.op=="FILTER_TYPE":inputs=[f"type:{node_types[node.node_id]}"]
            elif node.op=="FIND":inputs=[f"type:{node_types[node.node_id]}"]
            bound_nodes.append(ProgramNode(node.node_id,node.op,node.dependencies,tuple(inputs),node.raw_function))
        metadata=dict(program.metadata);metadata.update({"typed_binding":True,"binding_seed":seed,"diversity_guided_binding":guidance is not None,"semantic_binding_policy":{"version":2,"min_empirical_type_mass":MIN_EMPIRICAL_TYPE_MASS,"audit":audit}})
        if guidance is not None:metadata["binding_diversity_caps"]={"predicate":guidance.predicate_cap,"type_pair":guidance.type_pair_cap,"accepted_count":guidance.accepted_count}
        return BindingPlan(ProgramIR(bound_nodes,metadata=metadata),node_types,relation_bindings,attribute_bindings)
