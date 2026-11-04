"""Ground ProgramIR semantic slots with real shared-entity YAGO graph witnesses."""
from __future__ import annotations
from collections import Counter,OrderedDict,defaultdict
from copy import deepcopy
from hashlib import sha1
from itertools import combinations
from typing import Any,Mapping
from .predicate_binder import AttributeBinding,PredicateBinder,RelationBinding
from .program_binder import ATTRIBUTE_OP_KIND,ENTITY_OUTPUT_OPS,UNSUPPORTED_OPS,BindingError,BindingPlan,TypedProgramBinder
from .type_solver import TypeConstraintSolver
from ..ir import ProgramIR,ProgramNode
from ..semantic.profile import compile_attribute_profile
from ..semantic.typed_profile import update_binding_profile
GRAPH_WITNESS_SCHEMA="graph_witness_support.v1";_GRAPH_WITNESS_BACKING_CACHE_LIMIT=4;_GRAPH_WITNESS_BACKING_CACHE=OrderedDict()
def binding_snapshot(plan):return {"node_types":{str(k):v for k,v in sorted(plan.node_types.items())},"relation_bindings":{str(k):deepcopy(v.__dict__) for k,v in sorted(plan.relation_bindings.items())},"attribute_bindings":{str(k):deepcopy(v.__dict__) for k,v in sorted(plan.attribute_bindings.items())}}
class GraphWitnessIndex:
    @classmethod
    def clear_process_cache(cls):_GRAPH_WITNESS_BACKING_CACHE.clear()
    def __init__(self,catalog,taxonomy):
        payload=catalog.get("graph_witness_support",{})
        if payload.get("version")!=1:raise BindingError("graph_grounding:graph_witness_support_v1_required")
        cache_key=(id(catalog),id(taxonomy));backing=_GRAPH_WITNESS_BACKING_CACHE.get(cache_key)
        if backing is None or backing.get("catalog") is not catalog or backing.get("taxonomy") is not taxonomy:
            rows=defaultdict(list);motifs=payload.get("motifs",());indexed=0
            for row in motifs:
                ports=tuple(sorted(str(x) for x in row.get("ports",())))
                if len(ports)!=2:continue
                witnesses=tuple(str(x) for x in row.get("witnesses",()) if str(x))
                if witnesses:rows[ports].append(row);indexed+=1
            backing={"catalog":catalog,"taxonomy":taxonomy,"rows":rows,"pair_cache":{},"shared_cache":{},"stats":Counter({"index_builds":1,"motif_rows_scanned":len(motifs),"motif_rows_indexed":indexed})};_GRAPH_WITNESS_BACKING_CACHE[cache_key]=backing;_GRAPH_WITNESS_BACKING_CACHE.move_to_end(cache_key)
            while len(_GRAPH_WITNESS_BACKING_CACHE)>_GRAPH_WITNESS_BACKING_CACHE_LIMIT:_GRAPH_WITNESS_BACKING_CACHE.popitem(last=False)
        else:_GRAPH_WITNESS_BACKING_CACHE.move_to_end(cache_key);backing["stats"]["index_reuses"]+=1
        self.taxonomy=taxonomy;self._rows=backing["rows"];self._pair_cache=backing["pair_cache"];self._shared_cache=backing["shared_cache"];self._stats=backing["stats"]
    def cache_stats(self):return {**{str(k):int(v) for k,v in self._stats.items()},"pair_cache_entries":len(self._pair_cache),"shared_cache_entries":len(self._shared_cache)}
    def _type_supports(self,observed,requested):return bool(requested and self.taxonomy.is_subtype(str(observed),str(requested)))
    def pair_witnesses(self,center_type,ports):
        pair=tuple(sorted(str(x) for x in ports));key=(str(center_type),pair);self._stats["pair_queries"]+=1;cached=self._pair_cache.get(key)
        if cached is not None:self._stats["pair_cache_hits"]+=1;return deepcopy(cached)
        compatible=[row for row in self._rows.get(pair,()) if any(self._type_supports(t,center_type) for t in row.get("center_types",()))];witnesses=sorted({str(w) for row in compatible for w in row.get("witnesses",())});result={"ports":list(pair),"center_type":str(center_type),"accepted":bool(witnesses),"witnesses":witnesses,"observed_count":sum(int(row.get("count",0) or 0) for row in compatible)};self._pair_cache[key]=result;return deepcopy(result)
    def shared_entity(self,center_type,ports):
        ordered=tuple(sorted(set(str(x) for x in ports)));key=(str(center_type),ordered);self._stats["shared_queries"]+=1;cached=self._shared_cache.get(key)
        if cached is not None:self._stats["shared_cache_hits"]+=1;return deepcopy(cached)
        if len(ordered)<2:result={"status":"not_needed","accepted":True,"center_type":str(center_type),"ports":list(ordered),"witnesses":[],"pair_evidence":[]};self._shared_cache[key]=result;return deepcopy(result)
        pair_evidence=[self.pair_witnesses(center_type,pair) for pair in combinations(ordered,2)]
        if any(not row["accepted"] for row in pair_evidence):result={"status":"missing_pair_witness","accepted":False,"center_type":str(center_type),"ports":list(ordered),"witnesses":[],"pair_evidence":pair_evidence};self._shared_cache[key]=result;return deepcopy(result)
        common=set(pair_evidence[0]["witnesses"])
        for row in pair_evidence[1:]:common.intersection_update(row["witnesses"])
        witnesses=sorted(common);result={"status":"supported" if witnesses else "no_shared_center_witness","accepted":bool(witnesses),"center_type":str(center_type),"ports":list(ordered),"witnesses":witnesses,"pair_evidence":pair_evidence};self._shared_cache[key]=result;return deepcopy(result)
def _logical_roots(program,taxonomy):
    solver=TypeConstraintSolver.for_program(program,taxonomy);return solver,{node.node_id:solver.find(node.node_id) for node in program.nodes}
def _binding_ports(program,roots,relation_bindings,attribute_bindings):
    nodes=program.node_map();ports=defaultdict(set)
    for node_id,binding in relation_bindings.items():
        node=nodes[node_id];dep=node.dependencies[0];direction=node.inputs[1] if len(node.inputs)>1 else binding.direction
        if direction=="forward":ports[roots[dep]].add("out|"+binding.predicate);ports[roots[node_id]].add("in|"+binding.predicate)
        else:ports[roots[dep]].add("in|"+binding.predicate);ports[roots[node_id]].add("out|"+binding.predicate)
    for node_id,binding in attribute_bindings.items():
        node=nodes[node_id];entity_node=node.dependencies[0] if node.op=="QUERY_ATTR" else node_id;ports[roots[entity_node]].add("out|"+binding.predicate)
    return ports
def _schema_max_count(catalog,predicate):
    value=catalog.get("predicates",{}).get(predicate,{}).get("schema",{}).get("max_count");return value if isinstance(value,int) and not isinstance(value,bool) else None
def _binding_realizability_failure(program,roots,relation_bindings,attribute_bindings,catalog):
    nodes=program.node_map();attribute_slots=defaultdict(list)
    for node_id,binding in attribute_bindings.items():
        node=nodes[node_id];entity_node=node.dependencies[0] if node.op=="QUERY_ATTR" else node_id;attribute_slots[(roots[entity_node],binding.predicate)].append((node_id,node.op,binding.value_kind))
    for (root,predicate),slots in sorted(attribute_slots.items()):
        if len({kind for _,_,kind in slots})>1:return f"duplicate_attribute_kind_conflict:root={root}:predicate={predicate}"
        constraints=[item for item in slots if item[1]!="QUERY_ATTR"]
        if len(constraints)>1:return f"duplicate_attribute_constraint:root={root}:predicate={predicate}:nodes="+",".join(str(item[0]) for item in constraints)
    relation_subject_slots=defaultdict(list)
    for node_id,binding in relation_bindings.items():
        node=nodes[node_id];direction=node.inputs[1] if len(node.inputs)>1 else binding.direction;subject_node=node.dependencies[0] if direction=="forward" else node_id;relation_subject_slots[(roots[subject_node],binding.predicate)].append(node_id)
    for (root,predicate),node_ids in sorted(relation_subject_slots.items()):
        maximum=_schema_max_count(catalog,predicate)
        if maximum is not None and len(node_ids)>maximum:return f"relation_schema_max_count_conflict:root={root}:predicate={predicate}:slots={len(node_ids)}:max={maximum}"
    return None
def validate_graph_grounded_plan(plan,catalog,taxonomy):
    index=GraphWitnessIndex(catalog,taxonomy);_,roots=_logical_roots(plan.program,taxonomy);failure=_binding_realizability_failure(plan.program,roots,plan.relation_bindings,plan.attribute_bindings,catalog)
    if failure:raise BindingError("graph_grounding:world_realizability:"+failure)
    ports=_binding_ports(plan.program,roots,plan.relation_bindings,plan.attribute_bindings);members=defaultdict(list)
    for node_id,root in roots.items():members[root].append(node_id)
    proof=[]
    for root,bundle in sorted(ports.items()):
        if len(bundle)<2:continue
        center_type=next((plan.node_types.get(n) for n in members[root] if plan.node_types.get(n)),None)
        if not center_type:raise BindingError(f"graph_grounding:missing_center_type:root={root}")
        evidence=index.shared_entity(center_type,bundle);proof.append(evidence)
        if not evidence["accepted"]:raise BindingError("graph_grounding:no_shared_entity_witness:"+str(center_type)+":"+"+".join(sorted(bundle)))
    return proof
def validate_program_structure(program):
    """Reject malformed operator arity before the graph-witness search.

    ProgramIR validates DAG references/cycles, but does not validate
    operator-specific arities. In particular, RELATE must have one input.
    """
    fixed_arity={
        "FIND":0,"FIND_ALL":0,
        "RELATE":1,
        "FILTER_TYPE":1,"FILTER_STRING":1,"FILTER_NUMBER":1,
        "FILTER_YEAR":1,"FILTER_DATE":1,
        "COUNT":1,"QUERY_NAME":1,"QUERY_ATTR":1,
        "QUERY_RELATION":2,"SELECT_BETWEEN":2,"SELECT_AMONG":1,
        "VERIFY_STRING":1,"VERIFY_NUMBER":1,
        "VERIFY_YEAR":1,"VERIFY_DATE":1,
    }
    node_map=program.node_map()
    for node in program.nodes:
        if node.op.startswith("VERIFY_") and len(node.dependencies)==1 and node_map[node.dependencies[0]].op!="QUERY_ATTR":
            raise BindingError("malformed_program:VERIFY requires QUERY_ATTR scalar input")
        actual=len(node.dependencies)
        expected=fixed_arity.get(node.op)
        if expected is not None and actual!=expected:
            raise BindingError(
                f"malformed_program:op={node.op}:node={node.node_id}:"
                f"dependencies={actual}:expected={expected}"
            )
        if node.op in {"INTERSECT","UNION"} and actual<2:
            raise BindingError(
                f"malformed_program:op={node.op}:node={node.node_id}:"
                f"dependencies={actual}:expected_at_least=2"
            )

class GraphGroundedProgramBinder:
    def __init__(self,catalog,taxonomy,*,seed=17,min_predicate_count=25,candidate_limit=32,max_states=4096,predicate_usage=None,predicate_binder=None):
        self.predicate_binder=predicate_binder;self.predicate_usage=predicate_usage if predicate_usage is not None else {};self.catalog=catalog;self.taxonomy=taxonomy;self.seed=int(seed);self.min_predicate_count=int(min_predicate_count);self.candidate_limit=max(1,int(candidate_limit));self.max_states=max(1,int(max_states));self.index=GraphWitnessIndex(catalog,taxonomy);self.validator=TypedProgramBinder(catalog,taxonomy,seed=seed,max_attempts=1);self.states=0;self.rejections=defaultdict(int)
    def _stable_key(self,*parts):return sha1("|".join([str(self.seed),*(str(x) for x in parts)]).encode()).hexdigest()
    def _bundle_check(self,program,solver,roots,rels,attrs):
        failure=_binding_realizability_failure(program,roots,rels,attrs,self.catalog)
        if failure:self.rejections["world_realizability:"+failure.split(":",1)[0]]+=1;return False,0
        ports=_binding_ports(program,roots,rels,attrs);members=defaultdict(list)
        for node_id,root in roots.items():members[root].append(node_id)
        score=0
        for root,bundle in ports.items():
            if len(bundle)<2:continue
            center_type=next((solver.get(n) for n in members[root] if solver.get(n)),None)
            if not center_type:continue
            evidence=self.index.shared_entity(center_type,bundle)
            if not evidence["accepted"]:self.rejections[evidence["status"]]+=1;return False,0
            score+=len(evidence["witnesses"])
        return True,score
    def bind(self,program):
        self.states=0
        self.rejections.clear()
        validate_program_structure(program)
        unsupported=sorted({node.op for node in program.nodes if node.op in UNSUPPORTED_OPS})
        if unsupported:raise BindingError(f"qualifier/conditional operators not enabled yet: {unsupported}")
        base_solver,roots=_logical_roots(program,self.taxonomy);nodes=program.node_map();children={node.node_id:[] for node in program.nodes}
        for node in program.nodes:
            for dep in node.dependencies:children[dep].append(node)
        slots=[]
        for node_id in program.topological_order():
            node=nodes[node_id]
            if node.op=="RELATE":slots.append(("relation",node,None,None));continue
            desired_kind=ATTRIBUTE_OP_KIND.get(node.op);entity_node=node_id if desired_kind is not None else None
            if node.op=="QUERY_ATTR":
                entity_node=node.dependencies[0] if node.dependencies else None
                verify_kinds=[{"VERIFY_STRING":"string","VERIFY_NUMBER":"number","VERIFY_YEAR":"year","VERIFY_DATE":"date"}[c.op] for c in children[node_id] if c.op.startswith("VERIFY_")]
                desired_kind=verify_kinds[0] if verify_kinds else None
            if entity_node is not None:slots.append(("attribute",node,desired_kind,entity_node))
        binder=self.predicate_binder or PredicateBinder(self.catalog,self.taxonomy,seed=self.seed,min_count=self.min_predicate_count)
        self.validator.observed_support=binder.observed_support
        def rel_candidates(solver,rels,attrs,node):
            direction=node.inputs[1] if len(node.inputs)>1 else "forward";feasible=[]
            for count,option in binder._relation_options():
                in_type=option.subject_type if direction=="forward" else option.object_type;out_type=option.object_type if direction=="forward" else option.subject_type;trial=solver.clone()
                if not (trial.constrain(node.dependencies[0],in_type) and trial.constrain(node.node_id,out_type)):continue
                actual_subject=in_type if direction=="forward" else out_type;actual_object=out_type if direction=="forward" else in_type;edge=self.validator.observed_support.edge(option.predicate,actual_subject,actual_object)
                if edge.get("status")!="supported":self.rejections["edge_"+str(edge.get("status","unknown"))]+=1;continue
                binding=RelationBinding(option.predicate,option.subject_type,option.object_type,direction);new_rels=dict(rels);new_rels[node.node_id]=binding;ok,score=self._bundle_check(program,trial,roots,new_rels,attrs)
                if ok:feasible.append((score,int(count),self._stable_key(node.node_id,option.predicate,in_type,out_type),trial,binding))
            feasible.sort(key=lambda x:(self.predicate_usage.get(x[4].predicate,0),-x[0],-x[1],x[2]));return feasible[:self.candidate_limit]
        def attr_candidates(solver,rels,attrs,node,value_kind,entity_node):
            feasible=[]
            for count,predicate,info,domain,selected_kind,datatype in binder._attribute_specs(value_kind):
                trial=solver.clone()
                if not trial.constrain(entity_node,domain):continue
                profile_type=trial.get(entity_node) or domain;profile=binder._attribute_profile(str(predicate),info,entity_type=profile_type,value_kind=selected_kind)
                if not profile.safe:continue
                edge=self.validator.observed_support.edge(str(predicate),profile_type)
                if edge.get("status")!="supported":self.rejections["attribute_"+str(edge.get("status","unknown"))]+=1;continue
                binding=AttributeBinding(str(predicate),profile_type,selected_kind,datatype,profile.family,profile.canonical_unit,profile.minimum,profile.maximum,profile.integer_values,profile.confidence,profile.evidence);new_attrs=dict(attrs);new_attrs[node.node_id]=binding;ok,score=self._bundle_check(program,trial,roots,rels,new_attrs)
                if ok:feasible.append((score,int(count),self._stable_key(node.node_id,predicate,profile_type,selected_kind),trial,binding))
            feasible.sort(key=lambda x:(self.predicate_usage.get(x[4].predicate,0),-x[0],-x[1],x[2]));return feasible[:self.candidate_limit]
        def dfs(index,solver,rels,attrs):
            if self.states>=self.max_states:return None
            if index==len(slots):
                try:self.validator._check_final_consistency(program,solver.materialize(nodes),rels,attrs)
                except BindingError:return None
                return solver,rels,attrs
            self.states+=1;kind,node,value_kind,entity_node=slots[index];candidates=rel_candidates(solver,rels,attrs,node) if kind=="relation" else attr_candidates(solver,rels,attrs,node,value_kind,entity_node)
            for _,_,_,trial,binding in candidates:
                new_rels,new_attrs=dict(rels),dict(attrs);(new_rels if kind=="relation" else new_attrs)[node.node_id]=binding;result=dfs(index+1,trial,new_rels,new_attrs)
                if result is not None:return result
            return None
        result=dfs(0,base_solver,{}, {})
        if result is None:detail=",".join(f"{k}={v}" for k,v in sorted(self.rejections.items())) or "no_feasible_assignment";raise BindingError(f"graph_grounding:no_plan_within_{self.max_states}_states:{detail}")
        solver,relation_bindings,attribute_bindings=result
        for node in program.nodes:
            if node.op in ENTITY_OUTPUT_OPS and solver.get(node.node_id) is None:
                fallback=binder.choose_common_entity_type()
                if fallback is None or not solver.constrain(node.node_id,fallback):raise BindingError(f"cannot assign fallback type to node {node.node_id}")
        node_types=solver.materialize(node.node_id for node in program.nodes);preliminary=BindingPlan(program,node_types,relation_bindings,attribute_bindings);graph_proof=validate_graph_grounded_plan(preliminary,self.catalog,self.taxonomy);semantic_audit=self.validator._check_final_consistency(program,node_types,relation_bindings,attribute_bindings);bound_nodes=[]
        for node in program.nodes:
            inputs=list(node.inputs)
            if node.op=="RELATE":binding=relation_bindings[node.node_id];inputs=[binding.predicate,inputs[1] if len(inputs)>1 else binding.direction]
            elif node.node_id in attribute_bindings:binding=attribute_bindings[node.node_id];inputs=[binding.predicate,*inputs[1:]] if inputs else [binding.predicate]
            elif node.op in {"FILTER_TYPE","FIND"}:inputs=[f"type:{node_types[node.node_id]}"]
            bound_nodes.append(ProgramNode(node.node_id,node.op,node.dependencies,tuple(inputs),node.raw_function))
        metadata=dict(program.metadata);metadata.update({"typed_binding":True,"binding_source":"yago_graph_witness","binding_seed":self.seed,"semantic_binding_policy":{"version":4,"graph_witness_schema":GRAPH_WITNESS_SCHEMA,"world_realizability":"slot_multiplicity_and_schema_cardinality","search_states":self.states,"semantic_audit":semantic_audit,"graph_witness_entities":sum(len(r.get("witnesses",())) for r in graph_proof)}});return BindingPlan(ProgramIR(bound_nodes,metadata=metadata),node_types,relation_bindings,attribute_bindings)
