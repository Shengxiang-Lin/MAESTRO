"""Construct facts backwards from a bound goal and return its concrete program."""
from __future__ import annotations
import re
from dataclasses import replace
from typing import Any
from ..binding.program_binder import BindingPlan
from ..executor import SymbolicExecutor, _compare
from ..ir import ProgramIR, ProgramNode
from ..semantic.literal_factory import SemanticLiteralFactory
from ..world import Fact, SyntheticWorld


class WorldSynthesisError(RuntimeError):
    pass


def unresolved_slot(value):
    return bool(re.search(r'<(?:SYMBOL|YEAR|NUMBER|DATE|ENTITY_OR_TYPE|ATTR|REL|BOOL|STRING|VALUE)[^>]*>',str(value)))


class WorldSampler:
    def __init__(self, *, seed=17, catalog=None, taxonomy=None):
        self.seed = int(seed)
        self.literal_factory = SemanticLiteralFactory(seed=seed)
        self.catalog, self.taxonomy = catalog or {}, taxonomy
        self.concrete_program = None

    def synthesize(self, plan: BindingPlan) -> SyntheticWorld:
        self.concrete_program = None
        nodes = plan.program.node_map()
        inputs = {i:list(n.inputs) for i,n in nodes.items()}
        world = SyntheticWorld(metadata={'synthesis_seed':self.seed})
        cache, counter = {}, [0]
        schema = {p:i.get('schema',{}).get('max_count')
                  for p,i in self.catalog.get('predicates',{}).items()}
        world.metadata['schema_max_counts'] = {p:v for p,v in schema.items() if isinstance(v,int)}

        def entity(node_id):
            counter[0] += 1
            eid = f'e{counter[0]:04d}'
            world.entities[eid] = plan.node_types.get(node_id,'schema:Thing')
            return eid

        def fact(subject,predicate,value,kind):
            for f in world.facts:
                if (f.subject,f.predicate,f.object,f.object_kind)==(subject,predicate,value,kind):
                    return
            maximum = schema.get(predicate)
            existing = [f for f in world.facts if f.subject==subject and f.predicate==predicate]
            if isinstance(maximum,int) and len(existing)>=maximum:
                raise WorldSynthesisError(f'schema maxCount exceeded: {predicate}')
            world.add_fact(Fact(f'f{len(world.facts)+1:05d}',subject,predicate,value,kind,None))

        def attribute(subject,binding,ordinal=4):
            observed = [f for f in world.facts if f.subject==subject and f.predicate==binding.predicate]
            if observed:
                if any(f.object_kind!=binding.value_kind for f in observed):
                    raise WorldSynthesisError('incompatible existing literal kind')
                return observed[0].object
            value = self.literal_factory.value(binding,ordinal=ordinal)
            fact(subject,binding.predicate,value,binding.value_kind)
            return value

        def ensure(node_id, desired=None):
            node = nodes[node_id]
            op, deps = node.op, node.dependencies
            entity_op = op in {'FIND','FIND_ALL','RELATE','FILTER_TYPE','FILTER_STRING',
                'FILTER_NUMBER','FILTER_YEAR','FILTER_DATE','INTERSECT','UNION'}
            if entity_op:
                wanted = set(desired or cache.get(node_id,()) or {entity(node_id)})
                wanted.update(cache.get(node_id,()))
                if op in {'FIND','FIND_ALL'}:
                    if op=='FIND':
                        inputs[node_id] = ['entity:'+next(iter(wanted))] if len(wanted)==1 else ['type:'+world.entities[sorted(wanted)[0]]]
                elif op=='RELATE':
                    relation = plan.relation_bindings[node_id]
                    source = cache.get(deps[0]) or {entity(deps[0])}
                    ensure(deps[0],source)
                    direction = inputs[node_id][1] if len(inputs[node_id])>1 else relation.direction
                    for start in sorted(source):
                        for end in sorted(wanted):
                            fact(end,relation.predicate,start,'entity') if direction=='backward' else fact(start,relation.predicate,end,'entity')
                elif op=='FILTER_TYPE':
                    expected = inputs[node_id][0].split(':',1)[-1]
                    if any(world.entities[e]!=expected for e in wanted):
                        raise WorldSynthesisError('FILTER_TYPE output incompatible with requested entities')
                    ensure(deps[0],wanted)
                elif op.startswith('FILTER_'):
                    ensure(deps[0],wanted)
                    binding = plan.attribute_bindings[node_id]
                    comparator = inputs[node_id][1]
                    threshold, observed = self.literal_factory.comparison_pair(binding,comparator)
                    target = inputs[node_id][2]
                    if unresolved_slot(target):
                        target = threshold
                        inputs[node_id][2] = str(target)
                    for eid in sorted(wanted):
                        prior = [f.object for f in world.facts if f.subject==eid and f.predicate==binding.predicate]
                        if prior:
                            if not any(_compare(v,comparator,target) for v in prior):
                                raise WorldSynthesisError('conflicting attribute constraints')
                        else:
                            value = observed if _compare(observed,comparator,target) else target
                            if not _compare(value,comparator,target):
                                raise WorldSynthesisError('unsatisfiable attribute comparator')
                            fact(eid,binding.predicate,value,binding.value_kind)
                else:
                    # Both sides receive actual supporting facts for the shared goal.
                    for dep in deps:
                        ensure(dep,wanted)
                cache[node_id] = wanted
                return wanted
            if node_id in cache:
                return cache[node_id]
            if op in {'COUNT','QUERY_NAME','QUERY_ATTR'}:
                wanted = {entity(deps[0]) for _ in range(1+self.seed%3)} if op in {'COUNT','QUERY_NAME'} else None
                entities = ensure(deps[0],wanted)
                if op=='COUNT':
                    value = len(entities)
                elif op=='QUERY_NAME':
                    value = sorted(entities)
                else:
                    binding = plan.attribute_bindings[node_id]
                    value = [attribute(e,binding,3+i) for i,e in enumerate(sorted(entities))]
            elif op.startswith('VERIFY_'):
                if nodes[deps[0]].op!='QUERY_ATTR':
                    raise WorldSynthesisError('VERIFY requires a scalar QUERY_ATTR dependency')
                observed = ensure(deps[0])
                if not observed:
                    raise WorldSynthesisError('VERIFY has no observed values')
                target = inputs[node_id][1]
                if unresolved_slot(target):
                    binding = plan.attribute_bindings[deps[0]]
                    candidates = [self.literal_factory.value(binding,ordinal=i) for i in (0,4,8)]
                    want_true = bool(self.seed%2)
                    target = next((v for v in candidates if any(_compare(x,inputs[node_id][0],v) for x in observed)==want_true),observed[0])
                    inputs[node_id][1] = str(target)
                value = any(_compare(x,inputs[node_id][0],target) for x in observed)
            elif op in {'SELECT_AMONG','SELECT_BETWEEN'}:
                entities = set()
                for dep in deps:
                    entities.update(ensure(dep,{entity(dep),entity(dep)}))
                binding = plan.attribute_bindings[node_id]
                for i,eid in enumerate(sorted(entities)):
                    attribute(eid,binding,1+(i%3)*3)
                value = entities
            elif op=='QUERY_RELATION':
                left,right = ensure(deps[0]),ensure(deps[1])
                from ..binding.predicate_binder import PredicateBinder
                from ..quality.observed_support import ObservedSupport
                if self.taxonomy is None:
                    raise WorldSynthesisError('QUERY_RELATION needs a grounded predicate catalog')
                support = ObservedSupport(self.catalog,self.taxonomy)
                options = PredicateBinder(self.catalog,self.taxonomy,seed=self.seed)._relation_options()
                for a in sorted(left):
                    for b in sorted(right):
                        choices = [option.predicate for _,option in options
                            if self.taxonomy.compatible(world.entities[a],option.subject_type)
                            and self.taxonomy.compatible(world.entities[b],option.object_type)
                            and support.edge(option.predicate,world.entities[a],world.entities[b]).get('accepted')]
                        if not choices:
                            raise WorldSynthesisError('no observed relation for terminal endpoint types')
                        fact(a,choices[self.seed%len(choices)],b,'entity')
                value = []
            else:
                raise WorldSynthesisError('unsupported synthesis operator: '+op)
            cache[node_id] = value
            return value

        try:
            ensure(plan.program.topological_order()[-1])
            concrete = ProgramIR([replace(n,inputs=tuple(inputs[n.node_id])) for n in plan.program.nodes],
                metadata={**plan.program.metadata,'world_skeleton':True})
            if any(unresolved_slot(v) for n in concrete.nodes for v in n.inputs):
                raise WorldSynthesisError('unresolved literal or predicate slot')
            from ..quality.semantic_policy import normalize_program_world
            concrete,world = normalize_program_world(concrete,world)
            answer = SymbolicExecutor().execute(concrete,world)
        except (ValueError,TypeError,KeyError,RuntimeError) as exc:
            raise WorldSynthesisError(str(exc)) from exc
        self.concrete_program = concrete
        world.metadata['oracle_answer'] = answer
        world.metadata['attribute_semantics'] = {b.predicate:{'family':b.semantic_family,
            'canonical_unit':b.canonical_unit,'value_kind':b.value_kind,
            'minimum':b.value_min,'maximum':b.value_max,'integer_values':b.integer_values,
            'confidence':b.profile_confidence,'evidence':b.profile_evidence}
            for b in plan.attribute_bindings.values()}
        return world


__all__ = ['WorldSampler','WorldSynthesisError','unresolved_slot']
