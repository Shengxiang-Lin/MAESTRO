"""Preserve composition cuts and expose independent branches before their joins."""
from __future__ import annotations
from collections import Counter
import re
from .tools import tool_profile

def compile_fragments(realized, record, rendered, *, max_sources=8, scene='omnigaia'):
    from .compiler import (GoldCompilationError, _ancestors, _subprogram,
        _minimal_witness, _maximal_dependencies, _public_inputs, _public_answer,
        _source_claims, _same, json_value, ready_fragments,
        _evidence_world, _public_observable_entities)
    from .acquisition import resolve_dependencies
    from environment.task_construction.rendering.pipeline import program_from_dict, world_from_dict
    from environment.task_construction.executor import SymbolicExecutor
    from environment.task_construction.scenario.information import assess_goal_information_completeness
    tid=str(realized.get('task_id',''))
    if not tid or tid!=rendered.task_id:
        raise GoldCompilationError('question/world task_id mismatch')
    p=program_from_dict(record['program']);w=world_from_dict(record['world'])
    nodes=p.node_map();order=p.topological_order();trace=SymbolicExecutor().execute_trace(p,w)
    # A real Sub-Agent can enumerate only entities supported by published
    # Source facts. The private world may have additional unsourced distractors.
    # Project the *entire* computation first, never individual answers.
    public_world=_evidence_world(w,set(w.sources))
    public_full_trace=SymbolicExecutor().execute_trace(p,public_world)
    hidden=getattr(rendered,'hidden_metadata',{}) or {}
    def public_args(node):
        args=_public_inputs(node.inputs,rendered)
        predicate=None;slot=None
        if node.op.startswith('FILTER_') and node.op!='FILTER_TYPE':
            predicate=node.inputs[0];slot=2
        elif node.op.startswith('VERIFY_') and node.dependencies:
            parent=nodes[node.dependencies[0]]
            if parent.op=='QUERY_ATTR':predicate=parent.inputs[0];slot=1
        if slot is not None:
            raw=node.inputs[slot]
            surface=hidden.get('literal_surfaces_by_predicate',{}).get(predicate,{}).get(str(raw),hidden.get('literal_surfaces',{}).get(str(raw),args[slot]))
            args[slot]=str(surface)
        return args
    links=(p.metadata.get('deep_dependency',{}) or {}).get('composition_links')
    if not isinstance(links,list) or not links:
        raise GoldCompilationError('missing first-stage composition_links; no inferred fallback')
    outputs=[int(link['output_node_id']) for link in links]
    if len(set(outputs))!=len(outputs) or any(n not in nodes for n in outputs):
        raise GoldCompilationError('invalid/duplicate composition output nodes')
    resources=(realized.get('capability_world') or {}).get('resources',[])
    by_source={str(r['resource_id']):r for r in resources}
    if set(by_source)!=set(w.sources):
        raise GoldCompilationError('capability resources do not match frozen world')
    audit=(realized.get('capability_world') or {}).get('binding_audit',{})
    discovery=(audit.get('progressive_discovery',{}) or audit.get('p1_orchestration_profile',{}).get('progressive_discovery',{}))
    dependencies,_=resolve_dependencies(resources,discovery.get('anchor_registry',{}))
    profile=tool_profile(scene)
    question=str(realized.get('question',''))
    cut_names={out:f'fragment:{i}' for i,out in enumerate(outputs)}
    # Branch operands are independent work, even if Stage 1 left them outside a composition cut.
    consumers=Counter(dep for node in p.nodes for dep in node.dependencies)
    for node in p.nodes:
        if node.op in {'INTERSECT','UNION','SELECT_BETWEEN','QUERY_RELATION'}:
            for dep in node.dependencies:
                cut_names.setdefault(dep,f'fragment:branch:{dep}')
        if consumers[node.node_id]>1 and node.dependencies:
            cut_names.setdefault(node.node_id,f'fragment:shared:{node.node_id}')
    # Expose a separate join boundary when Stage 1 explicitly concealed a
    # singleton FIND in favor of a uniquely identifying intersection branch.
    # The public graph adapter can then remove that redundant private root
    # without changing any other ProgramIR calculation.
    concealed_ids={
        int(nid)
        for entry in hidden.get('public_query_plan',{}).get('latent_entities',[])
        if entry.get('rule')=='unique_intersection_clue'
        for nid in entry.get('find_node_ids',[])
    }
    # Independently certify a hidden singleton INTERSECT even when Stage 1
    # did not label its public-query metadata. This is only a *candidate*
    # for Stage-3 graph projection; the adapter still checks causal evidence,
    # consumers, intermediate outputs, and the public replay.
    candidate_intersections=[]
    for join in p.nodes:
        if join.op!='INTERSECT' or len(join.dependencies)!=2:continue
        for find_id,other_id in (
            (join.dependencies[0],join.dependencies[1]),
            (join.dependencies[1],join.dependencies[0]),
        ):
            find=nodes[find_id]
            if find.op!='FIND' or len(find.inputs)!=1 or not find.inputs[0].startswith('entity:'):
                continue
            alias=_public_inputs(find.inputs,rendered)[0]
            if re.search(r'(?<!\w)'+re.escape(alias)+r'(?!\w)',question,re.I):
                continue
            if find_id in _ancestors(p,other_id):
                continue
            entity_id=find.inputs[0].split(':',1)[1]
            expected={entity_id}
            if not all(
                isinstance(public_full_trace[nid],set) and public_full_trace[nid]==expected
                for nid in (find_id,other_id,join.node_id)
            ):
                continue
            candidate_intersections.append({
                'rule':'witness_verified_intersection',
                'find_node_ids':[find_id],
                'identifying_node_ids':[other_id],
                'join_node_id':join.node_id,
            })
            concealed_ids.add(find_id)
            break
    for node in p.nodes:
        if node.op=='INTERSECT' and any(dep in concealed_ids for dep in node.dependencies):
            cut_names.setdefault(node.node_id,f'fragment:public_join:{node.node_id}')
    final=order[-1];cut_names.setdefault(final,'fragment:terminal')
    if not _same(_public_answer(trace[final],rendered,nodes[final]),rendered.answer):
        raise GoldCompilationError('rendered final answer disagrees with verified program')
    if not _same(_public_answer(public_full_trace[final],rendered,nodes[final]),rendered.answer):
        raise GoldCompilationError('public_full_world_answer_disagrees_with_gold', {
            'output_node_id':final,
            'private_answer_matches_rendered':True,
            'public_source_domain_answer_matches':False,
        })
    # A shared root is safe to repeat; a non-root boundary is bound to a completed result.
    previous={};fragments=[];support={}
    latent={nid:entry for entry in (getattr(rendered,'hidden_metadata',{}) or {}).get('public_query_plan',{}).get('latent_entities',[]) for nid in entry.get('find_node_ids',[])}
    for out in order:
        if out not in cut_names:continue
        fid=cut_names[out];prev=_maximal_dependencies(p,out,previous)
        owned=_ancestors(p,out)
        for po,pf in previous.items():
            if pf in prev:owned-=_ancestors(p,po)
        local_order=[nid for nid in order if nid in owned];local={nid:i+1 for i,nid in enumerate(local_order)}
        operations=[]
        for nid in local_order:
            node=nodes[nid];args=public_args(node)
            if node.op=='FIND' and node.inputs[0].startswith('entity:'):
                alias=args[0]
                if not re.search(r'(?<!\w)'+re.escape(alias)+r'(?!\w)',question,re.I):
                    clues=[]
                    for cn in latent.get(nid,{}).get('identifying_node_ids',[]):
                        clue=nodes[cn];ci=public_args(clue)
                        if clue.op.startswith('FILTER_') and len(ci)==3:
                            clues.append(' '.join(ci))
                    args=['the entity matching the question clues'+(': '+'; '.join(clues) if clues else ' in the question')]
            bindings=[]
            for dep in node.dependencies:
                if dep in local:
                    bindings.append({'kind':'step','step':local[dep]})
                elif dep in previous:
                    bindings.append({'kind':'completed','fragment_id':previous[dep]})
                else:
                    raise GoldCompilationError(f'unbound operation input {dep} at {nid}')
            operations.append({'node_id':nid,'op':node.op,'dependencies':list(node.dependencies),'inputs':args,'input_bindings':bindings,'find_kind':'type' if node.op=='FIND' and node.inputs[0].startswith('type:') else 'entity'})
        subtree=_subprogram(p,out)
        # Full-world FIND_ALL and type filters include distractors that have
        # no frozen public evidence. These cannot be read by a real Sub-Agent.
        # Require entity coverage only for the public source domain, then check
        # the *whole fragment answer* against the original full-world oracle.
        observable_entities=_public_observable_entities(w)
        entity_reads=set()
        for node in subtree.nodes:
            if node.op=='FIND_ALL' or (node.op=='FIND' and node.inputs[0].startswith('type:')):
                entity_reads.update(set(public_full_trace[node.node_id]) & observable_entities)
            if node.op=='FILTER_TYPE':
                entity_reads.update(set(public_full_trace[node.dependencies[0]]) & observable_entities)
        witness=_minimal_witness(subtree,public_world,public_full_trace[out],dependencies,max_sources=max_sources,required_entity_ids=entity_reads)
        # A step's public intermediate value must come from public evidence,
        # not from invisible entities in the private symbolic universe.
        public_trace=SymbolicExecutor().execute_trace(
            subtree, _evidence_world(w,set(witness))
        )
        inherited=set().union(*(support[x] for x in prev)) if prev else set()
        new_sources=sorted(set(witness)-inherited)
        observations=_source_claims(w,rendered,new_sources)
        for obs in observations:obs['visibility']=by_source[obs['resource_id']].get('visibility','provided')
        # The simulated inspection narrative follows a reachable acquisition order.
        pending=set(new_sources);done=set(witness)&inherited;acquisition=[]
        while pending:
            ready=sorted(sid for sid in pending if all(g&done for g in dependencies[sid]))
            if not ready:raise GoldCompilationError('fragment evidence cannot be acquired causally')
            acquisition.append(ready);done.update(ready);pending.difference_update(ready)
        position={sid:i for i,wave in enumerate(acquisition) for sid in wave}
        observations.sort(key=lambda obs:(position[obs['resource_id']],obs['resource_id']))
        tools=sorted({tool for obs in observations for tool in profile.for_resource(obs)}) or list(profile.calculation)
        fragment={'fragment_id':fid,'stage':len(fragments),'output_node_id':out,
            'depends_on':prev,'operations':operations,
            'operation_results':[{'node_id':nid,'answer':_public_answer(public_trace[nid],rendered,nodes[nid]),
                'input_values':[_public_answer(public_trace[d],rendered,nodes[d]) for d in nodes[nid].dependencies]}
                for nid in local_order],
            'resource_ids':new_sources,'support_resource_ids':witness,
            'observations':observations,'acquisition_waves':acquisition,
            'known_observations':_source_claims(w,rendered,sorted(set(witness)&inherited)),
            'expected_answer':json_value(rendered.answer) if out==final else _public_answer(public_full_trace[out],rendered,nodes[out]),
            'answer_kind':'entity_set' if isinstance(trace[out],set) else 'value',
            'suggested_tools':tools,'available_tools':list(profile.available),
            'scene':scene,'bounded_realization':True}
        fragments.append(fragment);previous[out]=fid;support[fid]=set(witness)
    info=assess_goal_information_completeness(p,w)
    if not info.accepted:raise GoldCompilationError('frozen world lacks decision-complete information')
    waves=[];completed=set()
    while len(completed)<len(fragments):
        ready=ready_fragments({'fragments':fragments},completed)
        if not ready:raise GoldCompilationError('cyclic or unresolved fragment dependencies')
        wave=[f['fragment_id'] for f in ready];waves.append(wave);completed.update(wave)
    provided=[]
    for src in rendered.sources:
        if by_source[src.source_id].get('visibility') in {'attached','provided','initial','visible','provided_initial'}:
            provided.append({'resource_id':src.source_id,'source_type':src.source_type,'title':src.title})
    return {'task_id':tid,'question':question,'fragments':fragments,'waves':waves,
        'final_answer':json_value(rendered.answer),'final_fragment_id':cut_names[final],
        'answer_contract':{'operation':nodes[final].op,'collection_mode':
            'set' if nodes[final].op in {'QUERY_NAME','QUERY_RELATION','SELECT_BETWEEN','SELECT_AMONG'} else
            'multiset' if isinstance(trace[final],(list,tuple,set)) else 'scalar'},
        'terminal':{'op':nodes[final].op,'node_id':final,'depends_on':fragments[-1]['depends_on']},
        'symbolic_trace':{str(k):json_value(v) for k,v in trace.items()},
        'projection_candidates':candidate_intersections,
        'public_intermediate_projections':[
            {'node_id':nid,'private_count':len(trace[nid]),
             'public_count':len(public_full_trace[nid])}
            for nid in order if isinstance(trace[nid],set)
            and isinstance(public_full_trace[nid],set)
            and not _same(trace[nid],public_full_trace[nid])
        ],
        'final_proof_source_ids':fragments[-1]['support_resource_ids'],
        'policy':'first_stage_composition_links_fragment_gold','scene':scene,
        'available_tools':list(profile.available),'provided_resources':provided,
        'canonical_question':getattr(rendered,'question',question)}
