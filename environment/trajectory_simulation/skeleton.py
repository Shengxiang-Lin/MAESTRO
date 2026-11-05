"""Preserve composition cuts and expose independent branches before their joins."""
from __future__ import annotations
from collections import Counter
import re
from .tools import tool_profile

def compile_fragments(realized, record, rendered, *, max_sources=8, scene='omnigaia'):
    from .compiler import (GoldCompilationError, _ancestors, _subprogram,
        _minimal_witness, _maximal_dependencies, _public_inputs, _public_answer,
        _source_claims, _same, json_value, ready_fragments)
    from .acquisition import resolve_dependencies
    from environment.task_construction.rendering.pipeline import program_from_dict, world_from_dict
    from environment.task_construction.executor import SymbolicExecutor
    from environment.task_construction.scenario.information import assess_goal_information_completeness
    tid=str(realized.get('task_id',''))
    if not tid or tid!=rendered.task_id:
        raise GoldCompilationError('question/world task_id mismatch')
    p=program_from_dict(record['program']);w=world_from_dict(record['world'])
    nodes=p.node_map();order=p.topological_order();trace=SymbolicExecutor().execute_trace(p,w)
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
    cut_names={out:f'fragment:{i}' for i,out in enumerate(outputs)}
    # Branch operands are independent work, even if Stage 1 left them outside a composition cut.
    consumers=Counter(dep for node in p.nodes for dep in node.dependencies)
    for node in p.nodes:
        if node.op in {'INTERSECT','UNION','SELECT_BETWEEN','QUERY_RELATION'}:
            for dep in node.dependencies:
                cut_names.setdefault(dep,f'fragment:branch:{dep}')
        if consumers[node.node_id]>1 and node.dependencies:
            cut_names.setdefault(node.node_id,f'fragment:shared:{node.node_id}')
    final=order[-1];cut_names.setdefault(final,'fragment:terminal')
    if not _same(_public_answer(trace[final],rendered,nodes[final]),rendered.answer):
        raise GoldCompilationError('rendered final answer disagrees with verified program')
    # A shared root is safe to repeat; a non-root boundary is bound to a completed result.
    previous={};fragments=[];support={}
    latent={nid:entry for entry in (getattr(rendered,'hidden_metadata',{}) or {}).get('public_query_plan',{}).get('latent_entities',[]) for nid in entry.get('find_node_ids',[])}
    question=str(realized.get('question',''))
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
        entity_reads=set()
        for node in subtree.nodes:
            if node.op=='FIND_ALL' or (node.op=='FIND' and node.inputs[0].startswith('type:')):
                entity_reads.update(trace[node.node_id])
            if node.op=='FILTER_TYPE':entity_reads.update(trace[node.dependencies[0]])
        witness=_minimal_witness(subtree,w,trace[out],dependencies,max_sources=max_sources,required_entity_ids=entity_reads)
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
            'operation_results':[{'node_id':nid,'answer':_public_answer(trace[nid],rendered,nodes[nid]),
                'input_values':[_public_answer(trace[d],rendered,nodes[d]) for d in nodes[nid].dependencies]}
                for nid in local_order],
            'resource_ids':new_sources,'support_resource_ids':witness,
            'observations':observations,'acquisition_waves':acquisition,
            'known_observations':_source_claims(w,rendered,sorted(set(witness)&inherited)),
            'expected_answer':json_value(rendered.answer) if out==final else _public_answer(trace[out],rendered,nodes[out]),
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
        'final_proof_source_ids':fragments[-1]['support_resource_ids'],
        'policy':'first_stage_composition_links_fragment_gold','scene':scene,
        'available_tools':list(profile.available),'provided_resources':provided,
        'canonical_question':getattr(rendered,'question',question)}
