"""Bounded language realization: LLMs select phrasing, never rewrite execution semantics.

The public wire format remains native O1. All entity sets, operands, source
provenance and conclusions below come from the verified execution ledger.
"""
from __future__ import annotations
import json

STYLES=('concise','stepwise','goal','evidence')

def value_text(value):
    if value == []:return 'an empty result'
    if isinstance(value,list):
        items=[value_text(item) for item in value]
        return items[0] if len(items)==1 else ', '.join(items[:-1])+' and '+items[-1]
    return json.dumps(value,ensure_ascii=False) if not isinstance(value,str) else value

def input_label(binding, completed):
    if binding['kind']=='step':return f"the result of instruction {binding['step']}"
    previous=completed.get(binding['fragment_id'])
    if previous is None:raise ValueError('missing completed input for language realization')
    return f"the completed {previous.get('public_ref','subtask')} result ({previous['answer']})"

def task_sentence(op, sources):
    name,args=op['op'],op['inputs']
    inp=sources[0] if sources else 'the supplied records'
    if name=='FIND':return ('Identify all entities of type ' if op.get('find_kind')=='type' else 'Identify ')+args[0]+'.'
    if name=='FIND_ALL':return 'Start a separate candidate set from all supplied records.'
    if name=='FILTER_TYPE':return f'From {inp}, keep only entities of type {args[0]}.'
    if name.startswith('FILTER_'):return f'From {inp}, keep only entities whose {args[0]} {args[1]} {args[2]}.'
    if name=='RELATE':
        direction='objects back to their subjects' if 'backward' in args else 'subjects to their objects'
        return f'Using {inp}, follow the {args[0]} relation from {direction}.'
    if name in {'INTERSECT','UNION'}:
        verb='Intersect' if name=='INTERSECT' else 'Take the union of'
        return verb+' '+ ' and '.join(sources)+'.'
    if name=='COUNT':return f'Count the distinct entities in {inp}.'
    if name=='QUERY_NAME':return f'Return the names of entities in {inp}.'
    if name=='QUERY_ATTR':return f'Retrieve the {args[0]} values for entities in {inp}.'
    if name.startswith('VERIFY_'):return f'Check whether any value in {inp} satisfies {args[0]} {args[1]}.'
    if name=='QUERY_RELATION':return 'Return the relation names linking '+ ' to '.join(sources)+'.'
    if name in {'SELECT_AMONG','SELECT_BETWEEN'}:
        return f'Select entities from {" and ".join(sources)} by {args[0]} using {args[1]}.'
    raise ValueError('unsupported realization operation '+name)

def task_variants(fragment, completed, visible_resources=(), public_cache=None):
    sentences=[]
    for op in fragment['operations']:
        refs=[input_label(b,completed) for b in op.get('input_bindings',[])]
        sentences.append(task_sentence(op,refs))
    # Local instruction numbers have identical meaning in both presentations.
    instruction='\n'.join(f'{i}. {line}' for i,line in enumerate(sentences,1))
    tool_list=fragment['suggested_tools']
    context='Execute the instructions in dependency order; each input refers only to the result explicitly named in that instruction.'
    if any(b['kind']=='completed' for op in fragment['operations'] for b in op.get('input_bindings',[])):
        refs={b['fragment_id'] for op in fragment['operations'] for b in op.get('input_bindings',[]) if b['kind']=='completed'}
        context+=' Completed inputs: '+'; '.join(f"{completed[r].get('public_ref','subtask')}: {completed[r]['answer']}" for r in sorted(refs))+'.'
    if visible_resources:
        context+=' Available reference handles: '+'; '.join(f"{r['title']} ({r['resource_id']}, {r['source_type']})" for r in visible_resources)+'.'
    if 'GoogleSearchAction' in tool_list:
        context+=' Locate any further required records using the named entities and conditions, then inspect them with the assigned modality tools.'
    else:context+=' Use the supplied or previously inspected records with the assigned tools.'
    if public_cache is not None:
        from .evidence import task_packet,MARKER
        context+=MARKER+json.dumps(task_packet(fragment,completed,public_cache),ensure_ascii=False,separators=(',',':'))
    last=fragment['operations'][-1];op=last['op'];args=last['inputs']
    goal=('Determine the number of matching entities' if op=='COUNT' else
          'Check whether the retrieved values satisfy '+ ' '.join(args) if op.startswith('VERIFY_') else
          'Find the '+args[0]+' values for the specified entities' if op=='QUERY_ATTR' else
          'Identify the entities reached through '+args[0]+' in the required direction' if op=='RELATE' else
          'Identify the entities satisfying the specified constraints')
    headers=['','Perform these checks and return the final result:\n',
             goal+'. Return the result with the supporting references. Preserve the following input scopes and checks:\n',
             goal+'. Inspect the supplied evidence and report the relevant facts and source references. Required checks:\n']
    return [{'task_instruction':header+instruction,'context':context,'model':'model_1','tools':tool_list} for header in headers]

def action_options(payload):
    completed={r['fragment_id']:r for r in payload.get('completed_subtask_contracts',[])}
    return [task_variants(f,completed,payload.get('visible_resources',[]),payload.get('evidence_cache')) for f in payload['gold_guidance']['ready']]

def realize_action(payload, choice):
    guidance=payload['gold_guidance']
    style=choice.get('reasoning_style')
    if style not in STYLES:raise ValueError('invalid reasoning_style')
    if guidance['may_complete']:
        if set(choice)!={'reasoning_style','answer'} or not isinstance(choice['answer'],str):raise ValueError('invalid completion language plan')
        return {'action':'complete','reasoning':(('All assigned checks are complete. The final subtask returned '+choice['answer']+'.') if style=='concise' else ('The completed subtasks supply the required result: '+choice['answer']+'. Finish with that result.')),
                'params':{'answer':choice['answer']}}
    if set(choice)!={'reasoning_style','task_styles'}:raise ValueError('unexpected delegation language-plan fields')
    options=action_options(payload);styles=choice.get('task_styles')
    if not isinstance(styles,list) or len(styles)!=len(options) or any(s not in STYLES for s in styles):raise ValueError('one task style is required per ready subtask')
    reason=('The required inputs are available. Delegate the following '+('independent subtasks in parallel.' if len(options)>1 else 'subtask.')) if style=='concise' else ('All prerequisites for the next checks are available. '+('Assign these independent subtasks in one parallel batch.' if len(options)>1 else 'Assign the next subtask.'))
    return {'action':'delegate_task','reasoning':reason,'params':{'tasks':[variants[STYLES.index(s)] for variants,s in zip(options,styles)]}}

def main_style_prompt(payload):
    guidance=payload['gold_guidance'];done=[{'reference':r.get('public_ref'),'answer':r['answer']} for r in payload.get('completed_subtask_contracts',[])]
    state=dict(payload['public_state'])
    history=state.get('subtask_history','')
    if len(history)>9000:
        state['subtask_history']='[Earlier narrative compressed; all completed result bindings are listed below.]\n'+history[-9000:]
    if 'evidence_store' in state:
        state['evidence_store']={sid:{'title':r['title'],'source_type':r['source_type'],'fact_count':len(r['claims'])} for sid,r in state['evidence_store'].items()}
    packet={'public_o1_state':state,'completed_results':done}
    if guidance['may_complete']:
        packet['output_example']={'reasoning_style':'concise','answer':done[-1]['answer'] if done else ''}
        fields={'reasoning_style':{'type':'string','enum':list(STYLES)},'answer':{'type':'string'}}
    else:
        packet['private_gold_guidance']={'ordered_ready_task_variants':[[{'task_instruction':v['task_instruction'],'tools':v['tools']} for v in variants] for variants in action_options(payload)]}
        packet['output_example']={'reasoning_style':'concise','task_styles':['stepwise']*len(guidance['ready'])}
        fields={'reasoning_style':{'type':'string','enum':list(STYLES)},'task_styles':{'type':'array','items':{'type':'string','enum':list(STYLES)},'minItems':len(guidance['ready']),'maxItems':len(guidance['ready'])}}
    packet['required_output_schema']={'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
    return ('Choose language variants for the next O1 action. Tasks and their exact input bindings are fixed. '
        'Select a presentation style, never solve, rewrite, drop or merge a task. The controller renders params.tasks '
        'from the selected variants. No private IDs, analysis or additional fields. Return JSON only.\n'+json.dumps(packet,ensure_ascii=False,separators=(',',':')))

def report_value(value):
    return f"{len(value)} entities/values" if isinstance(value,list) and len(value)>6 else value_text(value)

def report_sentence(op, result):
    name,args=op['op'],op['inputs'];answer=report_value(result['answer'])
    values=result.get('input_values',[])
    def scope(i=0):
        if len(values)<=i:return 'the supplied input'
        v=values[i]
        if isinstance(v,list) and len(v)>8:return f'the {len(v)} input entities'
        return value_text(v)
    if name=='FIND':return f'Identified entities of type {args[0]}: {answer}.' if op.get('find_kind')=='type' else f'Identified {args[0]} as {answer}.'
    if name=='FIND_ALL':return 'Initialized a separate candidate set from all supplied records.'
    if name=='FILTER_TYPE':return f'Filtered {scope()} for type {args[0]}; retained {answer}.'
    if name.startswith('FILTER_'):return f'Filtered {scope()} using {args[0]} {args[1]} {args[2]}; retained {answer}.'
    if name=='RELATE':
        direction='objects back to subjects' if 'backward' in args else 'subjects to objects'
        return f'Followed {args[0]} from {direction}, starting with {scope()}; obtained {answer}.'
    if name=='INTERSECT':return f'Intersected {scope()} with {scope(1)}; the common entities were {answer}.'
    if name=='UNION':return f'Combined {scope()} and {scope(1)} into their union: {answer}.'
    if name=='COUNT':return f'Counted distinct entities in {scope()}; the count was {answer}.'
    if name=='QUERY_NAME':return f'Retrieved the names of the resulting entities: {answer}.'
    if name=='QUERY_ATTR':return f'Retrieved {args[0]} values for {scope()}: {answer}.'
    if name.startswith('VERIFY_'):return f'Checked whether any value in {scope()} satisfies {args[0]} {args[1]}; the result was {answer}.'
    if name=='QUERY_RELATION':return f'Checked relations from {scope()} to {scope(1)}; found {answer}.'
    if name in {'SELECT_AMONG','SELECT_BETWEEN'}:
        return f'Compared {" and ".join(scope(i) for i in range(len(values)))} by {args[0]} using {args[1]}; selected {answer}.'
    raise ValueError('unsupported report operation '+name)

def feedback_variants(fragment):
    observations=fragment['observations'];known=fragment.get('known_observations',[])
    evidence=[]
    if observations:
        from .tools import tool_profile
        profile=tool_profile(fragment.get('scene','omnigaia'))
        by_id={o['resource_id']:o for o in observations}
        for wave in fragment.get('acquisition_waves',[[o['resource_id'] for o in observations]]):
            refs=[by_id[sid] for sid in wave]
            tools=sorted({t for r in refs for t in profile.for_resource(r)})
            evidence.append('Inspected '+', '.join(f"{r['title']} ({r['resource_id']})" for r in refs)+' using '+', '.join(tools)+'.')
    if known:evidence.append('Reused the earlier observations from '+', '.join(f"{r['title']} ({r['resource_id']})" for r in known)+'.')
    sentences=[report_sentence(op,result) for op,result in zip(fragment['operations'],fragment['operation_results'])]
    answer=report_value(fragment['expected_answer']);last=fragment['operations'][-1]['op']
    if last=='COUNT':conclusion='The assigned checks yielded a count of '+answer+'.'
    elif last.startswith('VERIFY_'):conclusion='The requested condition evaluated to '+answer+'.'
    else:conclusion='Completed the assigned checks and returned '+answer+'.'
    from .evidence import evidence_lines
    predicates={op['inputs'][0] for op in fragment['operations'] if op['inputs'] and op['op'] not in {'FIND','FILTER_TYPE'} and not op['op'].startswith('VERIFY_')}
    facts=evidence_lines([*observations,*known],predicates) if predicates else []
    selected=facts[:6]
    support=(' Supporting observations: '+'; '.join(selected)+'.') if selected else ''
    variants=[
        {'summary':conclusion,'trace_summary':' '.join(evidence+sentences)},
        {'summary':sentences[-1],'trace_summary':'\n'.join('* '+line for line in evidence+sentences)},
        {'summary':conclusion,'trace_summary':' '.join(evidence+sentences)+support},
        {'summary':conclusion,'trace_summary':'\n'.join('* '+line for line in evidence+sentences+(['Supporting observations:']+selected if selected else []))},
    ]
    for variant in variants:
        if len(variant['summary'])>1600 or len(variant['trace_summary'])>4000:
            raise ValueError('bounded report exceeds compression budget')
    return variants

def realize_feedback(payload, choice):
    if set(choice)!={'report_style'} or choice['report_style'] not in STYLES:raise ValueError('invalid report language plan')
    return feedback_variants(payload['fragment'])[STYLES.index(choice['report_style'])]

def feedback_style_prompt(payload):
    return ('Choose a compressed Sub-Agent report style: concise uses connected prose; stepwise uses '
        'progress bullets, like an executed tool trace. Both reports below describe the same fixed '
        'evidence and verified operations. Select the clearest style; never alter facts or explain '
        'how training data was generated. Return exactly one key, report_style. Its value must be concise, stepwise, goal or evidence. Example: {"report_style":"stepwise"}.\n'+
        json.dumps({'assigned_task':payload.get('assigned_task',{'task_instruction':payload.get('task_instruction','')})['task_instruction'],'styles':list(STYLES),'report_variants':[{'summary':v['summary'],'trace_preview':v['trace_summary'][:1100]} for v in feedback_variants(payload['fragment'])]},ensure_ascii=False,separators=(',',':')))

def validate_bound_action(action, payload):
    if payload['gold_guidance']['may_complete']:
        variants=[realize_action(payload,{'reasoning_style':style,'answer':action.get('params',{}).get('answer','')}) for style in STYLES]
        return (action in variants,[] if action in variants else ['language_inconsistency:complete_reasoning_outside_bound_contract'])
    options=action_options(payload);tasks=action.get('params',{}).get('tasks',[])
    issues=[]
    if len(tasks)!=len(options):return False,['parallel_gold_fragment_count_mismatch']
    for i,(task,variants) in enumerate(zip(tasks,options)):
        if task not in variants:issues.append(f'language_inconsistency:task_{i}_outside_bound_execution_contract')
    reasons={realize_action(payload,{'reasoning_style':style,'task_styles':['concise']*len(options)})['reasoning'] for style in STYLES}
    if action.get('reasoning') not in reasons:issues.append('language_inconsistency:public_reasoning_outside_bound_contract')
    return not issues,issues

def validate_bound_feedback(value, fragment):
    if value not in feedback_variants(fragment):return False,['language_inconsistency:feedback_outside_bound_execution_contract']
    return True,[]
