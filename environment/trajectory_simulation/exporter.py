"""Export clean Stage-3 JSONL into the existing Main SFT record schema."""
from __future__ import annotations
import json,copy,re
from pathlib import Path
from .eligibility import digest
from .evidence import verify_known_packet
from .replay import replay_task
from .contract import answers_equal
from .history import format_o1_subtask_history

def validate_public_episode(episode,*,require_training=True):
    validation=episode.get('validation',{});eligibility=episode.get('input_eligibility',{})
    if require_training and not (validation.get('training_eligible') and eligibility.get('eligible') and eligibility.get('reviewed')):
        raise ValueError('episode has not passed training eligibility review')
    if digest(episode['question'])!=eligibility.get('question_sha256'):raise ValueError('question changed after eligibility review')
    if not validation.get('public_evidence_replay_verified'):raise ValueError('missing public replay validation')
    cache={};completed={};past=[];steps=episode.get('trajectory',[])
    if not steps or steps[-1].get('action')!='complete':raise ValueError('missing terminal complete action')
    for step in steps:
        state=step['state_before']
        public_action_text=json.dumps({k:step[k] for k in ('action','reasoning','params')},ensure_ascii=False)
        if re.search(r'gold_supervision|private_gold_guidance|symbolic_trace|fragment:',public_action_text):raise ValueError('private supervision in public target')
        if set(state)!={'question','subtask_history','available_tools','evidence_store'}:raise ValueError('unexpected public state fields')
        if state['evidence_store']!=cache:raise ValueError('public evidence store contains unavailable or altered records')
        if state['subtask_history']!=format_o1_subtask_history(past):raise ValueError('public history does not match preceding results')
        if step['action']=='complete':
            if set(step['params'])!={'answer'}:raise ValueError('unexpected completion fields')
            if step is not steps[-1]:raise ValueError('premature completion')
            if not past:raise ValueError('completion without a public subtask result')
            last=past[-1]['result']['subtask_results'][-1]
            if not answers_equal(step['params']['answer'],last['result_value'],{'collection_mode':'set' if last['result_kind']=='entity_set' else 'multiset'}):raise ValueError('final answer contradicts the final public result')
            break
        if step['action']!='delegate_task':raise ValueError('invalid public action')
        if set(step['params'])!={'tasks'}:raise ValueError('unexpected delegation fields')
        tasks=step['params']['tasks'];replies=step['result']['subtask_results']
        if len(tasks)!=len(replies) or not tasks:raise ValueError('task/reply mismatch')
        pending=[]
        for index,(task,reply) in enumerate(zip(tasks,replies)):
            if set(task)!={'task_instruction','context','model','tools'}:raise ValueError('unexpected public task fields')
            if task['task_instruction']!=reply['task_instruction']:raise ValueError('reply attached to a different task')
            packet=verify_known_packet(task,cache)
            for ref,item in packet['completed_inputs'].items():
                if completed.get(ref)!=item:raise ValueError('task uses a future or altered completed input')
            observed=reply['observed_evidence']
            for record in observed:
                if set(record)-{'resource_id','source_type','title','content','claims','entity_types'}:raise ValueError('unexpected evidence fields')
                for fact in record['claims']:
                    if set(fact)-{'subject','predicate','object','object_kind','observation_key'}:raise ValueError('unexpected public claim fields')
            replay=replay_task(task,observed)
            contract={'collection_mode':'set' if reply['result_kind']=='entity_set' else 'multiset'}
            from .compiler import json_value
            if not answers_equal(json_value(replay['result']),reply['result_value'],contract):raise ValueError('public replay result mismatch')
            if not answers_equal(reply['finish_result']['result'],reply['result_value'],contract):raise ValueError('public result transport mismatch')
            ref=f"Attempt {step['attempt']}, subtask {index+1}"
            pending.append((ref,{'value':reply['result_value'],'answer_kind':reply['result_kind']},observed))
        # Siblings may not access each other's same-round results.
        for ref,item,observed in pending:
            completed[ref]=copy.deepcopy(item)
            for record in observed:
                sid=record['resource_id']
                if sid in cache and cache[sid]!=record:raise ValueError('contradictory evidence records')
                cache[sid]=copy.deepcopy(record)
        past.append(step)

def public_prompt(state):
    return ('You are the Main Agent. Delegate independent tasks in parallel when their inputs are available. '
        'Use delegate_task with params.tasks (task_instruction, context, model, tools), or complete with params.answer. '
        'Return one native O1 JSON action with action, reasoning and params.\n\nQuestion:\n'+state['question']+
        '\n\nSUBTASK HISTORY:\n'+state['subtask_history']+'\n\nAVAILABLE TOOLS:\n'+json.dumps(state['available_tools'],ensure_ascii=False)+
        '\n\nPUBLIC EVIDENCE STORE:\n'+json.dumps(state['evidence_store'],ensure_ascii=False,separators=(',',':')))

def export_episode(episode,*,require_training=True):
    validate_public_episode(episode,require_training=require_training)
    records=[]
    for index,step in enumerate(episode['trajectory'],1):
        target={key:step[key] for key in ('action','reasoning','params')}
        record={'id':f"{episode['task_id']}:{index}",'task_id':episode['task_id'],'step':index,
            'input':public_prompt(step['state_before']),'target_json':json.dumps(target,ensure_ascii=False,separators=(',',':')),
            'macro_label':'complete' if step['action']=='complete' else 'advance','macro_label_source':'public_action',
            'episode_success':True,'episode_reward':0.0}
        from train_maestro_8b.schema import validate_stage1_record
        records.append(validate_stage1_record(record))
    return records

def export_jsonl(source,output,*,max_input_chars=None):
    from .pipeline import protect_paths
    source=Path(source);output=Path(output)
    files=[source] if source.is_file() else sorted({*source.rglob('*.json'),*source.rglob('*.jsonl')})
    files=[p for p in files if '.rejected.' not in p.name and '.quarantine.' not in p.name and not p.name.endswith('.excluded.jsonl')]
    excluded=output.with_suffix(output.suffix+'.excluded.jsonl')
    protect_paths(files,[output,excluded]);output.parent.mkdir(parents=True,exist_ok=True)
    counts={'episodes_exported':0,'episodes_excluded':0,'records_exported':0}
    seen=set()
    counts['duplicates_skipped']=0;counts['maximum_input_characters']=0
    with output.open('w',encoding='utf-8') as good,excluded.open('w',encoding='utf-8') as bad:
        for path in files:
            rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.suffix=='.jsonl' else [json.loads(path.read_text(encoding='utf-8'))]
            for episode in rows:
                if not isinstance(episode,dict) or 'trajectory' not in episode:continue
                tid=episode.get('task_id','unknown')
                if tid in seen:counts['duplicates_skipped']+=1;continue
                try:
                    records=export_episode(episode)
                    if max_input_chars is not None and any(len(r['input'])>max_input_chars for r in records):raise ValueError('public prompt exceeds maximum input length; not truncated')
                except (ValueError,KeyError,TypeError) as exc:
                    counts['episodes_excluded']+=1;bad.write(json.dumps({'task_id':tid,'reason':str(exc)},ensure_ascii=False)+'\n');continue
                seen.add(tid)
                counts['maximum_input_characters']=max(counts['maximum_input_characters'],*(len(r['input']) for r in records))
                for record in records:good.write(json.dumps(record,ensure_ascii=False)+'\n')
                counts['episodes_exported']+=1;counts['records_exported']+=len(records)
    return {**counts,'output':str(output),'excluded_output':str(excluded)}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory-path',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--max-input-chars',type=int)
    args=parser.parse_args();print(json.dumps(export_jsonl(args.trajectory_path,args.output,max_input_chars=args.max_input_chars),indent=2))
