"""Public evidence packets: no invisible cache, oracle, or graph required."""
from __future__ import annotations
import copy,json
from .tools import tool_profile

MARKER='\n[Public task packet]\n'

def public_record(observation):
    return {key:copy.deepcopy(observation[key]) for key in
        ('resource_id','source_type','title','content','claims','entity_types') if key in observation}

def bind_ready(fragments,cache):
    bound=[]
    for original in fragments:
        f=copy.deepcopy(original);all_obs={o['resource_id']:o for o in [*f['observations'],*f.get('known_observations',[])]}
        witness=set(f['support_resource_ids'])
        # A source is reused only when its complete public record is already available.
        f['known_observations']=[copy.deepcopy(cache[s]) for s in sorted(witness&set(cache))]
        f['observations']=[all_obs[s] for s in f['resource_ids'] if s not in cache]
        # Original inherited sources must also be backed by the public store.
        missing=witness-set(cache)-{o['resource_id'] for o in f['observations']}
        if missing:raise ValueError('private-only inherited evidence: '+','.join(sorted(missing)))
        f['resource_ids']=[o['resource_id'] for o in f['observations']]
        f['acquisition_waves']=[[s for s in wave if s not in cache] for wave in f.get('acquisition_waves',[])]
        f['acquisition_waves']=[wave for wave in f['acquisition_waves'] if wave]
        profile=tool_profile(f.get('scene','omnigaia'))
        f['suggested_tools']=sorted({t for o in f['observations'] for t in profile.for_resource(o)}) or list(profile.calculation)
        bound.append(f)
    return bound

def task_packet(fragment,completed,cache):
    refs={b['fragment_id'] for op in fragment['operations'] for b in op.get('input_bindings',[]) if b['kind']=='completed'}
    inputs={}
    for ref in sorted(refs):
        item=completed[ref]
        inputs[item['public_ref']]={'value':copy.deepcopy(item['value']),'answer_kind':item['answer_kind']}
    records=[]
    for source in fragment.get('known_resource_ids',[]):
        if source not in cache:raise ValueError('task requested unavailable public evidence '+source)
        records.append(public_record(cache[source]))
    return {'version':1,'scene':fragment.get('scene','omnigaia'),'completed_inputs':inputs,'known_records':records}

def read_packet(task):
    context=task.get('context','')
    if MARKER not in context:raise ValueError('missing public task packet')
    packet=json.loads(context.split(MARKER,1)[1])
    if set(packet) not in ({'version','completed_inputs','known_records'},{'version','scene','completed_inputs','known_records'}) or packet['version']!=1:
        raise ValueError('invalid public task packet')
    return packet

def verify_known_packet(task,public_cache):
    packet=read_packet(task)
    for record in packet['known_records']:
        if record!=public_cache.get(record['resource_id']):raise ValueError('unpublished or altered cached evidence')
    return packet

def evidence_lines(observations,predicates=()):
    lines=[]
    for obs in observations:
        for fact in obs['claims']:
            if predicates and fact['predicate'] not in predicates:continue
            lines.append(f"{fact['subject']} | {fact['predicate']} | {fact['object']} [{obs['title']}; {obs['resource_id']}]")
    return lines
