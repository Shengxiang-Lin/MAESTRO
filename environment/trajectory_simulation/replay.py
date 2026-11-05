"""Execute only public delegated instructions and public observations, never gold."""
from __future__ import annotations
import json,re,copy
from .evidence import read_packet,verify_known_packet
from .tools import tool_profile

def compare(left,op,right):
    def number(v):
        text=str(v).strip().replace(',','')
        if re.fullmatch(r'\d{4}-\d{2}(?:-\d{2})?',text):raise ValueError('date')
        match=re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\s+[A-Za-z%]+)?',text)
        if not match:raise ValueError('not numeric')
        return float(match[1])
    try:a,b=number(left),number(right)
    except ValueError:a,b=str(left),str(right)
    if op in ('=','=='):return a==b
    if op=='!=':return a!=b
    if op in ('>','greater'):return a>b
    if op in ('<','less'):return a<b
    if op=='>=':return a>=b
    if op=='<=':return a<=b
    raise ValueError('unsupported public comparison '+op)

def replay_task(task,new_observations=(),*,public_cache=None):
    # The public state is the sole source for previously observed evidence.
    # A v2 handle without a supplied public cache must never resolve through
    # private Gold state or silently skip the missing evidence.
    raw=read_packet(task)
    if raw['version'] in (2,3) and public_cache is None:
        raise ValueError('public evidence cache required for referenced task packet')
    packet=(verify_known_packet(task,public_cache)
            if public_cache is not None else raw)
    profile=tool_profile(packet.get("scene","omnigaia"))
    records=[*packet['known_records'],*new_observations]
    for record in new_observations:
        if not set(profile.for_resource(record))<=set(task['tools']):
            raise ValueError('new evidence without assigned inspection tool')
    entities={};facts=[];seen=set()
    for record in records:
        for name,kind in record.get('entity_types',{}).items():
            if name in entities and entities[name]!=kind:raise ValueError('conflicting public entity type')
            entities[name]=kind
        for fact in record['claims']:
            key=fact.get('observation_key') or json.dumps(fact,sort_keys=True)
            if key not in seen:facts.append(fact);seen.add(key)
    completed={}
    for ref,item in packet['completed_inputs'].items():
        value=copy.deepcopy(item['value'])
        completed[ref]=set(value) if item['answer_kind']=='entity_set' else value
    values=[];inputs=[]
    def replace_refs(line):
        refs={};index=0
        while match:=re.search(r'the completed (Attempt \d+, subtask \d+) result \(',line):
            ref=match[1]
            if ref not in completed:raise ValueError('unavailable completed input '+ref)
            start=match.end();depth=1;quoted=False;escaped=False;end=start
            while end<len(line) and depth:
                c=line[end]
                if escaped:escaped=False
                elif c=='\\' and quoted:escaped=True
                elif c=='"':quoted=not quoted
                elif not quoted:
                    if c=='(':depth+=1
                    elif c==')':depth-=1
                end+=1
            if depth:raise ValueError('unterminated completed input')
            token=f'@e{index}';index+=1;refs[token]=completed[ref]
            line=line[:match.start()]+token+line[end:]
        def local(match):
            n=int(match[1])-1
            if not 0<=n<len(values):raise ValueError('future local input')
            token=f'@s{n}';refs[token]=values[n];return token
        return re.sub(r'the result of instruction (\d+)',local,line),refs
    def as_set(value):return value if isinstance(value,set) else set(value) if isinstance(value,list) else {value}
    def attrs(candidates,predicate):
        return [(f['subject'],f['object']) for f in facts if f['subject'] in as_set(candidates) and f['predicate']==predicate and f['object_kind']!='entity']
    checks=packet['execution_checks'] if packet['version']==3 else task['task_instruction']
    lines=re.findall(r'^\d+\. (.+)$',checks,re.M)
    if not lines:raise ValueError('public task has no executable instruction list')
    for original in lines:
        line,refs=replace_refs(original);deps=[]
        def get(token):
            value=refs[token];deps.append(copy.deepcopy(value));return value
        if line.startswith('Identify all entities of type '):
            # Historical canonical wording remains accepted for frozen banks.
            kind=line[len('Identify all entities of type '):-1]
            value={e for e,t in entities.items() if t==kind}
        elif m:=re.fullmatch(
            r'(?:Search task-relevant sources to identify entities of type |'
            r'Examine accessible task evidence to identify entities of type )'
            r'(.+); do not assume a complete list was initially provided\.',line
        ):
            # Discovery changes only how Main describes the task. The actual
            # set must still come solely from the published source records.
            value={e for e,t in entities.items() if t==m[1]}
        elif line.startswith('Identify '):
            target=line[len('Identify '):-1]
            if target.startswith('the entity matching the question clues: '):
                candidates=set(entities)
                for condition in target.split(': ',1)[1].split('; '):
                    m=re.fullmatch(r'(.+?) (==|!=|>=|<=|=|>|<) (.+)',condition)
                    if not m:raise ValueError('unexecutable identification clue')
                    candidates={s for s,v in attrs(candidates,m[1]) if compare(v,m[2],m[3])}
                if len(candidates)!=1:raise ValueError('identification clue is not uniquely supported')
                value=candidates
            elif target.startswith('the entity matching '):raise ValueError('missing concrete identification clue')
            else:value={target}
        elif line in {
            'Start a separate candidate set from all supplied records.',
            'Search task-relevant sources and enumerate the candidate entities '
            'supported by the discovered evidence.',
            'Examine the accessible task evidence and enumerate the candidate '
            'entities it supports.',
        }:
            # FIND_ALL ranges over only published source-backed entity records.
            # This does not inject an oracle roster or assume initial names.
            value=set(entities)
        elif m:=re.fullmatch(r'From (@\w+), keep only entities of type (.+)\.',line):
            value={e for e in as_set(get(m[1])) if entities.get(e)==m[2]}
        elif m:=re.fullmatch(r'From (@\w+), keep only entities whose (.+?) (==|!=|>=|<=|=|>|<) (.+)\.',line):
            value={s for s,v in attrs(get(m[1]),m[2]) if compare(v,m[3],m[4])}
        elif m:=re.fullmatch(r'Using (@\w+), follow the (.+) relation from (.+)\.',line):
            candidates=as_set(get(m[1]));backward=m[3]=='objects back to their subjects'
            if m[3] not in ('objects back to their subjects','subjects to their objects'):raise ValueError('invalid public direction')
            value={f['subject'] if backward else str(f['object']) for f in facts if f['predicate']==m[2] and f['object_kind']=='entity' and (str(f['object']) if backward else f['subject']) in candidates}
        elif m:=re.fullmatch(r'(Intersect|Take the union of) (@\w+(?: and @\w+)*)\.',line):
            groups=[as_set(get(token)) for token in m[2].split(' and ')]
            value=set.intersection(*groups) if m[1]=='Intersect' else set.union(*groups)
        elif m:=re.fullmatch(r'Count the distinct entities in (@\w+)\.',line):value=len(as_set(get(m[1])))
        elif m:=re.fullmatch(r'Return the names of entities in (@\w+)\.',line):value=sorted(as_set(get(m[1])))
        elif m:=re.fullmatch(r'Retrieve the (.+) values for entities in (@\w+)\.',line):value=[v for _,v in attrs(get(m[2]),m[1])]
        elif m:=re.fullmatch(r'Check whether any value in (@\w+) satisfies (==|!=|>=|<=|=|>|<) (.+)\.',line):
            observed=as_set(get(m[1]))
            if not observed:raise ValueError('verification requires an observed value')
            value=any(compare(v,m[2],m[3]) for v in observed)
        elif m:=re.fullmatch(r'Return the relation names linking (@\w+) to (@\w+)\.',line):
            left,right=as_set(get(m[1])),as_set(get(m[2]))
            value=sorted({f['predicate'] for f in facts if f['subject'] in left and str(f['object']) in right and f['object_kind']=='entity'})
        elif m:=re.fullmatch(r'Select entities from (@\w+(?: and @\w+)*) by (.+) using (.+)\.',line):
            candidates=set.union(*(as_set(get(t)) for t in m[1].split(' and ')))
            scored=[]
            for entity,v in attrs(candidates,m[2]):
                try:scored.append((float(str(v).replace(',','').split()[0]),entity))
                except ValueError:continue
            minimum=m[3].lower() in ('min','smallest','less','<')
            best=(min if minimum else max)([v for v,_ in scored]) if scored else None
            value={e for v,e in scored if v==best}
        else:raise ValueError('unrecognized public instruction: '+line)
        values.append(value);inputs.append(deps)
    return {'result':values[-1],'step_values':values,'step_inputs':inputs}
