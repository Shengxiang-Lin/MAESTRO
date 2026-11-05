"""Optional linear fusion; forks, joins and all frozen operations are preserved."""
from __future__ import annotations
import copy
from .tools import tool_profile

def compact_plan(original):
    plan=copy.deepcopy(original);fragments=plan['fragments']
    while True:
        uses={f['fragment_id']:set() for f in fragments}
        for f in fragments:
            for parent in f['depends_on']:uses[parent].add(f['fragment_id'])
            for op in f['operations']:
                for binding in op['input_bindings']:
                    if binding['kind']=='completed':uses[binding['fragment_id']].add(f['fragment_id'])
        by_id={f['fragment_id']:f for f in fragments};pair=None
        for child in fragments:
            if len(child['depends_on'])!=1:continue
            parent=by_id[child['depends_on'][0]]
            if uses[parent['fragment_id']]=={child['fragment_id']}:
                pair=(parent,child);break
        if pair is None:break
        parent,child=pair;offset=len(parent['operations']);parent_id=parent['fragment_id']
        operations=copy.deepcopy(child['operations'])
        for op in operations:
            for binding in op['input_bindings']:
                if binding['kind']=='step':binding['step']+=offset
                elif binding['fragment_id']==parent_id:
                    binding.clear();binding.update(kind='step',step=offset)
        observed={o['resource_id']:o for o in [*parent['observations'],*child['observations']]}
        known={o['resource_id']:o for o in [*parent['known_observations'],*child['known_observations']] if o['resource_id'] not in observed}
        waves=[];seen=set()
        for wave in parent['acquisition_waves']+child['acquisition_waves']:
            new=[sid for sid in wave if sid not in seen];seen.update(new)
            if new:waves.append(new)
        child.update(depends_on=parent['depends_on'],operations=parent['operations']+operations,
            operation_results=parent['operation_results']+child['operation_results'],
            observations=list(observed.values()),known_observations=list(known.values()),
            resource_ids=list(observed),support_resource_ids=sorted(set(parent['support_resource_ids'])|set(child['support_resource_ids'])),acquisition_waves=waves)
        profile=tool_profile(plan['scene'])
        child['suggested_tools']=sorted({tool for o in child['observations'] for tool in profile.for_resource(o)}) or list(profile.calculation)
        fragments=[f for f in fragments if f['fragment_id']!=parent_id]
    plan['fragments']=fragments;done=set();waves=[]
    while len(done)<len(fragments):
        wave=[f['fragment_id'] for f in fragments if f['fragment_id'] not in done and set(f['depends_on'])<=done]
        if not wave:raise ValueError('invalid compact scheduling graph')
        waves.append(wave);done.update(wave)
    plan['waves']=waves;plan['schedule']='compact'
    return plan
