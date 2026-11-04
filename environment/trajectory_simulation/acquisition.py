"""Resolve AND-over-anchors / OR-over-providers without mutating frozen data."""
from __future__ import annotations
from itertools import combinations
from environment.task_construction.executor import SymbolicExecutor
from environment.task_construction.scenario.information import assess_goal_information_completeness
from environment.task_construction.validation import normalize_answer

INITIAL={'attached','provided','initial'}


def resolve_dependencies(resources,registry):
    known=set(registry.get('question_anchor_ids',()))
    by_id={str(r['resource_id']):r for r in resources}
    pending=set(by_id);done=set();dependencies={};levels={}
    has_anchors=bool(registry) or any('requires_anchors' in r for r in resources)
    providers={}
    wave=0
    while pending:
        ready={}
        for sid in sorted(pending):
            r=by_id[sid]
            if r.get('visibility') in INITIAL:
                ready[sid]=set()
            elif has_anchors:
                required=set(r.get('requires_anchors',()))
                if not required:
                    raise ValueError(f'discoverable resource {sid} has no frozen acquisition anchor')
                if not required.issubset(known):continue
                # One previously observed provider per required anchor suffices.
                ready[sid]={providers[a] for a in required if a in providers}
            else:
                parents=set(r.get('anchor_dependency_resource_ids',()))
                if not parents.issubset(by_id) or sid in parents:
                    raise ValueError(f'invalid acquisition parents for {sid}')
                if parents.issubset(done):ready[sid]=parents
        if not ready:
            raise ValueError('unreachable frozen acquisition anchors: '+','.join(sorted(pending)))
        wave+=1
        # Publish anchors only AFTER the whole ready wave is fixed.
        for sid,parents in ready.items():
            dependencies[sid]=parents;levels[sid]=wave
        for sid in sorted(ready):
            done.add(sid);pending.remove(sid)
            for anchor in by_id[sid].get('produces_anchors',()):
                if anchor not in known:providers[anchor]=sid
                known.add(anchor)
    return dependencies,levels


def closure(ids,dependencies):
    result=set(ids)
    while True:
        expanded=result|{p for sid in result for p in dependencies[sid]}
        if expanded==result:return result
        result=expanded


def evidence_world(world,source_ids):
    """Keep facts available from ANY selected provider, including shared facts."""
    result=world.copy()
    result.sources={sid:source for sid,source in world.sources.items() if sid in source_ids}
    fact_ids={fid for source in result.sources.values() for fid in source.fact_ids}
    result.facts=[fact for fact in world.facts if fact.fact_id in fact_ids or fact.source_id in source_ids]
    return result


def choose_evidence(program,world,dependencies,max_sources=8):
    full=assess_goal_information_completeness(program,world)
    if not full.accepted:raise ValueError('frozen world has incomplete decision information')
    required=set(full.required_fact_ids)
    expected=normalize_answer(SymbolicExecutor().execute(program,world))
    ids=sorted(world.sources)
    if len(ids)>max_sources:raise ValueError('source count exceeds exact evidence selection budget')
    options={}
    for size in range(1,len(ids)+1):
        for subset in combinations(ids,size):
            selected=tuple(sorted(closure(set(subset),dependencies)))
            if selected in options:continue
            visible=evidence_world(world,set(selected))
            if not required.issubset({f.fact_id for f in visible.facts}):continue
            try:answer=normalize_answer(SymbolicExecutor().execute(program,visible))
            except (ValueError,TypeError,RuntimeError):continue
            info=assess_goal_information_completeness(program,visible)
            if answer==expected and info.accepted:options[selected]=True
    if not options:raise ValueError('no reachable decision-complete evidence cover')
    selected=min(options,key=lambda x:(len(x),x))
    return list(selected),sorted(required)
