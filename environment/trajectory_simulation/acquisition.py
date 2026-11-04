"""Resolve AND-over-anchors / OR-over-providers without mutating frozen data."""
from __future__ import annotations
from itertools import combinations
from environment.task_construction.executor import SymbolicExecutor
from environment.task_construction.scenario.information import assess_goal_information_completeness
from environment.task_construction.validation import normalize_answer

INITIAL={'attached','provided','initial'}


def resolve_dependencies(resources,registry):
    by_id={str(r['resource_id']):r for r in resources}
    question_anchors=set(registry.get('question_anchor_ids',()))
    pending=set(by_id);done=set();dependencies={};levels={}
    has_anchors=bool(registry) or any('requires_anchors' in r for r in resources)
    providers={}
    for sid,r in by_id.items():
        for anchor in r.get('produces_anchors',()):
            providers.setdefault(anchor,set()).add(sid)
    for sid,r in by_id.items():
        if r.get('visibility') in INITIAL:
            dependencies[sid]=[]
        elif has_anchors:
            required=set(r.get('requires_anchors',()))
            if not required:
                raise ValueError(f'discoverable resource {sid} has no frozen acquisition anchor')
            dependencies[sid]=[
                providers.get(a,set())-{sid}
                for a in sorted(required-question_anchors)
            ]
        else:
            parents=set(r.get('anchor_dependency_resource_ids',()))
            if not parents.issubset(by_id) or sid in parents:
                raise ValueError(f'invalid acquisition parents for {sid}')
            dependencies[sid]=[{p} for p in sorted(parents)]
    wave=0
    while pending:
        ready={sid for sid in pending if all(group & done for group in dependencies[sid])}
        if not ready:
            raise ValueError('unreachable frozen acquisition anchors: '+','.join(sorted(pending)))
        wave+=1
        for sid in ready:levels[sid]=wave
        # A same-wave provider cannot satisfy another task's prerequisite.
        done.update(ready);pending.difference_update(ready)
    return dependencies,levels


def reachable_subset(ids,dependencies):
    pending=set(ids);done=set()
    while pending:
        ready={sid for sid in pending if all(group & done for group in dependencies[sid])}
        if not ready:return False
        done.update(ready);pending.difference_update(ready)
    return True


def evidence_world(world,source_ids):
    """Keep facts available from ANY selected provider, including shared facts."""
    result=world.copy()
    result.sources={sid:source for sid,source in world.sources.items() if sid in source_ids}
    fact_ids={fid for source in result.sources.values() for fid in source.fact_ids}
    result.facts=[fact for fact in world.facts if fact.fact_id in fact_ids or fact.source_id in source_ids]
    return result


def decision_complete_evidence(program, world, source_ids, required_fact_ids, expected):
    """Check the actually inspected cover, not just attribute support IDs."""
    if not source_ids:
        return False
    visible = evidence_world(world, source_ids)
    if not set(required_fact_ids).issubset({fact.fact_id for fact in visible.facts}):
        return False
    try:
        answer = normalize_answer(SymbolicExecutor().execute(program, visible))
        information = assess_goal_information_completeness(program, visible)
    except (ValueError, TypeError, RuntimeError):
        return False
    return answer == expected and information.accepted


def choose_evidence(program,world,dependencies,max_sources=8):
    full=assess_goal_information_completeness(program,world)
    if not full.accepted:raise ValueError('frozen world has incomplete decision information')
    required=set(full.required_fact_ids)
    expected=normalize_answer(SymbolicExecutor().execute(program,world))
    ids=sorted(world.sources)
    if len(ids)>max_sources:raise ValueError('source count exceeds exact evidence selection budget')
    for size in range(1,len(ids)+1):
        for subset in combinations(ids,size):
            selected=tuple(subset)
            if not reachable_subset(selected,dependencies):continue
            # Combinations are already ordered by size, then resource ID.
            if decision_complete_evidence(program,world,set(selected),required,expected):
                return list(selected),sorted(required)
    raise ValueError('no reachable decision-complete evidence cover')
