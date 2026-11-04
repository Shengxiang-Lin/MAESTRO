"""Audit actual executor results, concrete slots and entity-valued node types."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from ..executor import SymbolicExecutor
from ..rendering.pipeline import program_from_dict,world_from_dict


def assess_execution_contract(record):
    from ..binding.program_binder import ENTITY_OUTPUT_OPS
    from ..synthesis.world_sampler import unresolved_slot
    from ..validation import normalize_answer
    program=program_from_dict(record['program'])
    world=world_from_dict(record['world'])
    issues=[]
    try:
        trace=SymbolicExecutor().execute_trace(program,world)
        answer=trace[program.topological_order()[-1]]
    except Exception as exc:
        trace,answer={},None
        issues.append('executor_failure:'+str(exc))
    if any(unresolved_slot(v) for n in program.nodes for v in n.inputs):
        issues.append('unresolved_program_slot')
    if normalize_answer(answer)!=normalize_answer(record.get('answer')):
        issues.append('recorded_answer_mismatch')
    for node_id,expected in record.get('binding',{}).get('node_types',{}).items():
        node=program.node_map().get(int(node_id))
        if node is None:
            issues.append('missing_bound_node:'+str(node_id))
            continue
        if node.op not in ENTITY_OUTPUT_OPS:
            continue
        observed=trace.get(int(node_id))
        if not isinstance(observed,set):
            issues.append('missing_entity_trace:'+str(node_id))
        # FIND_ALL denotes the whole universe; downstream filters impose types.
        elif node.op!='FIND_ALL' and any(world.entities.get(str(e))!=str(expected) for e in observed):
            issues.append('bound_entity_type_mismatch:'+str(node_id))
    return {'candidate_id':record.get('candidate_id'),'accepted':not issues,
            'issues':issues,'answer':sorted(answer) if isinstance(answer,set) else answer}


def audit_execution_contract_jsonl(worlds_path,*,output_path=None,seed=17):
    source=Path(worlds_path)
    target=Path(output_path) if output_path is not None else None
    writer=target.open('w',encoding='utf-8') if target else None
    total=accepted=0
    reasons=Counter()
    try:
        for line in source.open(encoding='utf-8'):
            if not line.strip():continue
            report=assess_execution_contract(json.loads(line))
            total+=1
            accepted+=int(report['accepted'])
            reasons.update(issue.split(':',1)[0] for issue in report['issues'])
            if writer:writer.write(json.dumps({**report,'seed':seed},ensure_ascii=False)+'\n')
    finally:
        if writer:writer.close()
    return {'version':1,'input':str(source),'output':str(target) if target else None,
        'audited':total,'accepted':accepted,'acceptance_rate':accepted/total if total else 0.,
        'rejection_reasons':dict(reasons),'seed':seed}


__all__=['assess_execution_contract','audit_execution_contract_jsonl']
