"""Stage-3-only eligibility; frozen questions/programs are never repaired here."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

def digest(value):
    text=value if isinstance(value,str) else json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(text.encode()).hexdigest()

def check_question(gold,record,role,*,review=True):
    question=gold['question'];fingerprint={'question_sha256':digest(question),'program_sha256':digest(record['program'])}
    audited=json.loads(Path(__file__).with_name('question_reviews.json').read_text())
    for item in audited:
        if item['task_id']==gold['task_id'] and all(item[k]==v for k,v in fingerprint.items()):
            return {'eligible':item.get('eligible',False),'reviewed':True,'method':'audited_exact_input_review','issues':[] if item.get('eligible') else [item['reason']],'audit_reason':item['reason'],**fingerprint}
    if not review or getattr(role,'offline',False) or not getattr(role,'backend',None):
        return {'eligible':True,'reviewed':False,'method':'offline_structure_only','issues':[],**fingerprint}
    nodes={op['node_id']:op for f in gold['fragments'] for op in f['operations']}
    visited=set()
    def expression(nid):
        if nid in visited:return {'step_reference':nid}
        visited.add(nid);node=nodes[nid]
        return {'step':nid,'action':node['op'],'arguments':node['inputs'],'inputs':[expression(d) for d in node['dependencies']]}
    expression_root=expression(max(nodes))
    prompt=('Check whether the accepted question describes EXACTLY the supplied canonical computation. '
        'Do not solve it, inspect answers, or apply real-world plausibility. Check the owner of each constraint, '
        'relation direction and all necessary hops, branch inputs, set joins and final operation. '
        'Compare mathematical meaning, not number of steps. Redundant type filters, filter order, '
        'and explicitly naming an already implied type are allowed. Impossible numerical conditions '
        'are valid questions if the program asks that condition. Filtering people by any height > a, '
        'then checking their recorded heights for b, may be phrased as a height for a person whose '
        'height exceeds a that equals b; accept this even when b<a. Attributes may have multiple '
        'recorded values. Do not invent scalar uniqueness. '
        'Simplification is allowed only if the computation is unchanged. A person constraint cannot be moved '
        'to a place or vice versa. FALSE/empty outcomes do not make a question invalid. '
        'If ambiguity prevents establishing equivalence, choose uncertain. '
        'Return JSON only: {"decision":"eligible"|"mismatch"|"uncertain","issues":'
        '[{"question_quote":"exact span from accepted question","reason":"scope or hop discrepancy"}]}. '
        'At most two issues; quotes <=140 characters and reasons <=180 characters. For eligible, issues must be empty. No answer values or regenerated questions.\n'+json.dumps({
            'accepted_question':question,'canonical_question':gold['canonical_question'],
            'canonical_expression':expression_root,'canonical_steps':[{'step':nid,'action':op['op'],'arguments':op['inputs'],'input_steps':op['dependencies']} for nid,op in sorted(nodes.items())]},ensure_ascii=False,separators=(',',':')))
    cache_path=getattr(role,'eligibility_cache',None)
    if not isinstance(cache_path,(str,Path)):cache_path=None
    cache_key=digest({**fingerprint,'reviewer_model':getattr(role.backend,'model','unknown'),
        'reviewer_endpoint':getattr(role.backend,'base_url','unknown'),'review_contract':2})
    if cache_path and Path(cache_path).exists():
        for line in Path(cache_path).read_text().splitlines():
            item=json.loads(line)
            if item.get('cache_key')==cache_key:
                verdict=item['verdict'];break
        else:verdict=None
    else:verdict=None
    cache_hit=verdict is not None
    try:
        if verdict is None:verdict=(getattr(role,'check_question',None) or getattr(role,'review'))(prompt)
    except Exception as exc:
        return {'eligible':False,'reviewed':False,'method':'semantic_review_unavailable','issues':[str(exc)[:240]],**fingerprint}
    valid=isinstance(verdict,dict) and verdict.get('decision') in ('eligible','mismatch','uncertain') and isinstance(verdict.get('issues'),list)
    if valid and cache_path and not cache_hit:
        target=Path(cache_path);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('a',encoding='utf-8') as handle:handle.write(json.dumps({'cache_key':cache_key,'verdict':verdict},ensure_ascii=False)+'\n')
    if valid and verdict['decision']=='eligible' and verdict['issues']==[]:
        return {'eligible':True,'reviewed':True,'method':'llm_semantic_screen','cache_hit':cache_hit,'issues':[],**fingerprint}
    issues=[]
    for issue in verdict.get('issues',[]) if isinstance(verdict,dict) else []:
        if isinstance(issue,dict) and issue.get('question_quote') and issue['question_quote'] in question:
            issues.append(issue)
    return {'eligible':False,'reviewed':bool(valid),'method':'llm_semantic_screen','cache_hit':cache_hit,'issues':issues or ['equivalence_not_established'],'raw_verdict':verdict,**fingerprint}
