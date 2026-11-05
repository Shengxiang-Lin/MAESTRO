"""Stage-3-only eligibility; frozen questions/programs are never repaired here."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path

def digest(value):
    text=value if isinstance(value,str) else json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(text.encode()).hexdigest()

def check_question(gold,record,role,*,review=True):
    question=gold['question'];fingerprint={'question_sha256':digest(question),'program_sha256':digest(record['program'])}
    audit_path = Path(__file__).with_name('question_reviews.json')
    # This manually curated file is optional and not committed in all setups.
    # Missing file is NOT an approval: a real LLM semantic check is still
    # required before training_eligible can become true.
    audited = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.is_file() else []
    if not isinstance(audited, list):
        raise ValueError("question_reviews.json must be a JSON array")
    for item in audited:
        if item['task_id']==gold['task_id'] and all(item[k]==v for k,v in fingerprint.items()):
            return {'eligible':item.get('eligible',False),'reviewed':True,'method':'audited_exact_input_review','issues':[] if item.get('eligible') else [item['reason']],'audit_reason':item['reason'],**fingerprint}
    if not review or getattr(role,'offline',False) or not getattr(role,'backend',None):
        return {'eligible':True,'reviewed':False,'method':'offline_structure_only','issues':[],**fingerprint}
    # A certified Stage-3 public projection may omit an internal FIND and
    # its redundant INTERSECT. Their original ProgramIR dependencies remain
    # referenced by the final operation. Semantic review must use the *full*
    # frozen graph, not infer the question's semantics from the pruned O1
    # delegation graph (which would KeyError on an omitted node).
    original_nodes={
        int(node['node_id']):node
        for node in record['program']['nodes']
    }
    public_nodes={
        int(op['node_id']):op
        for f in gold['fragments'] for op in f['operations']
    }
    nodes={
        nid:public_nodes.get(nid,raw)
        for nid,raw in original_nodes.items()
    }
    def normalized(text):
        return re.sub(r'[^a-z0-9]+','',str(text).casefold())
    # Missing literal relation wording is a review trigger, not a rejection:
    # faithful paraphrases are allowed, but an approval must bind each such
    # hop to actual question evidence instead of overlooking it.
    relation_gaps=[{
        'step':nid,'relation':op['inputs'][0],
        'direction':op['inputs'][1] if len(op['inputs'])>1 else 'forward',
        'input_steps':op['dependencies'],
    } for nid,op in sorted(nodes.items()) if op['op']=='RELATE'
        and normalized(str(op['inputs'][0]).split(':')[-1]) not in normalized(question)]
    visited=set()
    def expression(nid):
        if nid in visited:return {'step_reference':nid}
        visited.add(nid)
        node=nodes[nid]
        return {'step':nid,'action':node['op'],'arguments':node['inputs'],
                'inputs':[expression(int(d)) for d in node['dependencies']]}
    expression_root=expression(max(nodes))
    prompt=('Check whether the accepted question describes EXACTLY the supplied canonical computation. '
        'Do not solve it, inspect answers, or apply real-world plausibility. Check the owner of each constraint, '
        'relation direction and all necessary hops, branch inputs, set joins and final operation. '
        'Compare mathematical meaning, not number of steps. Redundant type filters, filter order, '
        'FIND_ALL with no input steps initializes ALL entities, not an empty set. '
        'Read the numbered dependency graph literally: a filter before RELATE constrains '
        'the input entity; a filter after RELATE constrains the related entity. '
        'Repeated identical type filters do not change the set. '
        'and explicitly naming an already implied type are allowed. SELECT_AMONG and '
        'SELECT_BETWEEN return selected entities, not their ranking attribute values; '
        'QUERY_ATTR returns attribute values. Impossible numerical conditions '
        'are valid questions if the program asks that condition. Filtering people by any height > a, '
        'then checking their recorded heights for b, may be phrased as a height for a person whose '
        'height exceeds a that equals b; accept this even when b<a. Attributes may have multiple '
        'recorded values. Do not invent scalar uniqueness. '
        'Simplification is allowed only if the computation is unchanged. A person constraint cannot be moved '
        'to a place or vice versa. FALSE/empty outcomes do not make a question invalid. '
        'If ambiguity prevents establishing equivalence, choose uncertain. '
        'An explicitly certified public projection may remove a redundant private singleton FIND '
        'and its INTERSECT from agent delegation; the accepted question need not expose the '
        'private singleton name when another public branch identifies the same entity. '
        'Do not excuse missing, redirected, or changed constraints beyond that exact redundancy. '
        'Reject only a concrete mathematical discrepancy, never stylistic wording. '
        'Before claiming a missing condition, check the ENTIRE accepted question, including its ending. '
        'Identical canonical wording is not proof of equivalence: verify the actual steps. '
        'Return JSON only: {"decision":"eligible"|"mismatch"|"uncertain","issues":'
        '[{"kind":"missing_constraint"|"scope"|"direction"|"join"|"output",'
        '"program_steps":[1],"question_quote":"exact span from accepted question",'
        '"program_claim":"what these steps compute",'
        '"question_claim":"what the quoted wording means or omits",'
        '"reason":"specific mathematical discrepancy"}]}. '
        'At most two issues; quotes <=240 characters, claims and reasons <=180 characters each. '
        'Do not put analysis or deliberation inside the JSON strings. '
        'For eligible, issues must be empty. '
        'For mismatch, at least one issue must cite existing step IDs and an exact question span. '
        'No answer values or regenerated questions.\n'+json.dumps({
            'accepted_question':question,'canonical_question':gold['canonical_question'],
            'certified_public_projections':gold.get('public_projections',[]),
            'canonical_expression':expression_root,
            'canonical_steps':[{
                'step':nid,'action':op['op'],'arguments':op['inputs'],
                'input_steps':op['dependencies'],
                'projected_from_public_delegation':nid not in public_nodes,
            } for nid,op in sorted(nodes.items())]},ensure_ascii=False,separators=(',',':')))
    cache_path=getattr(role,'eligibility_cache',None)
    if not isinstance(cache_path,(str,Path)):cache_path=None
    # Test doubles and partially constructed clients can expose Mock objects
    # for unknown attributes. Cache identity must be stable, JSON-native and
    # never depend on a Mock repr or memory address.
    def identity(field):
        value = getattr(role.backend, field, None)
        return value if isinstance(value, str) and value else "unknown"
    cache_key=digest({**fingerprint,'reviewer_model':identity('model'),
        'reviewer_endpoint':identity('base_url'),'review_contract':5})
    if cache_path and Path(cache_path).exists():
        for line in Path(cache_path).read_text().splitlines():
            item=json.loads(line)
            if item.get('cache_key')==cache_key:
                verdict=item['verdict'];break
        else:verdict=None
    else:verdict=None
    cache_hit=verdict is not None
    coverage_required=cache_hit and bool(relation_gaps)
    def valid_verdict(value):
        if not (isinstance(value,dict)
                and value.get('decision') in ('eligible','mismatch','uncertain')
                and isinstance(value.get('issues'),list)):
            return False
        if value['decision']=='eligible':
            if value['issues']!=[]:return False
            if not coverage_required:return True
            coverage=value.get('relation_coverage')
            if not isinstance(coverage,list):return False
            return all(any(
                isinstance(c,dict) and c.get('step')==gap['step']
                and isinstance(c.get('question_quote'),str)
                and 0<len(c['question_quote'])<=240 and c['question_quote'] in question
                and isinstance(c.get('interpretation'),str) and c['interpretation'].strip()
                for c in coverage
            ) for gap in relation_gaps)
        if value['decision']=='uncertain':return True
        if not 1<=len(value['issues'])<=2:return False
        return all(
            isinstance(i,dict)
            and i.get('kind') in ('missing_constraint','scope','direction','join','output')
            and isinstance(i.get('program_steps'),list) and i['program_steps']
            and all(type(n) is int and n in nodes for n in i['program_steps'])
            and isinstance(i.get('question_quote'),str)
            and 0<len(i['question_quote'])<=240 and i['question_quote'] in question
            and all(isinstance(i.get(k),str) and i[k].strip()
                    for k in ('program_claim','question_claim','reason'))
            for i in value['issues']
        )
    audit=[]
    def call(text):
        from .backend import _LLMRole
        if isinstance(role,_LLMRole):
            result=role.check_question(text,audit=audit)
        else:
            result=(getattr(role,'check_question',None) or getattr(role,'review'))(text)
        audit.append({'stage':'semantic_verdict','verdict':result})
        return result
    try:
        if verdict is None:
            verdict=call(prompt)
            # A negative is a hypothesis, not sufficient proof. Re-read the
            # same frozen graph and full question with the concrete objection.
            # This does not vote or auto-approve matching canonical text.
            if not valid_verdict(verdict) or verdict.get('decision')=='mismatch':
                verdict=call(
                    'Re-audit this input. The previous review below may have misread '
                    'a branch, operation order, repeated type or an explicit condition. '
                    'Verify its claims directly against the full question and numbered steps. '
                    'Withdraw unsupported objections; retain any real discrepancy. '
                    'Return the same decision/issues schema. If equivalence remains ambiguous, '
                    'return uncertain. Do not approve merely because canonical wording matches.\n'
                    'Previous review: '+json.dumps(verdict,ensure_ascii=False)+'\n'+prompt
                )
            if valid_verdict(verdict) and verdict['decision']=='eligible' and relation_gaps:
                coverage_required=True
                verdict=call(
                    'Focused REQUIRED-HOP audit. These relation names do not occur literally '
                    'in the accepted question. They may be faithfully paraphrased, or actually '
                    'missing. Check EACH traversal and its owner/direction against the full '
                    'question. A publisher hop cannot stand in for a separate represented-by '
                    'hop; naming the destination type does not express the connecting relation. '
                    'For eligible, additionally return "relation_coverage":'
                    '[{"step":1,"question_quote":"exact short span expressing this hop",'
                    '"interpretation":"how this wording expresses the specific traversal"}]. '
                    'Cover every listed step. Do not reuse a span for unrelated hops. '
                    'If a necessary hop is absent, return mismatch with a concrete issue; '
                    'if wording is ambiguous, return uncertain. Never infer an unstated '
                    'relation from a destination type. FIND_ALL initializes all entities; '
                    'repeated identical type filters are redundant. '
                    'Return JSON ONLY: {"decision":"eligible"|"mismatch"|"uncertain",'
                    '"issues":[{"kind":"missing_constraint"|"scope"|"direction",'
                    '"program_steps":[1],"question_quote":"exact span <=240 characters",'
                    '"program_claim":"required traversal","question_claim":"expressed or omitted hop",'
                    '"reason":"specific discrepancy"}],'
                    '"relation_coverage":[{"step":1,"question_quote":"exact span <=240 characters",'
                    '"interpretation":"how the specific hop is expressed"}]}. '
                    'For eligible, issues=[] and relation_coverage must cover EVERY required hop. '
                    'For mismatch, issues must identify at least one concrete discrepancy.\n'
                    +json.dumps({
                        'accepted_question':question,
                        'required_hops':relation_gaps,
                        'canonical_steps':[{
                            'step':nid,'action':op['op'],'arguments':op['inputs'],
                            'input_steps':op['dependencies'],
                        } for nid,op in sorted(nodes.items())],
                        'certified_public_projections':gold.get('public_projections',[]),
                    },ensure_ascii=False,separators=(',',':'))
                )
    except Exception as exc:
        return {'eligible':False,'reviewed':False,'decision':'review_error',
                'method':'semantic_review_unavailable','issues':[str(exc)],
                'review_audit':audit,'backend_diagnostic':getattr(exc,'diagnostic',None),**fingerprint}
    valid=valid_verdict(verdict)
    if valid and cache_path and not cache_hit:
        target=Path(cache_path);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('a',encoding='utf-8') as handle:handle.write(json.dumps({'cache_key':cache_key,'verdict':verdict},ensure_ascii=False)+'\n')
    if valid and verdict['decision']=='eligible' and verdict['issues']==[]:
        return {'eligible':True,'reviewed':True,'decision':'eligible','method':'llm_semantic_screen','cache_hit':cache_hit,'issues':[],'review_audit':audit,**fingerprint}
    issues=[]
    for issue in verdict.get('issues',[]) if isinstance(verdict,dict) else []:
        if isinstance(issue,dict) and issue.get('question_quote') and issue['question_quote'] in question:
            issues.append(issue)
    return {'eligible':False,'reviewed':bool(valid),
            'decision':verdict['decision'] if valid else 'review_error',
            'method':'llm_semantic_screen','cache_hit':cache_hit,
            'issues':issues or ['equivalence_not_established'],
            'raw_verdict':verdict,'review_audit':audit,**fingerprint}
