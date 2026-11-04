"""Deterministic summary guards plus an evidence-scoped LLM review prompt."""
from __future__ import annotations
import json
import re


def text_issues(text,claims=None,*,pending_ids=None,allowed_ids=None,question=""):
    if not isinstance(text,str) or not text.strip():return ['empty_or_non_string_summary']
    issues=[]
    if len(text)>3000:issues.append('summary_not_compressed')
    if re.search(r'(?:read|observed|inspected|accessed).{0,45}(?:unavailable|unseen|future)',text,re.I):
        issues.append('claims_unavailable_observation')
    if pending_ids and re.search(r'\b(?:all|every)\s+(?:required\s+)?(?:tasks?|subtasks?|work)\s+(?:(?:is|are|have been|has been)\s+)?(?:done|complete|completed|finished)\b',text,re.I):
        issues.append('premature_completion_in_summary')
    if claims is not None:
        evidence=json.dumps({"claims":claims,"public_question_constraints":question,"observed_resource_count":len(allowed_ids or ()),"pending_task_count":len(pending_ids or ())},ensure_ascii=False)
        tokens=lambda s:set(re.findall(r'(?<!\w)[+-]?\d+(?:[.,]\d+)*(?!\w)',s))
        # Numeric task/resource identifiers are not factual quantities.
        clean=re.sub(r'(?:source_|subtask:source_|discover:source_)\d+','',text)
        known_references={int(v) for v in re.findall(r'(?:source_|(?:Source|Reference)\s+)(\d+)',evidence,re.I)}
        known_references.update(int(v) for v in re.findall(r'source_(\d+)',json.dumps(list(allowed_ids or ()))))
        def resource_list(match):
            ids={int(v) for v in re.findall(r'\d+',match.group(1))}
            if ids.issubset(known_references):return 'known resources'
            issues.append('unknown_resource_reference')
            return match.group(0)
        clean=re.sub(r'\b(?:sources?|references?)\s+(\d{1,3}(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+)\d{1,3})*)\b',resource_list,clean,flags=re.I)
        supported=tokens(evidence)
        if tokens(clean)-supported:issues.append('unsupported_numeric_claim_in_summary')
    if allowed_ids is not None:
        mentioned=set(re.findall(r'source_\d+',text))
        pending_sources=set(re.findall(r'source_\d+',json.dumps(list(pending_ids or ()))))
        if mentioned-set(allowed_ids)-pending_sources:issues.append('unknown_source_in_summary')
    return sorted(set(issues))


def review_prompt(text,claims,*,question='',pending_ids=(),allowed_ids=(),context=None):
    return (
        'Check a compressed execution/answer summary against the supplied evidence and state. verified_claims and source_observation are AUTHORITATIVE TRUE OBSERVATIONS. Do not challenge their provenance or require external corroboration. A statement directly matching them is supported. '
        'Reject contradictions, invented factual values/entities, claims to have read unavailable sources, '
        'or claims that pending work is already complete. Paraphrases and task objectives are allowed. '
        'Evaluate only the current subtask or summary, not the whole original question. Partial work is expected. Do not demand completion of pending tasks or repetition of all facts. Public question constraints can be mentioned as search/filter objectives without claiming they were newly observed. A verified resource descriptor proves resource existence, but not its contents. Execution context authorizes the stated inspection modality. Do not solve the original question again. '
        'Resource descriptors establish only discovery, not inspection. Use execution_context.completed_results.kind to distinguish discover from inspect and associate facts with their actual owning subtask; do not attach a flat claim to a neighboring descriptor. Only inspected_resource_ids (or the current inspect task) establish read contents. Scoped observations are complete, so a summary may say an attribute is not provided by that source, without saying it is globally absent. Return JSON only: {"accept":true|false,"issues":["short reasons"]}. '
        'At most three issues; no repair, no analysis.\n'+json.dumps({
            'execution_context':context or {},'question':question,'summary':text,'verified_claims':claims,
            'pending_task_ids':list(pending_ids),'known_resource_ids':list(allowed_ids)},
            ensure_ascii=False,separators=(',',':')))


def semantic_review(role,text,claims,*,question='',pending_ids=(),allowed_ids=(),context=None):
    issues=text_issues(text,claims,pending_ids=pending_ids,allowed_ids=allowed_ids,question=question)
    if issues:return False,issues
    review=role.review(review_prompt(text,claims,question=question,
                                   pending_ids=pending_ids,allowed_ids=allowed_ids,context=context))
    if review.get('accept') is not True:
        return False,['summary_semantic_review:'+str(x) for x in review.get('issues',[])][:3] or ['summary_semantic_review_rejected']
    return True,[]
