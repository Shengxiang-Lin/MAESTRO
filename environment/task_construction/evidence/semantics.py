"""Semantic classification and bundling helpers for synthetic evidence."""
from __future__ import annotations
import random
from collections import defaultdict
from typing import Any,Mapping,Sequence
from ..world import Fact
class EvidenceAllocationError(RuntimeError):pass
def _statement_key(fact:Fact):return fact.subject,fact.predicate,fact.object,fact.object_kind
def _predicate_semantics(predicate:str,predicate_catalog:Mapping[str,Any]|None):
    if not predicate_catalog:return None,None
    info=predicate_catalog.get("predicates",{}).get(predicate,{});schema=info.get("schema",{});value=schema.get("max_count")
    try:max_count=int(value) if value is not None else None
    except (TypeError,ValueError):max_count=None
    return info.get("kind"),max_count
def classify_noncritical_facts(facts:Sequence[Fact],critical_fact_ids:Sequence[str],*,predicate_catalog:Mapping[str,Any]|None=None)->dict[str,list[Fact]]:
    critical_set=set(critical_fact_ids);critical=[fact for fact in facts if fact.fact_id in critical_set];critical_keys={_statement_key(fact) for fact in critical};critical_subject_predicate={(fact.subject,fact.predicate) for fact in critical};critical_predicates={fact.predicate for fact in critical};critical_entities={fact.subject for fact in critical}
    for fact in critical:
        if fact.object_kind=="entity":critical_entities.add(str(fact.object))
    seen_noncritical=set();buckets=defaultdict(list)
    for fact in facts:
        if fact.fact_id in critical_set:continue
        key=_statement_key(fact)
        if key in critical_keys:role="corroborating"
        elif key in seen_noncritical:role="redundant"
        elif (fact.subject,fact.predicate) in critical_subject_predicate:
            kind,max_count=_predicate_semantics(fact.predicate,predicate_catalog);role="conflicting" if (fact.object_kind!="entity" or kind=="attribute" or max_count==1) else "competing"
        else:
            touches=fact.subject in critical_entities
            if fact.object_kind=="entity":touches=touches or str(fact.object) in critical_entities
            role="near_miss" if touches or fact.predicate in critical_predicates else "irrelevant"
        seen_noncritical.add(key);buckets[role].append(fact)
    return dict(buckets)
_classify_noncritical_facts=classify_noncritical_facts
def _mutation_groups(records):
    result={}
    for mutation in records or ():
        added=tuple(str(fid) for fid in mutation.get("added_fact_ids",()) if fid)
        if len(added)<2:continue
        for fact_id in added:result[fact_id]=added
    return result
def _fact_entities(fact):
    result={fact.subject}
    if fact.object_kind=="entity":result.add(str(fact.object))
    return result
def _grow_bundle(seed,*,unused,role_by_fact,mutation_groups,max_bundle_size,rng):
    bundle=[seed];unused.pop(seed.fact_id,None)
    if max_bundle_size<=1:return bundle
    candidates=[];seen={seed.fact_id}
    for fact_id in mutation_groups.get(seed.fact_id,()):
        if fact_id in unused and fact_id not in seen:candidates.append(unused[fact_id]);seen.add(fact_id)
    seed_entities=_fact_entities(seed);local=[fact for fact in unused.values() if fact.fact_id not in seen and seed_entities&_fact_entities(fact)];rng.shuffle(local);candidates.extend(local);seen.update(f.fact_id for f in local);same_role=[fact for fact in unused.values() if fact.fact_id not in seen and role_by_fact.get(fact.fact_id)==role_by_fact.get(seed.fact_id)];rng.shuffle(same_role);candidates.extend(same_role)
    for fact in candidates:
        if len(bundle)>=max_bundle_size:break
        if fact.fact_id in unused:bundle.append(fact);unused.pop(fact.fact_id,None)
    return bundle
def select_role_bundles(buckets:Mapping[str,Sequence[Fact]],*,mutation_records:Sequence[Mapping[str,Any]]|None,limit:int,rng:random.Random,max_bundle_size:int)->list[dict[str,Any]]:
    if limit<=0:return []
    role_by_fact={fact.fact_id:role for role,facts in buckets.items() for fact in facts};unused={fact.fact_id:fact for facts in buckets.values() for fact in facts};groups=_mutation_groups(mutation_records);diagnostic=[role for role in ("conflicting","competing","near_miss") if buckets.get(role)];background=[role for role in ("irrelevant","corroborating","redundant") if buckets.get(role)];rng.shuffle(diagnostic);rng.shuffle(background);order=[]
    if diagnostic:order.append(diagnostic.pop())
    if background and len(order)<limit:order.append(background.pop())
    rest=diagnostic+background;rng.shuffle(rest);order.extend(rest);bundles=[]
    def add_seed(seed,role):bundles.append({"role":role,"facts":_grow_bundle(seed,unused=unused,role_by_fact=role_by_fact,mutation_groups=groups,max_bundle_size=max_bundle_size,rng=rng),"origin":"hardener_mutation_bundle" if seed.fact_id in groups else "local_noncritical_bundle"})
    for role in order:
        if len(bundles)>=limit:break
        choices=[fact for fact in buckets.get(role,()) if fact.fact_id in unused]
        if choices:rng.shuffle(choices);add_seed(choices[0],role)
    leftovers=list(unused.values());rng.shuffle(leftovers)
    for seed in leftovers:
        if len(bundles)>=limit:break
        if seed.fact_id in unused:add_seed(seed,role_by_fact.get(seed.fact_id,"near_miss"))
    return bundles
_select_role_bundles=select_role_bundles
