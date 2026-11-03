"""Type-conditioned numeric profiles from actual joined reservoir samples."""
from collections import Counter
from decimal import Decimal
import math
from ..binding.semantic_support import MIN_EMPIRICAL_TYPE_MASS
_UNITS={'metre':('length',1.,'m'),'meter':('length',1.,'m'),'kilometre':('length',1000.,'km'),'kilometer':('length',1000.,'km'),'centimetre':('length',.01,'cm'),'centimeter':('length',.01,'cm'),'millimetre':('length',.001,'mm'),'micrometre':('length',.000001,'um'),'decimetre':('length',.1,'dm'),'hectometre':('length',100.,'hm'),'foot_unit':('length',.3048,'ft'),'international_foot':('length',.3048,'ft'),'inch':('length',.0254,'in'),'international_inch':('length',.0254,'in'),'yard':('length',.9144,'yd'),'mile':('length',1609.344,'mi'),'kilogram':('mass',1.,'kg'),'gram':('mass',.001,'g'),'milligram':('mass',.000001,'mg'),'tonne':('mass',1000.,'t'),'megagram':('mass',1000.,'t'),'pound_mass':('mass',.45359237,'lb'),'ounce':('mass',.028349523125,'oz'),'dalton_unit':('molecular_mass',1.,'Da'),'kilodalton':('molecular_mass',1000.,'kDa'),'second':('duration',1.,'s'),'minute':('duration',60.,'min'),'hour':('duration',3600.,'h'),'day':('duration',86400.,'days'),'week':('duration',604800.,'weeks'),'year':('calendar_year',1.,'years'),'month':('calendar_month',1.,'months'),'percent':('percentage',1.,'%'),'percentage':('percentage',1.,'%')};_CANONICAL={'length':(1.,'m'),'mass':(1.,'kg'),'molecular_mass':(1.,'Da'),'scalar':(1.,None),'percentage':(1.,'%'),'calendar_year':(1.,'years'),'calendar_month':(1.,'months')}
def unit_spec(unit):
    if unit is None:return ('scalar',1.,None)
    return _UNITS.get(str(unit).split(':')[-1].casefold())
def convert_surface_value(value,source_unit,target_unit):
    if source_unit==target_unit:return float(value)
    by_surface={spec[2]:spec for spec in _UNITS.values()};by_surface[None]=('scalar',1.,None);source=by_surface.get(source_unit);target=by_surface.get(target_unit)
    if source is None or target is None or source[0]!=target[0]:raise ValueError('incompatible_or_unknown_units')
    return float(Decimal(str(value))*Decimal(str(source[1]))/Decimal(str(target[1])))
def _canonical(info,dimension):
    if dimension in _CANONICAL:return _CANONICAL[dimension]
    counts=Counter()
    for row in info['typed_literal_support']['samples']:
        spec=unit_spec(row.get('unit'))
        if spec and spec[0]==dimension:counts[(spec[1],spec[2])]+=1
    if not counts:return None
    factor,label=min(counts,key=lambda key:(-counts[key],key[0],key[1] or ''));return factor,label
def compile_typed_numeric_profile(predicate,info,entity_type,*,family):
    rows=[r for r in info['typed_literal_support'].get('samples',[]) if entity_type in r.get('supported_types',r.get('subject_types',[]))];subjects={r['subject'] for r in rows};base={'family':family,'canonical_unit':None,'minimum':None,'maximum':None,'integer_values':False,'confidence':0.,'safe':False}
    def reject(reason):return dict(base,evidence='typed-observed:'+reason+':'+str(len(subjects)))
    if len(subjects)<MIN_EMPIRICAL_TYPE_MASS:return reject('sparse-subject-support')
    if entity_type in {'schema:Thing','owl:Thing','rdfs:Resource'}:return reject('uninformative-root-type')
    specs=[unit_spec(r.get('unit')) for r in rows]
    if any(s is None for s in specs):return reject('unknown-unit')
    dimensions={s[0] for s in specs}
    if len(dimensions)!=1:return reject('incompatible-dimensions')
    dimension=next(iter(dimensions));expected={'height':{'length'},'distance':{'length'},'mass':{'mass','molecular_mass'},'duration':{'duration','calendar_year','calendar_month'}}.get(family)
    if expected is not None and dimension not in expected:return reject('predicate-dimension-conflict')
    canonical=_canonical(info,dimension)
    if canonical is None:return reject('unknown-canonical-unit')
    target_factor,label=canonical;values=[]
    for row,spec in zip(rows,specs):
        value=float(Decimal(str(row['value']))*Decimal(str(spec[1]))/Decimal(str(target_factor)))
        if not math.isfinite(value):return reject('nonfinite-value')
        values.append(value)
    values.sort();lo=values[max(0,int(len(values)*.05)-1)] if len(values)>=10 else values[0];hi=values[min(len(values)-1,int(len(values)*.95))] if len(values)>=10 else values[-1]
    if hi<=lo:return reject('degenerate-range')
    if dimension in {'length','mass','molecular_mass','duration','calendar_year','calendar_month'} and lo<0:return reject('negative-physical-value')
    result=dict(base,canonical_unit=label,minimum=lo,maximum=hi,integer_values=dimension=='scalar' and all(float(v).is_integer() for v in values),confidence=round(min(.95,.6+.02*len(subjects)),4),safe=True,evidence='typed-observed:unit-range:'+dimension+':'+str(len(subjects)))
    if dimension=='scalar' and family in {'height','mass','distance','duration','speed','temperature'}:return reject('physical-unit-unobserved')
    return result
def update_binding_profile(binding,profile):
    from dataclasses import replace
    return replace(binding,entity_type=profile.entity_type,semantic_family=profile.family,canonical_unit=profile.canonical_unit,value_min=profile.minimum,value_max=profile.maximum,integer_values=profile.integer_values,profile_confidence=profile.confidence,profile_evidence=profile.evidence)
def binding_profile_matches(binding,profile):
    return profile.safe and all(getattr(binding,a)==getattr(profile,b) for a,b in [('entity_type','entity_type'),('semantic_family','family'),('canonical_unit','canonical_unit'),('value_min','minimum'),('value_max','maximum'),('integer_values','integer_values'),('profile_evidence','evidence')])
def bound_attribute_metadata(bindings):
    grouped={};typed={}
    for b in bindings:
        row={'family':b.semantic_family,'canonical_unit':b.canonical_unit,'value_kind':b.value_kind,'minimum':b.value_min,'maximum':b.value_max,'integer_values':b.integer_values,'confidence':b.profile_confidence,'evidence':b.profile_evidence};grouped.setdefault(b.predicate,[]).append((b.entity_type,row))
        if b.profile_evidence.startswith('typed-observed:'):typed.setdefault(b.predicate,{})[b.entity_type]=row
    global_profiles={}
    for predicate,entries in grouped.items():
        if predicate not in typed:global_profiles[predicate]=entries[-1][1];continue
        units={r['canonical_unit'] for _,r in entries};kinds={r['value_kind'] for _,r in entries}
        if len(units)!=1 or len(kinds)!=1:raise ValueError('typed_attribute_unit_or_kind_conflict:'+predicate)
        row=dict(entries[0][1]);row['minimum']=min(r['minimum'] for _,r in entries);row['maximum']=max(r['maximum'] for _,r in entries);row['entity_types']=sorted({t for t,_ in entries});global_profiles[predicate]=row
    return global_profiles,typed
def typed_fact_audit(world,catalog,cache=None):
    from .profile import compile_attribute_profile
    cache={} if cache is None else cache;violations=[];checked=0;conversions=[]
    for fact in world.facts:
        info=catalog.get('predicates',{}).get(fact.predicate,{})
        if fact.object_kind!='number' or info.get('typed_literal_support',{}).get('version')!=1:continue
        checked+=1;entity_type=world.entities.get(fact.subject);key=(fact.predicate,entity_type)
        if key not in cache:cache[key]=compile_attribute_profile(fact.predicate,info,entity_type=entity_type,value_kind='number')
        expected=cache[key];frozen=world.metadata.get('attribute_semantics',{}).get(fact.predicate);reason=None
        if not expected.safe:reason='unsupported_typed_numeric_profile'
        elif frozen is None or 'canonical_unit' not in frozen:reason='missing_frozen_numeric_profile'
        elif isinstance(fact.object,bool):reason='typed_numeric_invalid_value'
        else:
            try:value=float(fact.object)
            except (TypeError,ValueError):reason='typed_numeric_invalid_value'
            else:
                if not math.isfinite(value):reason='typed_numeric_nonfinite_value'
                else:
                    try:normalized=convert_surface_value(value,frozen.get('canonical_unit'),expected.canonical_unit)
                    except ValueError:reason='typed_numeric_unit_mismatch'
                    else:
                        if frozen.get('canonical_unit')!=expected.canonical_unit:conversions.append({'fact_id':fact.fact_id,'from_unit':frozen.get('canonical_unit'),'to_unit':expected.canonical_unit,'normalized_value':normalized})
                        if normalized<expected.minimum or normalized>expected.maximum:reason='typed_numeric_outside_generation_range'
        if reason:violations.append({'fact_id':fact.fact_id,'entity_type':entity_type,'predicate':fact.predicate,'value':fact.object,'reason':reason,'expected_profile':expected.to_dict(),'frozen_unit':None if frozen is None else frozen.get('canonical_unit')})
    report={'version':1,'checked_facts':checked,'accepted':not violations,'violations':violations,'scope':'type_conditioned_generation_ranges_not_universal_physical_laws'}
    if conversions:report['unit_conversions']=conversions
    return report
