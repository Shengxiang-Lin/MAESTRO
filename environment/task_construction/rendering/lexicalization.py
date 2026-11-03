"""Leakage-resistant deterministic lexicalization for synthetic literal values."""
from __future__ import annotations
import math,re
from decimal import Decimal
from typing import Any,Mapping,Sequence
from ..utils.determinism import stable_int as _stable_int
_PLACEHOLDER_RE=re.compile(r"^(?:value|val|string|text|label|literal|token|item|name)[_-]?\d+(?:[_-]?(?:alt|variant|distractor)\d*)?$",re.IGNORECASE);_CONTEXT_PLACEHOLDER_RE=re.compile(r"^context(?:[_-]\d+)+(?:[_-]?(?:alt|variant|distractor)\d*)?$",re.IGNORECASE);_ALT_MARKER_RE=re.compile(r"(?:^|[_-])alt(?:\d*)?(?:$|[_-])",re.IGNORECASE);_NUMBER_RE=re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$");_DATE_RE=re.compile(r"^\d{4}-\d{2}-\d{2}$")
def is_synthetic_placeholder(value):
    if not isinstance(value,str):return False
    text=value.strip();return bool(_PLACEHOLDER_RE.match(text) or _CONTEXT_PLACEHOLDER_RE.match(text) or _ALT_MARKER_RE.search(text))
def _predicate_key(predicate):
    if not predicate:return "value"
    text=str(predicate).split(":",1)[-1];text=re.sub(r"(?<=[a-z0-9])(?=[A-Z])"," ",text);return re.sub(r"[_-]+"," ",text).strip().lower()
def _predicate_family(predicate):
    key=_predicate_key(predicate)
    if "address" in key:return "address"
    if "slogan" in key or "motto" in key:return "slogan"
    if "sport number" in key or "jersey number" in key:return "sport_number"
    if any(t in key for t in ("code","identifier","icao","lei","isbn","issn","gtin")):return "code"
    if any(t in key for t in ("geo","region","area","location")):return "region"
    if any(t in key for t in ("logo","emblem","symbol")):return "emblem"
    if "student" in key and "count" in key:return "student_count"
    if "population" in key:return "population"
    if "age" in key:return "age"
    if "height" in key:return "height"
    if "mass" in key or "weight" in key:return "mass"
    if "duration" in key or "runtime" in key:return "duration"
    if "date" in key or "year" in key:return "temporal"
    return "generic"
def _as_number(value):
    if isinstance(value,bool):return None
    if isinstance(value,(int,float)):return float(value) if math.isfinite(float(value)) else None
    text=str(value).strip()
    if not _NUMBER_RE.match(text):return None
    try:number=float(text)
    except ValueError:return None
    return number if math.isfinite(number) else None
def _format_number(value,*,grouped=False):
    rounded=round(value,1)
    if abs(rounded-round(rounded))<1e-9:return f"{int(round(rounded)):,}" if grouped else str(int(round(rounded)))
    return f"{rounded:,.1f}" if grouped else f"{rounded:.1f}"
def _numeric_family_surface(number,family):
    if family=="age":return _format_number(18.+.10*number)
    if family=="height":return _format_number(150.+.25*number)+" cm"
    if family=="mass":return _format_number(50.+.25*number)+" kg"
    if family=="duration":return _format_number(30.+.50*number)+" min"
    if family=="population":return _format_number(50_000.+1_000.*number,grouped=True)
    if family=="student_count":return _format_number(2_000.+100.*number,grouped=True)
    return None
def _semantic_profile_surface(number,profile):
    family=str(profile.get("family","generic"));grouped=family in {"population","count","money"};exact=Decimal(str(number));surface=format(exact,",f" if grouped else "f")
    if "." in surface:surface=surface.rstrip("0").rstrip(".")
    unit=profile.get("canonical_unit")
    if unit:return f"{surface}%" if str(unit)=="%" else f"{surface} {unit}"
    return surface
def _looks_temporal(value,predicate,object_kind):return object_kind in {"date","year"} or bool(_DATE_RE.match(str(value).strip())) or _predicate_family(predicate)=="temporal"
_STREETS=("Cedar Lane","Harbor Road","Willow Street","Orchard Avenue","Summit Way","Lantern Court","Meadow Drive","Granite Terrace","Juniper Row","Riverside Walk","Maple Crescent","Beacon Street");_SLOGAN_LEFT=("Forward","Ideas","Built","Better","Beyond","Progress","Together","Future","Open","Driven");_SLOGAN_RIGHT=("Together","in Motion","for Tomorrow","by Design","with Purpose","without Limits","through Insight","for Everyone");_REGIONS=("North Basin","West Ridge","Lake District","River Quarter","Cedar Valley","Harbor Plain","Summit Region","East Terrace","Willow Corridor","Granite Coast");_EMBLEMS=("crescent emblem","silver compass mark","three-line crest","blue arch symbol","geometric leaf mark","four-point star emblem","interlocking ring symbol","mountain-line crest");_LABEL_LEFT=("Aster","Beacon","Cedar","Delta","Harbor","Juniper","Lumen","Meridian","Orchid","Pioneer","Summit","Willow");_LABEL_RIGHT=("North","Field","Point","Bridge","Circle","Gate","Line","Works","House","Archive","Project","Series")
class LiteralLexicalizer:
    def __init__(self,*,seed=17,contexts=None,profiles=None):
        self.seed=int(seed);self._contexts={str(raw):tuple((str(kind),str(predicate)) for kind,predicate in values) for raw,values in (contexts or {}).items()};self._profiles={str(predicate):dict(profile) for predicate,profile in (profiles or {}).items() if isinstance(profile,Mapping)};self._cache={};self._used=set()
    def _family_for(self,raw,predicate):
        if predicate and str(predicate) in self._profiles:
            family=str(self._profiles[str(predicate)].get("family",""))
            if family=="identifier":return "code"
            if family:return family
        predicates=[pred for _,pred in self._contexts.get(raw,())]
        if predicate:predicates.append(str(predicate))
        families={_predicate_family(pred) for pred in predicates if pred};return next(iter(families)) if len(families)==1 else "generic"
    def _candidate(self,raw,family,attempt):
        number=_stable_int(self.seed,raw,family,attempt)
        if family=="address":return f"{10+((number//len(_STREETS))%890)} {_STREETS[number%len(_STREETS)]}"
        if family=="slogan":return f"{_SLOGAN_LEFT[number%len(_SLOGAN_LEFT)]} {_SLOGAN_RIGHT[(number//len(_SLOGAN_LEFT))%len(_SLOGAN_RIGHT)]}"
        if family=="sport_number":return str(1+(number%98))
        if family in {"code","identifier"}:
            alphabet="ABCDEFGHJKLMNPQRSTUVWXYZ";letters="".join(alphabet[(number>>(shift*5))%len(alphabet)] for shift in range(3));return f"{letters}-{100+((number>>17)%900)}"
        if family=="region":return _REGIONS[number%len(_REGIONS)]
        if family=="emblem":return _EMBLEMS[number%len(_EMBLEMS)]
        return f"{_LABEL_LEFT[number%len(_LABEL_LEFT)]} {_LABEL_RIGHT[(number//len(_LABEL_LEFT))%len(_LABEL_RIGHT)]} {10+((number>>19)%90)}"
    def render(self,value,*,predicate=None,object_kind=None):
        if isinstance(value,bool):return "true" if value else "false"
        raw=str(value);number=_as_number(value);numeric_kind=object_kind!="string" and (object_kind=="number" or isinstance(value,(int,float)) or number is not None);profile=self._profiles.get(str(predicate)) if predicate else None
        if numeric_kind and number is not None and profile is not None:
            if object_kind in {"year","date"}:return raw
            return _semantic_profile_surface(value,profile)
        if raw in self._cache:return self._cache[raw]
        family=self._family_for(raw,predicate)
        if numeric_kind:
            if _looks_temporal(value,predicate,object_kind):return raw
            if family!="generic" and number is not None:
                surface=_numeric_family_surface(number,family)
                if surface is not None:self._cache[raw]=surface;return surface
            return raw
        if object_kind in {"year","date"}:return raw
        if not is_synthetic_placeholder(raw):return raw
        for attempt in range(64):
            candidate=self._candidate(raw,family,attempt)
            if candidate not in self._used:self._used.add(candidate);self._cache[raw]=candidate;return candidate
        candidate=f"Record {_stable_int(self.seed,raw)%100000:05d}";self._used.add(candidate);self._cache[raw]=candidate;return candidate
    def mapping(self):return dict(sorted(self._cache.items()))
