"""Deterministic natural public names for synthetic world entities.

The symbolic executor keeps opaque entity ids (e0001, ...). This module only
controls the public surface. Names are synthetic compositions, never copied
from YAGO instance labels, so relation/type realism can be retained without
making a benchmark answer directly searchable on the web.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_PERSON_FIRST=("Adrian","Amelia","Clara","Dorian","Elena","Elias","Iris","Jonas","Leona","Mara","Milo","Nadia","Noah","Rhea","Rowan","Selene","Theo","Tobias","Vera","Yara","Caleb","Lena","Mira","Soren")
_PERSON_LAST=("Alder","Bellamy","Corwin","Darrow","Ellery","Farrow","Hale","Kessler","Langley","Marwick","Nolan","Ormond","Pryce","Quill","Renn","Sayer","Sterling","Vale","Voss","Whitlock","Winslow","Wren","Yarrow","Zeller")
_ORG_PREFIX=("Alderstone","Arbor","Bluehaven","Cedarbridge","Clearwater","Evermont","Fairwind","Granite","Harborline","Hawthorne","Ironwood","Juniper","Kingswell","Meridian","Northbridge","Northstar","Oakridge","Redwood","Silvercrest","Stonegate","Westhaven","Westmere","Willowbrook","Wintermere")
_ORG_SUFFIX=("Institute","Foundation","Group","Laboratory","Research Center","Alliance","Consortium","Trust","Society","Holdings","Network","Council")
_PLACE_PREFIX=("Alder","Ash","Cedar","Clear","Dun","Fair","Fox","Glen","Harbor","Hawthorn","Lake","Maple","North","Oak","Pine","Raven","Red","Silver","Stone","West","Willow","Winter","Wren","York")
_PLACE_SUFFIX=("borough","bridge","crest","dale","field","ford","haven","mere","mont","port","ridge","stead","vale","view","wick","wood")
_EVENT_HEAD=("Aurora","Cedarbridge","Harborlight","Meridian","Northbridge","Northstar","Redwood","Silvercrest","Stonegate","Westhaven","Willowbrook","Wintermere")
_EVENT_TAIL=("Forum","Summit","Festival","Congress","Symposium","Challenge","Expo","Conference","Games","Assembly","Showcase","Convention")
_AWARD_HEAD=("Aurora","Beacon","Cedar","Horizon","Laurel","Meridian","Northstar","Pioneer","Silvercrest","Stonegate","Westhaven","Willow")
_AWARD_TAIL=("Prize","Medal","Award","Honor","Citation","Fellowship")
_WORK_HEAD=("Echoes","Shadows","Letters","Chronicle","Atlas","Songs","Tales","Notes","Fragments","Voyage","Signals","Memory")
_WORK_TAIL=("of Northbridge","from Cedar Bay","of the Last Harbor","under Silver Skies","of Westmere","beyond the Meridian","from Wintermere","of the Quiet Coast")
_FICTION_FIRST=("Aerin","Cael","Elara","Ilyra","Kael","Liora","Maren","Neris","Orin","Riven","Seren","Tarin","Vela","Ysra","Zorin","Avel")
_FICTION_LAST=("Ashfall","Briar","Dusk","Evermere","Frost","Grey","Hollow","Lark","Morrow","Quill","Rook","Thorne","Vale","Venn","Wilder","Wren")
_TAXON_GENUS=("Aurelia","Borealis","Caelora","Dendria","Erythra","Floravia","Luminella","Marivora","Nemoris","Orinthia","Silvania","Venturia")
_TAXON_SPECIES=("alba","aurata","borealis","cerulea","montana","nivalis","orientalis","rubra","silvatica","viridis","maritima","minor")
_AIR_HEAD=("Bluecrest","Cedar","Harbor","Meridian","Northstar","Silverwing","Skyhaven","Stonebridge","Westwind","Willow")
_AIR_TAIL=("Air","Airways","Aviation","Express")
_THING_HEAD=("Aster","Beacon","Cedar","Harbor","Meridian","Northstar","Orion","Pioneer","Silver","Stone","Vertex","Willow")
_THING_TAIL=("Module","Device","Instrument","Platform","Unit","System","Prototype")
_BASE_SALT_ATTEMPTS=512;_MAX_DISTINCT_BASES=96

def _kind_key(kind):return re.sub(r"[^a-z0-9]+"," ",str(kind).lower()).strip()
def _pick(values,token,offset):
    start=(offset*4)%(len(token)-3);return values[int.from_bytes(token[start:start+4],"big")%len(values)]
def _place_qualifier(index):
    size=len(_PLACE_PREFIX)*len(_PLACE_SUFFIX);value=index%size;return f"{_PLACE_PREFIX[value//len(_PLACE_SUFFIX)]}{_PLACE_SUFFIX[value%len(_PLACE_SUFFIX)]}"
def _qualified_surface(base,kind,qualifier):
    key=_kind_key(kind)
    if "person" in key or key in {"human","agent"} or "fictional" in key or "character" in key:return f"{base} of {qualifier}"
    if "event" in key or "award" in key or "prize" in key:return f"{base} at {qualifier}"
    if any(word in key for word in ("tvseries","tv series","creative work","book","film","movie","song")):return f"{base}, {qualifier} edition"
    if "taxon" in key or "species" in key:return f"{base} from {qualifier}"
    if "organization" in key or any(word in key for word in ("company","corporation","institution","airline")):return f"{base}, {qualifier}"
    return f"{base} of {qualifier}"
@dataclass(frozen=True)
class EntityNameLexicalizer:
    seed:int=17
    def _digest(self,entity_id,kind,salt=0):return hashlib.sha256(f"{self.seed}:{kind}:{entity_id}:{salt}".encode()).digest()
    def render(self,entity_id,kind,*,salt=0):
        key=_kind_key(kind);token=self._digest(str(entity_id),key,salt)
        if "fictional" in key or "character" in key:return f"{_pick(_FICTION_FIRST,token,0)} {_pick(_FICTION_LAST,token,1)}"
        if "person" in key or key in {"human","agent"}:return f"{_pick(_PERSON_FIRST,token,0)} {_pick(_PERSON_LAST,token,1)}"
        if "administrative area" in key or key in {"place","city","town","region","country"}:return _pick(_PLACE_PREFIX,token,0)+_pick(_PLACE_SUFFIX,token,1)
        if "airline" in key:return f"{_pick(_AIR_HEAD,token,0)} {_pick(_AIR_TAIL,token,1)}"
        if "organization" in key or any(word in key for word in ("company","corporation","institution")):return f"{_pick(_ORG_PREFIX,token,0)} {_pick(_ORG_SUFFIX,token,1)}"
        if "award" in key or "prize" in key:return f"{_pick(_AWARD_HEAD,token,0)} {_pick(_AWARD_TAIL,token,1)}"
        if "event" in key:return f"{_pick(_EVENT_HEAD,token,0)} {_pick(_EVENT_TAIL,token,1)}"
        if "taxon" in key or "species" in key:return f"{_pick(_TAXON_GENUS,token,0)} {_pick(_TAXON_SPECIES,token,1)}"
        if any(word in key for word in ("tvseries","tv series","creative work","book","film","movie","song")):return f"{_pick(_WORK_HEAD,token,0)} {_pick(_WORK_TAIL,token,1)}"
        if key=="thing" or any(word in key for word in ("product","device","artifact","object")):return f"{_pick(_THING_HEAD,token,0)} {_pick(_THING_TAIL,token,1)}"
        return f"{_pick(_ORG_PREFIX,token,0)} {str(kind).strip() or 'Entity'}"
    def unique(self,entity_id,kind,used):
        base_candidates=[];seen_bases=set()
        for salt in range(_BASE_SALT_ATTEMPTS):
            candidate=self.render(entity_id,kind,salt=salt);normalized=candidate.casefold()
            if normalized not in seen_bases:
                seen_bases.add(normalized);base_candidates.append(candidate)
                if normalized not in used:used.add(normalized);return candidate
                if len(base_candidates)>=_MAX_DISTINCT_BASES:break
        if not base_candidates:base_candidates.append(str(kind).strip() or "Entity")
        qualifier_count=len(_PLACE_PREFIX)*len(_PLACE_SUFFIX);token=self._digest(str(entity_id),_kind_key(kind),salt=1_000_003);offset=int.from_bytes(token[:4],"big")%qualifier_count
        for base in base_candidates:
            for step in range(qualifier_count):
                candidate=_qualified_surface(base,kind,_place_qualifier(offset+step));normalized=candidate.casefold()
                if normalized not in used:used.add(normalized);return candidate
        raise RuntimeError(f"unable to allocate unique public name for {entity_id}; exhausted {len(base_candidates)*qualifier_count} qualified surfaces")
