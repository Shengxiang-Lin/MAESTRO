"""Deterministic surface names for synthetic entities, predicates and values."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..utils.determinism import stable_int
from .entity_lexicalization import EntityNameLexicalizer
from .lexicalization import LiteralLexicalizer

_NAMESPACE_PREFIXES = ("schema:", "yago:", "rdf:", "rdfs:", "owl:", "xsd:")
_SYNTHETIC_CONTEXT_PREDICATE_RE = re.compile(r"^context(?:[_\-\s]*)descriptor(?:[_\-\s]*\d+)?$", re.IGNORECASE)
_CONTEXT_FIELD_LABELS = ("catalog note", "archive classification", "record annotation", "reference category", "index note", "catalog classification")

def strip_namespace(value: str) -> str:
    text = str(value)
    for prefix in _NAMESPACE_PREFIXES:
        if text.startswith(prefix): return text[len(prefix):]
    return text.split(":", 1)[-1] if ":" in text else text

def split_identifier(value: str) -> str:
    text = strip_namespace(value).replace("_", " ").replace("-", " ")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def type_label(value: str) -> str:
    text = strip_namespace(value); text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text); text = re.sub(r"[_-]+", " ", text).strip(); return text or "Entity"

def predicate_label(value: str) -> str:
    raw = strip_namespace(value)
    if _SYNTHETIC_CONTEXT_PREDICATE_RE.fullmatch(raw): return _CONTEXT_FIELD_LABELS[stable_int(17, raw) % len(_CONTEXT_FIELD_LABELS)]
    return split_identifier(value) or "relation"

def comparison_phrase(value: str) -> str:
    token = str(value).strip().lower(); mapping = {"=":"equal to","==":"equal to","<cmp:=>":"equal to","<cmp:==>":"equal to","!=":"different from","<cmp:!=>":"different from",">":"greater than","greater":"greater than","<cmp:>>":"greater than","<cmp:greater>":"greater than","<":"less than","less":"less than","<cmp:<>":"less than","<cmp:less>":"less than",">=":"at least","<cmp:>=>":"at least","<=":"at most","<cmp:<=>":"at most"}; return mapping.get(token, token)

def select_mode_phrase(value: str) -> str:
    return "smallest" if str(value).strip().lower() in {"min","smallest","less","<","<cmp:less>","<cmp:<>",} else "largest"

def literal_label(value: Any) -> str:
    return "true" if value is True else "false" if value is False else str(value)

class EntityAliasBook:
    def __init__(self, entities: Mapping[str,str], *, literal_contexts: Mapping[str,Sequence[tuple[str,str]]] | None=None, literal_profiles: Mapping[str,Mapping[str,Any]] | None=None, seed:int=17):
        self._aliases={}; lexicalizer=EntityNameLexicalizer(seed=seed); used=set()
        for entity_id in sorted(entities): self._aliases[entity_id]=lexicalizer.unique(entity_id, type_label(entities[entity_id]), used)
        self._literal_contexts={str(raw):tuple((str(kind),str(predicate)) for kind,predicate in contexts) for raw,contexts in (literal_contexts or {}).items()}
        self._literal_lexicalizer=LiteralLexicalizer(seed=seed, contexts=self._literal_contexts, profiles=literal_profiles)
    def is_entity(self,value): return str(value) in self._aliases
    def entity(self,entity_id): return self._aliases.get(str(entity_id),str(entity_id))
    def literal(self,value,*,object_kind=None,predicate=None):
        contexts=self._literal_contexts.get(str(value),()); matching_kind=next((kind for kind,pred in contexts if pred==predicate),None); resolved_kind=object_kind or matching_kind or (contexts[0][0] if contexts else None); return self._literal_lexicalizer.render(value,object_kind=resolved_kind,predicate=predicate)
    def value(self,value,object_kind=None,*,predicate=None): return self.entity(str(value)) if object_kind=="entity" else self.literal(value,object_kind=object_kind,predicate=predicate)
    def to_dict(self): return dict(self._aliases)
    def literal_surfaces(self): return self._literal_lexicalizer.mapping()
