"""Live batch-diversity guidance for diversity-aware semantic binding."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable,Mapping
@dataclass(frozen=True)
class BindingDiversityGuidance:
    predicate_usage:Mapping[str,int];type_pair_usage:Mapping[str,int];predicate_cap:int;type_pair_cap:int;accepted_count:int
_enabled=False;_predicate_usage=None;_type_pair_usage=None;_topology_usage=None;_cap_resolver=None
def enable_binding_diversity_guidance():
    global _enabled,_predicate_usage,_type_pair_usage,_topology_usage,_cap_resolver
    _enabled=True;_predicate_usage=_type_pair_usage=_topology_usage=_cap_resolver=None
def disable_binding_diversity_guidance():
    global _enabled,_predicate_usage,_type_pair_usage,_topology_usage,_cap_resolver
    _enabled=False;_predicate_usage=_type_pair_usage=_topology_usage=_cap_resolver=None
def publish_binding_diversity_guidance(*,predicate_usage,type_pair_usage,topology_usage,cap_resolver):
    if not _enabled:return
    global _predicate_usage,_type_pair_usage,_topology_usage,_cap_resolver
    _predicate_usage=predicate_usage;_type_pair_usage=type_pair_usage;_topology_usage=topology_usage;_cap_resolver=cap_resolver
def current_binding_diversity_guidance():
    if not _enabled or _predicate_usage is None or _type_pair_usage is None or _topology_usage is None or _cap_resolver is None:return None
    accepted_count=sum(int(v) for v in _topology_usage.values());caps=_cap_resolver(accepted_count);return BindingDiversityGuidance(_predicate_usage,_type_pair_usage,max(1,int(caps["predicate"])),max(1,int(caps["type_pair"])),accepted_count)
__all__=["BindingDiversityGuidance","current_binding_diversity_guidance","disable_binding_diversity_guidance","enable_binding_diversity_guidance","publish_binding_diversity_guidance"]
