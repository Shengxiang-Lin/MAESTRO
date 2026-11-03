"""Soft intra-program predicate diversity for semantic binding."""
from __future__ import annotations
from collections import Counter
from typing import Any
from .predicate_binder import PredicateBinder
_INSTALLED=False;_ORIGINAL_RELATION_POOL=PredicateBinder._guided_relation_pool;_ORIGINAL_ATTRIBUTE_POOL=PredicateBinder._guided_attribute_pool;_ORIGINAL_CHOOSE_RELATION=PredicateBinder.choose_relation;_ORIGINAL_CHOOSE_ATTRIBUTE=PredicateBinder.choose_attribute
def _usage(binder):
    current=getattr(binder,"_intra_program_predicate_usage",None)
    if current is None:current=Counter();setattr(binder,"_intra_program_predicate_usage",current)
    return current
def _copies(use_count):return max(1,4-min(3,int(use_count)))
def _relation_pool(self,feasible):
    pool=_ORIGINAL_RELATION_POOL(self,feasible);usage=_usage(self);expanded=[]
    for item in pool:expanded.extend([item]*_copies(usage[item[1].predicate]))
    return expanded
def _attribute_pool(self,feasible):
    pool=_ORIGINAL_ATTRIBUTE_POOL(self,feasible);usage=_usage(self);expanded=[]
    for item in pool:expanded.extend([item]*_copies(usage[item[1].predicate]))
    return expanded
def _choose_relation(self,*args:Any,**kwargs:Any):
    binding=_ORIGINAL_CHOOSE_RELATION(self,*args,**kwargs)
    if binding is not None:_usage(self)[binding.predicate]+=1
    return binding
def _choose_attribute(self,*args:Any,**kwargs:Any):
    binding=_ORIGINAL_CHOOSE_ATTRIBUTE(self,*args,**kwargs)
    if binding is not None:_usage(self)[binding.predicate]+=1
    return binding
def install_intra_program_predicate_diversity():
    global _INSTALLED
    if _INSTALLED:return
    PredicateBinder._guided_relation_pool=_relation_pool;PredicateBinder._guided_attribute_pool=_attribute_pool;PredicateBinder.choose_relation=_choose_relation;PredicateBinder.choose_attribute=_choose_attribute;_INSTALLED=True
__all__=["install_intra_program_predicate_diversity"]
