"""Interpretable task difficulty derived from coherent near-miss candidates."""
from __future__ import annotations
from dataclasses import dataclass
from statistics import mean
from typing import Any,Mapping,Sequence
@dataclass(frozen=True)
class CandidateEliminationProfile:
    near_miss_count:int;candidate_pool_lower_bound:int;distinct_broken_constraints:int;min_constraints_to_eliminate_near_misses:int;mean_matched_constraints:float;max_matched_constraints:int;max_support_degree:int;level:str;score:float
    def to_dict(self)->dict[str,Any]:return {"near_miss_count":self.near_miss_count,"candidate_pool_lower_bound":self.candidate_pool_lower_bound,"distinct_broken_constraints":self.distinct_broken_constraints,"min_constraints_to_eliminate_near_misses":self.min_constraints_to_eliminate_near_misses,"mean_matched_constraints":self.mean_matched_constraints,"max_matched_constraints":self.max_matched_constraints,"max_support_degree":self.max_support_degree,"level":self.level,"score":self.score}
def _answer_entity_count(answer):return len(answer) if isinstance(answer,(set,list,tuple)) else 1 if answer is not None else 0
def candidate_elimination_profile(mutations:Sequence[Mapping[str,Any]],answer:Any)->CandidateEliminationProfile:
    coherent=[m for m in mutations if m.get("kind")=="coherent_near_miss_subgraph"];broken={str(m.get("broken_from_fact")) for m in coherent if m.get("broken_from_fact") is not None};matched=[int(m.get("matched_constraints",0)) for m in coherent];support=[int(m.get("support_degree",0)) for m in coherent];near_miss_count=len(coherent);distinct_broken=len(broken);candidate_pool=max(1,_answer_entity_count(answer))+near_miss_count;avg=float(mean(matched)) if matched else 0.;max_matched=max(matched,default=0);max_support=max(support,default=0);score=float(near_miss_count)+float(distinct_broken)+.5*float(max_matched)+.25*max(0,candidate_pool-1);level="hard" if near_miss_count>=3 and distinct_broken>=2 and max_matched>=2 else "medium" if near_miss_count>=1 else "low";return CandidateEliminationProfile(near_miss_count,candidate_pool,distinct_broken,distinct_broken,avg,max_matched,max_support,level,score)
