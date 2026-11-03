"""Answer-independent source-count planning and leakage auditing."""
from __future__ import annotations
import math,random
from collections import Counter,defaultdict
from typing import Any,Sequence
SourceCountWeights=tuple[tuple[int,float],...]
DEFAULT_SOURCE_COUNT_WEIGHTS:SourceCountWeights=((4,.30),(5,.35),(6,.35))
def sample_target_source_count(*,min_sources:int,max_sources:int,rng:random.Random,weights:Sequence[tuple[int,float]]=DEFAULT_SOURCE_COUNT_WEIGHTS)->int:
    if min_sources<1:raise ValueError("min_sources must be >= 1")
    if max_sources<min_sources:raise ValueError("max_sources must be >= min_sources")
    candidates=[(int(c),max(0.,float(w))) for c,w in weights if min_sources<=int(c)<=max_sources]
    if not candidates:candidates=[(count,1.) for count in range(min_sources,max_sources+1)]
    if not any(weight>0 for _,weight in candidates):candidates=[(count,1.) for count,_ in candidates]
    return int(rng.choices([c for c,_ in candidates],weights=[w for _,w in candidates],k=1)[0])
def answer_shape_signature(answer:Any)->str:
    if isinstance(answer,bool):return f"bool:{str(answer).lower()}"
    if isinstance(answer,int) and not isinstance(answer,bool):return f"int:{answer}"
    if isinstance(answer,float):return f"float:{answer}"
    if isinstance(answer,(list,tuple,set)):return f"list_len:{len(answer)}"
    if answer is None:return "none"
    return type(answer).__name__
def _entropy(counter:Counter[str])->float:
    total=sum(counter.values())
    if total<=0:return 0.
    value=0.
    for count in counter.values():p=count/total;value-=p*math.log2(p)
    return value
class SourceCountLeakageAudit:
    def __init__(self)->None:self.total=0;self.source_count_histogram=Counter();self.by_operator=defaultdict(Counter);self.by_operator_source_count=defaultdict(Counter)
    def add(self,operator:str,source_count:int,answer:Any)->None:
        operator=str(operator);source_count=int(source_count);shape=answer_shape_signature(answer);self.total+=1;self.source_count_histogram[source_count]+=1;self.by_operator[operator][shape]+=1;self.by_operator_source_count[(operator,source_count)][shape]+=1
    @staticmethod
    def _majority_accuracy(groups):
        total=sum(sum(counter.values()) for counter in groups);correct=sum(max(counter.values(),default=0) for counter in groups);return correct,total,(correct/total if total else 0.)
    def summary(self)->dict[str,Any]:
        _,_,operator_only=self._majority_accuracy(list(self.by_operator.values()));_,_,conditioned=self._majority_accuracy(list(self.by_operator_source_count.values()));per_operator={}
        for operator in sorted(self.by_operator):
            groups={count:counter for (op,count),counter in self.by_operator_source_count.items() if op==operator};_,total,accuracy=self._majority_accuracy(list(groups.values()));per_operator[operator]={"count":total,"source_count_histogram":{str(count):sum(counter.values()) for count,counter in sorted(groups.items())},"answer_shape_by_source_count":{str(count):dict(counter.most_common()) for count,counter in sorted(groups.items())},"operator_source_count_prior_accuracy":accuracy,"operator_only_shape_entropy_bits":_entropy(self.by_operator[operator])}
        return {"task_count":self.total,"source_count_histogram":{str(count):value for count,value in sorted(self.source_count_histogram.items())},"operator_only_shape_prior_accuracy":operator_only,"operator_source_count_shape_prior_accuracy":conditioned,"source_count_prior_accuracy_gain":conditioned-operator_only,"by_final_operator":per_operator}
