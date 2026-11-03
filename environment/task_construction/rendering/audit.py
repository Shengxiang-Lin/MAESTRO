"""Batch-level deterministic rendering leakage/diversity diagnostics."""
from __future__ import annotations
import re
from collections import Counter
from typing import Any
from .schema import RenderedTask
class RenderingAudit:
    def __init__(self):
        self.task_count=0;self.question_templates=Counter();self.question_bigrams=Counter();self.source_type_sequences=Counter();self.answer_surfaces=Counter();self.public_placeholder_hits=0;self.public_hidden_label_hits=0
    @staticmethod
    def _tokens(text):return [token for token in re.findall(r"[A-Za-z0-9]+",str(text).lower()) if token]
    def add(self,task:RenderedTask):
        self.task_count+=1;template=str(task.hidden_metadata.get("question_template_id","unknown"));self.question_templates[template]+=1;tokens=self._tokens(task.question);self.question_bigrams.update(" ".join(tokens[i:i+2]) for i in range(max(0,len(tokens)-1)));self.source_type_sequences[" -> ".join(source.source_type for source in task.sources)]+=1;self.answer_surfaces[str(task.answer)]+=1;public=str(task.training_dict(include_hidden=False)).lower();self.public_placeholder_hits+=int(bool(re.search(r"\b(?:value|val|string|text|label|literal|token|item|name)[_-]?\d+\b",public,re.IGNORECASE)));self.public_hidden_label_hits+=int(any(token in public for token in ("proof_relevant","indispensable","critical_source_ids","minimal_proof","fact_ids")))
    def summary(self)->dict[str,Any]:
        total=max(1,self.task_count);most_template=max(self.question_templates.values(),default=0);most_bigram=max(self.question_bigrams.values(),default=0);most_source=max(self.source_type_sequences.values(),default=0);return {"task_count":self.task_count,"distinct_question_templates":len(self.question_templates),"dominant_question_template_rate":most_template/total,"distinct_question_bigrams":len(self.question_bigrams),"dominant_question_bigram_rate":most_bigram/max(1,sum(self.question_bigrams.values())),"distinct_source_type_sequences":len(self.source_type_sequences),"dominant_source_type_sequence_rate":most_source/total,"distinct_answer_surfaces":len(self.answer_surfaces),"public_placeholder_hit_count":self.public_placeholder_hits,"public_hidden_label_hit_count":self.public_hidden_label_hits}
