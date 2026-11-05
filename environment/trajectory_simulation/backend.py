"""Two LLM roles realize a verified execution contract into native O1 language.

Language choices are internal. Native tasks, summaries and answers are rendered
from the ledger, so wording cannot introduce a new condition or change an input.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib,random,threading
from collections import Counter
from .realization import STYLES
from .realization import realize_action, realize_feedback

class PassthroughPlanner:
    offline=True
    def decide(self,payload,prompt):
        g=payload['gold_guidance']
        choice={'reasoning_style':'concise'}
        if g['may_complete']:
            from .contract import answer_string
            choice['answer']=answer_string(payload['offline_oracle_answer'])
        else:choice['task_styles']=['concise']*len(g['ready'])
        return realize_action(payload,choice)

class PassthroughExecutor:
    offline=True
    def respond(self,payload,prompt):
        return realize_feedback(payload,{'report_style':'stepwise'})

class _LLMRole:
    offline=False
    def __init__(self,backend):
        self.backend=backend
        self.style_provider='llm';self.style_seed=17
        self.request_counts=Counter();self._lock=threading.Lock()
    def _count(self,key):
        with self._lock:self.request_counts[key]+=1
    def _rng(self,text):
        return random.Random(int.from_bytes(hashlib.sha256((str(self.style_seed)+text).encode()).digest(),'big'))
    def check_question(self,prompt):
        self._count('question_review')
        from environment.question_generation.backend import OpenAICompatibleBackend
        return OpenAICompatibleBackend(**{**asdict(self.backend),'temperature':0.0,'max_tokens':1024}).complete_json('Audit question/program equivalence. Return only the requested decision/issues JSON.',prompt)
    def review(self,prompt):
        # Kept for diagnostics; bounded production language needs no semantic judge.
        from environment.question_generation.backend import OpenAICompatibleBackend
        return OpenAICompatibleBackend(**{**asdict(self.backend),'temperature':0.0,'max_tokens':1024}).complete_json('Return a short accept/issues JSON verdict.',prompt)

class LLMPlanner(_LLMRole):
    def naturalize_action(self, payload, canonical):
        from .natural_language import (
            main_writer_prompt, realize_main, main_reviewer_prompt, check_reviewer,
        )
        self._count('natural_main_write')
        candidate=self.backend.complete_json(
            'Rewrite a verified O1 Main action naturally without changing its semantics. JSON only.',
            main_writer_prompt(payload,canonical),
        )
        action=realize_main(payload,canonical,candidate)
        self._count('natural_main_review')
        verdict=self.backend.complete_json(
            'Strictly check equivalence between a fixed public task and its natural rewrite. JSON only.',
            main_reviewer_prompt(payload,canonical,action),
        )
        check_reviewer(verdict,flag='equivalent')
        return action

    def decide(self,payload,prompt):
        if self.style_provider in ('template','natural'):
            rng=self._rng(payload['public_state']['question']+str(len(payload.get('completed_subtask_contracts',[]))))
            choice={'reasoning_style':rng.choice(STYLES)}
            if payload['gold_guidance']['may_complete']:choice['answer']=payload['completed_subtask_contracts'][-1]['answer']
            else:choice['task_styles']=[rng.choice(STYLES) for _ in payload['gold_guidance']['ready']]
            return realize_action(payload,choice)
        self._count('main_style')
        choice=self.backend.complete_json('Select the next Main action language plan. Return only the requested small JSON object.',prompt)
        try: return realize_action(payload,choice)
        except ValueError as exc: raise ValueError(f'{exc}; model_language_plan={choice!r}') from exc

class LLMExecutor(_LLMRole):
    def naturalize_feedback(self,payload,canonical):
        from .natural_language import (
            feedback_writer_prompt, realize_feedback_natural,
            feedback_reviewer_prompt, check_reviewer,
        )
        self._count('natural_feedback_write')
        proposal=self.backend.complete_json(
            'Write a concise natural O1 Sub-Agent report from verified public evidence. JSON only.',
            feedback_writer_prompt(payload,canonical),
        )
        result=realize_feedback_natural(payload,proposal)
        self._count('natural_feedback_review')
        verdict=self.backend.complete_json(
            'Strictly verify every claim against the provided observations and operations. JSON only.',
            feedback_reviewer_prompt(payload,result),
        )
        check_reviewer(verdict,flag='grounded')
        return result

    def respond(self,payload,prompt):
        if self.style_provider in ('template','natural'):
            return realize_feedback(payload,{'report_style':self._rng(payload['task_instruction']).choice(STYLES)})
        self._count('subagent_style')
        choice=self.backend.complete_json('Select a natural Sub-Agent summary/trace-summary style. Preserve every supplied execution step. Return only the requested small JSON object.',prompt)
        try: return realize_feedback(payload,choice)
        except ValueError as exc: raise ValueError(f'{exc}; model_language_plan={choice!r}') from exc

__all__=['PassthroughPlanner','PassthroughExecutor','LLMPlanner','LLMExecutor']
