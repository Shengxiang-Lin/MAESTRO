"""Two LLM roles realize a verified execution contract into native O1 language.

Language choices are internal. Native tasks, summaries and answers are rendered
from the ledger, so wording cannot introduce a new condition or change an input.
"""
from __future__ import annotations
from dataclasses import asdict
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
    def review(self,prompt):
        # Kept for diagnostics; bounded production language needs no semantic judge.
        from environment.question_generation.backend import OpenAICompatibleBackend
        return OpenAICompatibleBackend(**{**asdict(self.backend),'temperature':0.0,'max_tokens':1024}).complete_json('Return a short accept/issues JSON verdict.',prompt)

class LLMPlanner(_LLMRole):
    def decide(self,payload,prompt):
        choice=self.backend.complete_json('Select the next Main action language plan. Return only the requested small JSON object.',prompt)
        try: return realize_action(payload,choice)
        except ValueError as exc: raise ValueError(f'{exc}; model_language_plan={choice!r}') from exc

class LLMExecutor(_LLMRole):
    def respond(self,payload,prompt):
        choice=self.backend.complete_json('Select a natural Sub-Agent summary/trace-summary style. Preserve every supplied execution step. Return only the requested small JSON object.',prompt)
        try: return realize_feedback(payload,choice)
        except ValueError as exc: raise ValueError(f'{exc}; model_language_plan={choice!r}') from exc

__all__=['PassthroughPlanner','PassthroughExecutor','LLMPlanner','LLMExecutor']
