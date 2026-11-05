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
            main_single_task_prompt, realize_main,
            main_reviewer_prompt, check_reviewer,
        )
        # Real O1 traces frequently use empty Main reasoning. The Gold
        # controller already fixes the answer, so do not waste a completion
        # request or invite the model to add an accidental params field.
        if canonical['action']=='complete':
            self.last_natural_audit={'rewritten_tasks':0,'task_count':0,
                                     'errors':[]}
            return realize_main(payload,canonical,{'reasoning':''})
        tasks=canonical['params']['tasks']
        choices=[task['task_instruction'] for task in tasks]
        errors=[]
        rewritten=0
        # One writer request per immutable ready fragment: a small LLM cannot
        # accidentally drop a parallel sibling from a generated JSON array.
        # Each task is checked independently; one rejected rewrite must not
        # erase other accepted rewrites from the same parallel wave.
        for index in range(len(tasks)):
            try:
                self._count('natural_main_write')
                answer=self.backend.complete_json(
                    'Write exactly one faithful natural O1 subtask instruction. '
                    'Return JSON with only task_instruction.',
                    main_single_task_prompt(payload,canonical,index),
                )
                if not isinstance(answer,dict) or set(answer)!={'task_instruction'}:
                    raise ValueError('natural_single_task_schema')
                candidate=answer['task_instruction']
                if not isinstance(candidate,str):
                    raise ValueError('natural_single_task_not_text')
                if candidate == tasks[index]['task_instruction']:
                    raise ValueError('natural_single_task_unchanged')
                trial=choices.copy()
                trial[index]=candidate
                draft=realize_main(payload,canonical,{
                    'reasoning':'','task_instructions':trial,
                })
                self._count('natural_main_review')
                verdict=self.backend.complete_json(
                    'Strictly verify one rewritten public delegation against its '
                    'canonical source-only execution contract. JSON only.',
                    main_reviewer_prompt(payload,canonical,draft,task_index=index),
                )
                check_reviewer(verdict,flag='equivalent')
                choices[index]=candidate
                rewritten+=1
            except (ValueError,TypeError,KeyError) as exc:
                errors.append({'task_index':index,
                               'reason':f'{type(exc).__name__}:{str(exc)[:200]}'})
            except Exception as exc:
                # Network/backend exceptions must retain a short diagnostic,
                # but must not silently approve an unreviewed rewrite.
                from environment.question_generation.backend import BackendError
                if not isinstance(exc,BackendError):
                    raise
                errors.append({'task_index':index,
                               'reason':f'{type(exc).__name__}:{str(exc)[:200]}'})
        self.last_natural_audit={
            'rewritten_tasks':rewritten,'task_count':len(tasks),'errors':errors,
        }
        return realize_main(payload,canonical,{
            'reasoning':'','task_instructions':choices,
        })

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
            feedback_writer_prompt, feedback_repair_prompt,
            feedback_output_diagnostic, realize_feedback_natural,
            feedback_reviewer_prompt, check_reviewer,
        )
        # The local small-model JSON writer can return empty required fields.
        # Retry ONCE with a compact source-grounded prompt, rather than
        # silently dropping an otherwise valid Sub-Agent example. Both paths
        # still pass the identical deterministic AND semantic evidence guards.
        for attempt in range(2):
            self._count('natural_feedback_write')
            if attempt:
                self._count('natural_feedback_retry')
            prompt = (
                feedback_writer_prompt(payload,canonical)
                if attempt == 0
                else feedback_repair_prompt(payload,canonical)
            )
            proposal=self.backend.complete_json(
                'Return JSON with exactly two NONEMPTY strings: summary and '
                'trace_summary. Report only facts supported by public evidence.',
                prompt,
            )
            try:
                result=realize_feedback_natural(payload,proposal)
            except ValueError as exc:
                # Only the model's response schema/empty-field failures can
                # trigger a retry. Hallucinated facts, unauthorized actions,
                # incorrect results or semantic errors remain hard failures.
                reason=str(exc)
                if reason in {'natural_language_missing',
                              'invalid_natural_report_fields'}:
                    if attempt == 0:
                        continue
                    raise ValueError(
                        reason+';writes=2;'+feedback_output_diagnostic(proposal)
                    ) from exc
                raise
            self._count('natural_feedback_review')
            verdict=self.backend.complete_json(
                'Strictly verify every claim against the provided observations '
                'and operations. JSON only.',
                feedback_reviewer_prompt(payload,result),
            )
            check_reviewer(verdict,flag='grounded')
            return result
        raise AssertionError('natural_feedback_retry_loop_exhausted')

    def respond(self,payload,prompt):
        if self.style_provider in ('template','natural'):
            return realize_feedback(payload,{'report_style':self._rng(payload['task_instruction']).choice(STYLES)})
        self._count('subagent_style')
        choice=self.backend.complete_json('Select a natural Sub-Agent summary/trace-summary style. Preserve every supplied execution step. Return only the requested small JSON object.',prompt)
        try: return realize_feedback(payload,choice)
        except ValueError as exc: raise ValueError(f'{exc}; model_language_plan={choice!r}') from exc

__all__=['PassthroughPlanner','PassthroughExecutor','LLMPlanner','LLMExecutor']
