"""Epistemic-state + prospective-deliberation prompt for MAESTRO."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from orchestra.prompts.omnigaia import OmniGAIAMainAgentPrompt as BaseOmniGAIAPrompt


class ProspectiveOmniGAIAPrompt:
    @staticmethod
    def build_prompt(
        instruction: str,
        meta: Dict[str, Any],
        prior_context: str,
        attempt_index: int,
        max_attempts: int,
        sub_models: List[str],
        subtask_history: str = "",
        model_to_alias: Dict[str, str] | None = None,
        tools: List[Any] | None = None,
        observed_state: Dict[str, Any] | None = None,
        max_future_requirements: int = 4,
        max_critical_dependencies: int = 3,
        max_tasks_per_action: int = 5,
    ) -> str:
        base = BaseOmniGAIAPrompt.build_prompt(
            instruction=instruction,
            meta=meta,
            prior_context=prior_context,
            attempt_index=attempt_index,
            max_attempts=max_attempts,
            sub_models=sub_models,
            subtask_history=subtask_history,
            model_to_alias=model_to_alias,
            tools=tools,
        ).split("==== OUTPUT FORMAT ====")[0]
        state = json.dumps(observed_state or {}, ensure_ascii=False, indent=2)
        models = json.dumps(sub_models, ensure_ascii=False)
        return f"""{base}

==== MAESTRO: EVIDENCE-GROUNDED PROSPECTIVE DELIBERATION ====
The execution history below is append-only evidence. Do not rewrite, erase, or retroactively reinterpret an old `result_id`. Maintain a separate CURRENT EPISTEMIC STATE that may be revised when new evidence arrives.

Perform this sequence in ONE response:
1. revise current epistemic bindings from visible evidence;
2. identify which bindings matter for future task success;
3. predict what evidence/state should change next;
4. emit exactly ONE outward macro-action: `delegate_task` or `complete`.

Do not classify a source, Sub-Agent, provider, tool, or communication path with hidden trust/reliability labels. Do not invent numeric confidence, trust scores, or causal probabilities.

==== 1. PERSISTENT EPISTEMIC STATE / DELAYED COMMITMENT ====
Maintain compact symbolic bindings for decision-relevant variables/claims. Each binding has:
- `key`: stable symbolic name reused across rounds for the SAME predicate;
- `predicate`: exact operational meaning;
- `value`: current best candidate value, or empty when not justified;
- `status`: exactly `unresolved`, `provisional`, or `supported`;
- `evidence_refs`: visible result IDs relevant to this binding;
- `open_issue`: what remains insufficient for current downstream use.

Prefer delayed commitment. One coherent result does not force `supported`; prior disagreement does not permanently ban a route. Keep the same `key` when revising the same predicate.

==== 2. PROSPECTIVE FUTURE-IMPACT STATE ====
Use at most {max_future_requirements} `future_requirements` describing WHAT must still become known/resolved. Use at most {max_critical_dependencies} `critical_dependencies`, each with `belief_key`, `reason`, `consequence_if_wrong`, and `evidence_gap`. `desired_next_evidence_state` describes the TARGET STATE after the next useful work, not a tool name. `completion_readiness` is `ready`, `not_ready`, or `uncertain`.

==== 3. EVIDENCE ATTEMPT HISTORY ====
Use `evidence_attempt_history` to distinguish fresh retry, same-provider re-sampling, diverse-provider evidence, repeated correlated agreement, and strict diversity requests that could not execute. Same-provider retries can still be informative; decide from future consequence, evidence gap, previous attempts, and cost.

==== 4. SEMANTIC EVIDENCE REQUEST ====
All outward work remains a normal task inside `delegate_task`. Do NOT introduce new outward actions. A task may optionally include `evidence_request` with:
- `target_key`;
- `target_requirement`;
- `sampling`: `reuse_ok`, `prefer_diverse`, or `require_diverse`;
- `based_on`: prior result IDs relevant to route diversity;
- `blind`: whether prior candidate values should be removed from worker context.

`reuse_ok`: reusing the old provider/path is allowed.
`prefer_diverse`: prefer a different provider but keep old providers as fallback.
`require_diverse`: rare strict projection; exclude providers used by `based_on` only when provider-disjoint evidence is necessary. If unavailable, execution may return `evidence_projection_unavailable`.

==== 5. BLIND RE-QUERY / PREDICATE PRESERVATION ====
`blind` is independent of sampling diversity. Use it when an old candidate may anchor a new attempt. `target_requirement` must not contain the candidate answer/value and must preserve the same predicate.

==== 6. EXECUTABLE-SPACE PROJECTION ====
The Main outputs a SEMANTIC evidence request. A deterministic projector maps it onto the executable tool space. The projector does not decide whether more evidence is needed and does not assign trust.

==== DECISION RULE ====
- Revise epistemic bindings before deciding the action.
- Focus on load-bearing bindings whose wrong value materially changes future task success.
- If such a binding is unresolved/provisional and evidence is insufficient, delegate the highest-value evidence/task bottleneck.
- Reuse an old path when another sample still has information value.
- Prefer diversity when correlation is the main remaining evidence gap.
- Do not gather evidence merely to increase source/tool counts.
- If `complete`, load-bearing bindings needed for the answer should not remain unresolved, `completion_readiness` should be `ready`, and `future_requirements` should be empty.

==== OBSERVED APPEND-ONLY EVIDENCE + PRIOR EPISTEMIC STATE ====
{state}

Allowed model aliases: {models}

==== OUTPUT FORMAT ====
Return exactly one JSON object:
{{
  "epistemic_state": {{
    "bindings": [{{
      "key": "stable_symbolic_key",
      "predicate": "exact meaning of the variable/claim",
      "value": "current candidate value or empty",
      "status": "unresolved|provisional|supported",
      "evidence_refs": ["result_1"],
      "open_issue": "what remains unresolved for current use"
    }}]
  }},
  "prospective_state": {{
    "future_requirements": ["what still must become known/resolved"],
    "critical_dependencies": [{{
      "belief_key": "stable_symbolic_key",
      "reason": "why this belief is load-bearing",
      "consequence_if_wrong": "what future answer/branch changes if wrong",
      "evidence_gap": "what remains insufficient"
    }}],
    "desired_next_evidence_state": "what should be resolved next",
    "completion_readiness": "ready|not_ready|uncertain"
  }},
  "action": "delegate_task|complete",
  "params": {{
    "tasks": [{{
      "task_instruction": "self-contained task",
      "model": "one exact allowed alias",
      "tools": ["exact tool names"],
      "context": "only context needed for the task",
      "evidence_request": {{
        "target_key": "stable_symbolic_key",
        "target_requirement": "predicate-only requirement; no candidate answer value",
        "sampling": "reuse_ok|prefer_diverse|require_diverse",
        "based_on": ["result_1"],
        "blind": true
      }}
    }}]
  }},
  "reasoning": "brief explanation linking epistemic revision, future consequence, and the chosen action"
}}

`evidence_request` is OPTIONAL for ordinary task-progress delegation. For `complete`, replace `params` with {{"answer": "precise concise final answer"}}. Do not generate candidate action portfolios, rankings, hidden trust labels, numeric trust/confidence scores, or a second-stage critique.
""".strip()
