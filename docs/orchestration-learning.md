# MAESTRO Orchestration Design

MAESTRO uses the same orchestration idea in two forms.

## API Main: explicit scaffold

For closed/API models we cannot add trainable hidden-state heads. The Main therefore writes a compact explicit state in one API call:

1. revise evidence-grounded beliefs;
2. identify load-bearing dependencies;
3. choose an internal orchestration mode;
4. perform a qualitative consequence check;
5. emit the normal Orchestra runtime action.

The internal modes are:

- `advance`
- `verify`
- `retry`
- `reformulate`
- `switch`
- `rollback`
- `complete`

They are **not** new environment actions. The runtime still executes only `delegate_task` or `complete`.

The consequence check should be qualitative rather than fake calibrated probabilities: what useful state change is expected, what can fail if the bottleneck is ignored, and what cost is being spent.

This lets the API experiment test whether the orchestration decomposition itself helps a strong model, without changing model parameters.

## Trainable small Main: internalized scaffold

For Qwen3-8B, the public input remains Orchestra-o1-equivalent. A special final `<|orch|>` token provides a dedicated readout position:

```text
Question + Subtask History + Budget + <|orch|>
                         |
                      h_orch
                 /        \
            macro head   critic
              pi(m|s)    Q(s,m)
                 \
             original LM head
                 |
        delegate_task / complete JSON
```

Stage 1 trains the original LM objective together with macro classification. Stage 2 trains action values from rollout outcomes/counterfactual branches. Stage 3 performs budget-aware trajectory RL.

The key experimental control is that the small Main receives no extra oracle state compared with Orchestra-o1. Any structured state is learned internally or used only as training-side supervision.

## Main hypotheses

1. Explicit orchestration structure should improve API models over the same model under the O1 prompt.
2. The same structure can be internalized by a smaller trainable Main.
3. Recovery should improve without collapsing into verify/retry everywhere.
4. Gains should persist under matched delegation/tool/token budgets.

Recommended metrics include final task success, recovery rate, premature completion rate, redundant retry rate, average delegations, token cost, and success per unit cost.
