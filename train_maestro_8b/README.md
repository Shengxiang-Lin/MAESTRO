# Learnable MAESTRO Main Agent

This directory contains the trainable small-Main path for MAESTRO. It is deliberately **outside `baseline/`**: baseline methods remain comparison implementations, while this directory trains the proposed MAESTRO Main policy.

## Observation contract

The trainable Main must receive the same public information available to the Orchestra-o1 Main:

- original question;
- accumulated subtask history;
- SubAgent result / summary / trace summary;
- model/tool choices;
- remaining attempt budget.

Do not place privileged simulator labels, ground-truth error types, or oracle belief states into the Main input.

Real MAESTRO runtime trajectories store `state_before.o1_equivalent_prompt`. The synthetic Stage-3 `rounds` generator independently builds the same native `OmniGAIAMainAgentPrompt` into each JSONL row's `input` field (it does not require a complete episode).

## Model

The first implementation is intentionally minimal:

```text
O1-equivalent Main prompt
        |
      <|orch|>
        |
   Qwen shared backbone
      /         \
 Macro head    Critic head
 pi(m|s)        Q(s,m)
        \
     original LM head
          |
   O1-compatible JSON
```

Semantic macro modes:

`advance`, `verify`, `retry`, `reformulate`, `switch`, `rollback`, `complete`.

The original Qwen LM head is retained. The two new heads read the hidden state of the final `<|orch|>` token. Public runtime actions remain only `delegate_task` and `complete`.

## Data views

One episode/trajectory can be projected into different supervision views.

### Clean teacher-forced Main rounds (recommended starting point)

The frozen Stage-1 ProgramIR and Stage-2 accepted question define the
correct Fragment DAG. Stage 3 now first builds a **source-backed Subtask Bank**
(all correct task instructions, symbolic answers, and grounded Sub-Agent
summaries), then composes Main states from topologically earlier feedback.
The student learns one `(Main prompt, O1 JSON action)` per selected decision
round. **No full episode is required** to accept that row.

```bash
python environment/simulate_trajectories.py rounds \
  --scene omnigaia --provider openai-compatible \
  --style-provider natural --schedule compact \
  --max-records 100 --output work/clean_main_rounds_100.jsonl \
  --bank-output work/private_subtask_bank_100.jsonl

accelerate launch -m train_maestro_8b.train_stage1 \
  --model /path/to/Qwen3-8B \
  --train_file work/clean_main_rounds_100.jsonl \
  --output_dir workspace/checkpoints/maestro-clean-sft \
  --gradient_checkpointing
```

The JSONL is **already in Stage-1 training schema**. It needs no additional
episode export. Private bank JSONL is for teacher audit only; it must not
be provided to the Main Agent. Gold results and evidence are revealed to
the Main state strictly in already-completed wave order.

Clean-data macro targets are only `advance` / `complete`. Do not invent
`verify`, `retry`, `switch` or `rollback` labels before erroneous Sub-Agent
feedback and actual interventions are generated and evaluated.

A rejected natural rewrite falls back to its correct canonical wording to
build subsequent history. In strict default mode, a current Main delegation
with such fallback is **not** exported as a natural SFT target; use
`--include-fallback-rounds` only for separate controlled experiments.
The Sub-Agent's exact answers remain correct regardless of language fallback.
To avoid leakage, split by task or source World before sampling rounds.

### Stage 1: LM + macro supervision

JSONL:

```json
{
  "id": "task-1:2",
  "input": "<O1-equivalent Main prompt>",
  "macro_label": "verify",
  "target_json": "{\"action\":\"delegate_task\",...}"
}
```

Objective:

```text
L_stage1 = L_LM + lambda_macro * L_macro
```

Run a first Stage-1 experiment with:

```bash
accelerate launch -m train_maestro_8b.train_stage1 \
  --model /path/to/Qwen3-8B \
  --train_file workspace/train_maestro/stage1.jsonl \
  --output_dir workspace/checkpoints/maestro-8b-stage1 \
  --gradient_checkpointing
```

The LM learns how to formulate the concrete O1 delegation. The macro head learns the higher-level intervention category.

Export current MAESTRO trajectories with:

```bash
python -m train_maestro_8b.export_stage1 \
  --trajectory_path workspace/logs/trajectories \
  --output workspace/train_maestro/stage1.jsonl
```

### Stage 2: consequence critic

JSONL:

```json
{
  "id": "task-1:2",
  "input": "<same O1-equivalent Main prompt>",
  "macro_values": {
    "verify": 0.91,
    "retry": 0.28,
    "advance": 0.05
  }
}
```

`macro_values` is intentionally sparse. Only actions that were actually rolled out / evaluated should receive targets; missing actions are masked, not silently assigned zero.

Recommended first pass:

1. load the Stage-1 checkpoint;
2. freeze the LM and macro head;
3. train only the 7-way critic head;
4. optionally unfreeze the last few backbone layers with a much smaller LR.

```bash
accelerate launch -m train_maestro_8b.train_critic \
  --model workspace/checkpoints/maestro-8b-stage1 \
  --train_file workspace/train_maestro/critic.jsonl \
  --output_dir workspace/checkpoints/maestro-8b-critic
```

Counterfactual action values should be added only for selected decision states, not invented from the chosen action.

### Stage 3: trajectory RL

After Stage 1/2 are stable, optimize complete episodes. A starting reward is:

```text
R = final_correctness
    - lambda_call * extra_delegations
    - lambda_token * token_cost
    - lambda_repeat * redundant_retry
```

This stage is intentionally not hard-coded yet because the rollout dataset and budget-controlled RL protocol are still experimental.

## Files

- `schema.py`: dataset validation and macro indexing.
- `data.py`: Stage-1 and critic datasets/collators.
- `modeling.py`: Qwen wrapper with macro + critic heads.
- `export_stage1.py`: trajectory -> Stage-1 JSONL exporter.
- `train_stage1.py`: Accelerate Stage-1 trainer.
- `train_critic.py`: frozen-backbone critic trainer with optional last-layer unfreezing.

The model/data code is separated from the rollout environment so future training backends (Transformers/Accelerate, DeepSpeed, verl/GRPO) can be swapped without changing the benchmark runtime.
