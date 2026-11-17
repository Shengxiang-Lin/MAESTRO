# Trusted Main-Agent orchestration (Stage 3)

This module generates clean Main decisions for training and testing, using
reviewed Stage-2 reports and the same O1-compatible format as the baseline.

```bash
python -m environment generate-main-turns \
  --config environment/configs/main_orchestration.yaml
```

The default output is `environment/data/stage3/`. For an isolated pilot,
specify `--max-scenes 10 --output-dir environment/data/pilots/stage3`.

## Compact generation-only Author/Critic context

In `freeze_dispatch: true` mode, Stage 3 sends the prose Author and
independent Critic a compact, **lossless evidence view** rather than the
baseline O1 prompt. It retains the entire user question, provided initial
observations and source catalog (including source handles for legacy Scenes),
current attempt budget, and the complete cumulative O1-formatted Sub-Agent
history. Each pending investigation still carries its permitted sources,
required output fields, and prior-result context, and the Critic still sees
the round-specific evidence and task-prose contracts.

Only redundant static O1 instructions are omitted: parallel-decomposition
tutorials, model-pricing tables, full tool descriptions/JSON schemas and output
examples. Nothing in the Main-visible history is truncated or summarized.

The **training prompt is unchanged**: each turn still stores
`input.subtask_history`, and `o1_prompt_from_turn` rebuilds the original
`OmniGAIAMainAgentPrompt` exactly. Validation of proposed prose uses the
original full O1 prompt for identifier allowlisting. Private review audits
record `full_o1_prompt_chars` and `author_context_chars` for before/after
size comparison; these are character counts, not token counts or measured
speedups. Existing run journals have code signatures and should be
regenerated in a new output directory after this change.

The deterministic compiler owns the legal Macro batches, dependencies,
task count and order, model and tool routing. With `freeze_dispatch: true`,
**the same Qwen Author call** now writes `reasoning` plus natural
`task_instruction`/`context` for each preassigned slot. The Author receives
the current baseline O1 question/history, allowed source IDs and public
titles/modalities, required output fields and the fixed slot's existing
instruction as a semantic reference. No hidden future result, DAG or
private teacher alignment is passed to the Author.

Before publication, programmatic Guards check task count/order, permitted
source references, invented identifiers and fixed model/tools. The compiler
adds required return fields as needed and appends *only already delivered*
public Sub-Agent result text verbatim to dependent contexts. An independent
Critic reviews **all three natural fields** for source specificity,
stage-local reasoning and no premature claims. The actual Macro dispatch,
execution sequence and Stage-2 feedback are not altered. Stage 4's existing
frozen reasoning-only author path remains unchanged.

The output remains `input.subtask_history` →
`output.{action,reasoning,params}`, while the internal authoring record
indicates `authored_fields: [reasoning, task_instruction, context]`.
Private contracts and reviewer diagnostics stay in
`private/teacher_alignment.jsonl`.

Sub-Agent summaries are normalized as **copies** of previously reviewed
reports. No structured evidence, Result, status, FactRef or source value is
modified. A rejected Episode never leaks its partial turns into public
training data.

Each accepted Episode also contains a stable `clean_reference_id`, derived
from the frozen World, Macro DAG and reviewed Stage-2 Finish Results. The
reference ID does not depend on stylistic Main reasoning.

Stage 4 normally compiles an equivalent deterministic trusted prefix. To
reuse the exact reviewed Stage-3 trajectory instead, configure
`input.clean_episodes: environment/data/stage3/episodes.jsonl` in Stage 4.
The runner validates its returned evidence, action batches and completion
against an independently compiled clean reference before any fault injection.
