# Trusted Main-Agent orchestration (Stage 3)

This module generates clean Main decisions for training and testing, using
reviewed Stage-2 reports and the same O1-compatible format as the baseline.

```bash
python -m environment generate-main-turns \
  --config environment/configs/main_orchestration.yaml
```

The default output is `environment/data/stage3/`. For an isolated pilot,
specify `--max-scenes 10 --output-dir environment/data/pilots/stage3`.

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
