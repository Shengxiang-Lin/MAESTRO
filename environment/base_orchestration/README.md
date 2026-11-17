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
model and tool routing. When `freeze_dispatch: true`, the LLM writes only
brief Main Reasoning. The current-round Evidence Contract guards invalid
output fields, wrong delegate counts, and premature completion. An
independent Critic reviews the authored prose. The public turn remains
`input.subtask_history` → `output.{action,reasoning,params}`;
private contracts and reviewer diagnostics stay in
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
