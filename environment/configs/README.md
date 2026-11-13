# MAESTRO configuration

**Current supported path: Stage 1 → Stage 2 → Stage 4.**

```bash
python -m environment run \
  --questions-config environment/configs/questions.yaml \
  --reports-config environment/configs/subagent_reports.yaml \
  --untrusted-config environment/configs/untrusted_orchestration.yaml \
  --mode llm
```

Or invoke each stage separately:

```bash
python -m environment generate-questions --config environment/configs/questions.yaml
python -m environment naturalize-subtasks --config environment/configs/subagent_reports.yaml
python -m environment generate-untrusted-turns --config environment/configs/untrusted_orchestration.yaml --mode llm
```

`questions.yaml` controls accepted questions, executable scenes and World
construction. `subagent_reports.yaml` runs both Stage-2 Oracle evidence
materialization and independently reviewed Sub-Agent report naturalization.
`untrusted_orchestration.yaml` uses **only** the Stage-1 Scene path and
Stage-2 reviewed report path; it compiles trusted Main prefixes internally.

The production Stage-4 config is a single-atom, synchronous-batch,
causally replayed run with deterministic 4A candidate ranking, 4B
report naturalization and 4C Main reasoning-only review. This is not an
asynchronous message-passing or arbitrary multi-fault generator.

Paths in YAML are relative to the repository root. Each output directory
has a source-code/input signature for safe resumption; use a **new**
directory after updating pipeline code. No external Stage-3 episode input
is accepted by the public Stage-4 CLI.

## Pilot: typed Stage-1 operator grammar

For a fresh, non-overwriting Stage 1→2→4 experiment, use the separate
`*_typed_operators.yaml` configs. They select bounded, deterministic
typed relational query-plan variants while preserving the same Macro DAG,
goal, source scopes, generated World and Gold answer.

```bash
python -m environment run \
  --questions-config environment/configs/questions_typed_operators.yaml \
  --reports-config environment/configs/subagent_reports_typed_operators.yaml \
  --untrusted-config environment/configs/untrusted_typed_operators.yaml \
  --mode llm
```

The existing default config trio remains unchanged. The Stage-1 setting
`generation.operator_grammar_share` is an integer from 0 to 100, only
meaningful for Medium/Hard Macro-first V2 Scenes. Old Scene/Program files
without `operator_grammar_version` are replayed by the legacy compiler.
The pilot outputs go to separate directories and can be compared to the
existing frozen 100-Scene baseline.

The new quality report includes a role-level variant histogram, unique
operator sequences, and execution-based equivalence to the original
Macro solver on the *same* synthesized World; it does not claim that
operator reordering by itself increases reasoning difficulty.
