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
