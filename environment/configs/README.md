# MAESTRO configuration

The supported path is **Stage 1 → Stage 2 → Stage 4**. There is
one canonical YAML per stage; the separate typed-operator pilot config
trio and deprecated Stage-1 CLI aliases are retired.

```bash
python -m environment run \
  --questions-config environment/configs/questions.yaml \
  --reports-config environment/configs/subagent_reports.yaml \
  --untrusted-config environment/configs/untrusted_orchestration.yaml \
  --mode llm
```

Stage 1 enables `macro_planning_share: 100`,
`operator_grammar_share: 100`, `numeric_goal_diversity_share: 100`,
and `question_semantic_lock_share: 100`. The historical
`macro_first_v2_share` config spelling is rejected.

Stage 2 builds Oracle results and reviewed Sub-Agent reports; Stage 4
compiles clean O1 history internally, then simulates and reviews
single-fault causal recoveries. Each default YAML points to the same
new Scene bank, with independent output folders and replay signatures.
