# Stage 1 — canonical V1.3

From the repository root:

```bash
python -m environment generate-questions --config environment/configs/questions.yaml
```

Stage 1 deterministically builds an executable synthetic World, frozen Macro DAG,
Gold Program, Gold Answer and source-grounded FactRefs. Existing Scenario Author/
Critic and Question Author/Critic LLMs only naturalize the user-facing context
and question; **no LLM is used for evidence operators, relations or warrant
states**. The Easy approval baseline is unchanged.

## Evidence relations (V1.3)

- `evidence_operators.py` implements `extract_claim`, `align_claims`,
  `relate_claims`, `resolve_claim` and `check_warrant`.
- `evidence_relations.py` derives a **private** Claim Ledger from actual
  `report_a` and `report_b` source observations already consumed by
  `first_report` and `second_report` Macros. It verifies real
  same-subject, same-attribute and same-temporal-scope pairs, including
  **corroborates** and **contradicts**.
- Each source has an explicitly generated provenance origin. Same-origin
  duplicates are not independent confirmation. `supersedes` is available
  only with an explicit same-lineage higher-version `revision_of` claim;
  ordinary timestamp differences cannot create revisions.
- `warrant_transitions` evaluate holding out one source and restoring
  independent evidence. These are **source-ablation counterfactual states**,
  not a fabricated sequence of actual Main-Agent message arrivals.
- `quality.py` checks strict reproducibility of this private ledger
  alongside ordinary World/Program/FactRefs replay; `quality_report.json`
  summarizes comparable pairs and evidence-state sensitivity.

This first vertical slice deliberately **does not** manufacture supersession
events or delayed Main-visible rounds. Existing Stage 1 question semantics,
Gold answers and Macro operator plans remain unchanged. Subsequent work
can add actual versioned sources and a policy/availability-aware decision
oracle without weakening the existing answer counterfactual checks.

Old V1.2 Scenes without an `evidence_relation_version` marker retain
their original World and replay semantics; new V1.3 output paths are isolated.
