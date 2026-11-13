# Stage 1 — V1.4 evidence-aware Macro programs

```bash
python -m environment generate-questions --config environment/configs/questions.yaml
```

Stage 1 constructs frozen synthetic sources, a fully executable Macro DAG,
Gold Program, Gold Answer, FactRefs, and (for report-dispute patterns) a
private evidence relation ledger. The existing Scenario/Question Author and
Critic LLMs only naturalize the user-facing task; **no LLM judges Claim
relations, provenance or trust-state reference updates**.

## New in V1.4

- `surface_quality.py` and `question.py` accept grounded `locate` requests
  and compositional two-source higher-count paraphrases without relaxing
  mandatory output and numerical constraints.
- New eligible disputed-report Specs sample `claim_verification_mode` as
  `parallel` or `keyed_followup` using a separate deterministic hash stream.
  Keyed follow-up can only retrieve records from independent report B
  **after** report A has supplied candidate case IDs. It is a real typed
  `lookup_by_key` with Source A provenance, not a manufactured delay.
- `dispute_resolution` now executes `claim_relation_filter`, a typed
  provenance-checked primitive. Rows must carry FactRefs from both report
  sources; compared year fields must be comparable. Historical V1.3
  scenes without `evidence_trust_version` retain the old `compare_fields`
  program and exact Replay behavior.
- `evidence_relations.py` restricts the Claim Ledger to facts actually
  accessible to their source Macro. New `reference_wave_trust`
  annotations track when evidence becomes **eligible** following each
  frozen Macro wave and which target-related Claim pairs gain independent
  support or become disputed.
- `quality_report.json` separates raw Claim-pair counts from
  answer-relevant, genuinely cross-wave pairs and state transitions.
  All ledgers and reference schedule annotations are **private**, never
  added to the user Question or Orchestra-o1 Main prompt.

`reference_wave_trust` means *source evidence became available to a
Macro*, **not** that every item was necessarily stated in a natural-language
Sub-Agent finish report. Main-visible state supervision must be computed
from actual delivered reports, with no hidden World-to-prompt shortcut.
Evidence source independence means independent authorship/lineage, not the
absence of task data dependencies: report B can depend on case IDs from
report A without copying its year assertion.

Stage 2 materializes frozen Macro WorkProducts and naturalizes reviewed
Sub-Agent reports. For V1.4 source-report Macros, a compact list of actual
returned case/year rows is added deterministically to the reviewed summary.
Stage 4 separately parses literal returned text and records private
`visible_claim_state` transitions (including faulty/missing reports and
retries) without reconstructing unseen facts from the World. Stage 4 provides adversarial feedback and runtime
Main-Agent recovery behavior; truthful source disagreements are not
themselves Sub-Agent faults.

Previous V1.3 data are retained in distinct directories. Strict Replay
must continue to work on old frozen scenes; use a fresh output directory
for any new generation run.
