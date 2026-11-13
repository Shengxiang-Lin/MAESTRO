# Stage 4 — Untrusted Main orchestration

The current public runner is `python -m environment generate-untrusted-turns`.
It reads **Stage-1 scenes + Stage-2 reviewed reports**, not an externally
generated trusted Main episode file.

## Internal execution

1. `trusted_prefix.py` deterministically constructs the clean O1 history
   needed for each Scene. This is an internal bootstrap, not a separately
   generated or published Stage-3 training dataset. Rejected or incomplete
   scenes are recorded under `private/trusted_prefix_checks.jsonl`.
2. `candidate_exploration.py` probes source-grounded single faults, and
   `candidate_selection.py` ranks recovered causal variants before LLM
   language generation. Fault labels remain private.
3. `faults.py` and `state_machine.py` replay actual operator outputs,
   including wrong retrieval/entity binding, partial/missing delivery,
   misleading reports and Result/Summary conflict. A changed upstream
   WorkProduct invalidates previously used dependent results.
4. `report_author.py` + `report_contract.py` emit reviewed, natural
   Sub-Agent reports for the actual replayed evidence, except deliberately
   untrusted report faults which retain their injected claim.
5. `main_decision_contract.py` extracts public Result/Summary
   disagreements, including explicit values absent from multi-row Result
   sets. The **same delivered report** must contain both claims; merely
   seeing a planner's VERIFY request cannot prove a conflict.
6. `offline_pipeline.py` passes only the question, already returned
   history, observed conflicts, remaining evidence needs and substantive
   task objectives to the 4C Author. The Author generates **only natural
   reasoning**. The legal O1 action and delegated tasks are compiler-owned,
   not freely chosen by the Author. The complete public Reasoning is
   independently guarded and reviewed.
7. `publication_guard.py` rescans all published input/output prose
   for internal review, scheduling and generation-process vocabulary.

There are no automatically prepended factual prose templates in released
Main Reasoning; conflict facts are presented as grounding *input* to the
Author, not copied directly into its output.

## Supported faults

`no_result`, `incomplete_retrieval`, `wrong_retrieval`,
`wrong_entity_binding`, `report_missing`, `fabricated_claim`,
`result_summary_conflict`. The public configuration deliberately uses
**one injected atomic fault per branch**. It can still cause multiple
downstream tasks to fail or require revalidation.

The Main receives a **synchronous batch of all delegated Sub-Agent
results** before making its next decision. A future valid investigation
can be scheduled in a later round, but transport-delayed messages and
asynchronous partial-batch decisions are not implemented.

## Run

```bash
python -m environment generate-untrusted-turns \
  --config environment/configs/untrusted_orchestration.yaml \
  --mode llm \
  --output-dir environment/data/maestro_untrusted_rounds_v10_4_2
```

Useful artifacts:

- `summary.json`: total faults, evidence recovery, Main/Report audit
  decisions, Main-visible conflicts, SFT publication counts
- `main_turns.jsonl`: *only* globally recovered and fully reviewed Main
  SFT decisions; each input stores cumulative O1 `subtask_history`
- `stage4a/journal/`, `stage4b/journal/`, `journal/`: resumable
  execution, Sub-Agent review and Main decision records respectively
- `private/trusted_prefix_checks.jsonl`: accepted/rejected internal
  deterministic clean prefixes
- `private/candidate_selection.jsonl`,
  `private/main_turns_local_review.jsonl`: diagnostic selection and
  per-decision review metadata, not positive training targets

A known limitation is the scarcity of *independent same-claim sources*
in current Stage-1 scenes. The environment intentionally does not
fabricate unseen cross-source verification evidence in Stage 4.
