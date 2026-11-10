# Untrusted orchestration: round-local evidence faults and recovery

Run from the MAESTRO repository root:

```bash
python -m environment generate-untrusted-turns \
  --config environment/configs/untrusted_orchestration.yaml
```

The generator uses the three existing trusted products:
1. Frozen Scene + Macro DAG + World from Question generation
2. Complete, reviewed Sub-Agent Macro reports
3. Accepted Base Orchestration Episodes and exact O1 returned-batch history

A branch chooses a **Main decision round t >= 2**. All history through
round t-2 stays clean and is replayed as the original accepted O1 records.
Only one or more Sub-Agent results from the batch returned in round t-1
are changed. The simulator modifies *primitive execution results* and
re-runs the rest of the Macro operator chain. The resulting WorkProduct
(not a rewritten Oracle report) is passed to any downstream tasks.

## Current functionality

- `faults.py`: Deterministic `missing`, `partial` or `wrong` outcomes.
  The `wrong` mode uses actual alternative source observations or
  mismatched **real** parent candidates at a join; unsupported cases are
  skipped instead of inventing values. Fault target selection weighs
  downstream dependency impact, with optional multi-target batches.
- `state_machine.py`: Versions individual WorkProducts, derives necessary
  parent fields for readiness, identifies conflicts **only if visible in
  previous Sub-Agent reports**, and marks dependent results stale after
  re-execution. Original reports remain visible in the cumulative O1
  history and are never silently overwritten.
- `trajectory_builder.py`: Constructs new Main `delegate_task` actions
  for targeted rechecks, unfinished evidence and independent ready tasks
  **in the same decision batch**. All tasks are actually re-executed on
  the resulting WorkProducts. Unresolved or incorrect outcomes are
  diagnostic only and never published as positive SFT labels.
- `report_author.py`: A deterministic faithful result string stays fixed.
  In `llm` mode, an Author/Critic pair may naturalize only the summary
  and trace prose; failure uses an explicitly labeled grounded fallback.
- `generate_turns.py`: Multiple independent branch variants per Scene,
  configurable worker pool, atomic journals, input/code signatures for
  exact resume and positive-label publication.

The runner also executes a private **no-repair counterfactual** along the
old Macro DAG using the poisoned WorkProduct, to distinguish an incorrect
upstream key from a correct arithmetic calculation on that key.

## Pilot vs production

Default YAML uses `generation.mode: template` for a repeatable first
pass without local API calls. To naturalize actions and reports, set
`mode: llm` and configure the local OpenAI-compatible Author/Critic
endpoint. Only reviewed or explicitly labeled teacher-template output is
created; this is **not** an online learned Main-Agent rollout. Persistent
`same_path` faults that have no independent authorized recovery route
remain unresolved rather than being magically repaired.

The `input` fields in the YAML point at the *existing* accepted trusted
outputs. Output paths must differ from the existing inputs.

Published files:
- `episodes.jsonl`: visible O1-style branched episodes, including
  unresolved outcomes but **no hidden injected-fault labels**.
- `main_turns.jsonl`: only post-fork Main decisions from fully recovered
  Episodes, one record per decision, with compact cumulative
  `input.subtask_history`.
- `private/fault_alignment.jsonl`: target primitive, actual per-attempt
  relations, uncorrected propagation/counterfactual, invalidations,
  reason for recovery or failure, and measured simulation cost.
- `failed.jsonl`, `summary.json`, `manifest.json`, and `journal/`
  for per-branch diagnosis and resumable runs.

The supported model deliberately starts with one injected last-batch event
per independent branch. Later follow-up rounds can remain affected by a
persistent same-path fault. This is **not** a general-purpose adversarial
agent attack library or independent-source retrieval engine yet. The
frozen World is synthetic, and media observations remain simulated.
