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

- `faults.py`: Deterministic `missing`, `partial` or `wrong` outcomes, with a **per-attempt observed operator trace** built from the actual post-injection execution. The missing mode stops at the failed operation; partial is currently a final-delivery truncation (not a mid-chain partial source read); wrong changes a primitive Relation then recomputes downstream primitives.
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
- `report_author.py`: Stage-2-style Sub-Agent Author/Guard/Critic
  consumes the current attempt's actual `observed_operator_steps`, output
  rows and evidence atoms. It shares Stage 2's observed-step representation,
  result normalization, surface normalization and high-precision stage-scope
  checks; it generates natural-language `finish_result.result`, `summary`
  and `trace_summary` **for the actual attempt**. Executor-owned `status`
  is never requested from the model and a legacy model `status` is ignored. The Guard verifies values,
  identifiers, report status, evidence scope and partial delivery against
  the deterministic WorkProduct, not the private clean Gold. For a missing
  deliverable, `finish_result` remains absent. Review failures are
  explicitly labeled and excluded from positive LLM-mode SFT.
- Main Action Author/Critic reuses the Stage-3 O1 Guard, but with
  recovery-specific prose prompts. The deterministic planner still chooses
  every legal target, tool/model routing and parallel batch; LLM never
  changes scheduling decisions or makes a hidden attack label visible.
- `generate_turns.py`: Multiple independent branch variants per Scene,
  configurable worker pool, atomic journals, input/code signatures for
  exact resume and positive-label publication.

The public report Author never receives the private fault label, clean
reference WorkProduct or overall Gold answer. In a wrong-result attempt,
actual observed (possibly wrong) records are authoritative, while the
complete frozen Gold remains private for episode outcome evaluation.

The `summary.json` explicitly separates `execution_recovered_branches` and
`execution_outcomes` from `recovered`/positive SFT publication. It also
records `subagent_executor_result_normalized`,
`subagent_guard_failure_causes` and `subagent_review_failure_stages`.
A branch
can be deterministically recovered but fail LLM language review; it remains
diagnostic and is never silently treated as reviewed training data.
For changed trace/Author code, generate into a new output directory:
existing signed journals cannot be resumed across these changes.

The runner also executes a private **no-repair counterfactual** along the
old Macro DAG using the poisoned WorkProduct, to distinguish an incorrect
upstream key from a correct arithmetic calculation on that key.

## Pilot vs production

Default YAML uses `generation.mode: template` for a repeatable first
pass without local API calls. To naturalize **both** actions and reports,
set `generation.mode: llm`. Two independent settings,
`generation.report_mode` and `generation.main_mode`, accept
`inherit` (use `generation.mode`), `template`, or `llm`. The
`subagent_model` YAML section configures the Stage-2-style report
Author/Critic; `model` configures the Stage-3-style Main Action
Author/Critic. They can use different Qwen models, temperatures and
revision budgets via `report_max_revisions` / `main_max_revisions`.

Example: run only the Sub-Agent Author/Critic first:

```yaml
generation:
  mode: template
  report_mode: llm
  main_mode: template
```

The deterministic fault and decision planner never invoke an LLM.

### Progress display

The CLI displays **two independent live progress bars**:
`Sub-Agent feedback` (each new report Author/Guard/Critic call) and
`Main orchestration` (each new Main Action Author/Guard/Critic call).
Unlike trusted Stage 2 and Stage 3, recovery execution is causally
interleaved: Main decisions can require another report and vice versa.
It would be incorrect to force the two bars into sequential full passes.
Each bar estimates its total work from the original DAG, expands as
dynamic recovery schedules extra calls, and displays the exact total
on completion. Postfixes show accepted/fallback report counts and
accepted/failed Main action counts. A final branch-level line reports
**execution recovered** separately from **fully reviewed SFT**.

Sub-Agent reports use the **complete delivered relation** for factual
grounding. Source evidence atoms and intermediate operator rows are
explicitly *previews*, not indications that the scan was incomplete.
A `partial` handoff may omit values observed before delivery; it remains
partial even if the internal earlier step found those values. An LLM
`result=null` for a partially delivered WorkProduct can be replaced with
a readable, deterministic statement of exactly that delivered Relation
before undergoing the full Guard and independent Critic; no arbitrary
wrong or fabricated value is silently corrected.

`summary.json` reports independent
`subagent_report_generation_modes`,
`subagent_report_review_statuses`,
`main_action_generation_modes` and
`main_action_review_statuses`. Their per-attempt details remain only in
`private/fault_alignment.jsonl`. An LLM report that fails deterministic
Guard or independent Critic may be retained as a clearly labeled
diagnostic template, but its branch is **not** published as positive LLM
SFT. This is **not** an online learned Main-Agent rollout. Persistent
`same_path` faults without an authorized recovery route remain unresolved.

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
