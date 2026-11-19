# Stage 4 — Untrusted orchestration and causal recovery

Stage 4 produces **per-round Main-Agent decisions** from executable,
Main-visible Sub-Agent histories. Stage 1 supplies the frozen task graph
and source contracts; Stage 2 supplies reviewed reports; Stage 3 supplies
the clean reference. Stage 4 runs separate private counterfactual and
recovery executions and publishes only validated recovery decisions.

## Entry point

```bash
python -m environment generate-untrusted-turns \
  --config environment/configs/untrusted_orchestration.yaml \
  --max-scenes 10 --workers 4 --mode template \
  --output-dir environment/data/stage4_pilot_10
```

The canonical config enumerates **all eligible feedback rounds** and
all executable single-fault source variants. Set explicit numeric
`branches_per_scene`, `max_candidates_per_round`,
`max_distractors_per_fault`, `max_selected_per_round` or
`max_language_attempts_per_round` only for bounded smoke runs.
`max_post_fork_rounds` is a **causal replay horizon**, not a sampling
cap; expiry is not successful recovery. Run `--mode llm` only after
checking template-mode source/recovery diagnostics.

## Performance and interrupted v10 recovery

4A is deterministic **CPU-heavy** replay. It now uses a process pool
bounded by `--workers` and the actual available CPU quota, whereas 4B/4C
continue using threads for Author/Critic calls. Frozen Stage-1/2 report
validation is reused once per accepted scene; per-candidate atomic
journals are unchanged. Runtime gains depend on real allocated CPU cores;
20 workers do not imply 20 available cores.

The v10 rollout from PR #270 crashed after 4,231 of 4,381 candidates
due to a missing `relation_rows` import in the final delayed audit.
To **explicitly resume** that exact directory after pulling the fix:

```bash
python -m environment generate-untrusted-turns \
  --config environment/configs/untrusted_orchestration.yaml \
  --max-scenes 10 --workers 20 --mode template \
  --output-dir environment/data/stage4_delayed_v10_template_10 \
  --resume-after-audit-fix
```

The migration refuses changed input hashes, settings, candidate ledgers,
or other changes to source executor/claim logic. Each old 4A journal must
match the exact candidate identity. It preserves `manifest.before_v10_audit_fix.json`,
records `private/v10_resume_audit.json`, retains only already completed
**non-delayed** 4A journals, and *always recomputes* delayed candidates.
It refuses to mix any existing 4B or 4C journals. Subsequent reruns use
the standard signed manifest (no flag necessary). For any other branch
or changed semantics, use a **new** output directory instead.

## Safety and causal invariants

1. The initial fault is **one** real executor or delivery fault at an
   explicit, actual Sub-Agent execution wave. No arbitrary wrong values,
   new data sources or invented FactRefs are permitted.
2. Main receives only delivered reports and its preceding actions.
   Fault identity, attempts, World/Gold, source provenance and
   deterministic teacher state are **private**.
3. A report cannot be inserted without an associated legal Main
   delegation and executable task. Same-source repeated queries are
   **report discrepancies**, not independent corroboration.
4. Every output or claim whose input changes must be replayed on the
   **actual communicated WorkProduct**. Clean Stage-3 downstream
   values cannot be reused after an earlier fault.
5. Retry, verification and rollback use visible claims, admitted source
   permissions, versioned dependencies and legal scheduling.
   Final correctness alone is insufficient to label a recovered episode.

The optional reviewed Stage-3 `input.clean_episodes` must match the
validated Stage-1/2 reference fingerprint. Without it, Stage 4 builds a
deterministic clean prefix. Each candidate retains the same immutable
clean reference and a scenario-local private prefix augmentation.

## Scenario construction

**A — Existing latest-batch fault.** Mutate an actual Sub-Agent in the
latest returned batch. Missing, incomplete, acquisition, entity binding,
arithmetic and report-only faults all require executable preflight.
Main may retry when the returned evidence itself is anomalous.

**B — Later same-source recheck.** Find a complete *publicly returned*
source-query result, then explicitly add a later authorized Main
assignment to repeat that exact query with the same upstream versions.
A legitimate fault in the *new* report may cause a discrepancy with an
old report. B may update future, not-yet-executed consumers' input
binding when that binding remains legal. Old completed results are not
rewritten.

**C — Early fault with delayed discovery and actual propagation.**
Choose an old report with at least one **already executed intermediate
descendant**. Poison that old execution, replay the intermediate
descendant(s) using the wrong communicated result, and delegate a
**clean later repeated query**. Keep original downstream dependency
edges. After Main sees the conflict, the bounded teacher verifies the
new report first, then the older side if disagreement remains.
An older corrected WorkProduct must invalidate genuinely consumed
descendant versions, which must then be recomputed. A case is
publishable as recovered **only** when the private audit proves:

- A real intermediate WorkProduct differed from its clean reference
- An old/new contradiction was visible from public reports
- A completed intermediate consumer was actually invalidated
- The invalidated consumer was actually re-executed

If any condition fails, keep it as a private unsuccessful diagnostic;
do not label a speculative rollback as positive SFT.

**History-first keyed handoff.** An earlier public Sub-Agent can provide
a genuine lookup key for a later authorized lookup. The old report is
never fabricated to force a match. Optional earlier-source surveys
require legally browseable sources and an explicit historical Main
assignment; keyed-only sources cannot be scanned without a public key.
When no comparable public evidence exists, the case is ineligible.

An explicit independent-source claim dispute requires distinct
source/observation origins. A same-source repeated-query discrepancy
instead compares the complete public result sets for the same frozen
query and parent-version binding; it never uses Gold to infer a
missing public claim.

## Stage 4 pipeline and package responsibilities

- `generate_turns.py` / `config.py`: production orchestration,
  manifest, resumable private journals, accepted publication.
- `trusted_prefix.py`: executable clean-reference validation.
- `scenario/candidate_exploration.py`: fault and recheck preflight;
  `scenario/candidate_selection.py`: causal 4A selection and dedup.
- `scenario/same_source_recheck.py`: shared query replay and explicit
  scenario-local Main dispatch; `scenario/history_conflicts.py`,
  `handoff_conflicts.py` and `prior_observation.py`: public
  historical evidence alignment.
- `trajectory_builder.py`: prefix replay, Main's recovery decisions,
  counterfactual continuation and causal acceptance checks.
- `state_machine.py`: versioned Attempts, observable issues,
  two-sided dispute handling, invalidation and recomputation.
- `runtime/`: executable faults, legal evidence scheduling;
  `reporting/`: actual Sub-Agent reports and immutable result
  contract; `orchestration/`: O1 Main decision contracts and
  reasoning review; `inspection/`: public claim extraction and
  generation/privacy guard; `rewards/`: private per-turn reward.

There are no shadow report-only agents, no alternate state machine,
and no compatibility wrappers for deleted top-level module paths.

## Data and audit interpretation

`episodes.jsonl` and `main_turns.jsonl` contain only globally
accepted, fully reviewed O1-compatible positives. A private
`stage4a/journal` retains every attempted execution (including
incorrect and unresolved); `stage4b/journal` contains re-authored
actually executed reports; `journal` records final Main decisions.

Useful private outputs:

- `private/candidate_selection.jsonl`: one row per candidate,
  `scenario_blueprint`, actual replay outcome, historical gaps,
  `delayed_causal_audit`, `behavior_signature` (case-rich) and
  `policy_signature` (abstract operations).
- `private/paired_trajectories.jsonl`: clean-reference identity and
  private unrepaired continuation; it is **not** shown to Main.
- `private/claim_state_supervision.jsonl` and
  `private/reviewed_observable_states.jsonl`: next-round public
  Claim–Source–Dependency belief/risk labels. Negative or unresolved
  cases are **not** silently exported as positive action labels.
- `private/decision_reward_states.jsonl`: pre-action executable
  tasks, evidence needs, dependency/parallel legality and completion
  eligibility for deterministic reward; never a Main prompt field.

Key `summary.json` checks:
`candidate_delayed_propagation_trials`,
`execution_recovered_delayed_propagation`,
`selected_recovered_delayed_propagation`,
`published_delayed_propagation_recoveries`,
`actual_downstream_invalidations`,
`published_with_downstream_invalidation`,
`published_historical_conflict_round_gap_distribution`,
`distinct_selected_case_signatures` and
`distinct_selected_policy_signatures`.

The `template` teacher is a deterministic causal reference, **not**
an online trained Main-Agent policy result. For language-based SFT,
4B/4C Author+Critic must accept report and decision prose. Failure
and fallback remain private diagnostics; they cannot be promoted
through a successful deterministic outcome.

## Implementation checks

Run the environment unit tests and small paired Stage 2–4 smoke
pilot before any full-scale run. In particular, exercise: no phantom
dispatch; no early clean-answer reuse; same-source discrepancies not
counted as independent votes; bounded two-sided retries; actual
invalidated versions; non-recomputed consumers failing publication;
unchanged Stage 1–3 contracts; and no future/private data in O1 inputs.
