# Stage 3 — Clean Sub-Agent Reports → Main Scheduling SFT

Stage 3 consists of **two explicitly separate, sequential phases**. It does
not require or train on an entire autonomously executed task episode.

The frozen Stage-1 ProgramIR / Fragment DAG and accepted Stage-2 question
supply the immutable semantics. The subtask answers always come from public
source-replayed symbolic execution; the LLM writes only language.

## Phase 1 — Complete and verify every Sub-Agent subtask

Run **once** after constructing the frozen Worlds and accepted questions:

```bash
python environment/simulate_trajectories.py subtasks \
  --scene omnigaia \
  --provider openai-compatible \
  --schedule compact \
  --max-records 20 \
  --feedback-workers 4 \
  --output work/subtask_bank_20.jsonl
```

Inputs: frozen Stage-1 Worlds and Stage-2 accepted questions. The defaults
come from `environment/config.yaml`; override with `--worlds` and `--realized`.

For each question, this phase:

1. Compiles and preflights the complete Gold Fragment DAG against **public**
   Source evidence. Hidden private entities cannot satisfy a public lookup.
2. Revalidates the question/program semantic equivalence using the configured
   question reviewer. Unreviewed or mismatched questions are not exported.
3. Visits **all** fragments in topological waves. For each subtask, records
   the canonical task, source-replay-verified exact answer, `summary`,
   `trace_summary`, observations, dependencies, and report-language audit.
4. Reconstructs an auditable **simulated O1 tool-observation sequence**
   from the public Source acquisition waves and assigned native tool profile.
   A discoverable Source can yield a `GoogleSearchAction` discovery event
   followed by an `ExtractUrlContentAction`, `ImageAnalysisAction`,
   `VideoAnalysisAction` or `ParseAudioAction` inspection event, as
   supported by its modality; previously known Sources are reused without
   creating a new tool call. No fabricated search queries, timestamps, URLs,
   failures or unobserved facts are added.
5. Pairs each permitted tool/source event with that source's EXACT
   observed claims in a grouped `source_evidence_guide`. This reduces
   cross-source attribution errors when the LLM writes its report.
6. Attempts fluent, evidence-grounded Sub-Agent report wording: concise
   `summary` plus real-O1-style `trace_summary` that naturally describes
   what was examined, observed, verified and returned. Full class names
   such as `VideoAnalysisAction` are **optional**, never mandatory. A trace
   may omit tool mechanics if it accurately explains the supported result.
   Both literal and natural-language claims of *unpermitted* tool actions
   remain forbidden. Semantic review still checks source-specific facts,
   values and relation directions. If the local LLM returns missing/blank
   `summary` or `trace_summary`, Stage 3.1 retries the writer **once** using
   a shorter prompt with predicate-relevant source facts and the same fixed
   answer. The retry is not a bypass: both deterministic fact checks and the
   separate semantic reviewer still must pass. Persistent missing fields
   are recorded as `summary=blank/null/...;trace_summary=...` in the
   private bank's `feedback_fallback_reason` instead of losing the cause.
   All other invalid facts/tool actions immediately fall back without retry.
   A rejected rewrite falls back to the correct canonical report; the
   answer is never changed by the LLM.
   Private bank entry `simulated_tool_provenance` explicitly marks
   `live_tool_calls=false` for audit.
7. Only after all parallel siblings pass does the next wave see their public
   results. Siblings never inspect each other's unreleased outputs.

The output `subtask_bank_20.jsonl` is a **complete private teacher bank**:
one independently validated record per accepted question. It includes future
Gold answers for the controller and must **never** be passed to the Main
model as prompt text. It has schema ID `maestro.clean_subtasks.v1` and a
content-integrity hash. Inspect it and the accompanying rejected/quarantine
JSONL files before starting Phase 2.

**Phase 1 does not generate Main natural-language delegations, scheduling SFT
targets, or any simulated Sub-Agent errors.** It never calls the Main writer;
the planner role is used only for question-equivalence review.

## Phase 2 — Compose causal Main states and gold-correct delegations

After Phase 1 has completed, run Phase 2 **using the bank only**:

```bash
python environment/simulate_trajectories.py rounds \
  --provider openai-compatible \
  --bank work/subtask_bank_20.jsonl \
  --round-indices 2,3 \
  --output work/main_rounds_20.jsonl
```

Phase 2 does **not** read the original World / question JSONL, recompile the
ProgramIR, or call the Sub-Agent writer. For every bank:

1. Rechecks the bank's hash, semantic review, completed public replay and
   frozen Gold answer. A corrupted or incomplete bank is rejected.
2. Traverses DAG waves. A Main state contains the original question and
   only previously completed, verified Sub-Agent results and public evidence;
   the current and future waves remain invisible.
3. Uses the stored canonical Gold delegation as immutable teacher guidance.
   The Main LLM may rewrite each task into a natural O1 instruction, subject
   to strict semantic review and deterministic public-source replay.
4. Emits one **Main** SFT sample per selected decision round: the real
   `OmniGAIAMainAgentPrompt.build_prompt` input and the correct O1
   `delegate_task` JSON action, or a fixed-answer `complete` target.
5. Writes each accepted round immediately. A failed Round 1 natural rewrite
   does not prevent a correct Round 2 state, because the previously verified
   teacher action and Sub-Agent report still reconstruct a valid history.

A Main rewrite that does not pass review falls back to the public-executable
canonical action for **history construction**, but strict natural Main SFT
omits that current action by default. `--include-fallback-rounds` includes these
template-style targets only when explicitly requested.

`--round-indices 2,3` selects the target decision rounds; unselected earlier
waves still contribute their correct feedback. Omit it to include every
delegation wave and the final `complete` decision. `--exclude-complete` excludes the
final action.

The output `main_rounds_20.jsonl` is **already in the
`train_maestro_8b` Stage-1 SFT schema**. Each row has:
`input`, `target_json`, `macro_label` (`advance` or `complete`),
`round_index`, validity fields, and `history_origin`. Do **not** export
it again from a full-episode trajectory. Internal public replay checks are
removed from natural Main training targets only **after verification**.

## Correctness boundaries and current scope

- **Gold Answer**: exact symbolic value, supported by public Source evidence.
- **Subtask feedback**: always completed correctly. The natural-language
  summary is attempted, reviewed and may fall back to an accurate template.
- **Main scheduling**: exactly the Gold-compatible ready subtask set per wave;
  the LLM writes phrasing, never changes the task graph, tools, operands or
  return values.
- **Causality**: completed predecessor waves are Main-visible; parallel
  siblings and future answers are not.
- **No errors/attacks yet**: do not generate fabricated retries, switches,
  rollbacks or unsupported `partial` reports. Clean-data macro labels
  are only `advance` / `complete`.
- **No actual live tool traces**: the private simulated tool-observation
  sequence represents only what is supported by frozen Source witnesses and
  native tool assignments. Report prose may use realistic O1 action/observation
  style, but do not interpret its action counts as real browser, media-model
  or Python tool executions.
- **Scheduling policy**: the current implementation follows deterministic
  ready-wave topological ordering, not all possible linear extensions of
  the DAG. Alternative feasible schedules can be added independently.
- **Validation**: unit tests verify source replay, O1 prompt/tool contracts,
  per-wave visibility, independent phase execution, bank tampering
  rejection and persistence of successful earlier Main rounds.

For a local smoke check without real LLM calls, use `--provider passthrough`
only for low-level fixture tests. It cannot supply independent question
semantic review, so it should not be treated as training-ready production
data.

## Training

```bash
accelerate launch -m train_maestro_8b.train_stage1 \
  --model /path/to/Qwen3-8B \
  --train_file work/main_rounds_20.jsonl \
  --output_dir workspace/checkpoints/maestro-clean-sft \
  --gradient_checkpointing
```

Split train and test by task/World **before** sampling rounds, to prevent
sibling decision states from the same source World leaking across splits.

## Tests

```bash
python -m unittest discover -s environment/tests -v
```

GitHub Actions runs this suite and compiles the environment on every relevant PR.
