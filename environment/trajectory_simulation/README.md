# Stage 3 — Clean Sub-Agent Reports → Main Scheduling SFT

Stage 3 consists of **two explicitly separate, sequential phases**. It does
not require or train on an entire autonomously executed task episode.

The frozen Stage-1 ProgramIR / Fragment DAG and accepted Stage-2 question
supply the immutable semantics. The subtask answers always come from public
source-replayed symbolic execution; the LLM writes only language.

### Answer-blind simulated Web Search (newly constructed tasks)

New `web_search` and `hybrid` tasks opt into
`task_local_ranked_public_index_v1`. Stage 3 builds a **teacher-private
search catalog** from ALL publicly rendered task sources, including
distractor pages. Only the public Main question, its assigned task instruction,
and snippets returned by earlier simulated searches may form queries.
The ranker never accepts a Gold answer or Gold source list.

`GoogleSearchAction` is simulated as query → deterministically ranked hits
(title, resource handle, snippet) with pagination and bounded query expansion.
The matching resources can then be inspected with the assigned O1 tool
(`ExtractUrlContentAction`, `VideoAnalysisAction`, etc). Every Gold-required
discoverable source must actually appear among independently retrieved hits
before public Source replay can certify the subtask. Otherwise the world is
quarantined with `public_search_witness_unreachable` rather than silently
inserting the answer-bearing source. A Source from an actual initial attachment
is inspected without a fabricated search.

The Sub-Agent bank retains the logged simulated queries/hits in
`simulated_tool_provenance.search_calls` and a private `search_audit`.
Main prompt/history never receives the index or future search hits. The
inspection set remains teacher-guided by the frozen proof AFTER retrieval;
this is not an autonomous search-policy benchmark. It tests whether a
public-only search *could locate* the necessary evidence, not whether an
untrained agent would choose the same queries or click decisions. The corpus
is local to each synthetic world and uses synthetic Wiki/YAGO aliases, not
real online pages.

Old frozen tasks lack the new retrieval policy and remain on the legacy
replay pathway for reproducibility. To exercise web-first simulation, rebuild
Task Construction/freeze and Stage 2 questions in a **new output workspace**,
then rerun Stage 3. Merely rerunning a Stage-3 command on an existing v6 Bank
does not retrofit historical search actions.

### Candidate discovery is not an initially supplied roster

An unbound `FIND_ALL` or `FIND`-by-type operation starts with **unknown
candidate names**. The canonical Main delegation asks the Sub-Agent to find
entities in the task-related evidence first, then apply the frozen filters.
The exact root result is reconstructed from **published source claims and
entity types**, not injected into the Main prompt as an oracle list.
Sub-Agent `summary` / `trace_summary` may name the candidates AFTER
describing how the evidence identified them; they must not falsely claim
that a complete candidate list was supplied at the outset when other
sources had to be discovered. In contrast, a later-round Main delegation
may explicitly name entities in earlier, already published Sub-Agent results.

This is source-backed *simulation*. Public aliases (e.g. synthetic city
names) deliberately differ from original YAGO entity labels, and the
`simulated_tool_provenance` log records `live_tool_calls=false`.
A real public web search may not resolve these aliases. Do not claim this
pipeline measures successful real-web search, or that the task corpus is
the whole Internet. A truly live-search benchmark would require an
independently retrievable evidence index or resolvable original names.

The new canonical instruction text is supported by the public replay
parser as the same `FIND_ALL` / `FIND` operation; the older canonical
sentences remain readable for existing banks. Old Stage-2 questions and
already written Stage-3 banks are **not** silently rewritten. To change
question wording, regenerate/reauthor the Stage-2 artifacts against the
same verified ProgramIR, then rebuild the Stage-3 Subtask Bank.

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
   remain forbidden. The output contract has **exactly two nonempty JSON
   strings**, `summary` and `trace_summary`; trace bullets belong inside
   the latter string, separated by escaped newlines. For local vLLM, Stage 3.1
   requests a JSON Schema response for those two fields and falls back to
   `json_object` only if the endpoint explicitly rejects schema support.
   A valid nonempty `trace_summary: list[str]` is losslessly joined into
   one string before the SAME deterministic and semantic checks, avoiding
   unnecessary re-generation. `natural_feedback_normalized` counts
   successfully checked writer responses that required such conversion.
   The reviewer distinguishes verified FIND/FILTER/INTERSECT/SELECT_BETWEEN
   and backward relation results from unsupported direct Source claims;
   facts and relation directions must still be correctly attributed.
   Internal observation keys and invented quoted search queries are rejected.
   If the local LLM returns missing/blank
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
