# Trusted Main-Agent orchestration (Stage 3)

This module generates clean Main decisions for training and testing, using
reviewed Stage-2 reports and the same O1-compatible format as the baseline.

```bash
python -m environment generate-main-turns \
  --config environment/configs/main_orchestration.yaml
```

The default output is `environment/data/stage3/`. For an isolated pilot,
specify `--max-scenes 10 --output-dir environment/data/pilots/stage3`.

Stage 3 selects its input **from complete, published Stage-2 Scene reports**,
not from the first `max_scenes` Stage-1 rows. The Stage-1 file provides the
canonical Scene definitions and ordering; the Stage-2 `subtasks.jsonl` bank
determines which Scenes are eligible. The selector checks that every Macro ID
is present exactly once and its report was reviewed and accepted, before
taking up to `max_scenes` eligible Scenes. The normal full replay/source/
provenance verification still happens in `build_episode`.

Stage-2 rejected/unpublished Scenes are **skipped**, not counted as Stage-3
rejections or consumed from the Stage-3 scene limit. It is valid for the
number selected to be less than `max_scenes` when Stage 2 published fewer
complete Scenes; the run-level summary separates `stage2_selection` counters
from actual Stage-3 `accepted_scenes` / `rejected_scenes`. If no complete
Stage-2 Scene is available, Stage 3 raises a clear input error before
creating output. The manifest binds the full Stage-1 and Stage-2 input
files, and a previously journaled run still cannot silently change its
selection when either input file changes.

## Compact generation-only Author/Critic context

In `freeze_dispatch: true` mode, Stage 3 sends the prose Author and
independent Critic a compact, **lossless evidence view** rather than the
baseline O1 prompt. It retains the entire user question, provided initial
observations and source catalog (including source handles for legacy Scenes),
current attempt budget, and the complete cumulative O1-formatted Sub-Agent
history. Each pending investigation still carries its permitted sources,
required output fields, and prior-result context, and the Critic still sees
the round-specific evidence and task-prose contracts.

Only redundant static O1 instructions are omitted: parallel-decomposition
tutorials, model-pricing tables, full tool descriptions/JSON schemas and output
examples. Nothing in the Main-visible history is truncated or summarized.

For rejected authored task prose, Stage 3 gives targeted revision
hints for invalid context, generic repeated task instructions,
`invented_visible_identifier`, and `source_scope_missing`. Identifier
hints show literal IDs from the current Main-visible question/history
and current public task context (local matches first); source hints give
the exact permitted source handles and their public titles. They do
not reveal private Oracle values, guess the correct identifier, or
silently repair an unsupported claim. The original strict Guard still
rejects ungrounded IDs and missing source handles. The failed-action
journal records context/instruction lengths and types (not candidate
text), to distinguish a short context from an overlong one. No guard,
source permission, frozen dispatch or Critic requirement is relaxed.

The Stage-3 Author now reasons as an investigator rather than describing
private scheduling machinery. A generation-time Guard rejects phrases such as
"fixed dependency graph", "current fixed slot", and "fixed task graph";
the rejection yields an evidence-grounded rewrite hint on the next attempt.
Ordinary task-domain phrases like "fixed deadline" remain valid. Pure
reference-instruction boilerplate is still rejected, but adding genuine
source-specific clues visible in the question/history is not rejected merely
because an instruction also contains a copied generic clause; the independent
Critic remains responsible for its semantic specificity.

## Grounding and O1 writing style

A direct comparison against the two supplied real OmniGAIA/O1 trajectories
(4 Delegate rounds, 11 instructions) showed a useful distinction: real O1
`task_instruction` text gives concrete evidence actions, query clues and
requested findings, while `context` is a short situation handoff. The real
example Main `reasoning` fields were empty; our MAESTRO training target
intentionally retains evidence-driven reasoning, rather than imitating
that empty field. The **stable, concise Delegate Author prompt** requests
actionable investigations grounded in visible source IDs, named clues and
required return values; it deliberately avoids long prescriptive lists of
extra actions/tools and repeated example success phrases. This restores
the Author language used before PR #250 after a paired local-Qwen 48-scene
pilot regressed from 42/48 to 31/48 complete Scenes and from 4 to 20
generic-template Guard errors. The first local run is evidence of a
regression, not proof of a prompt-only causal effect. The hard tool/model
scope, frozen dispatch, Stage-3 Guard, O1 history and code-attached prior
results remain unchanged.

Stage-3 independent review now preserves two distinct boundaries:

- **Delegation:** the Critic checks the AUTHOR's reasoning, instructions and
  contexts against the code-frozen current-round task and its public
  evidence. Rejection suggestions to split this round or replace output
  fields trigger one independent re-review, never automatic acceptance.
  Genuine invented relations, unsupported source claims, dependencies and
  unassigned fields are still rejected.
- **Completion:** the final answer and deterministic evidence quotation
  are not subject to LLM recomputation. The Critic checks only NEW
  Author reasoning against previously returned public results, without
  conflating a 2025+ event-ledger filter with independent metric dates.
  An apparent demand to recalculate/change the frozen answer triggers a
  second independent prose-only review. Continued disagreement still
  produces the explicitly non-reviewed `teacher_grounded_completion_fallback`.

The Delegate Prompt remains in its stable wording; independent
Critic scope and optional out-of-scope re-review are unchanged.
`llm_reviewed_completion` now retains the Author's whole sentences,
never cuts a word off at character 420, and appends only a short verbatim
Sub-Agent Result citation when one is missing. It does **not** append the
long, deterministic generic completion conclusion to every reviewed
sample. The Critic sees exactly the retained authored sentences,
not the added citation; the final `completion_issues` Guard validates
the published text and code-frozen answer as before. If no complete
sentence fits the safe budget, the Author may retry; exhausted retries
now reject the strict Stage-3 Scene rather than publishing a fallback.
No new model role,
dispatch heuristic, Oracle mutation or relaxation of the complete-Scene
publication contract.

**Strict Completion policy:** `llm_reviewed_completion` is an
independently Critic-approved final rationale. By default, exhausted
Author/Critic retries that would yield
`teacher_grounded_completion_fallback` cause the entire Scene to fail.
Only explicit legacy `allow_delegate_fallback: true` enables publishing
that non-reviewed Completion mode. All strict successes therefore have
LLM-reviewed Delegate and Completion rounds without post-export filtering.
Completion refusal details remain in `failed.jsonl`/the Scene journal.

The **training prompt is unchanged**: each turn still stores
`input.subtask_history`, and `o1_prompt_from_turn` rebuilds the original
`OmniGAIAMainAgentPrompt` exactly. Validation of proposed prose uses the
original full O1 prompt for identifier allowlisting. Private review audits
record `full_o1_prompt_chars` and `author_context_chars` for before/after
size comparison; these are character counts, not token counts or measured
speedups. Existing run journals have code signatures and should be
regenerated in a new output directory after this change.

The deterministic compiler owns the legal Macro batches, dependencies,
task count and order, model and tool routing. With `freeze_dispatch: true`,
**the same Qwen Author call** now writes `reasoning` plus natural
`task_instruction`/`context` for each preassigned slot. The Author receives
the current baseline O1 question/history, allowed source IDs and public
titles/modalities, required output fields and the fixed slot's existing
instruction as a semantic reference. No hidden future result, DAG or
private teacher alignment is passed to the Author.

Before publication, programmatic Guards check task count/order, permitted
source references, invented identifiers and fixed model/tools. The compiler
adds required return fields as needed and appends *only already delivered*
public Sub-Agent result text verbatim to dependent contexts. An independent
Critic reviews **all three natural fields** for source specificity,
stage-local reasoning and no premature claims. The actual Macro dispatch,
execution sequence and Stage-2 feedback are not altered. Stage 4's existing
frozen reasoning-only author path remains unchanged.

The output remains `input.subtask_history` →
`output.{action,reasoning,params}`, while the internal authoring record
indicates `authored_fields: [reasoning, task_instruction, context]`.
Private contracts and reviewer diagnostics stay in
`private/teacher_alignment.jsonl`.

Sub-Agent summaries are normalized as **copies** of previously reviewed
reports. No structured evidence, Result, status, FactRef or source value is
modified. A rejected Episode never leaks its partial turns into public
training data.

Each accepted Episode also contains a stable `clean_reference_id`, derived
from the frozen World, Macro DAG and reviewed Stage-2 Finish Results. The
reference ID does not depend on stylistic Main reasoning.

Stage 4 normally compiles an equivalent deterministic trusted prefix. To
reuse the exact reviewed Stage-3 trajectory instead, configure
`input.clean_episodes: environment/data/stage3/episodes.jsonl` in Stage 4.
The runner validates its returned evidence, action batches and completion
against an independently compiled clean reference before any fault injection.

## Complete local task contracts and strict no-fallback Stage 3

With frozen Stage-3 dispatch, `task_contract.py` projects only the current
Macro operators into a generation-only contract. It preserves source/key
bindings, local data flow, filter owners and thresholds, joins, computations
and output fields. No source rows, future results or executable node IDs are
projected. Unsupported operators fail closed. The training prompt and Stage-1/2
inputs remain unchanged.

The Critic checks the full local contract and returns located issues. Repairs
resolve those locations back to authoritative contract requirements and include
the previous public-language draft. Unvalidated reviewer instructions cannot
remove required output fields. Candidate/review records remain in the private
attempt audit, including attempts that eventually succeed. Lexical condition
checks are necessary checks, not a proof of general natural-language equivalence.
Style findings are advisory; factual and scope errors still require revision.

**Default: no fallback.** Set `generation.allow_delegate_fallback: false`
(the maintained config default) to treat any exhausted Delegate Author/Critic
revision as a **failed Scene**. The generator does not construct or publish a
`contract_frozen_delegate_fallback` turn in this mode. A Completion that
exhausts its revisions and would have returned
`teacher_grounded_completion_fallback` is also **counted as a failed Scene**,
so the displayed accepted-Scene rate directly measures fully LLM-reviewed
trusted trajectories. The last failed round and its Author/Critic diagnostics
remain in the private failure journal; no incomplete Scene is included in
`episodes.jsonl` or `main_turns.jsonl`. Previously completed rounds remain
checkpointed and are not published when the Scene fails. This is one success
rate and one standard public dataset; no extra export categories are needed.

The **single explicit legacy opt-in** `generation.allow_delegate_fallback: true`
(or CLI `--allow-delegate-fallback`) restores the older behavior in which
supported contracts can produce `contract_frozen_delegate_fallback` and
Completion may use `teacher_grounded_completion_fallback`, both labeled
unreviewed. Its accepted count is not a strict LLM-reviewed rate; do not
compare it directly to the new strict default. Run manifest signatures
include the fallback policy and tracked code, so rerun into a new output
directory instead of resuming an older fallback-enabled journal.

Previously returned evidence is attached only once when the Author copied
the same handoff verbatim.

Capable backends receive a strict JSON schema requiring located Critic issues.
Legacy reviews can identify an exact quoted candidate claim but cannot supply
unvalidated repair instructions. For a local join of two previously returned
complete dossier-ID lists, the contract may contain a deterministic literal
intersection witness. Only fully recognized list forms are accepted; examples,
partial narratives and count mismatches produce no witness. This never reads
private evidence rows or infers a downstream project/entity value.
