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

For rejected authored task prose, Stage 3 now gives targeted revision
hints for short/invalid context and generic repeated task instructions,
using only current-round public sources and fields. The failed-action
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
that empty field. Stage-3 prose prompts now prioritize specific current
evidence and next unresolved needs over repetitive "successfully /
next logical step" narration. An Author may name an **already assigned**
tool to clarify a task, but can never change tools, models, source access
or invent steps; existing O1 history and code-attached prior results stay
unchanged.

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

These are prompt/review-scope changes only: no new model role, dispatch
heuristic, Oracle mutation or relaxation of the complete-Scene publication
contract.

**Completion quality tiers:** `generation_mode=llm_reviewed_completion`
means an independently Critic-approved final rationale. When the Author/
Critic cannot agree, Stage 3 may still publish a complete, code-verified
clean reference as `generation_mode=teacher_grounded_completion_fallback`.
Those fallback turns are explicitly **not LLM-reviewed positive training
labels**. Keep them for trusted replay and Stage-4 clean/fault pairing, but
exclude them when selecting strictly LLM-reviewed SFT/GRPO examples, e.g.
filter `main_turns.jsonl` on `generation_mode`. Completion fallback
failures and the non-reviewed designation remain in the private audit; the
`summary.json` reports completion modes separately.

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
