# V4 Question naturalization

V4 keeps the Macro TaskGraph, evidence nodes and clean feedback frozen.
Only public Question text can be rewritten. The pipeline follows
`Author → Critic → local repair (up to 2) → accept/fallback`, adapted from
`environment/question_generation/realization.py`.

## Naturalize the existing V4 pilot

```bash
python -m envir.macro.v4_cli naturalize \
  --scenes envir/data/macro_v4/private_clean_scenes_v4.jsonl \
  --output-dir envir/data/macro_v4_naturalized_10 \
  --max-records 10 \
  --question-api-base http://127.0.0.1:8807/v1 \
  --question-model local-qwen3.5-9b \
  --question-workers 4 --max-revisions 2
```

Inspect the first 10 Question outputs and audit records. For all 100, use
a new `--output-dir` and omit `--max-records`.

Critic defaults to the same model with independent reviewer instructions
and temperature zero. Optional flags are `--question-critic-model` and
`--question-critic-api-base`. `build` also accepts these flags, while
unconfigured builds remain deterministic and do not call an LLM.

The Author sees only a projection of the original Task goals, deliverables,
goal-level handoffs, user setting, issue, and public resource categories.
No private graph IDs, EvidenceGraph, source facts, Feedback, or final answers.
An independent Critic assesses naturalness and whether the same Macro work
goals still support the question. Up to three local issues feed revisions.
Explicit source/setting/numeric/intent changes are guarded deterministically.
Failed or uncertain revisions fall back to the original question.

Outputs include `private_clean_scenes_v4.jsonl`, `public_questions_v4.jsonl`,
`macro_v4_summary.json`, and `question_review_audit_v4.jsonl`.
The audit contains every attempted question, reviewer verdict, repair,
task contract fingerprint and accept/fallback reason.

```bash
python -m envir.macro.v4_cli replay \
  --scenes envir/data/macro_v4_naturalized_10/private_clean_scenes_v4.jsonl
```

Critic approval is a model judgment, not a formal proof of semantic
equivalence. The source-world simulation and the clean-task assumptions
remain unchanged; no evidence conflict or replanning is introduced.

## Live progress, incremental writes, and automatic resume

V4 question naturalization now shows a live `tqdm` progress bar (total
questions, accepted, fallback, accepted after revision), including the
already completed prefix when resuming. LLM-enabled `build` uses the same
writer; deterministic `build` remains a batch operation.

Every completed Question is committed in original Task order to three
JSONL files: `private_clean_scenes_v4.jsonl`,
`public_questions_v4.jsonl`, and `question_review_audit_v4.jsonl`.
The default `--flush-every 1` flushes and fsyncs every completed record,
then atomically updates `question_progress_state_v4.json`. You can raise
`--flush-every` to reduce disk sync overhead, at the risk of repeating
at most the last uncommitted group after an abrupt process crash.

If interrupted, **run the same command again, with the same output
directory and no `--overwrite`**. The process verifies the committed
input/model/code signature, byte offsets and hashes, discards any
uncommitted tails, and continues from the first incomplete Question.
Already committed Questions are not sent to the model again. The
summary JSON is published after the entire stream completes.

Example:

```bash
python -m envir.macro.v4_cli naturalize \
  --scenes envir/data/macro_v4/private_clean_scenes_v4.jsonl \
  --output-dir envir/data/macro_v4_naturalized_100 \
  --question-api-base http://127.0.0.1:8807/v1 \
  --question-model local-qwen3.5-9b \
  --question-workers 4 --max-revisions 2 --flush-every 1
```

To resume, rerun exactly that command; the progress bar starts at the
committed count. `--no-progress` hides only the display. Use
`--overwrite` **only** when you intentionally want to discard existing
Question outputs and regenerate them from the beginning.

A changed input source, model setting or reviewer code will deliberately
block resume; use a new output directory or explicitly restart. The
writer uses a single-writer lock to prevent overlapping runs against the
same output directory.

## V4.2 naturalness and task-semantic fidelity update

After a 100-Question Qwen3.5-9B pilot (84 accepted, 16 fallback),
the reviewer distinguishes harmless wording changes from factual additions.
Author and Critic now explicitly screen for (a) unsupported specificity,
such as changing "laboratory instrument" into "spectrometer" or inventing
specific record types; (b) adding a "currently" or "still" state not present
in the Task; (c) changing the object being ranked or assessed; and
(d) exposing a hidden "first investigate, then rank" work schedule.

Surface checks are less dependent on exact keywords or the literal
phrase "documented findings". For instance, "our museum" can preserve
the case setting of "municipal museum" if the reviewer finds the context
consistent. This can reduce false rejections, while the independent
Critic still checks the complete immutable Macro Task contract.

Newly accepted LLM Questions should not keep copied "documented findings"
boilerplate. The original deterministic Question is **not** changed for
existing inputs: it remains the reference semantic authority. If the
original accidentally repeated the exact support request twice, failed
LLM generations may use a narrowly defined deterministic cleanup of
only that duplicate clause, recorded in the review metadata.

V4.2 reviews are versioned. Old V4.1 Questions can still pass structural
replay checks but are **not retrospectively certified** to have passed
V4.2's stricter semantic guards. Because the review code has changed,
use a NEW directory rather than resuming an old V4.1 checkpoint.

Run the same frozen 100 TaskGraph/Evidence scenarios again:

```bash
python -m envir.macro.v4_cli naturalize \
  --scenes envir/data/macro_v4/private_clean_scenes_v4.jsonl \
  --output-dir envir/data/macro_v4_naturalized_v42_100 \
  --question-api-base http://127.0.0.1:8807/v1 \
  --question-model local-qwen3.5-9b \
  --question-workers 4 --max-revisions 2 --flush-every 1
```

Compare existing and new naturalizations with the READ-ONLY diagnostic:

To assess the old 100-record pilot, point `audit-questions` to its
existing `private_clean_scenes_v4.jsonl` file. Then use the same command
on the new v4.2 output for a paired comparison:

```bash
python -m envir.macro.v4_cli audit-questions \
  --scenes envir/data/macro_v4_naturalized_v42_100/private_clean_scenes_v4.jsonl \
  --output envir/data/macro_v4_naturalized_v42_100/quality_audit.json
```

The audit reports questionable added details, internal-order leakage,
location paraphrases, stock phrasing and fallback reasons. Warnings are
heuristic signals for manual comparison, not a formal proof of semantic
equivalence. No Macro Task, EvidenceGraph, Feedback or runtime policy
changes are introduced by this phase.

## V4.3: simpler Macro Task-aware LLM Critic

V4.3 asks one core semantic question: would the SAME Macro Tasks
still be a reasonable and sufficient work plan for the rewritten
user Question? The Author has strict instructions not to invent entities,
sources, facts, current states or new work, but broad freedom to write
natural MMQA/OmniGAIA-style user requests.

Author and Critic receive the overall goal, scenario, Task goals,
deliverables, and semantic prerequisite links from the Macro TaskGraph.
The model does not see private evidence, summary traces, full internal
Task IDs or reference execution waves. Unlike earlier versions, this
view is for semantic understanding, not a strict lexical writing contract.
Questions do not need to name every Task or preserve original wording.

The deterministic guards now check only basic length, malformed formatting
and private Task structure leakage. The LLM Critic judges meaning, task
coverage and naturalness. It may request up to three local repairs; the
Author can revise twice. Failed or uncertain reviews safely fall back.

Macro TaskGraph SHA-256 and model-view hashes still protect data integrity
and reproducibility. They do not enforce rigid wording. The old V4.1/V4.2
reviews remain replay-compatible, but do not claim the newer review policy.
Model Critic PASS is still a fallible judgment, not a semantic proof.

Use a NEW output directory for the same 100 fixed Macro TaskGraphs:

    python -m envir.macro.v4_cli naturalize \
      --scenes envir/data/macro_v4/private_clean_scenes_v4.jsonl \
      --output-dir envir/data/macro_v4_naturalized_v43_100 \
      --question-api-base http://127.0.0.1:8807/v1 \
      --question-model local-qwen3.5-9b \
      --question-workers 4 --max-revisions 2 --flush-every 1

Checkpoint resume, live progress, ordered writes and read-only quality
audits continue to work. A changed review policy prevents resuming old
checkpoints; it does not invalidate completed old data.

## V4.4: goal-first Author and diagnostic skeptical Critic

The earlier 100-record V4.3 local Qwen3.5-9B run accepted all 100
Questions with only one revision, but manual inspection found procedural
subtask checklists and invented specific source names. V4.4 is intended
to improve actual user-question quality, NOT to optimize a passing rate.

Author still sees all semantic Macro Tasks, prerequisites, and final
deliverable. Author must describe the USER'S END GOAL rather than the
Agent's internal workflow; intermediate Tasks are not new deliverables.
Crucially, generic resource categories do not authorize invented named
source types, even if the specific type is plausible in that domain.

The Critic must separately return four diagnostic axes:
- goal_alignment: same / changed / uncertain
- task_sufficiency: sufficient / insufficient / uncertain
- user_framing: user_goal / workflow_instructions / robotic
- unsupported_additions: candidate-only phrases, or an empty list

A PASS is accepted only when goal_alignment=same,
task_sufficiency=sufficient, user_framing=user_goal,
unsupported_additions=[], naturalness=acceptable and differences=[].
Contradictory PASS results are treated as invalid Critic protocol and
retried once on the same candidate, then safely fall back if still invalid.
Critic issues can be locally fixed by Author, up to --max-revisions.
No arbitrary target acceptance rate or mandatory question mark exists.

Question generation remains semantic, not keyword-contract based:
only simple format and internal-structure leakage checks are deterministic.
V4.1/V4.2/V4.3 records retain backward-compatible replay validation;
new V4.4 accepts only its own diagnostic Critic review records. The
clean Macro TaskGraphs, Evidence, Feedback and summary traces do not change.

Use a NEW output directory for the same original 100 TaskGraph scenes:

    python -m envir.macro.v4_cli naturalize \
      --scenes envir/data/macro_v4/private_clean_scenes_v4.jsonl \
      --output-dir envir/data/macro_v4_naturalized_v44_100 \
      --question-api-base http://127.0.0.1:8807/v1 \
      --question-model local-qwen3.5-9b \
      --question-workers 4 --max-revisions 2 --flush-every 1

Verify the output:

    python -m envir.macro.v4_cli replay \
      --scenes envir/data/macro_v4_naturalized_v44_100/private_clean_scenes_v4.jsonl

Use the read-only paired quality report for naturalness and model review
diagnostics, including mean word count, procedural connectors, question
marks, leading 'please', workflow-review rounds, goal-shift rounds,
and unsupported-addition rounds:

    python -m envir.macro.v4_cli audit-questions \
      --scenes envir/data/macro_v4_naturalized_v44_100/private_clean_scenes_v4.jsonl \
      --output envir/data/macro_v4_naturalized_v44_100/quality_audit.json

These diagnostic axes and review rates are fallible LLM judgments,
not independent proof of semantic equivalence. Inspect actual text
alongside the frozen TaskGraph when evaluating research quality.


## V4.5 — MMQA / OmniGAIA question style, Macro TaskGraph as semantic authority

**Authority:** the FINAL Macro Task goal and deliverable plus typed prerequisite
work define an acceptable question. The old deterministic Question is an
optional, sometimes unnatural writing hint, NOT an equivalence target.
The Author can freely change wording, rhetorical structure, and presentation
so long as the same graph remains a reasonable way to answer the final
information need. The Critic sees the Macro TaskGraph and the candidate,
**not** the old deterministic Question.

The Prompt's style priors come from official examples of:

- [MultiModalQA](https://github.com/allenai/multimodalqa) (Talmor et al.,
  ICLR 2021): concise, referential/relational questions; comparisons and
  cross-source dependencies are implicit rather than procedural task lists.
  Original questions were human-rephrased after symbolic composition.
- [OmniGAIA](https://huggingface.co/datasets/RUC-NLPIR/OmniGAIA)
  (Li et al., 2026): situational and descriptive clues, precise information
  requests, sometimes linked conditions/compound outputs requiring several
  hidden reasoning/tool-use steps. Do not copy dataset question contents
  or claim the simulated world contains real video/audio observations.

The Author's *optional_reference_hint* remains available for a weak wording
cue but may be ignored. It has no binding semantic authority. The Critic
evaluates the final target, graph sufficiency, user-style phrasing and
unsupported additions, without comparing against the reference text. This
removes the old rule that an unchanged deterministic sentence must be rejected
before Critic review: a good final question is judged on its merits.

The 4-axis V4.4 review schema stays the same, but V4.5 records carry
`v4.5_graph_primary_benchmark_style` and preserve replay compatibility
with older V4.1–V4.4 data. The reference remains in **teacher-only audit**
metadata for provenance and safe fallback, not as user-visible content.
This does **not** prove actual evidence sufficiency: V4 clean source worlds,
Sub-Agent Feedback and TaskGraph are frozen, and whether a task truly
computes the intended answer remains a separate execution-level question.

To run a paired question-only comparison on the same 100 fixed graphs:

```bash
python -m envir.macro.v4_cli naturalize \
  --scenes envir/data/macro_v4/private_clean_scenes_v4.jsonl \
  --output-dir envir/data/macro_v4_naturalized_v45_100 \
  --question-api-base http://127.0.0.1:8807/v1 \
  --question-model local-qwen3.5-9b \
  --question-workers 4 --max-revisions 2 --flush-every 1

python -m envir.macro.v4_cli replay \
  --scenes envir/data/macro_v4_naturalized_v45_100/private_clean_scenes_v4.jsonl

python -m envir.macro.v4_cli audit-questions \
  --scenes envir/data/macro_v4_naturalized_v45_100/private_clean_scenes_v4.jsonl \
  --output envir/data/macro_v4_naturalized_v45_100/quality_audit.json
```

`audit-questions` still includes older **reference-relative lexical warnings**
(e.g. changed setting words); these are diagnostic signals, not V4.5 graph-
semantic rejection criteria. Manual comparison should inspect the candidate
against the terminal Macro Task work product and its typed dependencies,
especially operations such as 'rank responses' vs 'rank reservations'.
