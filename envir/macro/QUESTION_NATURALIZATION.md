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
