# Stage 2 — Certified Subtask Goals → Natural Main Questions

Stage 2 is now **two independently resumable jobs**. Stage 1 freezes
ProgramIR, rendered resources, the canonical question and the Gold fragment
DAG. Neither Stage 2 job edits task semantics or generates Gold answers.

## Stage 2A — independently generate and verify Subtask Goals

\`\`\`text
Frozen ProgramIR + compact Gold DAG
    └─ compile_delegation_outline (deterministic PUBLIC projection)
        └─ each fragment concurrently/independently:
            ├─ Structured Intent compiler (action, target, output; no LLM)
            ├─ Local Goal Author (intent + local steps + direct dependencies)
            ├─ deterministic checks (numbers, comparisons, foreign branch literals)
            ├─ Local Goal Critic (main intent + output alignment + no contradiction)
            └─ bounded local repair, max --max-goal-revisions
        └─ require ALL fragments PASS; otherwise quarantine whole task
            └─ deterministic DAG assembly + goal_certificate
\`\`\`

The writer and Critic **do not** see the full mechanical reference,
other siblings' authored Goals, private world facts, Gold answers or
future observed results. The LLM returns only \`{"goal": "..."}\`;
fragment IDs, \`depends_on\` and \`waves\` are determined by the compiler.
The Goal is a short **Subtask Intent**, not a complete executable query.
It should describe the main work, principal target and expected output, in
roughly 15–45 words. Exact numeric thresholds, dates, intermediate filters and
other details may be omitted because the full public semantic contract is
saved separately. The Local Critic checks only: (1) main-task relevance,
(2) expected output alignment, and (3) no *explicit* contradictions.
For TERMINAL tasks, the writer/reviewer receives a NARROW public result
hint (COUNT vs QUERY_NAME vs QUERY_ATTR, output kind, and unambiguous
entity type where available). A COUNT goal must describe counting, and an
Event-name result cannot become names of Individuals. Explicitly written
attribute owners are also checked (e.g. `male population` is not a person's
sex, and a Place end date must not become an Event end date).
This remains a lightweight intent check; numeric thresholds and intermediate
steps can still be omitted.
A model summary that says "matching the specified date conditions" may
PASS; one that explicitly changes "not equal to 1160" into "after 1160"
must REVISE. Deterministic guards no longer require any local number to
appear in the Goal. They check explicit comparison contradictions, foreign-branch
numerals, and narrowly defined output / attribute-owner errors. An invalid Critic JSON is retried on the SAME Goal, including a
JSON-object fallback after schema-mode retries, without rewriting it.
A PASS remains an auditable model judgment, not semantic proof.

This is a bounded **automatic fidelity review**, not a formal mathematical
proof of natural-language equivalence. A reviewer may miss semantic
changes; ambiguous or repeatedly failed tasks are quarantined, not promoted.

Run from the MAESTRO root with the existing configured local vLLM:

\`\`\`bash
python environment/generate_goals.py \
  --max-records 50 \
  --max-goal-revisions 2 \
  --output-dir environment/data/stage2a_intents_v5_50 \
  --question-workers 4 \
  --blueprint-workers 2 \
  --flush-every 1
\`\`\`

Stage 2A outputs are separate and checkpointed:

| File | Purpose |
| --- | --- |
| \`certified_blueprints.jsonl\` | Only tasks whose **every** Goal passed, with Blueprint, per-fragment reviews and \`goal_certificate\` |
| \`rejected_blueprints.jsonl\` | Failed tasks and isolation reason; never forwarded |
| \`goal_review_audit.jsonl\` | Every writer/reviewer attempt, local feedback and per-task provenance |
| \`goal_generation_summary.json\` | Task yield, reviewer and author calls, failure reasons |
| \`goal_progress_state.json\` | Crash-safe ordered checkpoint; resumable with identical inputs/code/model settings |

Older v1-v4 Stage-2A Goal certifications used different Author/Reviewer
contracts. Preserve them for comparison; create a new v5 output directory.
A prior **v5** audit can be reused using --reuse-goal-audit pointing to
goal_review_audit.jsonl. Passed Goals are reused only when model roles,
review version and exact public outline hash match. Cached fragments are
revalidated before any new certificate is issued.

To intentionally overwrite an earlier Stage-2A run, add
\`--restart-goals\`. To preserve previous experiments, prefer a new
output directory. A failed Question in Stage 2B does **not** invalidate
the completed Stage-2A Goal dataset.

## Stage 2B — author and review only certified Questions

\`\`\`text
Stage 2A certified_blueprints.jsonl
    └─ join frozen task IDs / discard non-certified originals
        └─ independently rebuild compact Gold DAG and replay 2A certificate
            └─ deterministic public scope Brief
                ├─ group filters by their owning scope
                ├─ preserve directed relation endpoints, anchors and set joins
                ├─ attach Stage 2A Goals as context for the Author
                └─ Question Author (Brief only; no mechanical reference)
                    └─ deterministic guards + Critic (FULL public contract)
                    └─ Stage 2B certified Question, including signed 2A review
                        └─ Stage 3: verified source-based Sub-Agent reports
                                      → just-in-time Main delegations
\`\`\`

\`\`\`bash
python environment/generate_questions.py \
  --max-records 50 \
  --blueprints environment/data/stage2a_intents_v5_50/certified_blueprints.jsonl \
  --max-revisions 2 \
  --output-dir environment/data/stage2b_scoped_v5_50 \
  --question-brief-mode branch \
  --question-workers 4 \
  --flush-every 1
\`\`\`

**Stage 2B requires \`--blueprints\`.** Of the selected 50 originals, only
the Stage-2A certified task IDs are submitted; the summary includes
\`input_candidates\`, \`skipped_uncertified\`, and
\`certified_input_count\`. A tampered or stale Goal certificate is checked
against the frozen ProgramIR **before any Question Author call**. Each
accepted Question carries the source \`delegation_blueprint\`,
\`goal_reviews\`, \`goal_certificate\` and its own Stage-2B
\`surface_certificate\`. Stage 2B has **zero Goal Author/Goal Critic calls**;
it can be rerun against the same frozen certified Goal dataset to improve
Question prompts without regenerating Goals.

Stage 2B Author defaults to `--question-brief-mode full`. To try the
compact input shown above, pass `--question-brief-mode branch`. The v2 Brief
projects the public scopes already checked by the scoped reference builder.
Each scope identifies its inputs and owns its filters. Directed relations
carry their subject and object, and identification clues expose the public
type and conditions. The Critic receives the full scoped reference and
semantic contract. Stage 3 recomputes the Brief using the version recorded
in the review. Earlier Branch Brief hashes replay with the v1 compiler.

For a paired A/B baseline on the SAME v5 certified Goals, run
`--question-brief-mode full` in a separate output directory to use
the full-reference Question Author with the same Critic.

Stage-2B outputs and checkpoints retain the existing names:
\`realized_capability_worlds.jsonl\`,
\`question_rejected.jsonl\`,
\`question_review_audit.jsonl\`,
\`question_generation_summary.json\`,
and \`question_progress_state.json\`.

### Contract versions and Stage 3

- Goal Blueprint: \`maestro.delegation_blueprint.v2\`.
- Stage-2A Goal certificate: \`maestro.stage2a.certified_blueprint.v5\`.
- Stage-2B Question certificate for these Goals:
  \`maestro.stage2_surface_cert.certified_goals.v3\` (review contract 32).
- Existing earlier v2 Question certificates continue to replay for
  compatibility. **New Stage-2B CLI tasks can only publish v3** certificates.
- Stage 3 recompiles the original compact DAG and verifies both
  certificates; its Gold answers, public source replay and Main state
  visibility boundaries are unchanged.

A Goal review certificate represents an audited model judgement, not a
guarantee of perfect semantic fidelity. Continue measuring Stage-2A
fidelity and Stage-2B false-pass/false-reject rates independently on
unseen frozen tasks. No one-to-one natural-language operator mirroring
is required of the final Question; it should remain a coherent user request.

## Offline Stage 2B shadow audit

The shadow audit uses saved first and final drafts, with optional labeled
cases. It rebuilds the public graph from the matching frozen inputs and
verified blueprints. The parser sees the candidate wording alone. The
structural comparison keeps directed edges, condition scope, set operations,
typed comparisons, and the answer operation. Same-range AND and consecutive
filters can normalize to the same expression. Each filter still quantifies
over its own matching recorded value.

Run from the repository root with the usual Stage 2 model configuration:

```bash
python -m environment.question_generation.shadow_audit \
  --audit environment/data/stage2b_full_v5_50/question_review_audit.jsonl \
  --audit environment/data/stage2b_scoped_v5_50/question_review_audit.jsonl \
  --cases environment/tests/fixtures/stage2_shadow_v5_cases.jsonl \
  --blueprints environment/data/stage2a_intents_v5_50/certified_blueprints.jsonl \
  --output environment/data/stage2b_shadow_v1/paired.jsonl
```

For a small labeled run, supply only `--cases` with the same blueprints.
`--recheck-legacy` runs the existing Critic for drafts without a stored
semantic decision. The included cases use three tasks from the v5 development
batch. For a new benchmark, use JSONL rows with `task_id`, `question`,
`label` (`equivalent`, `different`, or `ambiguous`), and `category`.

Each result keeps historical acceptance and naturalness observations
separately from the shadow semantic status. `match` means the supported
normalized expressions agreed given the parser output. `mismatch` is a
structural discrepancy for review, including possible parser errors.
`unresolved` covers ambiguity, unsupported constructs, or invalid parser
output. The summary compares those statuses with the labeled cases, broken
down by category. The command creates a new diagnostics file and has no
effect on publication, candidate revisions, or certificate replay.

## Performance and troubleshooting

The two stages intentionally have distinct model-call budgets. Stage 2A
uses one brief Goal writer and one light intent Critic per attempt per Fragment;
\`--max-goal-revisions 2\` allows up to three local attempts for a failed
Fragment. Stage 2B uses Author and Question Critic only.

\`--question-workers\` is task-level parallelism; in Stage 2A,
\`--blueprint-workers\` is additional Fragment-level parallelism. Their
product bounds in-flight Goal requests; use smaller values if local vLLM
is memory constrained. Keep separate output directories for both stages
and use the **same frozen input files and seed** throughout Stage 2A/2B.

The existing Stage-3 commands are documented in
\`../trajectory_simulation/README.md\`.

Run all no-network tests:

\`\`\`bash
python -m unittest discover -s environment/tests -v
\`\`\`
