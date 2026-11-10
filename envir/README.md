# MAESTRO Environment

Current supported workflow: **generate validated questions with frozen Macro
plans, then produce grounded Sub-Agent reports**. Every World and media
observation is synthetic. A successful Replay verifies internal evidence
consistency, not independent multimodal perception or Agent performance.

## Generate questions (Stage 1)

Configure `envir/config.yaml`, then run:

```bash
python -m envir generate-questions \
  --config envir/config.yaml \
  --target 100 \
  --output-dir envir/data/maestro_current_100
```

Active settings under `generation`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `target` | 100 | **Exactly 100 initial candidates**, not 100 guaranteed accepted Questions |
| `macro_first_v2_share` | 100 | Apply the Macro-first design to Medium/Hard; Easy retains its approved baseline |
| `difficulty_mix` | `[17, 33, 50]` | **Initial sampling** distribution across Easy/Medium/Hard; never a post-review quota |
| `seed` | 17 | Reproducible initial pool and frozen World |

**One pass, no quota refill.** Stage 1 submits the initial fixed candidate
pool once to Scenario Author/Critic and then passes the validated survivors
once to Question Author/Critic. All accepted final Questions are published:
a legitimate Question is **never discarded** to hit a target difficulty
mix, and rejected candidates are **never replaced** just to reach a requested
number. For example, 100 candidates yielding 72 accepted Questions produces
`status=done`, `generated=72`, `rejected=28`, and the observed
difficulty distribution, with `selection_policy:
fixed_initial_candidates_no_refill_no_quota_filter`. These counters
are persisted to `summary.json`, while detailed failed cases remain
in the respective `failed.jsonl` files.

The default CLI displays **three meaningful item-level progress bars**:
`Scenario Author/Critic` (reviewed Scenario candidates),
`Compile & difficulty audit` (compiled Scenes with accepted/rejected
counters), and `Question Author/Critic` (reviewed Questions, also with
accepted/rejected counters). Each appears once; there are **no quota-refill
rounds**. Fast preliminary validation and the final quality/Replay audit
still run but do not add redundant progress bars. Use
`--verbose-progress` to display these auxiliary checks, or
`--no-progress` to suppress all progress bars for batch logs. The deterministic compiler and independent reviewing standards
remain unchanged; genuine failures do not become accepted automatically.

### Canonical Stage 1 outputs

```text
maestro_current_100/
├── summary.json                   # candidate_count, accepted, rejected, observed difficulty
├── quality_report.json            # strict Replay and difficulty proxies
├── constructed/
│   ├── scenes.jsonl                # Scenario-accepted, Replay-validated candidates
│   └── failed.jsonl                # Scenario / program validation failures
└── questions/
    ├── scenes.jsonl                # all accepted final Questions (input to Stage 2)
    ├── questions.jsonl             # Main-visible Question records
    ├── difficulty_audit.jsonl
    └── failed.jsonl                # Question review failures
```

The candidate pool size is fixed at `--target` **before** the
LLM review. The final `questions/scenes.jsonl` is the authoritative
Stage-2 input; its size may be smaller than `--target`. Accepted
records include typed Goal, scoped Macro DAG, deterministic Gold Solvers,
evidence World, provenance and final Question. There is no `rounds/`
directory for newly generated batches.

## Build Sub-Agent reports (Stage 2)

```bash
python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml \
  --scenes envir/data/maestro_current_100/questions/scenes.jsonl \
  --output-dir envir/data/maestro_current_reports_100
```

This creates exact Oracle WorkProducts and checked natural-language
Sub-Agent finishes. Stage 2 never changes the underlying Gold Answer.
The deterministic `oracle_review.py` maps Entry IDs to cited FactRefs
and their observed Active status, including full answer provenance when
evidence atom samples are truncated. A provably mistaken Critic assessment
may trigger one targeted independent re-review; it **never** auto-passes.
A wrong citation in an Author summary is rejected.

Reports are published under `complete_scene_only`: one rejected Macro
prevents publishing an incomplete scene.

## Contracts, validation and complexity

The current generation pipeline is organized around:

- `macro_first_v2.py`, `macro_first_v2_world.py`: frozen Macro DAG,
  typed Goal, synthetic, modality-sensitive observations and witness generation.
- `semantic_evidence.py`, `question.py`: executable predicate and
  **single-answer-slot** contracts. Question text may paraphrase, not add
  unlicensed outputs, time windows or media requirements.
- `source_title_contract.py`, `scenario_fields.py`, `scenario.py`:
  immutable source modality/title contract; audited, field-level repair
  of overlong/missing narrative fields; independent semantic Critic.
- `information_necessity.py`: check that independent multimodal branches
  each retain ambiguity and the intersection uniquely identifies a target
  with a distinct possible final answer.
- `difficulty_proxy.py`: observe Macro depth, evidence volume, primitive
  operation count, index hops and candidate ambiguity. Structural proxies
  **do not** establish true Hard-vs-Medium Agent failure rates.
- `stages.py`, `pipeline.py`: one bounded candidate pass,
  checkpointed Scenario/Question review and publish-all-valid output.
- `implement_subtask/oracle_review.py`, `implement_subtask/reporting.py`:
  deterministic Oracle evidence checks and independently reviewed prose.

The active CLI/config no longer exposes superseded
`clue_chain_share`, `evidence_graph_share` or `macro_first_share`
generation experiments. Frozen historical scene readers and underlying
operators remain because they are still necessary to audit prior
experiments. They are not part of the current user-facing generation API.

Run a read-only audit of an accepted or historical scene bank:

```bash
python -m envir quality \
  --config envir/config.yaml \
  --scenes envir/data/maestro_current_100/questions/scenes.jsonl \
  --output envir/data/maestro_current_100/quality_recheck.json
```

## Limits

The current Media World uses synthetic structured observations, not
original decoded audio, images, video or external web browsing. Macro
information necessity is checked for the frozen candidate World, not
against every imaginable independent plan. Scene/Question Author-Critic
acceptance and 100% Oracle replay do **not** measure real Main-Agent
solving difficulty, cost, accuracy or user-perceived naturalness.

## Stage 3 readiness: executable semantics rather than real-world answers

The simulated environments are designed for **learning orchestration**
(reuse, delegate, seek evidence, independently check, retry, switch, accept,
hold, terminate). They are not a benchmark of whether external websites
or real videos can be solved by an independent browsing Agent. No live
Agent accuracy run is a prerequisite for Stage 3.

New Stage-1 scenes freeze `question_fidelity_version`:
- `frame_time_ms >= t` = event/frame observed **at or after** timestamp t;
  neither a duration nor an exact action start time
- `footage_time_ms <= commentary_time_ms` = video event **no later than**
  audio utterance; an exact time gap is not implied
- the final user response has exactly one typed Goal; matching clip/record
  identities are intermediate conditions rather than extra output slots

Newly generated scenes also freeze a `topic_domain_version` categorical
cue. Unambiguous topic wording in MMQA (episode, voice actor, stadium,
football club, etc.) can override an unrelated broad dataset domain label,
but **no original MMQA answer, event fact or question wording is reused**
as the new problem or Goal. Ambiguous category cues fall back to the
previous domain selection.

The versioned new Macro grammar supports five families, adding
`corroborated_dispute`: two year reports produce a nonunique disputed
candidate set; an independent event log produces a different nonunique
set. The two sets' intersection is unique, and alternative candidates
lead to different final Goal values. The existing deterministic auditor
checks actual Macro WorkProducts before accepting the scene. Earlier
frozen V2 snapshots continue using their original four-pattern grammar.

Stage 2 now compiles an *internal stage-local claim contract* from exact
executed Macro steps, accessible source IDs and Oracle output. For
example, `scan(report_a)` cannot claim that the two reports disagree,
while the later `compare_fields(reported_year_a,reported_year_b,ne)`
can. LLM-authored finishes are checked against this contract before the
independent Critic. Known internal words such as 'oracle output' are
rewritten in **prose only**, with an audit trail, then passed through the
same guards and Critic. Answer rows and provenance stay immutable.

### Proposed Stage 3 trust boundary

- **Public Main-Agent view:** user Question; names/capabilities of simulated
  Sub-Agents; reports and source references *already delivered* in the
  active trajectory; remaining budget and action history
- **Private simulator/teacher:** original World observations; Gold Program;
  complete TaskGraph and reference waves; exact Oracle answers and hidden
  future reports; attack source/state; action costs and consequence labels
- **Main-Agent actions:** delegate, request_evidence, verify_independent,
  retry_same, switch_backend, accept, hold, retract, terminate (exact action
  set can be fixed in the Stage 3 design)
- **Feedback validation:** every report fact must be stage-local; never
  promote an upstream result to a verified conclusion just because the
  *overall Question* requires that conclusion

In particular, never expose the full Gold Solver/Oracle trace as a Main-Agent
prompt. The correct next step is to construct an interactive simulator that
reveals observations only after a selected action and computes rewards from
private state, not to run unrelated real-world search evaluations.

### New deterministic regressions

```bash
python -m unittest envir.tests.test_stage3_readiness
python -m unittest discover -s envir/tests
```

Local Qwen Author/Critic acceptance still requires a fresh generation pilot;
passing programmatic tests is NOT a proof of unrestricted language entailment.

## Versioned Goal Resolution: direct, comparison and calculation

New V2 scenarios freeze `goal_resolution_version = maestro.goal_resolution.v1`.
The sampler selects among:
- `direct_lookup`: retrieve the authoritative Goal value from the keyed
  archive (legacy V2 behavior)
- `comparison`: obtain `metric_a` and `metric_b` from two independently
  keyed count sources, join their proven identity keys, and derive
  `higher_count_source` (first, second or equal)
- `calculation`: obtain `completed_count` and `total_count` from two
  independently keyed count sources and derive
  `completion_rate_percent = floor(100 * completed_count / total_count)`.
  The operator rejects zero denominators and invalid count domains.

All three variants preserve the earlier visual/temporal/parallel/dispute
identity-discovery DAG. The two derived variants **do not** store the
answer directly in an archive table. Each source is accessible only
through a previously established upstream key, and the derived Goal
retains both branches' source provenance. Counterfactual audit checks
that changing each cited count-source witness changes the final answer.
The `macro_first_v2.goal_resolution_distribution` report shows the
actual mix. Source roles remain simulated observations; no original
MMQA answer or YAGO facts are used to populate the counts.

Frozen old V2 specs without `goal_resolution_version` reproduce
the previous direct-lookup plan and World without migration. Unit tests
cover 100 new specs, replay, all three modalities, independent evidence
dependencies, impossible denominator rejection and old marker-less replay.
These data are intended for downstream orchestration policy training,
not external-benchmark model question answering.
