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
| `macro_first_v2_share` | 100 | Use the current Macro-first World for Medium/Hard; Easy keeps the approval baseline |
| `difficulty_mix` | `[17, 33, 50]` | Easy/Medium/Hard desired **accepted** counts for a 100-item dataset |
| `quota_refill` | `true` | Refill missing difficulty groups after both Scenario and Question review |
| `max_candidate_multiplier` | `3` | Hard upper limit: attempted candidates ≤ target × multiplier |
| `seed` | 17 | Reproducible seed for dataset sampling and frozen World construction |

You may set `--max-candidate-multiplier 4` to raise the budget without
changing the requested difficulty ratio. A quota is never filled by relabeling
a problem, duplicating an answer, or inventing an accepted Critic decision.
If the budget is exhausted, `status=partial` and
`summary.json.difficulty_quotas.deficit` reports the shortfall.

Completed batches are saved under `rounds/01`, `rounds/02`, etc.
with independent Scenario/Question checkpoints. Runs interrupted between
batches can reuse already verified batches if their signatures match.
Use a **new output directory** when changing generation settings.

### Canonical Stage 1 outputs

```text
maestro_current_100/
├── summary.json                   # achieved quotas and accepted counts
├── quality_report.json            # strict Replay and difficulty proxies
├── constructed/
│   ├── scenes.jsonl                # selected frozen scenes
│   └── failed.jsonl                # Scenario failures across rounds
├── questions/
│   ├── scenes.jsonl                # accepted final Questions + frozen evidence
│   ├── questions.jsonl             # Main-visible Question records
│   ├── difficulty_audit.jsonl
│   └── failed.jsonl                # Question failures across rounds
└── rounds/
    ├── 01/                         # full intermediate stage output
    └── 02/                         # as many bounded rounds as necessary
```

The `constructed/scenes.jsonl` root file represents selected scenes after
quota selection; intermediate attempted/fallback records remain in each
round's folder. The accepted `questions/scenes.jsonl` is the authoritative
input for Stage 2. Its records include typed Goal, scoped Macro DAG,
deterministic Gold Solvers, evidence World, provenance and final Question.

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
- `quota_pipeline.py`, `stages.py`, `pipeline.py`: bounded,
  checkpointed generation rounds and final accepted-quota publication.
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
