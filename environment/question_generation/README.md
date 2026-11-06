# Question Generation

Natural-language realization over a frozen programmatic task batch.

```text
frozen manifest + public authoring projection
        ↓
public/private boundary validation (including SHA-256)
        ↓
parallel LLM Author (per-task sequential execution)
        ↓
deterministic preflight
        ↓
semantic Critic / wording-only repair
        ↓
deterministic postflight
        ↓
independent full-ProgramIR graph-aware semantic certification
        ↓
if unverified: retry Author, then certified Canonical Renderer fallback
        ↓
publish Surface + review certificate + ordered streaming checkpoint
```

The LLM may change wording only. It may not add facts, resources,
proof requirements, hidden criteria, answer hints, or change acquisition
visibility. **No quality gate has been relaxed.**

## Discovery-root questions: candidate names are findings, not input

A ProgramIR `FIND_ALL` enumerates entities from the **bounded, source-backed
task evidence domain**. It is not a user-supplied complete roster, and it
does not imply exhaustive search of Wikipedia/the open web. Questions about
these roots should ask which entities can be **identified from relevant task
references** and then filtered by the stated predicates. The Author/Critic
are instructed not to pre-enumerate answer-like entity names; an extra
preflight check rejects explicit complete-supplied-list claims in tasks
with discoverable sources.

The generator's entity names are deliberately **synthetic aliases**, not
the original YAGO labels. A real Internet Google search cannot be assumed
to find the aliases; source discovery in Stage 3 is witness-backed
**simulation**, not live search evaluation.

Changing the deterministic question renderer affects *newly rendered/frozen*
authoring inputs. Previously frozen `authoring_inputs_*.jsonl` retain their
old draft strings. Restarting Stage 2 reauthors against those frozen inputs
using the new discovery wording rules, but does **not** itself rewrite the
old drafts. To update the draft templates themselves, regenerate the
upstream rendering/freeze outputs in a new batch and then re-run Stage 2.
Do not hand-edit existing accepted questions: Stage 3 fingerprints both
the accepted question and ProgramIR for review.

## Semantic ownership and multi-hop fidelity

The Author and Critic now compare every public constraint against the
deterministic draft with special attention to **which entity owns each
number, date or attribute**, and to the direction and preservation of all
relation hops. Preserving the word "1912" is insufficient when a Way's
start date has been rewritten as a City's start date; preserving entity
names is insufficient when an intermediate death-place/spouse hop was
removed. Author/Critic instructions also explicitly require retaining
publicly named starting entities (such as an airport named in the frozen
draft), rather than replacing them with a potentially non-unique IATA
code or shifting dates across the relation chain. A Critic response with `accept=true` but nonempty `issues`
is rejected, including during wording-repair review.

### Stage 2 is the only full Surface semantic review boundary

The public-only Author/Critic remains an inexpensive writing filter, **not**
the final correctness authority. Before Stage 2 publishes a question, a
teacher-private graph-aware reviewer now checks the entire ProgramIR,
its controlled renderer reference, relation direction, filter ownership,
hops, joins and output type. The reviewer is reused from the earlier
Stage 3.1 implementation; it receives graph meanings and the candidate
question, **not** private World answer records. Only `eligible` with a
valid review protocol is accepted. Critic errors, uncertain decisions,
missing witnesses and reviewer output errors are all unverified.

The bounded Author retry loop may attempt a new rewrite. When no safe rewrite
emerges, Stage 2 falls back to the exact controlled-renderer question,
**only** if the renderer's dependency/hidden-clue certificate and deterministic
postflight both pass. An unsafe or unverified renderer is rejected rather
than emitted. Every published row includes `surface_certificate` and a
private `authoring.surface_review`, linked by exact question, ProgramIR and
renderer hashes and a versioned review contract. The certificate is an
auditable pipeline assertion, **not** a formal mathematical equivalence
proof or a cryptographic signature.

Stage 3.1 no longer calls a Surface semantic LLM Reviewer. It validates
this certificate against the frozen inputs; exact Canonical questions can
be independently certified by the renderer. Stage 3 still checks whether
public evidence is executable and discoverable; that is a different gate
from question semantic equivalence.

Existing files in `questions_2000` do not automatically gain certificates.
To build a fresh 50-question batch without overwriting the old run:

```bash
python environment/generate_questions.py \
  --max-records 50 \
  --output-dir environment/data/questions_cert_v10_50
```

Then use Stage 3.1 with
`--realized environment/data/questions_cert_v10_50/realized_capability_worlds.jsonl`.
Do **not** combine old checkpoints and new contract versions; use a new
directory or explicitly restart a chosen output directory. The frozen
Stage 1 World and authoring inputs are not regenerated.

Neither an LLM semantic judge nor literal-string checks can guarantee
zero incorrect paraphrases. The Canonical fallback is the conservative
route when the reviewer cannot establish equivalence.

## Run

From the MAESTRO root, after task construction/freeze:

```bash
python environment/generate_questions.py
```

Defaults in `environment/config.yaml`:

```yaml
question_generation:
  author_profile: local_qwen35_9b
  critic_profile: local_qwen35_9b
  question_workers: 4
  flush_every: 5
  restart_questions: false
```

To increase in-flight Qwen3.5-9B/vLLM requests:

```bash
python environment/generate_questions.py --question-workers 8
```

`question_workers` counts concurrent tasks; within each task the
Author → preflight → Critic → postflight → ProgramIR certification sequence remains ordered.
The vLLM server's GPU throughput and KV cache determine whether 8 is faster
than 4. Start with 4, then benchmark 8 without changing generation checks.

## Streaming, ordering and recovery

- Tasks run concurrently with at most `2 * question_workers` tasks queued.
- Results can finish out of order, but the writer only emits contiguous results
  in frozen task order.
- Every accepted result is written to
  `realized_capability_worlds.jsonl`; rejected tasks go to
  `question_rejected.jsonl`; every attempt has an ordered record in
  `question_critic_audit.jsonl`.
- Every `flush_every` ordered tasks, the writer flushes and fsyncs all three
  outputs, then atomically commits `question_progress_state.json`. The
  progress checkpoint records input/model/code signature, committed byte
  offsets, SHA-256 and accepted/rejected counts.
- Rerunning the same command reuses committed tasks and only schedules the
  unfinished suffix. Any uncommitted file tails left by a process crash are
  truncated to the recorded offsets.
- A changed frozen input, author/critic model setting or relevant validation
  code invalidates the checkpoint. The command stops rather than mixing
  differently generated records. The model and source files must remain
  unchanged to resume safely.
- The `question_generation.lock` file prevents two simultaneous writers from
  targeting the same output directory.

To **explicitly discard** previous output and regenerate all questions:

```bash
python environment/generate_questions.py --restart-questions
```

This deletes the question-generation JSONLs, checkpoint and final summary
in the configured `output_dir`, not the upstream frozen input files.
Do not use this flag to continue an interrupted run.

`--flush-every 1` commits every result at a higher I/O cost; the default
of 5 reduces checkpoint overhead. The summary includes `requested`,
`accepted`, `rejected`, `streaming.question_workers`, `streaming.flush_every`
and `streaming.resumed_records`. The final summary is only produced after
every frozen input task is accounted for.

If this directory contains outputs from a **pre-streaming** MAESTRO version,
there is no compatible checkpoint. Use a new output directory, or explicitly
start over with `--restart-questions`. No implicit overwrite occurs.

Run the no-network regressions:

```bash
python -m unittest discover -s environment/tests -p "test_question_streaming.py" -v
```
