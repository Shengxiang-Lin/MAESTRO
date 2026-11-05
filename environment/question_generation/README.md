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
ordered streaming JSONL + resumable checkpoint
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

These stronger language checks do not retroactively alter previously
accepted questions, and are not a formal proof of semantic equivalence.
For already frozen/question-generated batches, Stage 3 still quarantines
cases failing its independent graph-aware semantic review. Rerun Stage 2
in a separate output directory or explicitly restart it if you need to
regenerate those questions, and compare against the same frozen worlds.

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
Author → preflight → Critic → postflight sequence remains ordered.
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
