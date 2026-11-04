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
