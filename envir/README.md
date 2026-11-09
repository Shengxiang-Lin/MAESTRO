# MAESTRO Environment — Stage 1: Generate Question

This directory contains the **single maintained Question-generation pipeline**
for MAESTRO. The obsolete Task-First, V2 experiment and separate naturalize
commands have been removed; no old command compatibility is maintained.

## Usage

Edit [config.yaml](config.yaml) for input, output, difficulty mix, Author,
Critic, concurrency and revision settings. Then:

~~~bash
python -m envir generate-questions --config envir/config.yaml
~~~

For a read-only audit:

~~~bash
python -m envir quality --config envir/config.yaml
~~~

The implementation and detailed output contract are documented in
[generate_question/README.md](generate_question/README.md).

Stage 2, **Implement Subtask**, now materializes the frozen Macro
execution into per-Subtask `answer` + evidence-grounded `trace_summary`.
It reuses accepted Stage-1 private scenes and does not call a model:

~~~bash
python -m envir implement-subtasks \
  --config envir/implement_subtask/config.yaml
~~~

See [implement_subtask/README.md](implement_subtask/README.md).

A separate Stage-2 language pass rewrites the frozen Oracle responses into
realistic Sub-Agent `finish_result.result/status/summary` messages with
an LLM Author and independent Critic. The typed answer, primitive evidence
steps, and fact refs are never regenerated or edited:

~~~bash
python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml
~~~

Stage 3, Subtask scheduling and composition, remains separate and is not
implemented here.

Stage 1 remains Generate Question.
Stage 2 (implement Subtask) and Stage 3 (compose/schedule Subtasks) have
distinct future interfaces. They do not belong in this generator.

The accepted public Question bank never contains a Scenario/Question
fallback. Every successful Question is audited against the compiled
source-scoped Macro TaskGraph and terminal answer fields. Failed candidates
are private diagnostics. All media evidence is fictional structured
simulation, not perception of original images/audio/videos.


### Streaming and checkpoint recovery

Scenario Author/Critic and Question Author/Critic now append reviewed
decisions to dedicated accepted/rejected/audit JSONL journals AS THEY
FINISH, in stable input order. The live progress bar shows accepted,
rejected, and written (the committed ordered prefix).

Accepted Questions also appear immediately in the ordinary files
questions/scenes.jsonl and questions/questions.jsonl; rejected
Questions appear in questions/failed.jsonl.

Each stage has a signed, SHA256-verified checkpoint and a writer lock.
Interrupting and rerunning the SAME command and output directory
(with unchanged frozen inputs/model/prompt code) skips completed LLM
decisions. Uncommitted tails are discarded and published files are
reconstructed from the durable journal.

The config key generation.flush_every defaults to 1 (fsync every
decision); the CLI may override this:

~~~bash
python -m envir generate-questions \
  --config envir/config.yaml --flush-every 5
~~~

A different model, input scene bank, generator contract or code must
use a new output directory or explicitly overwrite the old run.
Completed legacy outputs from the old all-at-once implementation have
no checkpoint and cannot be resumed retrospectively.
