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


### Single-command Stage 2 (Oracle + naturalized Sub-Agent reports)

The recommended new-run entrypoint can now execute both Stage-2 passes
without an intermediate manual command:

~~~bash
python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml \
  --scenes envir/data/maestro_question_fresh_100/questions/scenes.jsonl \
  --output-dir envir/data/maestro_subagent_reports_all_in_one_100
~~~

The CLI creates a PRIVATE deterministic Oracle bank in
<report-output>/oracle/{subtasks.jsonl,scene_index.jsonl,...}, then
generates normal Sub-Agent finish reports in <report-output>/
with the existing Author/Critic and streaming journal. An Oracle
manifest locks the scene input hash, execution implementation, and
teacher output hashes. On interruption, the SAME command/directory
reuses this valid Oracle; the report checkpoint skips already
committed Macro reviews. Incorrect or changed inputs fail closed.

The separate implement-subtasks command is still supported for
Oracle-only inspection. The old naturalize-subtasks command without
--scenes continues to consume pre-materialized Oracle files configured
under reporting.source_subtasks and reporting.source_scene_index.


## Opt-in clue-driven evidence acquisition (clean pilot)

The normal generator defaults to the original static computation patterns.
Use --clue-chain-share to replace a controlled percentage of MEDIUM
scenes with a DIFFERENT information-acquisition dependency:

- Browse a fictional simulated illustration. Two exhibit features
  (emblem color + motif) are needed to distinguish the intended item;
  decoys deliberately share one feature each
- Read that item's marker code; use it as a typed, evidence-backed query
  into a KEYED (non-browseable) catalog index
- Use the accession identifier returned by the index to query a
  second KEYED (non-browseable) archive
- Answer the target's accession year; retain exact fact provenance from
  illustration -> index -> archive

The index and archive IDs/titles are advertised to Main-Agent, but
their records are NOT returned by full read_source or scan. The
lookup_by_key primitive requires an upstream typed WorkProduct and
emits only matching records. The compiler preserves lookup operations
as separate dispatchable Macros, each with a real dependency. The
source observations remain synthetic; there is NO real media decoding.

Pilot with 20 Stage-1 tasks, approx. 10 clue-chain tasks with the
default 20/50/30 difficulty mix (all medium tasks become clue-chain):

~~~bash
python -m envir generate-questions \
  --config envir/config.yaml \
  --target 20 --clue-chain-share 100 \
  --output-dir envir/data/maestro_clue_pilot_20
~~~

Then Stage 2A + Stage 2B via the existing unified one-command entry:

~~~bash
python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml \
  --scenes envir/data/maestro_clue_pilot_20/questions/scenes.jsonl \
  --output-dir envir/data/maestro_clue_reports_20
~~~

Quality reports include evidence_acquisition keyed_lookup_count,
mean_evidence_acquisition_depth, clue_counterfactual_pass_rate and
unauthorized_full_read_success_rate. The last should be zero for
synthetic keyed stores. Check the private source/answer/evidence data,
and inspect several public questions for scenario variety before
raising the clue share or introducing new retrieval motifs.

The clue-chain pilot intentionally uses only MEDIUM tasks; Easy and Hard
keep their existing programs for now. 100 means 100 percent of MEDIUM,
not all difficulty levels. Defaults remain unchanged when the option is
omitted (generation.clue_chain_share: 0).
