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

The current package only implements **Stage 1, Generate Question**.
Stage 2 (implement Subtask) and Stage 3 (compose/schedule Subtasks) have
distinct future interfaces. They do not belong in this generator.

The accepted public Question bank never contains a Scenario/Question
fallback. Every successful Question is audited against the compiled
source-scoped Macro TaskGraph and terminal answer fields. Failed candidates
are private diagnostics. All media evidence is fictional structured
simulation, not perception of original images/audio/videos.
