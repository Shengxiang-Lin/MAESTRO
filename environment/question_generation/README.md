# Stage 2 · Natural-language question realization

Stage 2 expresses a **frozen, executable Stage-1 task** as an understandable
question. It must not modify the ProgramIR, private World, Gold Answer,
Subtask DAG, or the Stage-3 execution plan. Question quality and semantic
fidelity are **separate** dimensions; grammatical compliance is not by
itself a human naturalness score.

## The three explicit generation strategies

| \`--surface-mode\` | Model contribution | Publication guard | Role |
| --- | --- | --- | --- |
| \`scoped\` (default) | Select among compiler-generated sentence alternatives | Reparse the exact bounded grammar and compare the typed query | Fast, source-replayable baseline |
| \`free\` | Generate a complete open-ended rewrite of an already certified Scoped question | Public boundary guards + **two blind text-only readbacks** + typed-tree equality and exact textual witnesses | Conservative natural-language experiment |
| \`graph\` | Historical direct Author/ProgramIR/repair workflow | Graph/contract reviewer | Explicit A/B research baseline; not the default |

The active production path is:

\`\`\`text
frozen World + ProgramIR + public task projection
                         |
                controlled renderer
                         |
          checked public Question Contract
                         |
             Scoped Grammar alternatives
                         |
           Scoped text + grammar certificate
                         |
     [if --surface-mode free and models configured]
                         |
             LLM writes NEW English question
                         |
        date/code/restriction and leak checks
                         |
        INDEPENDENT readback of free text ONLY
          /                                 \
       pass 1                             pass 2
          \                                 /
           normalized typed query comparison
                         |
           reference Question Contract match
                         |
       accepted free candidate OR Scoped fallback
                         |
  Stage-2 question + versioned certificate + audit
                         |
  Stage 3: revalidate certificate WITHOUT an LLM call
\`\`\`

### What is allowed into model prompts?

* The **free Author** receives only the previously certified Scoped question.
  It does not receive private World facts, hidden \`FIND\` identities, evidence
  answers, source identifiers, or the Gold Subtask DAG.
* Each independent **semantic reader** receives the *free candidate question
  only*, plus generic operator instructions and output-format requirements.
  Neither reader sees the expected query tree, Canonical question, previous
  parser response, or Author notes.
* The deterministic comparator holds the expected typed expression *after*
  the readers have returned. A node or condition is only admissible if its
  supporting quotation occurs in the actual question text. Dates normalize to
  calendar dates, numbers to numeric values, and \`+650\` remains a string code.
* An uncertainty, malformed parse, missing quote, changed owner, relation
  direction, conjunction/disjunction, output type, added literal or extra
  restriction is not approved. Every candidate is checked; no unchecked
  free rewrite is silently admitted to Main-Agent training.

The two parses may still share model bias (both can use the same Qwen
checkpoint). This is a **conservative, model-assisted** semantic verification
experiment, **not** a formal equivalence proof or a guarantee of zero errors.
For that reason, keep independent human challenge sets and controlled
counterexamples in the evaluation.

### Why keep the Scoped fallback?

The current Scoped Grammar is a bounded **generator/parser**: its predefined
productions carry typed semantics. It can prevent unlicensed edits, but its
readback is not independent natural-language understanding. In contrast,
a free-form rewrite must be interpreted from scratch; it is harder and may
fail. When it does, we publish the already validated Scoped version (or the
exact Canonical renderer if the grammar is unsupported).

**Do not confuse these successes:**

* \`free_rewrite_certified\`: unrestricted text passed two blind semantic parses
  and the deterministic comparison.
* \`scoped_grammar_certified\`: text matches compiler-generated productions;
  **not** a verified free rewrite.
* \`canonical_only\`: safe task/question fallback, **not** naturalization.
* \`rejected\`: no safely publishable surface.

The historical \`publication_status=naturalized_certified\` also covers
Scoped Grammar, so use the **method-level counters** above for experiments.
\`meaningful_rewrite_rate\` is only an inexpensive text heuristic, not a
human naturalness metric.

## Run a paired 10-question experiment

From the same frozen Stage-1 input, use **new output directories** to avoid
incompatible checkpoints:

\`\`\`bash
cd ~/code/MAESTRO

# Optional baseline: a new Scoped run over the first 10 frozen tasks
python environment/generate_questions.py \
  --max-records 10 \
  --surface-mode scoped \
  --output-dir environment/data/questions_scoped_v12_10 \
  --flush-every 1

# New experiment: Scoped → free Author → two blind semantic parses
python environment/generate_questions.py \
  --max-records 10 \
  --surface-mode free \
  --output-dir environment/data/questions_free_v12_10 \
  --flush-every 1
\`\`\`

The default \`environment/config.yaml\` resolves both roles to the configured
local model. In \`free\` mode, the question Author uses the configured Author
backend and the independent text reader uses the Critic backend with
temperature 0 and at least 4096 reserved output tokens. To avoid model-only
correlated failures, consider a separate Critic checkpoint for later runs.
Do not use the explicit \`--no-semantic-critic\` override when evaluating free
rewrites; without a real reader, the safe Scoped fallback remains in force.

The summary separates:
\`free_rewrite_certified\`, \`scoped_grammar_certified\`,
\`free_author_calls\`, \`blind_readback_calls\`,
\`free_attempt_failure_reasons\`, \`canonical_only\`, and \`rejected\`.
Review these alongside actual question examples and human ratings. A high
grammar-certified fraction is **not** a high free-naturalization success rate.

Stage 3 remains an independent consumer. To check the final output boundary
for these same ten accepted questions:

\`\`\`bash
python environment/simulate_trajectories.py subtasks \
  --scene omnigaia --provider openai-compatible \
  --schedule compact --feedback-workers 4 \
  --max-records 10 \
  --realized environment/data/questions_free_v12_10/realized_capability_worlds.jsonl \
  --output work/subtask_bank_free_v12_10.jsonl
\`\`\`

Stage 3 reconstructs the Canonical question from the frozen World, compiles
Gold from ProgramIR, validates Stage-2's certificate and separately checks
public source reachability under the actual Surface question. Old Scoped
**v17** certificates remain verifiable; new publications use **v18**.

## Module ownership

| Module | One primary responsibility |
| --- | --- |
| \`pipeline.py\` | CLI, frozen-input mapping, streaming, resume and metrics |
| \`realization.py\` | Per-task selection, Scoped/Free fallback policy, publication |
| \`scoped_surface.py\` | Deterministic grammar, productions and exact roundtrip |
| \`free_surface.py\` | Author's public-text rewrite and style/literal guards |
| \`semantic_readback.py\` | Blind text-only parsing, quotation witnesses, typed equality |
| \`verification.py\` | Versioned certificates and Stage-3 local revalidation |
| \`public_semantics.py\` | Safe public-only Question Contract projection |
| \`graph_realization.py\` | Historical graph-review baseline, isolated from default |
| \`streaming.py\` | Ordered durable JSONL/checkpoint writes |

**Safety boundary:** The Author can only rephrase information already
exposed by the Controlled Renderer. Structural changes to the task itself
belong in Stage 1. Deleting an answer-defining condition requires a fresh
task and Gold verification, not a permissive Stage-2 style check.

## Recovery and diagnostics

The generator validates the frozen manifest and SHA-256 of Stage-1 inputs.
It writes accepted questions, rejected tasks, and private audit records in
input order even when task generation runs concurrently. Checkpoint
signatures cover the code and model configuration, and \`--flush-every 1\`
provides the strictest crash/restart boundary.

A different input/model/verification version will reject resume against an
old checkpoint; choose another output directory or explicitly restart with
\`--restart-questions\`. This restart removes only the chosen Stage-2
output files, **never** the frozen Stage-1 World.

Run no-network regressions:

\`\`\`bash
python -m unittest discover -s environment/tests -p "test_*question*.py"
python -m unittest discover -s environment/tests -p "test_semantic_readback.py"
python -m unittest discover -s environment/tests -p "test_scoped_surface.py"
\`\`\`

Limitations: independently parsed English can still be ambiguous; two
parsers sharing one model are not statistically independent. A grammar
production might itself encode the wrong English relation despite carrying
the right typed label. Validate a representative set of sentences manually,
with explicit adversarial relation/owner/quantifier probes, before scaling
beyond the first 10–50 samples. Improving the *motivation* and coherence of
the underlying question is a separate Stage-1 Task Design problem.
