# MAESTRO — Reasoning-First Question Generation

This is the canonical executable environment pipeline, not a separate
experimental V2 command set. The Python implementation lives under
envir/reasoning/. Prior names containing "-v2" remain supported as
backwards-compatible CLI aliases for existing experiments. Existing teacher
scene schema versions remain frozen to preserve replay of prior data.

## 0. Inputs and local LLM

MultiModalQA is used for topic and source-role metadata, not original answers
or media perception. Use a verified local path:

~~~bash
cd ~/code/MAESTRO
git pull origin main
test -f data/world-source/MultiModalQA/MMQA_train.jsonl.gz && echo "MMQA ready"
~~~

For local LLM use an already running OpenAI-compatible vLLM service, e.g.
http://127.0.0.1:8807/v1 and its actually served model ID.

## 1. One command: dataset to final questions and quality

~~~bash
python -m envir generate \
  --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
  --output-dir envir/data/maestro_reasoning_100 \
  --target 100 --max-records 1000 --seed 17 \
  --difficulty-mix 20,50,30 \
  --scenario-api-base http://127.0.0.1:8807/v1 \
  --scenario-model local-qwen3.5-9b \
  --scenario-workers 4 --scenario-max-revisions 2 \
  --question-api-base http://127.0.0.1:8807/v1 \
  --question-model local-qwen3.5-9b \
  --question-workers 4 --max-revisions 2
~~~

Progress bars display actual COMPLETED concurrent Scenario Author/Critic
and Question Author/Critic requests, as well as deterministic compilation
and replay. Individual responses finish out of order; output scene rows
retain the original deterministic order. Use --no-progress for log-only jobs.

If you only want a reproducible non-LLM baseline, omit BOTH model and API
base for each optional stage. Input --scenes can be used with an existing,
real locally generated Task-First scenes.jsonl instead of --mmqa, but
a fresh install does not contain that local file automatically.

## 2. Outputs

The unified --output-dir has this structure:

~~~text
maestro_reasoning_100/
  constructed/
    scenes.jsonl                  # private teacher World, Program, DAG, Gold
    questions.jsonl               # public question and source descriptions
    difficulty_audit.jsonl        # per-task counterfactual/replay diagnostics
    summary.json
  questions/                      # only if Question Author/Critic enabled
    scenes.jsonl                  # private final, reviewed Question
    questions.jsonl               # FINAL PUBLIC QUESTION BANK
    difficulty_audit.jsonl
    summary.json
  quality_report.json             # quality and diversity report
  summary.json                    # overall run summary and exact output paths
~~~

The final public dataset is exactly the path in
summary.json["final_public_questions"]. Do not send the private
constructed/scenes.jsonl to ordinary Agent evaluation: it contains Gold,
World observations and the solution program.

## 3. Read-only quality evaluation

For a full generation run with the question model enabled:

~~~bash
python -m envir quality \
  --scenes envir/data/maestro_reasoning_100/questions/scenes.jsonl \
  --output envir/data/maestro_reasoning_100/quality_recheck.json
~~~

Without the question model, point --scenes at constructed/scenes.jsonl.

To strictly replay a previously generated bank:

~~~bash
python -m envir validate \
  --scenes envir/data/maestro_reasoning_100/questions/scenes.jsonl
~~~

To refine a pre-existing constructed bank separately:

~~~bash
python -m envir refine-questions \
  --scenes envir/data/maestro_reasoning_100/constructed/scenes.jsonl \
  --output-dir envir/data/maestro_questions_refined \
  --question-api-base http://127.0.0.1:8807/v1 \
  --question-model local-qwen3.5-9b \
  --question-workers 4 --max-revisions 2
~~~

## 4. What "quality" actually tests

- Reproducible Program, generated source facts, result trace and Macro Task
  execution from the private frozen specification.
- Required source coverage, controlled source counterfactual effects and
  source citation provenance.
- One-source register-only shortcut baseline; separately reported for Easy,
  Medium and Hard.
- Distribution of target difficulty, goal family and simulated modality.
- Program structural signature diversity, duplicate/templated public Question
  surfaces (heuristics only), scenario/question Author-Critic acceptance.
- Macro operation depth, dependency depth and task count.

IMPORTANT: These are internal quality gates, not proof that an LLM will fail
a shortcut, that a program is globally minimal, or that an LLM Critic has
formally proven question/answer equivalence. To establish practical
difficulty, evaluate candidate solver policies with matched source access
and compute budgets. The currently generated source facts follow a bounded
synthetic relational grammar and may still be repetitive.

Image/video/audio sources are simulated structured observations; no
original media is viewed or claimed to have been viewed. Full World and
Gold are private teacher objects. Agents only see initial public source
descriptions and explicitly retrieved source-scoped results.

## 5. Developer tests

~~~bash
python -m unittest discover -s envir/tests -p 'test_*.py' -v
~~~

Legacy CLI commands build, realize-world, naturalize, replay and
audit-questions remain unchanged, as do aliases build-v2,
naturalize-v2, replay-v2 and audit-v2.
