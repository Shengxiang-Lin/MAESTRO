# MAESTRO — Reasoning-First Question Generation

This is the canonical executable environment pipeline, not a separate
experimental V2 command set. The Python implementation lives under
envir/reasoning/. Prior names containing "-v2" remain supported as
backwards-compatible CLI aliases for existing experiments. Existing teacher
scene schema versions remain frozen to preserve replay of prior data.

## Quality repair after the 100-question pilot

The first full Qwen run (100 tasks) showed high Scenario approval but poor
Question acceptance. The author was originally not given the frozen
reference question. The corrected Author now receives that exact
program-derived question, with requirements and bounded-revision feedback.
Rejected candidates are captured in PRIVATE question_authoring.failures
with separate length/punctuation errors and previews. These are not public.

To measure only this fix, **reuse the existing constructed bank**. You do
not have to call Scenario Author again:

~~~bash
git pull origin main

python -m envir refine-questions \
  --scenes envir/data/maestro_reasoning_100/constructed/scenes.jsonl \
  --output-dir envir/data/maestro_question_repair_100 \
  --question-api-base http://127.0.0.1:8807/v1 \
  --question-model local-qwen3.5-9b \
  --question-workers 8 --max-revisions 2

python -m envir quality \
  --scenes envir/data/maestro_question_repair_100/scenes.jsonl \
  --output envir/data/maestro_question_repair_100/quality_report.json
~~~

Compare question_review, question_rejections.guard_reasons and
question_surface with the old report. Do not reuse an already
naturalized questions/scenes.jsonl input.

For the new **v3 fact sampler and reasoning program variants**, generate to a
NEW directory. It intentionally changes the sampled synthetic answers,
so compare distributions, not row-for-row Gold labels with the old batch:

~~~bash
python -m envir generate \
  --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
  --output-dir envir/data/maestro_diverse_100 \
  --target 100 --max-records 1000 --seed 17 \
  --difficulty-mix 20,50,30 \
  --scenario-api-base http://127.0.0.1:8807/v1 \
  --scenario-model local-qwen3.5-9b \
  --scenario-workers 8 --scenario-max-revisions 2 \
  --question-api-base http://127.0.0.1:8807/v1 \
  --question-model local-qwen3.5-9b \
  --question-workers 8 --max-revisions 2
~~~

The quality report now includes answer_diversity with exact answer-set
repetition counts, and question_rejections with rejection stage and guard
reason breakdown. New scenes use versioned v3 synthetic sampling; older
v2 scenes retain legacy program and World regeneration for replay
compatibility. Repeated answers, synthetic category labels, and step-level
Macro Tasks remain research limitations; real Agent difficulty and truly
3-9 coarsened Macro Tasks are not yet established.

## Semantic relation grammar and Question Intent experiment

This generator freezes a v4 ProblemSpec with a meaningful semantic pattern,
rather than treating every multi-source task as category approval:

- approval: registered entries are mapped and independently approved;
  evidence-reconcile answers include source fact references.
- time_window: registered event steps fall inside independently supplied
  simulated source intervals, using BOTH lower and upper boundaries.
- status_disagreement: two independent status observations differ after
  cross-source group resolution.
- dynamic_threshold: each measurement meets the independently supplied
  minimum threshold for its mapped group.

Video, audio and image annotation-specific fields enter actual Gold Program
predicates where those simulated source roles are selected. No real media
content is accessed.

Structured Question Intent specifies mandatory predicates and the typed
answer. The Author receives Intent and rhetorical guidance instead of a
ready-made Question in its first draft; the Critic receives the frozen
reference and can enforce equivalence. A template may be supplied to the
Author for a repair attempt. Critic acceptance is not proof of equivalence.

New Macro Tasks fuse up to three consecutive unary dependent IR nodes into
fully EXECUTABLE composite tasks. Source grants are still explicit, and
all primitive operations are replayed; the latest scenes require <=9
Macro Tasks. Old v2/v3 scenes replay unchanged.

Create a NEW experiment output directory:

~~~bash
git pull origin main

python -m envir generate \
  --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
  --output-dir envir/data/maestro_semantic_100 \
  --target 100 --max-records 1000 --seed 17 \
  --difficulty-mix 20,50,30 \
  --scenario-api-base http://127.0.0.1:8807/v1 \
  --scenario-model local-qwen3.5-9b \
  --scenario-workers 8 --scenario-max-revisions 2 \
  --question-api-base http://127.0.0.1:8807/v1 \
  --question-model local-qwen3.5-9b \
  --question-workers 8 --max-revisions 2
~~~

Inspect envir/data/maestro_semantic_100/quality_report.json for true semantic
pattern distributions, simulated media predicates, answer diversity, task
budgets, copied Question templates and repeated openings. Manually sample
final Question surfaces too: surface diagnostics are heuristic.

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
