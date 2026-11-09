# MAESTRO V2 — Reasoning-First Synthetic Environment

V2 is an opt-in build → execute → audit → naturalize → replay pipeline.
Existing Task-First scene banks and CLI commands remain available.

## 1. Build a 100-question V2 pilot without an LLM

From the MAESTRO repository root:

First verify the MultiModalQA source file exists locally:

~~~bash
ls -lh data/world-source/MultiModalQA/MMQA_train.jsonl.gz
~~~

Then use this **preferred command**. V2 does not require an old TaskGraph bank:

~~~bash
python -m envir build-v2 \
  --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
  --output-dir envir/data/v2_reasoning_100 \
  --target 100 --max-records 1000 --seed 17 \
  --difficulty-mix 20,50,30

python -m envir replay-v2 \
  --scenes envir/data/v2_reasoning_100/scenes.jsonl

python -m envir audit-v2 \
  --scenes envir/data/v2_reasoning_100/scenes.jsonl
~~~

Alternatively, reuse a **real, locally existing** Task-First
`scenes.jsonl` (including a bank from the earlier 859-question run).
Generated data under `envir/data/` is not guaranteed to be downloaded
by `git pull`. Locate the actual file first:

~~~bash
find envir environment -name scenes.jsonl -type f 2>/dev/null
# Then substitute an existing path:
python -m envir build-v2 \
  --scenes /actual/path/to/scenes.jsonl \
  --target 100 --seed 17 --output-dir envir/data/v2_from_scenes_100
~~~

Only scene topics and source-modality hints are reused, not the old MMQA
question, answer, World, DAG or original media. If the MMQA file is
also absent, locate it with:

~~~bash
find data environment envir -iname '*mmqa*train*' -type f 2>/dev/null
~~~

## 2. Optional richer pre-World scenario design (local Qwen)

An LLM Scenario Author + independent Critic propose fictional setting,
user motivation and source display titles BEFORE numerical World values
are sampled. They cannot change the Goal, source grants or operators.
The frozen deterministic scenario is the safe fallback.

~~~bash
python -m envir build-v2 \
  --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
  --output-dir envir/data/v2_scenario_100 \
  --target 100 --seed 17 --difficulty-mix 20,50,30 \
  --scenario-api-base http://127.0.0.1:8807/v1 \
  --scenario-model local-qwen3.5-9b \
  --scenario-workers 4 --scenario-max-revisions 2
~~~

## 3. Optional Question Author/Critic

~~~bash
python -m envir naturalize-v2 \
  --scenes envir/data/v2_scenario_100/scenes.jsonl \
  --output-dir envir/data/v2_scenario_natural_100 \
  --question-api-base http://127.0.0.1:8807/v1 \
  --question-model local-qwen3.5-9b \
  --question-workers 4 --max-revisions 2

python -m envir replay-v2 \
  --scenes envir/data/v2_scenario_natural_100/scenes.jsonl

python -m envir audit-v2 \
  --scenes envir/data/v2_scenario_natural_100/scenes.jsonl
~~~

Never naturalize the same bank twice. Use distinct directories for
different code/model/prompt experiments. LLM Critic acceptance is a
model judgment rather than a formal proof of semantic equivalence.

## Generation stages

1. MultiModalQA topic/source hints + frozen ProblemSpec and difficulty
2. Optional independently reviewed semantic scenario (no sampled values)
3. Typed executable Reasoning Program (filter, join, count, sort, group-count)
4. Synthetic Source–Entity–Observation World with multiple rows per source
5. Dependency-aware Macro TaskGraph compiled from the executable Program
6. Source-scoped Sub-Agent execution with typed derived WorkProducts
7. Gold replay, single-source baseline, source counterfactuals, provenance
8. Scenario-aware Question text + optional independent Author/Critic

## Files

- scenes.jsonl — private ProblemSpec, World, Program, DAG, Gold and evidence
- questions.jsonl — public questions and discoverable source descriptions
- difficulty_audit.jsonl — per-question complexity, baselines and mutations
- summary.json — counts, modality distribution and scenario review outcomes

The private teacher has all facts and the Gold Program. The Main-Agent
initial view contains neither. Source scan tasks can only read the
specifically authorized source, and downstream tasks consume only typed
WorkProducts, never the whole World.

All image, video and audio roles are fictional structured annotations.
No original MultiModalQA photograph, video or audio is perceived or used.
No real-world property or historical event is asserted by the simulator.

## Difficulty interpretation

Easy has one source. Medium has three sources and real two-step alias
resolution/approved-category joins. Hard has four sources and a further
entry-specific clearance condition. Goal families vary answer shape,
including counts, group aggregation, priority and relative chronology.

The single-source baseline is intentionally limited and does NOT
exhaustively search for an optimal shortcut. Counterfactual tests show
that the sampled critical source facts affect the substantive answer,
not that the Program is a globally minimal explanation. Node count,
program depth and orchestration cost are reported separately.

V2Session simulates an honest source-scoped orchestration process with
tool costs. Its verify action checks evidence closure only, not
independent-source verification or adversarial robustness. Training
an adaptive orchestration policy and adding attacks/retries remain
future extensions after measuring empirical V2 difficulty.

## Unit tests

~~~bash
python -m unittest discover -s envir/tests -p 'test_*.py' -v
~~~
