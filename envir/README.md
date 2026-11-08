# MAESTRO envir: Subtask-First experimental environment

A separate implementation under MAESTRO/envir/. The original environment/
pipeline is not changed or used for question/world construction. The only
direct code reuse is the tested low-level YAGO Turtle stream parser.

## Concept

KQA-Pro -> logical graph skeletons (operators/dependencies only).
YAGO-4.6 -> REAL fact triples -> searchable subject pages -> parameterized
            relation/attribute subtask templates.
MultiModalQA -> official QA annotations + supporting source IDs and available
                text/table/image assets, with explicit verification tiers.

Build a large SOURCE / FACT BANK; cache concrete YAGO subtasks only when a
TaskGraph needs them. Never pre-generate 100K summaries or fake evidence.
A task-local searcher ranks PUBLIC source documents using only the query.
YAGO answers are checked against observed facts, public page content, and
public search discovery before becoming source_replay_verified.

MultiModalQA question answers are dataset_labeled (NOT equivalent to YAGO
source-replay verification). Image captions alone are caption_proxy; an
available image path is file_available_unverified and still does not prove
raw image recognition. There is no claim of image/audio/video model execution.

Build TaskGraphs with actual typed/identity-consistent dependency bindings.
Initial supported graph shapes:
  chain             YAGO relation -> YAGO attribute/relation
  parallel          two factual lookups from distinct YAGO subject pages
  media_to_search   official MMQA labeled answer -> unique exact YAGO name
Use a KQA-Pro skeleton bank as prior for allowed patterns, not an assumed
equivalent KoPL full-program proof. Invalid joins are rejected.

Produce:
  bank.sqlite             source documents, original facts, KQA motifs,
                          labeled MMQA tasks, lazy verified YAGO task cache
  subtask_templates.json  compact parameterized YAGO task templates
  prepare_summary.json    ingestion statistics and evidence-tier policies
  gold_graphs.jsonl       PRIVATE: answers, tool events, summary traces, graph
  public_questions.jsonl  PUBLIC: question, provided material IDs, tool names
  compose_summary.json    yield, rejections, coverage/diversity statistics

The natural-language Question is currently a deterministic fallback generated
from the graph's final goal. QuestionBrief stores the necessary constraints
and explicit public anchors, not the complete factual trace or Gold answer.
The new environment does not yet perform online LLM question authoring,
conditional Main decisions, noisy subagent attacks or real multimodal
perception. Those are subsequent extensions, not implicit claims of v0.1.

## Run

From the MAESTRO root:

~~~bash
python -m pip install tqdm

python -m envir.cli prepare \
  --kqa data/world-source/KQA-Pro/train.json \
  --yago data/world-source/YAGO-4.6/yago-facts.ttl \
  --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
  --output-dir envir/data/pilot \
  --max-yago-triples 300000 \
  --max-yago-subjects 100000 \
  --max-kqa-records 10000 \
  --max-mmqa-records 2000

python -m envir.cli inspect \
  --bank envir/data/pilot/bank.sqlite

python -m envir.cli compose \
  --bank envir/data/pilot/bank.sqlite \
  --output-dir envir/data/pilot \
  --target 20 --max-attempts 300 \
  --patterns chain,parallel,media_to_search \
  --seed 17

python -m envir.cli search \
  --bank envir/data/pilot/bank.sqlite \
  --query "director film"

python -m unittest discover -s envir/tests -p 'test_*.py' -v
~~~

IMPORTANT: YAGO is very large. The prototype's bounded PREFIX scan
may give few usable connected edges if early triples are skewed, so the
CLI returns no_graphs/exit code 2 rather than inventing a valid graph.
If necessary, increase max-yago-triples or run a follow-up indexed
entity-neighborhood sampler; do not blindly assume bigger Subtask count
means more composability.

Graph checks reject cycles, missing/unused required subtasks, incorrect
bindings, missing public sources, broken YAGO replay, and incorrect
two-result composition. Parallel means two independent public lookups,
not necessarily a deep AND/OR query over two sets. The current KQA motif
selector is an approximate pattern prior.

## Next improvements and evaluation

1. Measure real dataset yields, especially overlap between MMQA entity names
   and YAGO subjects. Do not force a synthetic identity match when absent.
2. Build type-aware and diversity-aware candidate indices; avoid storing
   subject-centered pages beyond the budget while retaining fact consistency.
3. Map KQA operators to certified multi-subtask relation/filter/join programs.
4. Use genuine image files with independently verified perception actions,
   never present captions as visual perception.
5. Model optional/redundant sources and cost-sensitive Main-Agent choices.
6. Audit Question semantic fidelity and single-source ablations, alongside
   human ratings of naturalness.

Maintain strict entity/evidence/task-family splits during training/evaluation,
not just random question splits. Gold graphs are never public prompts.
