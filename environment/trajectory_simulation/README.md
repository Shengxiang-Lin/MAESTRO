# Fragment-Centric Gold O1 Trajectories

Stage 3 **does not generate a new solution**. Stage 1 has already composed
KQA ProgramIR fragments into an executable task and Stage 2 has generated
an accepted natural-language question. This stage turns those **existing
answered fragments** into a clean, official Orchestra-o1 collaboration
trajectory, one interaction round at a time.

## Ground truth comes from Stage 1

`compiler.py` reads:
- `program.metadata.deep_dependency.composition_links[]` (fragment boundaries,
  stage, `output_node_id` and predecessor structure);
- `SymbolicExecutor.execute_trace()` (exact intermediate answers);
- the frozen `World` and rendered source observations (grounded clues);
- capability resource dependencies (evidence-discovery prerequisites).

Every fragment's complete ancestor subprogram is evaluated on candidate
source subsets. The compiler selects a **reachable evidence witness** that
reproduces its full-world result, checks decision-information completeness and
retains evaluated facts, including values that exclude a candidate.
It separates new source observations from previously read witness evidence,
so later fragments can reuse predecessor facts without claiming new retrieval.

A terminal fragment is added when operations remain outside the composed
fragments, or when additional evidence is required. It includes every remaining
operation in dependency order, not just the final COUNT/VERIFY. The terminal
calculation and final oracle answer remain programmatically verified.

The compiler **rejects** tasks without `composition_links`; it never falls
back to source-by-source exploration or invents a new decomposition.

## Role 1: Main Teacher

The public O1 state supplied at each decision contains only:

```json
{
  "question": "...",
  "subtask_history": "[Attempt 1] 🔄 Parallel Batch (2 subtasks)\\n  [1] ✅ done | Model: ...\\n  ├─ Task: ...\\n  ├─ Result: ...\\n  ├─ Summary: ...\\n  └─ Trace summary: ...",
  "available_tools": ["GoogleSearchAction", "ExtractUrlContentAction"]
}
```

This is the **cumulative, human-readable SUBTASK HISTORY format** used by
`orchestra/base_main_agent.py::_format_subtask_history()`. It includes
**every previous round** (not only the last one) and every parallel
Sub-Agent's task, result/answer, `finish_result.summary` and
`trace_summary`. Before the first delegation the history reads
`No subtasks completed yet.`

The simulator leaves unavailable raw tool-step counts as `?`; it never
manufactures steps or costs. The full tool registry is supplied in real
episodes; the example is abbreviated.

The **Main generation prompt** additionally receives a **PRIVATE** teacher
guidance block, compiled from the frozen composition graph:
`fragment_graph`, `completed_fragments`, `ready[]` (current fragment operations,
resource descriptors, prerequisites, permitted tools), remaining fragments
and terminal operator. Future answer values, future resource contents and
the oracle final answer are **not** included.

The actual prompt uses ordered plan slots and source types rather than raw
fragment/source handles. Operations retain dependency edges, readable comparison
symbols and public aliases for question entities. This keeps the gold planning
reference while reducing private-handle copying and branch ambiguity.

Main emits only the next native O1 action:
- `delegate_task`: one task per currently-ready independent gold fragment;
  all ready tasks are dispatched in the same parallel round. Main writes
  actionable `task_instruction`, `context` and selects tools. The controller
  binds task array slots to fragment IDs **privately**; the model no longer
  generates private task binding IDs.
- `complete`: only after all required fragments have succeeded, with a
  final answer checked against the frozen oracle.

The public `state_before.subtask_history` for each round contains **no private
guidance, symbolic trace, oracle answer or fragment IDs**. The output is an
O1-format trajectory, but the labeled action is gold-guided rather than a
free exploration rollout.

## Role 2: Sub-Agent Teacher

One feedback-generation request is made **per assigned fragment**, followed
by a scoped language review. Independent current-round tasks run in parallel.
Generation inputs include:
- the delegated O1 task;
- completed prerequisite fragment answers;
- the gold fragment operations;
- its incremental evidence packet (rendered source content and public facts);
- previously read witness observations, separately marked as reused evidence;
- its **authoritative intermediate answer** and per-operation results from Stage 1.

The simulator prompt focuses on the assigned task rather than repeating the full
original question, so future filters do not contaminate an intermediate report.

The LLM generates only:

```json
{
  "summary": "What this particular subtask established.",
  "trace_summary": "A concise narrative of the relevant clues and operations."
}
```

The program inserts the **exact frozen subtask answer** as
`finish_result.result` and `status: done`. It never asks the LLM to reproduce
the entire fact list, a private fact enum or a regex-constrained evidence
prefix. Consequently the JSON schema is small, while the factual answer is
strictly correct by construction.

Natural-language summaries are bounded and checked for numeric assertions,
unknown source references, exact title aliases and private IDs. LLM-generated
task instructions and reports then receive a temperature-zero fixed-contract
review. It uses readable dependency steps, typed completed results, observed
claims and authoritative step outputs; rejections must quote generated text.
The reviewer must not re-solve the question or substitute general world
knowledge for the synthetic world's verified operations. Offline reference
roles skip this LLM review. It is **not** a formal semantic entailment proof;
sample manually before using as training data. Neither role fabricates raw
tool-call traces, cost, failures or attacks.

The Main must assign every required source-modality tool for a fragment rather
than substitute a search/code tool because the predicate appears structured.

## Commands

From the MAESTRO repository root, after task construction and question
generation:

```bash
# Compile the fragment DAGs without invoking an LLM.
python environment/simulate_trajectories.py plan \
  --max-records 10 --output /tmp/maestro_fragment_plans.jsonl

# Offline structural smoke test (not LLM training data).
python environment/simulate_trajectories.py synthesize \
  --provider passthrough --max-records 10 \
  --output /tmp/maestro_fragment_offline.jsonl

# Real Qwen3.5-9B Main/Sub-Agent teacher generation.
python environment/simulate_trajectories.py synthesize \
  --max-records 10 --output /tmp/maestro_fragment_llm.jsonl

# Regression tests.
python -m unittest discover -s environment/tests -p "test_gold_trajectory.py" -v
```

Both roles default to the same local Qwen3.5-9B vLLM service, but use
different prompts and can have independently selected model profiles.

```yaml
trajectory_simulation:
  planner_profile: local_qwen35_9b
  executor_profile: local_qwen35_9b
  planner_temperature: 0.2
  executor_temperature: 0.2
  subagent_workers: 4
  max_attempts: 3
  flush_every: 5
```

`subagent_workers` parallelizes current-round fragments; rounds and
different questions are still sequential. `max_attempts` retries
structurally invalid Main actions or Sub-Agent summaries with rejection
feedback, **never** changing gold facts or fragment dependencies.

## Output and privacy boundary

`trajectories_2000.jsonl` contains one accepted episode per JSONL row:

- `trajectory[]`: native O1 public `delegate_task` actions, corresponding
  `subtask_results` with `finish_result` and `trace_summary`, followed by
  `complete`.
- `trajectory[].state_before`: public O1 question, formatted cumulative
  `subtask_history` (all prior results/trace summaries), and available tools only.
- `validation`: final-answer and fragment-completion checks.
- `gold_supervision`: **PRIVATE** composition fragment IDs, node IDs,
  source witnesses, exact intermediate answers, symbolic trace and task-slot
  bindings. Exclude this field and any plan-only outputs from student prompts.

`trajectories_2000.jsonl.rejected.jsonl` records rejection stages and
attempt diagnostics. These may contain private data and must remain in
audit storage. `flush_every` periodically writes whole episodes; Stage 3
does not yet have a cross-run checkpoint. Re-running synthesis overwrites
the chosen output paths. Use a distinct `--output` for trials.

This implementation intentionally **removes** the old Main free-selection
of arbitrary resource scopes, source-centric Sub-Agent answers, optional
source exploration, dynamic JSON enum/fact-prefix decoding, redundant
LLM fact copying and the previous dual simulation path.
