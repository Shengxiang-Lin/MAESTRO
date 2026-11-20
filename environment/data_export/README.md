# MAESTRO Unified Paired Dataset

## Canonical data contract

There is exactly one Scene-level split and **two granularity levels**:

1. `cases/<split>.jsonl`: each row is a self-contained *paired Case*.
   `clean` is the original accepted Stage-3 Episode. For an injected
   Case, `untrusted.episode` contains the full actually executed Stage-4
   trajectory. Labels include *where* the real fault happened,
   `fault.atoms` and `fault.applied_events`, the Main-observable
   `recovery.observable_issues`, follow-up
   `recovery.main_decisions`, dependency
   `invalidations/revalidations`, counterfactual status and final
   publication outcome. An additional `clean:<scene_id>` Case records
   the no-fault control.
2. `decisions/<split>.jsonl`: **one Main decision** with full exact
   Orchestra-o1 `prompt` (user message only), `reference_action`
   (full `action`, `reasoning`, `params`), Scene ID, candidate Case ID,
   source phase and decision round. Stage-3 clean decisions appear
   **once**, not repeated for each Stage-4 fork. Stage-4 decisions are
   the accepted post-fork ones only.

The canonical stable identity of a fault Case is
`round_sample_id`, **not** the display `sample_id` that can repeat
between different fault variants. A Decision uses its Case identity
plus its Main decision round.

**Do not pass the complete paired Case to a policy as input.** It is
a private evaluation/teacher container and may contain future
observations, fault labels and recovery outcomes. Only
`decisions[*].prompt` is the student-visible input; `reference_action`
is a supervised completion/teacher label. More reward-only state is
stored under `private/decision_labels/<split>.jsonl`.

## Files

```text
output/
  manifest.json                 # source SHA-256 / counts / policies
  quality_report.json           # Scene coverage, label conflicts, distribution audit
  splits.json                   # frozen Scene -> train/val/test assignment
  cases/{train,val,test}.jsonl  # clean+fault paired private cases
  decisions/{train,val,test}.jsonl
  private/decision_labels/{train,val,test}.jsonl
  adapters/da-grpo/{train,val,test}.jsonl
  adapters/da-sft/{train,val,test}.jsonl
  adapters/da-grpo/{train,val,test}.parquet        # if --parquet
  adapters/da-sft/{train,eval,test}.parquet         # if --parquet
```

The paired Case contains reviewer/debug data. A non-recovered or
language-rejected Episode is **diagnostic only**; it must never be
promoted as a positive SFT target. Stage-4A candidates that were
never selected for formal 4B/4C language review remain in the
original Stage-4A journals and candidate index, rather than being
duplicated as accepted Cases.

## Run

Use the **same** frozen Stage-1 Scene file that Stage 3 and Stage 4
used. Input manifests and prompt-renderer compatibility are verified,
and every fault Case is checked against its Stage-3
`clean_reference_id`.

```bash
python -m environment.data_export \\
  --scenes environment/data/stage1/questions/scenes.jsonl \\
  --clean-episodes environment/data/pilots/qwen_stage234_10/stage3/episodes.jsonl \\
  --clean-turns environment/data/pilots/qwen_stage234_10/stage3/main_turns.jsonl \\
  --stage4-dir environment/data/stage4_llm_pilot_10_1000 \\
  --output-dir environment/data/unified_llm_pilot_10_1000 \\
  --seed 17 --parquet
```

The exporter automatically audits Scene isolation, Case identity and
publication status, exact SFT prompt/completion parity, source/action/mode
counts, per-Scene imbalance, identical prompts, contradictory completions,
and Stage-4 fault distribution. It writes `quality_report.json`, and
verifies any `--parquet` SFT adapter against the canonical JSONL output.
Cross-split identical prompts fail; within-split repeated examples and
conflicting completions are counted (not silently removed). To reject the
latter as well, add `--strict-conflicts`.

To re-audit existing artifacts without rebuilding:

```bash
python -m environment.data_export.audit   --output-dir environment/data/unified --check-parquet --strict-conflicts
```

Parquet mode requires nonempty train/val/test decision splits after
positive-label filtering. If a pilot has too few reviewed decisions to cover
all three partitions, use a larger dataset rather than introducing template
fallback labels or silently reassigning held-out Scenes.

The default is **LLM-reviewed decisions only**. `--allow-template`
exists for explicit debugging, not production-quality training.
Parquet adapters require `pyarrow`. Run without `--parquet` to
produce canonical JSONL and JSONL adapters with Python alone.

Default split is 80/10/10 **by Scene**, *before* expanding a Scene
into clean, multiple faulty Episodes and decision rounds. For ten
accepted Scenes this gives 8/1/1. Every method uses the exact same
`splits.json`, not its own random split. Freeze/reuse an existing
map via `--split-map /path/to/splits.json`. The exporter fails if its
scene coverage differs. Create the final map from the full intended
Scene universe before final experiments; a new ten-Scene pilot
should not be treated as a stable full-1000-Scene map.

`--overwrite` intentionally replaces only the chosen output
directory. By default a nonempty output directory is rejected.

## Method-specific adapters and split reuse

- **DA-SFT**: use `adapters/da-sft/train.parquet` and
  `adapters/da-sft/eval.parquet` as the existing
  `baseline/da-sft` config paths (with `--parquet`). They contain
  **only** `prompt` and `completion`. `test.parquet` is held out.
- **DA-GRPO**: use `adapters/da-grpo/train.parquet` as the
  `baseline/da-grpo` input. Gold/expert stay in reward-only
  `reward_model` and `extra_info`; nothing leaks into `prompt`.
  Validation/test Parquet are separately available.
- **MAESTRO Risk Head**: keep its existing report-span construction;
  pass `--split-map output/splits.json` to
  `python -m train_maestro_8b.data.export_risk`. It now respects the
  exact `train/val/test` allocation instead of independently hashing
  Scenes. **MAESTRO policy GRPO** joins risk and decision-reward
  state by `(scene_id, round_sample_id, decision_round)` and now also
  propagates `test` when present.
- Other reliability baselines should select the same
  `split == train|val|test` and `case_id` views; no model is
  allowed to consult the private `fault` or `recovery` labels
  while constructing its online policy prompt.

Keep manifest, split map and source data frozen for all compared
methods. Number of *training decisions* and number of *paired Cases*
are different statistics; both are reported.
