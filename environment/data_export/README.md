# MAESTRO method-agnostic unified data export

`python -m environment.data_export` exports only **base evidence and
Main-Agent decisions**. It does not pick a training/evaluation split, choose
loss/reward labels, filter generation modes, or write any trainer-specific
Parquet. These are responsibilities of the individual `baseline/*` methods.

## Input

Use the matching accepted Stage-1 Scenes, Stage-3 clean episodes and Main
turns, and Stage-4 publication output. The exporter checks both generation
manifests, clean-reference IDs, recovered Stage-4 publication state, case
identity and exact Orchestra-o1 Main prompt reconstruction.

```bash
python -m environment.data_export \
  --scenes environment/data/stage1/questions/scenes.jsonl \
  --clean-episodes environment/data/pilots/qwen_stage234_10/stage3/episodes.jsonl \
  --clean-turns environment/data/pilots/qwen_stage234_10/stage3/main_turns.jsonl \
  --stage4-dir environment/data/pilots/qwen_stage234_10/stage4 \
  --output-dir environment/data/unified
```

The example paths are the current pilot configuration; replace them with
the real matching generation files for a production dataset. Repeat only with
an explicit `--overwrite` (or use a new output directory).

## Output (no method-specific datasets)

```text
environment/data/unified/
  manifest.json
  quality_report.json
  cases.jsonl
  decisions.jsonl
  private/
    decision_labels.jsonl
```

- `cases.jsonl`: one clean Case per accepted Scene plus reviewed Stage-4
  fault Cases. Includes private paired Episode/fault/recovery provenance;
  unrecovered/diagnostic cases never supply positive Main decisions.
- `decisions.jsonl`: one **accepted/published Main decision** per row, with
  unique `decision_id`, `scene_id`, `case_id`,
  `generation_mode`, exact `prompt` (a single O1 user message),
  `visible_state.subtask_history`, and complete `reference_action`
  (`action`, `reasoning`, `params`). Stage-3 decisions appear once;
  Stage-4 distinct fault candidates are keyed by
  `(round_sample_id, decision_round)`.
- `private/decision_labels.jsonl`: offline `ground_truth`,
  `reference_action`, and optional Stage-4 reward state, keyed by
  `decision_id`. **Never expose this object to a policy prompt.**
- `quality_report.json`: structural integrity, sources, actions, generation
  modes, duplicate/conflicting prompts, per-Scene imbalance and fault kinds.
- `manifest.json`: frozen input/output hashes and counts. Baselines
  validate base-file hashes before transforming data.

The base export retains all published positive Main decisions, including
explicitly labeled fallback/template modes. Method-specific label eligibility
is decided downstream; private unrecovered/dropped 4A candidates are not
silently turned into positive decisions.

Check a persisted base dataset:

```bash
python -m environment.data_export.audit \
  --output-dir environment/data/unified --verify-manifest
```

The base layer deliberately has **no** `splits.json`, `train.parquet`,
`eval.parquet`, `adapters/da-sft`, or `adapters/da-grpo`.

## Method-owned conversion

```bash
python baseline/da-sft/prepare_data.py \
  --source-dir environment/data/unified \
  --output-dir baseline/da-sft/data
python baseline/da-grpo/prepare_data.py \
  --source-dir environment/data/unified \
  --output-dir baseline/da-grpo/data
```

Each baseline independently defines its example format, reviewed-label
policy, Scene-group train/val/test selection, quality checks and storage.
Both start from the same deterministic `seed=17` split algorithm for fair
default comparisons, but each writes its **own** `data/splits.json`.
Use `--split-map baseline/da-sft/data/splits.json` on DA-GRPO (or vice
versa) to pin exactly the same assignments. Existing MAESTRO risk exports
can also accept that file via their `--split-map` flag.

Train/eval/test split is **by Scene**, never by decision, branch or fault
variant. A Scene's clean and untrusted decisions cannot cross splits.

The two data preparation scripts write their *own* `quality_report.json`.
They omit unreviewed/fallback actions by default, report duplicate/conflicting
labels without auto-deletion and reject identical prompts shared across
different splits. `--strict-conflicts` additionally rejects conflicting
expert actions for the same prompt.
