## Frozen private Global Judge context (new)

The canonical exporter now freezes private evaluator inputs alongside
the unchanged method-independent Main decision dataset:

```text
environment/data/unified/
  cases.jsonl
  decisions.jsonl
  private/
    decision_labels.jsonl
    scenes.jsonl                     # private Stage1 World + Macro Graph
    claim_state_supervision.jsonl   # published Stage4 pre-action claims
```

The private files are SHA256-bound to `manifest.json.private_context`.
They are **evaluator-only**: the O1 policy prompts remain unchanged;
no World, gold answer or unseen claims are inserted into a Main prompt.
Only claim snapshots for published Stage4 decision rounds are retained.
Claims from the current/future round are rejected.

**Upgrade an existing unified export without regenerating Cases/Decisions:**

```bash
python -m environment.data_export.private_context \
  --unified-dir environment/data/unified
```

The default command automatically discovers SHA256-matched original Stage1/4 files under `environment/data/` (override with `--discover-root`).
You can alternatively pass both `--scenes` and `--stage4-dir` explicitly.
The one-time upgrade checks the original Stage1 Scene file and Stage4
manifest AND main_turns hashes from `unified/manifest.json`.
If they no longer match, do not bypass this check: recover the exact
source versions or regenerate unified with matching Stage1/3/4.
Once upgraded, the original Environment paths are not needed for GRPO.
Existing unified exports without `private_context` remain valid for
DA-SFT and legacy DA-GRPO, but Global Judge training requires the new
private sidecars. Missing required selected Stage4 claim snapshots
are reported explicitly rather than fabricated.

Run `python -m environment.data_export.audit --output-dir
environment/data/unified --verify-manifest` to verify all hashed files.

---

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

Data preparation now shows phase-specific `tqdm` bars with counts, throughput
and ETA; completed Case/decision conversion stages also show useful
acceptance or action statistics. Use `--no-progress` on the `python -m environment.data_export`
command to suppress progress bars when running CI or redirecting logs.
This changes only the display, not the generated records or split policy.


## Single-turn publication semantics

Current Stage 4 writes one reviewed Main next-decision per selected candidate,
using journal status `single_turn_accepted`. This is **not** a completed
recovery episode. The exporter accepts both old `recovered` records and the
current single-turn status, setting `recovery.decision_accepted=true` for
published Stage-4 cases. `recovery.globally_recovered` is `null` for
single-turn cases: there is no future Sub-Agent execution to evaluate.
Dataset preparation should use `decision_accepted` when testing publication
eligibility rather than treating null global recovery as failure.
