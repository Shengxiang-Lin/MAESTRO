# DA-SFT: Qwen3-8B + Unsloth + TRL SFTTrainer

**One reviewed Main-Agent turn -> one full expert action JSON label.**
Only completion tokens are supervised. There is no GRPO-to-SFT conversion.

## 1. Build the generic environment dataset

Use the one method-independent `environment.data_export` command from
[the exporter README](../../environment/data_export/README.md) once.
It writes `environment/data/unified/{cases,decisions}.jsonl` and private
teacher state, but no SFT/GRPO Parquet and no train/val/test split.

## 2. Prepare *DA-SFT* training data here

```bash
python baseline/da-sft/prepare_data.py \
  --source-dir environment/data/unified \
  --output-dir baseline/da-sft/data
```

This method-local converter:
- verifies the base manifest/hashes and Main prompt/action contract;
- filters to LLM-reviewed positives by default (no deterministic fallback);
- deterministically splits **Scene IDs** 80/10/10 for train/val/test;
- writes only `prompt` and `completion` to the Parquets;
- keeps decision identity/source/mode in separate metadata JSONL;
- writes `splits.json`, `quality_report.json` and `summary.json`.

```text
baseline/da-sft/data/
  train.parquet
  eval.parquet
  test.parquet
  train_metadata.jsonl
  eval_metadata.jsonl
  test_metadata.jsonl
  splits.json
  quality_report.json
  summary.json
```

The trainer's `config.yaml` now reads `baseline/da-sft/data/train.parquet`
and `baseline/da-sft/data/eval.parquet` by default. Test stays held out.
Default split seed is 17. For exact method comparisons, use
`--split-map baseline/da-grpo/data/splits.json` on this converter.
Use `--dry-run` for a no-write inspection; `--overwrite` is required
to replace a previously generated baseline-local dataset.

The quality audit reports Scene imbalance, Stage-3 vs Stage-4 coverage,
repeated identical prompt/label pairs and conflicting completions.
`--strict-conflicts` rejects the latter. Unreviewed/fallback labels are
excluded unless explicitly requested with `--include-fallback`.
No rows are silently reweighted, deduplicated or truncated.
All three splits must contain reviewed positive labels.

## 3. Validate and train

```bash
python -m pip install -r baseline/da-sft/requirements.txt
python -m unittest baseline.test_dataset_preparation
python -m unittest discover -s baseline/da-sft -p 'test_sft.py'
python baseline/da-sft/train.py --dry-run
python baseline/da-sft/train.py --check
python baseline/da-sft/train.py --max-steps 1 \
  --output-dir baseline/da-sft/output_smoke
bash baseline/da-sft/run.sh
```

The model is local `base_models/Qwen3-8B`, BF16 LoRA rank 32 alpha 64,
16,384-token full context, 12,288-token prompt cap, 8-bit AdamW,
batch 1/gradient accumulation 8, two epochs. `--check` refuses oversized
prompt or total conversation. Startup checks the actual TRL completion mask
instead of assuming that user context is not supervised.
Packing stays disabled. Optional `--load-in-4bit` enables QLoRA.

```bash
python baseline/da-sft/train.py --resume \
  baseline/da-sft/output/checkpoint-100
```

No vLLM, online Judge, KL term, reference model or RL rollout is used. The
completion contains the original `action`, `reasoning` and `params`
fields but is not a hidden thinking trace. Compare on matching held-out
Scenes using full orchestration and decision metrics, not SFT loss alone.

Data preparation now shows phase-specific `tqdm` bars with counts, throughput
and ETA; completed Case/decision conversion stages also show useful
acceptance or action statistics. Use `--no-progress` on the `prepare_data.py`
command to suppress progress bars when running CI or redirecting logs.
This changes only the display, not the generated records or split policy.
