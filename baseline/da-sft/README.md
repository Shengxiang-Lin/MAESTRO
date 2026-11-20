# DA-SFT — Qwen3-8B Main-Agent supervised fine-tuning

One **original Orchestra-o1 Main-Agent decision** is one supervised example.
DA-SFT learns the full reviewed `{"action", "reasoning", "params"}` JSON;
the prompt is the exact model-visible O1 Main input, including the accumulated
Sub-Agent history. Only assistant completion tokens contribute to the loss.

## Canonical data entrypoint (Stage 3 + Stage 4)

Run from the MAESTRO repository root. Always use the same Stage-1 Scene file
recorded in the Stage-3 and Stage-4 generation manifests:

```bash
python -m environment.data_export \
  --scenes environment/data/stage1/questions/scenes.jsonl \
  --clean-episodes environment/data/pilots/qwen_stage234_10/stage3/episodes.jsonl \
  --clean-turns environment/data/pilots/qwen_stage234_10/stage3/main_turns.jsonl \
  --stage4-dir environment/data/pilots/qwen_stage234_10/stage4 \
  --output-dir environment/data/unified \
  --seed 17 --parquet
```

Adjust the input paths for your production run. The defaults in
`config.yaml` read **only** these canonical adapter files:

- `environment/data/unified/adapters/da-sft/train.parquet`
- `environment/data/unified/adapters/da-sft/eval.parquet`

`test.parquet` is held out. **Do not** perform an independent DA-SFT split
and do not convert a DA-GRPO Parquet into SFT. All methods share the frozen
`splits.json` keyed by Scene, including every fault branch and decision round.

The exporter checks generation manifests, accepted/recovered Stage-4 Cases,
positive LLM-reviewed actions, source-to-Scene mapping, prompt reconstruction
and adapter consistency. Fallback/template actions are excluded by default;
`--allow-template` is for diagnostic ablations only.

Every run produces `quality_report.json` with source/mode/action distributions,
per-Scene imbalance, exact duplicate and conflicting Prompt labels, source
concentration, Scene split integrity and Stage-4 fault-kind coverage.
Structural errors and cross-split prompt collisions fail the export; ordinary
within-split duplicates/conflicts are reported, not silently deleted. Use
`--strict-conflicts` to reject identical prompts with different completions.
When `--parquet` is set, written Parquet adapters are round-trip verified.

Audit a previously exported dataset without rebuilding it:

```bash
python -m environment.data_export.audit \
  --output-dir environment/data/unified --check-parquet --strict-conflicts
```

## Install and preflight

Use an isolated Python 3.11+ GPU environment:

```bash
python -m pip install -r baseline/da-sft/requirements.txt
python -m unittest environment.tests.test_data_audit environment.tests.test_data_export
python -m unittest discover -s baseline/da-sft -p 'test_sft.py'
python baseline/da-sft/train.py --dry-run
python baseline/da-sft/train.py --check
```

The CPU tokenizer preflight rejects overlong prompts or entire conversations
instead of truncating examples. Verify the full distribution before adjusting
`max_prompt_tokens=12288` or `max_seq_length=16384`.
During GPU startup the trainer also checks the actual TRL completion mask.

## Single-GPU BF16 LoRA

```bash
python baseline/da-sft/train.py --max-steps 1 \
  --output-dir baseline/da-sft/output_smoke
bash baseline/da-sft/run.sh
```

The default is Qwen3-8B, BF16 LoRA rank 32, gradient checkpointing, 8-bit
AdamW, batch 1, accumulation 8, two epochs. Optional QLoRA:

```bash
python baseline/da-sft/train.py --load-in-4bit \
  --output-dir baseline/da-sft/output_qlora
```

Resume only a DA-SFT TRL checkpoint (not a GRPO optimizer checkpoint):

```bash
python baseline/da-sft/train.py --resume baseline/da-sft/output/checkpoint-100
```

No vLLM, Judge, reference policy, rollout generation or policy-gradient loop
is created. Packing stays disabled to preserve checked completion boundaries.
Qwen3 uses its model chat template and assistant EOS `<|im_end|>`; examples
contain the action JSON with its `reasoning` field, not a hidden thinking trace.

For comparisons with DA-GRPO, evaluate on identical held-out Scene IDs using
parseable O1 JSON, valid tool/action sets, decision quality, and latency.
Training loss alone is not a Main-Agent orchestration metric.
