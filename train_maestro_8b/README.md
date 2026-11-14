# MAESTRO 8B — latent window-risk pretraining

This is the proposed MAESTRO trainable Main-Agent path, **outside baseline/**.
It replaces the former macro-action SFT and consequence-critic code with
**Stage 1: non-generative, weakly supervised evidence-risk pretraining**.
Stage 2 will train the same PEFT causal-LM adapter with an interactive GRPO
policy objective; the auxiliary risk head does **not** appear in the runtime
prompt or required policy output.

## Learning problem

At Main decision round t the student sees the **full native O1 prompt**,
including question, previous dispatches and every returned Sub-Agent
Result / Summary / Trace summary. Each report can contain multiple implicit
claims and may depend on previous reports. No claim extraction is performed
at student inference time.

The student processes the full input **once**, then partitions each existing
Sub-Agent report into approximately 128-token windows (stride 96). Local
last-layer window representations are conditioned on the last valid
full-history hidden state. A small shared MLP scores each window, and a
top-k mean over window logits predicts **one observable-warning label per
report**, with masked BCE. This is weakly supervised MIL: the highest-scored
window is **not** guaranteed to be the erroneous fact. Training updates both
the LoRA backbone and the MLP, not just a frozen probe.

Risk label semantics:

- risk=1: an observable partial/missing delivery or a delivered claim in a
  visible disagreement / intra-report conflict.
- risk=0: a completed, parseable report with no such observed warning.
- mask=0: unparsable reports, superseded deliveries, ambiguous dependent
  reports, or cases without dependable evidence-based supervision.

**Risk=0 does not mean the report is factually true, corroborated, or safe to
accept without further verification.** The simulator may know a private
single root fault, but its hidden identity and true answer are NOT supervised
as if Main could observe them. There can be several affected reports from
a single root fault. Current release learns observable risk; future work
can add independently justified support and context-pair ranking targets.

## Inputs

Environment paths from the current canonical pipeline:

- environment/data/stage1/questions/scenes.jsonl
- environment/data/stage4/episodes.jsonl
- environment/data/stage4/private/claim_state_supervision.jsonl

The exporter accepts **globally recovered, reviewed** branches only, because
that is what the current published claim-state supervision covers. It does
not silently promote Stage-4 diagnostics / non-reviewed candidates to
positive labels. The exporter joins by scene ID, fault-branch decision round
and Main decision round; it rebuilds the complete O1 prompt via
environment.base_orchestration.o1_compat and maps the formatted report-block
offsets to weak supervision. It excludes private state/Gold from the prompt.

Potentially important current limitation: typed claim extraction does not
cover every free-form Result/Summary format. Reports with no parsed claims
and no explicit partial/missing status are **masked**, not negatives.

## Export

Run the environment first using the canonical YAMLs, then:

    python -m train_maestro_8b.export_reliability \
      --scenes environment/data/stage1/questions/scenes.jsonl \
      --episodes environment/data/stage4/episodes.jsonl \
      --states environment/data/stage4/private/claim_state_supervision.jsonl \
      --output_dir environment/data/stage4/private/window_risk

Output:

- train.jsonl / val.jsonl: Main prompt, report offsets, risk labels/masks
- summary.json: number of samples and label-reason coverage

The split is stable **per scene**, not per history row or failure candidate.
Use the summary to check parsed coverage and positive/negative balance
before committing GPU time.

## Train

Install torch, transformers, peft, accelerate and a fast tokenizer.
For 24 GB GPUs try QLoRA with small microbatches; 16K training still needs
memory profiling. Mixing shorter real history prefixes with a few 16K
histories is preferable to forcing everything to 16K.

    accelerate launch -m train_maestro_8b.train_reliability \
      --model /path/to/Qwen3-8B \
      --train_file environment/data/stage4/private/window_risk/train.jsonl \
      --eval_file environment/data/stage4/private/window_risk/val.jsonl \
      --output_dir workspace/checkpoints/maestro-window-risk \
      --max_length 16384 --window_size 128 --window_stride 96 \
      --batch_size 1 --grad_accum 8 --load_in_4bit \
      --gradient_checkpointing

The loader indexes JSONL byte offsets (not all prompts in memory), validates
a fast tokenizer's character-to-token offset mapping, does not cross report
boundaries, pads on the right, and **skips overlong histories** instead of
silently truncating dependencies. The model invokes the inner Qwen
transformer rather than the language-model head to avoid large vocabulary
logits. The head uses report-wise top-2 window-logit pooling; loss is first
averaged across labeled reports within a history.

The checkpoint contains:

- adapter_config.json and adapter_model.safetensors: **standard PEFT LoRA**
  adapter, loadable onto the original CausalLM for Stage-2 GRPO.
- window_risk_head.pt and window_risk_config.json: optional diagnostic
  readout, not part of the inference policy.
- tokenizer files.

No special <orch> token is added and no novel LM head is required; Stage-2
should load the original causal LM and attach this adapter. The auxiliary
head may be dropped after pretraining.

## Testing

    python -m unittest train_maestro_8b.test_reliability -v
    python -m compileall -q train_maestro_8b

**Not yet established:** throughput on 16K / Qwen3-8B GPU, learned
localization accuracy, additional gains from counterfactual ranking, and
GRPO transfer. Probe accuracy is not sufficient evidence that policy actions
use the latent representation; compare against GRPO-only and compute-matched
SFT+GRPO baselines.
