# A-Trust (attention trust score): MAESTRO online adapter

Paper: Attention Knows Whom to Trust / To Trust or Not to Trust,
https://arxiv.org/abs/2506.02546

Implementation: a frozen Hugging Face causal LM with eager attentions
produces layer-level attention-to-message features. Six separately trained
logistic classifiers score message trust dimensions. The lowest dimension
sets message risk; high-risk reports trigger advisory verification.

This is an A-Trust-inspired **feature adaptation**, not an exact reproduction
of the authors' attention feature engineering, data or published thresholds.
There is NO LLM-as-attention substitution.

Dependencies: torch, transformers, accelerate, numpy, scikit-learn, joblib.
Train on a JSONL containing current visible task_entries, report_id and six
binary labels (see train.py). Label 1 means reliable.

    python baseline/a-trust/train.py --train /path/train.jsonl \
      --attention-model /models/Qwen3-8B --output /path/a_trust_head

From repository root:

    python baseline/reliability_benchmark.py --method a-trust \
      --max-tasks 3 --max-concurrency 1 \
      --options '{"attention_model":"/models/Qwen3-8B","classifier_dir":"/path/a_trust_head","threshold":0.5}'

Requires *local* model weights and memory to expose attentions. Inference
can be expensive: review recomputes a score for every prior report.
Classifier checkpoints are not distributed. Pickled joblib weights should
only be loaded from a trusted source. The Main planner need not be trained.
