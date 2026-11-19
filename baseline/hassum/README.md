# HASSUM: semantic uncertainty guided online orchestration

Paper: Semantic Uncertainty-Guided Orchestration in Hierarchical
Multi-Agent Systems, https://arxiv.org/abs/2608.14707

The module computes normalized semantic entropy over candidate answer
clusters, a pairwise semantic-similarity density proxy, and a risk-guided
advisory verify decision. Semantic equivalence is evaluated by a metered
LLM judge (not keyword equality).

For an appropriate worker-level experiment, provide independently sampled
Sub-Agent outputs per report via options.samples_by_report. If omitted, this
adapter performs TEXT-ONLY LLM resampling from delivered task/report text.
It does not rerun video/audio tools and must not be reported as paper-exact
worker uncertainty. Default: four samples, one sampled answer + comparisons
per report; high inference cost.

    python baseline/reliability_benchmark.py --method hassum \
      --max-tasks 2 --max-concurrency 1 \
      --options '{"num_samples":4,"threshold":0.5,"entropy_weight":0.6}'
