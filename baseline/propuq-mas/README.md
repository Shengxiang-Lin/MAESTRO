# PropUQ-MAS: propagation-aware uncertainty online adapter

Paper: PropUQ-MAS: Propagation-Aware Uncertainty Quantification for
LLM Multi-Agent Systems, https://arxiv.org/abs/2608.22130

Core risk propagation is r_v = 1 - (1-u_v) product(1-alpha_pv*r_p).
The input graph is ONLY prior, visibly evidenced agent-message dependencies,
not the private MAESTRO task DAG. Local uncertainties and edge uptake rates
can be provided as external estimates, otherwise a metered LLM estimates
them from visible history. The latter is an adaptation and NOT necessarily
the same uncertainty estimator / graph as the original paper.

    python baseline/reliability_benchmark.py --method propuq-mas \
      --max-tasks 3 --options '{"threshold":0.55}'

For controlled comparisons pass full local_uncertainty and edges mappings.
The baseline fails on missing local risk rather than filling fake confidence.
No additional Main training; runtime inference budget must include estimator.
