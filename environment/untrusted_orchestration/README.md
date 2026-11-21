# Stage 4 — Main-observable unreliable feedback (single-turn)

```bash
python -m environment generate-untrusted-turns \
  --config environment/configs/untrusted_orchestration.yaml
```

From a **reviewed Stage-3 Clean Reference**, perturb one already returned
Sub-Agent feedback batch and produce **one** independently reviewed Main
next-decision label. Historical Stage-3 Main dispatches and earlier reports
stay unchanged. No hidden Oracle truth enters the Main prompt; no future
Sub-Agent task is executed to certify recovery.

## Observable scenario targets

| Target | Main-visible condition | Necessary evidence |
| --- | --- | --- |
| A | Evidence insufficiency | missing / partial / falsely finished delivery |
| B1 | Internal inconsistency | two contradictory claims within one delivered report |
| B2 | Constraint inconsistency | public key, source scope, relation or arithmetic contradiction |
| B3 | Inter-report inconsistency | same subject, field and scope across **actually delivered** reports |

B3 additionally records source relationship: a genuine
`independent_source_claim_conflict` requires explicit independent direct
provenance. Repeated queries to one source are not independent corroboration.
A candidate fault/operator is **never** accepted as proof of a visible
conflict. Stage 4A independently reconstructs the public witness.

## Runtime modules

- `trusted_prefix.py`: validates the supplied Stage-3 Episode against
  independently compiled Stage-1/2 trusted evidence
- `scenario/`: A/B1/B2/B3 feasibility, one-fault candidates,
  round sampling and scenario-first selection
- `runtime/`: actual atomic faults, evidence requests and replay
- `inspection/`, `state_machine.py`: Main-visible claim/source/dependency
  state and independent contradictions
- `reporting/`, `offline_pipeline.py`: Sub-Agent feedback naturalization,
  Main Author/Critic and guarded next action
- `publication.py`: separate positive turns, failed trials and private
  audit/supervision labels
- `generate_turns.py`: bounded candidate orchestration and resumable
  Stage 4A/4B/4C journals

The only execution unit is `single_turn`, not multi-round recovery. The
internal `scenario_mode` label also uses `single_turn`. A positive sample
means a legal reviewed Main decision based on the **currently visible**
feedback; it does not establish eventual answer correctness or global task
recovery.

## Publication and diagnostics

The 500-batch defaults bound candidate exploration to 24 variants per
feedback round and 8 distractors per fault, selecting up to 2 per round
against a global **2,000 positive Main-turn budget**. These are upper
bounds, not a guaranteed number of positive samples or exhaustive coverage.

- `main_turns.jsonl`, `episodes.jsonl`: **accepted** public one-turn labels
- `private/candidate_selection.jsonl`: 4A eligibility and separate 4B/4C
  primary/backfill outcomes
- `private/scene_evidence_affordances.jsonl`: structural opportunity, not
  proof of public B3 conflict
- `private/claim_state_supervision.jsonl`: private decision-time evidence
  states for accepted turns only
- `summary.json`: input prefixes, A/B1/B2/B3 realized and published
  distribution, language-review status and per-candidate failure reasons

Select actual published candidates with
`language_review_outcome == "single_turn_accepted"`, **not** merely
`selected_for_language_review == true`, because failed candidates and
backfill attempts are also present. Template mode only tests the executor
and selection logic; it is not LLM-reviewed supervision. Fallback reports
never become positive training labels.

For evaluation, use disjoint Scene (or stronger shared-template) splits and
measure decisions from an independently running Main Agent, not just the
offline Author/Critic acceptance rate.
