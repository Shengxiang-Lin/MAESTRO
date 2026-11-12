"""4B: compile executed facts, naturalize wording, and review additions.

Execution/Delivery/Observed Discrepancy are owned by report_contract.py;
fact validation is isolated in report_validation.py. This module does not
inspect hidden clean answers. Report-only adversarial feedback deliberately
bypasses normal factual naturalization and is NOT silently repaired.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re

from environment.implement_subtask.reporting import _public_field_text
from environment.implement_subtask.stage_local_claims import (
    normalize_report_language,
)
from .report_contract import (
    build_report_contract, render_delivered_result,
)
from .report_validation import (
    validate_report_draft, claim_tokens, _explicit_final_value_issues,
)
from .public_language import sanitize_public_prose
from .state_machine import deterministic_report, Attempt

# Private aliases remain available for older environment tests; these do
# not contain duplicate implementation paths.
_observed_contract = build_report_contract
_guard = validate_report_draft
_claim_tokens = claim_tokens


_AUTHOR_SYSTEM = """Naturalize a Sub-Agent's executed evidence into concise
Orchestra-o1 feedback. You are NOT deciding which facts occurred.
The fact_spine in observed_current_attempt is CODE-OWNED and authoritative
about observable executed steps and the ACTUAL returned WorkProduct.

Return JSON ONLY with TWO prose fields:
{"summary":"1-2 natural sentences on delivery status and observations",
 "trace_summary":"1-2 natural sentences on executed operations"}
Do NOT generate a result field: the executor supplies immutable result
prose from fact_spine.delivery.result_text. If your output nevertheless
contains result, it will be independently checked before being ignored.

Rules:
- Execution observations and final delivery are different categories.
  A value seen during scan but null in the final delivery is STILL NULL
  in the final delivery. Never silently restore it.
- A discrepancy between two COMPLETE executed steps is an objective
  observation when fact_spine says so. You may report this mismatch
  without declaring either side correct, verified, compromised or attacked.
- The final returned WorkProduct is authoritative ONLY as a record of what
  was delivered; it has NOT been independently validated as correct.
- The program will insert mandatory factual clauses and will render the
  final result from delivered rows. Treat authored text as natural phrasing,
  not as permission to add or correct values.
- No invented source, operation, number, date, causal explanation or
  outside website/media/tool access. Do not mention hidden Gold, teacher,
  critic, synthetic internals or any attack/fault-injection process.
- Missing, partial and completed empty returns are distinguished by
  fact_spine.delivery. You are not asked to author their result fields.
"""

_CRITIC_SYSTEM = """Review ONLY model-authored prose extensions of the
code-owned execution/delivery/discrepancy facts.

The deterministic fact_spine has already established what was executed,
what was actually DELIVERED, what fields are missing, and what specific
inter-step identity differences are PROVABLE from complete snapshots.
These facts are not hypotheses to re-evaluate. No Oracle/Gold is present.

A report may accurately state 'the attempt returned X' without claiming
'X was independently verified correct'. When earlier observed values
conflict with later delivered values, BOTH facts can be reported; no hidden
cause or correct replacement can be inferred.

Evaluate:
grounded: No UNSUPPORTED additional fact or cause is asserted.
scope_preserved: No unexecuted tool action or independent validation is claimed.
natural: Concise specialist feedback rather than schema boilerplate.
useful: Delivery status, relevant missing data and observable discrepancies
are conveyed without presenting preliminary observations as delivered values.

If rejecting, identify the EXACT unsupported model-written assertion.
Do not reject a statement that is explicitly present in fact_spine merely
because it seems implausible, final delivery differs, or a sampled preview
omits a row.

Return JSON ONLY:
{"verdict":"pass|fail","grounded":true|false,
 "scope_preserved":true|false,"natural":true|false,"useful":true|false,
 "reason":"short quote-based explanation","repair_hint":"prose-only fix"}
"""


def _attach_mandatory_facts(candidate: dict, contract: dict) -> dict:
    """Keep observed/undelivered facts visible without letting LLM invent.

    Caller must first check the Author JSON and raw factual guard. This
    only inserts deterministic, current-attempt facts. No private Gold,
    attack origin or unreleased observation is introduced.
    """
    facts = contract["fact_spine"]["mandatory_summary_clauses"]
    if not facts or not isinstance(candidate.get("summary"), str):
        return candidate
    modified = dict(candidate)
    anchor = " ".join(facts)
    natural = modified["summary"].strip()
    # Deterministic facts take priority over optional commentary, and
    # optional text is retained when it fits the existing review budget.
    # This must not silently remove a required fact from the model output.
    if len(anchor) + 1 + len(natural) <= 750:
        modified["summary"] = anchor + " " + natural
    else:
        modified["summary"] = anchor
        modified["_optional_summary_omitted_due_to_length"] = True
    return modified


def _prepare_result(candidate: dict, attempt: Attempt,
                    contract: dict) -> tuple[dict, bool]:
    """Keep structured result exactly matched; repair only absent prose.

    Numeric scalar is accepted as a single-field result only if identical
    to the actually delivered value. Arbitrary numbers are never coerced
    into a different answer.
    """
    # The requested V10.3.1 Author schema is summary/trace ONLY. When
    # the model follows it, compile the result from the actual delivery.
    # If it still supplies a result, the existing strict Guard will
    # inspect it; an invented value is never silently corrected.
    if "result" not in candidate:
        completed = dict(candidate)
        completed["result"] = (
            render_delivered_result(contract)
            if attempt.product is not None else None
        )
        return completed, True
    raw = candidate.get("result")
    if attempt.product is None or raw is not None:
        if (attempt.product is not None
                and type(raw) in (int, float)
                and len(attempt.product.columns) == 1
                and len(attempt.product.rows) == 1
                and raw == attempt.product.rows[0]["values"][
                    attempt.product.columns[0]
                ]):
            new = dict(candidate)
            new["result"] = render_delivered_result(contract)
            return new, True
        return candidate, False
    new = dict(candidate)
    new["result"] = render_delivered_result(contract)
    return new, True


def _review_confuses_intermediate_and_delivery(review: dict) -> bool:
    """Identify a *possible* reviewer scope error, never auto-accept it."""
    if not isinstance(review, dict) or review.get("verdict") != "fail":
        return False
    reason = (
        str(review.get("reason") or "") + " "
        + str(review.get("repair_hint") or "")
    ).lower()
    return (
        bool(re.search(
            r"intermediate|sampled|observed|projection|project|join|lookup",
            reason,
        ))
        and bool(re.search(
            r"delivered|returned|final|partial|null|missing|"
            r"no rows|zero rows|should yield|should return",
            reason,
        ))
    )


def _review_denies_witnessed_handoff(review: dict) -> bool:
    """An optional *second review* of an observed discrepancy, never a pass."""
    if not isinstance(review, dict) or review.get("verdict") != "fail":
        return False
    critique = (
        str(review.get("reason") or "") + " " +
        str(review.get("repair_hint") or "")
    ).casefold()
    return any(phrase in critique for phrase in (
        "falsely claims a contradiction", "falsely claims a discrepancy",
        "falsely claims a logical inconsistency",
        "not a contradiction", "no contradiction",
        "no mismatch", "no discrepancy", "not inconsistent",
        "does not constitute a contradiction",
        "remove the claim of contradiction",
        "remove the claim of logical inconsistency",
    ))


def naturalize_report(task: dict, attempt: Attempt, *,
                      instruction: str, model_alias: str,
                      reviewed: dict | None,
                      author=None, critic=None,
                      max_revisions: int = 2) -> tuple[dict, dict]:
    """Generate all three public prose fields from this executed attempt.

    Template fallback is explicitly labeled; in LLM dataset mode the
    caller can fail closed rather than treating fallback as reviewed.
    """
    base = deterministic_report(
        task, attempt, instruction=instruction,
        model_alias=model_alias, reviewed=reviewed,
    )
    provenance = base.pop("_report_mode")
    # Controlled report faults alter *only what Main sees*, not the
    # executed WorkProduct. They are intentional, so the normal report
    # author/critic cannot silently "fix" the malicious claim.
    if attempt.runtime.get("report_fault_kind"):
        from .report_faults import render_report_fault
        return render_report_fault(base, attempt)
    if author is None:
        return base, {"status": "accepted", "mode": provenance,
                      "attempts": 0}
    # In LLM mode even a repaired retry whose WorkProduct equals the clean
    # Oracle is freshly authored: its *current* delegated instruction and
    # completed verification differ from the old Stage 2 trace.
    if critic is None:
        raise ValueError("subagent_naturalization_requires_critic")
    contract = build_report_contract(task, attempt, instruction)
    failures = []
    witness_review_attempted = False
    for revision in range(max_revisions + 1):
        request = {
            "observed_current_attempt": contract,
            "previous_prose_feedback": failures[-2:],
        }
        try:
            candidate = author.complete_json(
                _AUTHOR_SYSTEM,
                json.dumps(request, ensure_ascii=False),
            )
        except Exception as exc:
            failures.append({"stage": "author_backend", "reason": str(exc)[:200]})
            continue
        # Stage 2's surface-only normalization removes internal schema
        # jargon without changing any structured result or evidence claim.
        if isinstance(candidate, dict):
            author_result_omitted = "result" not in candidate
            known_fields = set(contract["output_fields"]) | {
                step["operation"] for step in
                contract["observed_operator_steps"]
            }
            candidate = dict(candidate)
            for field in ("result", "summary", "trace_summary"):
                if isinstance(candidate.get(field), str):
                    revised = _public_field_text(
                        candidate[field], known_fields,
                    )
                    revised, _ = normalize_report_language(revised)
                    # Synthetic source handles are internal affordances,
                    # not phrases to hand back to Main; turn them into a
                    # grounded source role without changing substantive facts.
                    revised = re.sub(
                        r"(?:simulated-web-index|workspace)://"
                        r"([A-Za-z][A-Za-z0-9_-]*)",
                        lambda m: m.group(1) + " source", revised,
                        flags=re.I,
                    )
                    candidate[field] = sanitize_public_prose(revised)
            candidate, result_normalized = _prepare_result(
                candidate, attempt, contract,
            )
        else:
            result_normalized = False
            author_result_omitted = False
        # First vet the Author's ORIGINAL claims. Never use a deterministic
        # prefix to conceal invented identities, numeric claims or a wrong
        # final field value.
        raw_issues = validate_report_draft(candidate, contract, attempt)
        if raw_issues:
            issues = raw_issues
        else:
            candidate = _attach_mandatory_facts(candidate, contract)
            issues = validate_report_draft(
                {k: v for k, v in candidate.items()
                 if not k.startswith("_")}, contract, attempt,
            )
        if issues:
            failures.append({
                "stage": "guard", "errors": issues,
                # Private diagnostics: no attempted text is put into the
                # published Main turn. A mere error count is insufficient
                # to distinguish false positives from actual invention.
                "candidate_result": (
                    str(candidate.get("result"))[:600]
                    if isinstance(candidate, dict) else None
                ),
                "candidate_summary": (
                    str(candidate.get("summary"))[:600]
                    if isinstance(candidate, dict) else None
                ),
            })
            continue
        # The structured WorkProduct remains in the execution ledger.
        # LLM's structured answer, when supplied, must match exactly and is
        # normalized to Stage-2-style concise readable result prose.
        # The executor, not a stylistic LLM, owns the delivered values.
        # The Author's raw result was independently validated above.
        result = contract["fact_spine"]["delivery"]["result_text"]
        try:
            decision = critic.complete_json(
                _CRITIC_SYSTEM,
                json.dumps({
                    "observed_current_attempt": contract,
                    "locked_fact_spine": contract["fact_spine"],
                    "candidate_main_facing_report": {
                        "result": result,
                        "summary": candidate["summary"],
                        "trace_summary": candidate["trace_summary"],
                        "status": attempt.status,
                    },
                }, ensure_ascii=False),
            )
        except Exception as exc:
            failures.append({"stage": "critic_backend", "reason": str(exc)[:200]})
            continue
        # A critic may mistake an intermediate operator preview for the
        # final returned WorkProduct, particularly under partial/wrong
        # execution. Re-review this narrow class of objections using the
        # SAME immutable actual-attempt contract. Never auto-accept or
        # rewrite the result just because the first critique appears wrong.
        scope_rereviewed = False
        witness_rereviewed = bool(
            contract["observed_handoff_discrepancies"]
            and _review_denies_witnessed_handoff(decision)
        )
        if (_review_confuses_intermediate_and_delivery(decision)
                or witness_rereviewed):
            scope_rereviewed = True
            witness_review_attempted |= witness_rereviewed
            try:
                decision = critic.complete_json(
                    _CRITIC_SYSTEM + (
                        "\nSECOND REVIEW: Distinguish the authoritative "
                        "final actual_delivered_rows from earlier intermediate "
                        "operator outputs and sampled evidence atoms. "
                        "A returned value may differ from an intermediate; "
                        "do not rewrite the returned value as a correction. "
                        "Check whether the report claims verification or "
                        "only faithfully describes the return. The "
                        "observed_handoff_discrepancies are deterministic "
                        "witnesses derived ONLY from complete adjacent "
                        "executed step snapshots: if listed, do not deny "
                        "those specific identity differences. They do "
                        "not justify a claimed cause or clean answer. "
                        "If the summary invents a successful match or "
                        "operation, still FAIL. Do not auto-pass.\n"
                    ),
                    json.dumps({
                        "observed_current_attempt": contract,
                        "locked_fact_spine": contract["fact_spine"],
                        "candidate_main_facing_report": {
                            "result": result,
                            "summary": candidate["summary"],
                            "trace_summary": candidate["trace_summary"],
                            "status": attempt.status,
                        },
                        "first_critique_for_reconsideration": {
                            "reason": str(decision.get("reason", ""))[:300],
                        },
                    }, ensure_ascii=False),
                )
            except Exception as exc:
                failures.append({
                    "stage": "critic_boundary_rereview_backend",
                    "reason": str(exc)[:200],
                })
                continue
        if (isinstance(decision, dict)
                and decision.get("verdict") == "pass"
                and all(decision.get(k) is True for k in (
                    "grounded", "scope_preserved", "natural", "useful",
                ))):
            report = deepcopy(base)
            if report.get("finish_result") is not None:
                report["finish_result"] = {
                    "status": attempt.status,
                    "result": result,
                    "summary": candidate["summary"].strip(),
                }
                report["trace_summary"] = candidate["trace_summary"].strip()
            else:
                # Preserve real missing finish_result. Place the status
                # explanation in the observable trace, never invent one.
                report["trace_summary"] = (
                    candidate["summary"].strip() + " " +
                    candidate["trace_summary"].strip()
                )
            return report, {
                "status": "accepted", "mode": "llm_reviewed",
                "attempts": revision + 1,
                "author_status_ignored": candidate.get("status") is not None,
                "trace_from_actual_execution":
                    contract["trace_from_actual_execution"],
                "executor_result_normalized": result_normalized,
                "report_contract_version": "v10_3",
                "delivered_result_locked": True,
                "result_compiled_after_author_omission":
                    author_result_omitted,
                "mandatory_fact_clauses": len(
                    contract["fact_spine"]["mandatory_summary_clauses"]
                ),
                "optional_summary_omitted_due_to_length": bool(
                    candidate.get("_optional_summary_omitted_due_to_length")
                ),
                "review": decision,
                "critic_boundary_rereviewed": scope_rereviewed,
                "critic_witness_rereviewed": witness_rereviewed,
                "observed_handoff_discrepancy_count": len(
                    contract["observed_handoff_discrepancies"]
                ),
                "actual_workproduct_immutable": True,
            }
        failures.append({
            "stage": "critic",
            "reason": str(decision.get("reason", ""))[:250]
                      if isinstance(decision, dict) else "invalid_review",
            "repair_hint": str(decision.get("repair_hint", ""))[:250]
                           if isinstance(decision, dict) else "",
        })
    return base, {
        "status": "fallback",
        "mode": "grounded_template_fallback",
        "attempts": max_revisions + 1,
        "failures": failures,
        "critic_witness_rereviewed": witness_review_attempted,
        "observed_handoff_discrepancy_count": len(
            contract["observed_handoff_discrepancies"]
        ),
        "actual_workproduct_immutable": True,
        "report_contract_version": "v10_3",
        "mandatory_fact_clauses": len(
            contract["fact_spine"]["mandatory_summary_clauses"]
        ),
    }
