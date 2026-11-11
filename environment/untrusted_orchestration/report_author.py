"""Stage-2-style Author/Guard/Critic for executed untrusted Sub-Agent attempts.

The frozen World and its Gold answer are NEVER available to this author or
reviewer. An actual (possibly wrong or partial) WorkProduct is authoritative
for the report. The LLM writes full Main-facing result, summary and trace
prose, while the deterministic Guard checks field values, identifiers,
execution scope, status and report completeness. No result is fabricated
when an execution produced no deliverable.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re

from environment.implement_subtask.reporting import (
    _normalize_author_result, _stage_scope_issues, _public_field_text,
    _readable_verified_answer, _readable_value,
)
from environment.implement_subtask.stage_local_claims import (
    normalize_report_language,
)
from .state_machine import deterministic_report, Attempt, relation_rows


_AUTHOR_SYSTEM = """Write a concise, natural Orchestra-o1 Sub-Agent finish report,
following the same evidence-local writing convention as Stage 2.

The authoritative input describes THIS executed attempt. Its result may be
incomplete or wrong relative to an unknown global answer: do not correct it.
Write only the actually delivered observation and the operations actually
recorded in observed_operator_steps. Do not infer that planned steps were
performed when a missing result halted the attempt.

Return JSON ONLY with the three prose fields:
{"result": "short actionable finding, or null if no result was delivered",
 "summary": "1-2 grounded sentences on this attempt and its evidence",
 "trace_summary": "1-2 grounded sentences on the actually observed steps"}

DO NOT GENERATE status. The executor owns status, structured rows, fact
references, partial-field availability and the existence of finish_result.
For a completed empty relation, report no matches; it is not missing.
For partial delivery, the attempt DOES have a finish_result and some
delivered WorkProduct. result MUST NOT be null: report available fields and
explicitly identify missing fields or missing remaining rows. For missing
delivery (finish_result_exists=false), result MUST be null.
If a returned field has value null, do NOT infer its original value from an
earlier internal operation or a sampled source atom. Its delivered value
remains unavailable.
For a wrong but delivered record, report that returned record faithfully,
without comparing it to a private clean reference or claiming independent
verification. An executed intermediate observation may disagree with a later
delivered value. Describe the final DELIVERED value as returned, not as
independently validated, and do not silently replace it with a seemingly
more plausible intermediate value.

Prefer concrete source roles and returned identifiers over generic backend
phrases. Keep intermediate handoffs local: a source scan does not complete
future filters and a join does not establish later eligibility checks.
Use observed_operator_steps for actions, row counts, evidence atoms and lookup
keys. IMPORTANT: sampled_evidence_atoms and sampled_result_rows are JUST
PREVIEWS. Their size does not indicate a truncated scan, missing work, or
partial execution. Trust delivery_status, rows_complete, and the actual
delivered relation for completeness. Earlier intermediate operator outputs
may differ from later projected/filtered fields normally.
Never claim to have browsed a real website, watched video or transcribed audio
when only the synthetic evidence observation was supplied. Do not invent facts,
identities, measurements, dates, sources, operations or supporting references.
Never mention Oracle, Gold, Teacher, fault injection, drafts or Critic.
"""

_CRITIC_SYSTEM = """Independently review the natural Sub-Agent report against
THIS ATTEMPT's actual returned relation and actually observed steps, using
Stage-2-style stage-local factual review.

The report is not a new solver. Do not compare it with a clean Oracle or
reject it merely because an observed result is wrong relative to a hidden
answer. The executor owns status and WorkProduct; check the prose only.
Evaluate:
1. grounded: facts, identities and measured values are supported by actual
   delivered records and observed evidence, with no invented operations;
2. scope_preserved: no later filter, lookup, eligibility or independent
   verification is falsely claimed, and incomplete deliveries are acknowledged;
3. natural: concise useful specialized feedback, not a raw relation dump;
4. useful: handoff identifies the actual returned records or unavailable parts.

Only executed operations can justify claims. A failed acquisition cannot
justify describing its planned downstream operations as completed.
A structured result has already been checked against the complete actual
delivered relation; abbreviated sampled evidence does not invalidate it.
CRITICAL: sampled_evidence_atoms / sampled_result_rows contain illustrative
previews only. A scan returning N rows DID execute on all N rows even if the
preview shows fewer. Never infer lost rows, incomplete search, or unknown
values from preview size. Earlier operator outputs are intermediate evidence;
they are not necessarily final delivered fields after project/filter/take.
For a partial_delivery with a null final field or omitted final rows, earlier
successful operations may have observed a value subsequently NOT DELIVERED.
This is a partial handoff, not a contradiction or an error in the program.
Review whether the prose correctly distinguishes observed intermediates from
actual delivered WorkProduct; never demand missing values be restored.
If an executed step and the final delivered result conflict, do NOT demand
that the report replace the actually delivered result with the step's value.
This feedback may describe an unreliable return. A report can truthfully
state "the attempt returned X" without asserting "X was verified correct".
Judge fidelity to the executed attempt, not hypothetical clean values. If
a contradiction is relevant, request neutral wording or explicit uncertainty,
not alteration of the delivered record.

Return JSON ONLY:
{"verdict":"pass|fail","grounded":true|false,
 "scope_preserved":true|false,"natural":true|false,"useful":true|false,
 "reason":"short factual explanation","repair_hint":"prose-only fix"}
"""


_TOKEN = re.compile(
    r"\b(?:KEY[12]-\d+|MK-\d+|RK-\d+|Case-\d+|Entry-\d+"
    r"|\d+(?:\.\d+)?)\b", re.I,
)
_ENTITY = re.compile(
    r"\b(?:KEY[12]-\d+|MK-\d+|RK-\d+|Case-\d+|Entry-\d+)\b", re.I,
)
_PRIVATE = re.compile(
    r"\b(?:oracle|gold\s+(?:answer|program|dag)|teacher|simulat\w*|"
    r"fault\s+inject\w*|previous\s+draft|critic\s+(?:said|noted)|"
    r"benchmark\s+answer|clean\s+(?:reference|result))\b", re.I,
)
_UNPERFORMED = re.compile(
    r"\b(?:i\s+watched|i\s+listened|i\s+searched\s+(?:the\s+)?web|"
    r"i\s+browsed\s+(?:the\s+)?internet)\b", re.I,
)


def _observed_contract(task: dict, attempt: Attempt, instruction: str) -> dict:
    """Build the Stage-2-shaped contract from THIS execution, never Oracle.

    In the real runner observed_operator_steps comes from run_attempt's
    per-primitive post-injection trace. Legacy hand-made test Attempts may
    lack steps; planned_operator_steps are then clearly marked as plans.
    """
    delivered = relation_rows(attempt.product)
    # Present the actual trace with Stage-2-style evidence examples, while
    # removing preview-truncation counters that can be mistaken for real
    # missing rows. Full runtime evidence remains unchanged in private audit.
    observed = []
    for raw in attempt.runtime.get("observed_operator_steps", []):
        step = deepcopy(raw)
        for key in ("remaining_rows_omitted", "evidence_atoms_truncated",
                    "evidence_atoms_total", "remaining_atoms_omitted"):
            step.pop(key, None)
        if "evidence_atoms" in step:
            step["sampled_evidence_atoms"] = step.pop("evidence_atoms")
        step["evidence_preview_only"] = True
        observed.append(step)
    planned = [
        {"operation": node["op"], "direct_source": node.get("source_id")}
        for node in task.get("operator_chain", [])
    ]
    executed = [
        row["operation"] for row in observed
        if row.get("execution_status") == "done"
    ]
    refs = sorted({
        ref for row in delivered for ref in row["source_fact_refs"]
    })
    refs = sorted(set(refs) | {
        ref for step in observed
        for ref in step.get("supporting_fact_refs", [])
    })
    return {
        "assigned_task_instruction": instruction,
        "observed_operator_steps": observed,
        "planned_operator_steps": planned,
        "trace_from_actual_execution": bool(observed),
        "executed_operations_in_order": executed,
        # As in Stage 2, the actual complete delivered relation is the
        # authoritative answer. Truncation applies to evidence PREVIEWS,
        # not to execution. Do not hide the rest of the delivered relation.
        "actual_delivered_rows": delivered,
        "total_delivered_rows": len(delivered),
        "delivered_relation_is_complete_in_prompt": True,
        "evidence_samples_are_preview_only": True,
        "output_fields": (
            list(attempt.product.columns) if attempt.product else []
        ),
        "authorized_direct_sources": list(task["source_ids"]),
        "available_provenance_sources": sorted({
            ref.split(":", 1)[0] for ref in refs
        }),
        "returned_source_fact_refs": refs,
        "available_fields": list(attempt.available_fields),
        "unresolved_fields": list(attempt.runtime.get("missing_fields", [])),
        "rows_complete": attempt.runtime.get("rows_complete", True),
        "delivery_issue": (
            "partial_field_delivery"
            if attempt.runtime.get("delivery_fault_kind") == "field_not_delivered"
            else "incomplete_delivered_record_set"
            if attempt.runtime.get("delivery_fault_kind") == "row_set_truncated"
            else None
        ),
        "execution_ended": True,
        "delivery_status": attempt.status,
        "finish_result_exists": attempt.product is not None,
        "delivery_semantics": (
            "The task returned NO finish_result. Only actually completed "
            "operations can be described."
            if attempt.product is None else
            "An internal operator may have observed more than the final "
            "handoff delivered. The final delivered WorkProduct is PARTIAL: "
            "report only available fields/rows, not missing values."
            if attempt.status == "partial" else
            "The complete returned WorkProduct is authoritative. Sample "
            "previews do NOT limit which rows the executor processed."
        ),
        "attempt_version": attempt.version,
        "used_parent_versions": attempt.used_parent_versions,
        "upstream_handoffs": list(task.get("depends_on", [])),
        "stage_scope_note": (
            "Only actually observed operations and this delivered result "
            "may be reported. Later Macro operations are not yet performed."
        ),
    }


def _claim_tokens(value: str) -> set[str]:
    """Normalize thousands separators, not facts or entity identifiers."""
    value = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value)
    return {token.casefold() for token in _TOKEN.findall(value)}


def _executor_result(attempt: Attempt, contract: dict) -> str:
    """Stage-2-style readable result from THIS returned WorkProduct only.

    Used solely when the Author omitted a required result. A wrong or
    incomplete model-supplied finding is never silently rewritten.
    """
    rows = [row["values"] for row in relation_rows(attempt.product)]
    if len(rows) == 1 and len(rows[0]) <= 5:
        pairs = ", ".join(
            f"{str(k).replace('_', ' ')}: {_readable_value(v)}"
            for k, v in rows[0].items()
        )
        result = "The delivered record contains " + pairs + "."
    else:
        result = _readable_verified_answer(rows)
    if attempt.status == "partial":
        missing = attempt.runtime.get("missing_fields", [])
        if "remaining_rows" in missing:
            result += " This is an incomplete delivered row set."
        elif missing:
            result += (
                " The following values were not delivered: "
                + ", ".join(x.replace("_", " ") for x in missing) + "."
            )
    return result


def _prepare_result(candidate: dict, attempt: Attempt,
                    contract: dict) -> tuple[dict, bool]:
    """Keep structured result exactly matched; repair only absent prose.

    Numeric scalar is accepted as a single-field result only if identical
    to the actually delivered value. Arbitrary numbers are never coerced
    into a different answer.
    """
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
            new["result"] = _executor_result(attempt, contract)
            return new, True
        return candidate, False
    new = dict(candidate)
    new["result"] = _executor_result(attempt, contract)
    return new, True


def _explicit_final_value_issues(text: str, contract: dict) -> list[str]:
    """Distinguish a claimed delivered field value from observed provenance.

    Numeric values inside an actually executed filter/lookup condition may
    legitimately be discussed in `result`, even if the final projection
    drops that field. But a value explicitly assigned to a FINAL output
    field must belong to the actual delivered rows, not just an intermediate
    preview. This especially protects null fields in partial handoffs.
    """
    issues = []
    rows = [entry["values"] for entry in contract["actual_delivered_rows"]]
    for field in contract["output_fields"]:
        forms = {field, field.replace("_", " ")}
        valid = {
            str(row[field]).casefold() for row in rows
            if row.get(field) is not None
        }
        for name in forms:
            # Match declarative field-value syntax, not incidental context
            # such as "marker code field remains unavailable" or a
            # source/filter operation that did not deliver the field.
            pattern = re.compile(
                r"\b" + re.escape(name) +
                r"\b\s*(?::|=|is|was)\s*"
                r"(?P<value>KEY[12]-\d+|MK-\d+|RK-\d+|"
                r"Case-\d+|Entry-\d+|-?\d+(?:\.\d+)?)\b",
                flags=re.I,
            )
            for match in pattern.finditer(text):
                value = match.group("value").casefold()
                if value not in valid:
                    issues.append("claimed_final_field_value_not_delivered")
    return sorted(set(issues))


def _guard(draft: dict, contract: dict, attempt: Attempt) -> list[str]:
    if not isinstance(draft, dict):
        return ["invalid_report_json"]
    issues = []
    required = {"result", "summary", "trace_summary"}
    # Old Stage-4 Authors may still supply status. It is ignored rather than
    # compared with the executor, and never appears in the final report.
    if not required <= set(draft) or set(draft) - (required | {"status"}):
        issues.append("invalid_report_fields")
    missing = attempt.product is None
    result = draft.get("result")
    exact_rows = [row["values"] for row in relation_rows(attempt.product)]
    normalized = None
    if missing:
        if result is not None:
            issues.append("missing_result_must_be_null")
    else:
        normalized, errors = _normalize_author_result(
            result, exact_rows, list(attempt.product.columns),
        )
        issues.extend("result_" + e for e in errors)
    summary = draft.get("summary")
    trace = draft.get("trace_summary")
    if not isinstance(summary, str) or not 20 <= len(summary.strip()) <= 750:
        issues.append("invalid_summary")
    if not isinstance(trace, str) or not 20 <= len(trace.strip()) <= 1000:
        issues.append("invalid_trace_summary")
    prose = " ".join(
        v for v in (normalized, summary, trace) if isinstance(v, str)
    )
    if _PRIVATE.search(prose):
        issues.append("internal_generation_or_teacher_leak")
    if _UNPERFORMED.search(prose):
        issues.append("claims_unperformed_media_or_web_action")

    # Reuse Stage 2's narrow stage-local factual scope guard. The reference
    # is this ATTEMPT's delivered relation, never the frozen Gold.
    stage_local = {
        "exact_oracle_answer": exact_rows,
        "executed_operations_in_this_macro":
            contract["executed_operations_in_order"],
        "terminal_output_schema_of_this_macro": contract["output_fields"],
        "macro_is_terminal": False,
        "upstream_handoffs": contract["upstream_handoffs"],
        "available_provenance_sources":
            contract["available_provenance_sources"],
    }
    for field, text in (
        ("result", normalized), ("summary", summary),
        ("trace_summary", trace),
    ):
        if isinstance(text, str):
            issues.extend(
                field + "_" + issue
                for issue in _stage_scope_issues(text, stage_local)
            )

    # Keep exact structured result comparison above and protect against
    # hallucinated IDs/measurements in prose. Unlike the old Stage 4 check,
    # actual observed intermediate steps, lookup keys and output counts
    # also authorize a statement; none are drawn from the clean Oracle.
    observed = contract["observed_operator_steps"]
    available = (
        json.dumps(contract["actual_delivered_rows"], ensure_ascii=False)
        + " " + json.dumps(observed, ensure_ascii=False)
    )
    result_tokens = _claim_tokens(json.dumps(
        contract["actual_delivered_rows"], ensure_ascii=False,
    ))
    result_tokens.add(str(contract["total_delivered_rows"]))
    # Stage 2 normalizes schema names like record_key_1 to "record key 1".
    # The digit 1 is a FIELD LABEL, not a fabricated numeric observation.
    # Authorize only actual output/observed schema labels, not arbitrary
    # numbers from the question or hidden clean WorkProduct.
    schema_fields = set(contract["output_fields"]) | {
        field for step in observed
        for field in step.get("output_schema", [])
    }
    result_tokens.update(_claim_tokens(
        " ".join(name.replace("_", " ") for name in schema_fields)
    ))
    allowed_tokens = _claim_tokens(available)
    allowed_tokens |= result_tokens | {
        str(len(contract["executed_operations_in_order"])),
    }
    instruction_tokens = _claim_tokens(
        contract["assigned_task_instruction"]
    )
    # An upstream identity can be used as an explanatory parent/context
    # label in a result, but never invented out of thin air. Only explicit
    # entity-shaped IDs already visible in both the instruction AND the
    # current actual observation are admitted; independent numeric values
    # remain scoped to the final delivered WorkProduct.
    contextual_entities = (
        {x.casefold() for x in _ENTITY.findall(
            contract["assigned_task_instruction"]
        )}
        & {x.casefold() for x in _ENTITY.findall(available)}
    )
    for field, text in (
        ("result", normalized), ("summary", summary),
        ("trace_summary", trace),
    ):
        if not isinstance(text, str):
            continue
        # A clean upstream identifier may be visible in a preceding
        # operation, but it must not replace the actually DELIVERED result.
        # A result may explain a grounded threshold, observed lookup key,
        # or intermediate row count as context. Those tokens originate in
        # the ACTUAL executed trace, not from the future clean World.
        # The separate final-field check prevents intermediate values
        # from masquerading as the delivered output.
        allowed = (allowed_tokens if field == "result"
                   else allowed_tokens | instruction_tokens)
        if _claim_tokens(text) - allowed:
            issues.append(field + "_unobserved_numeric_or_entity")
        if field == "result":
            issues.extend("result_" + issue for issue in
                          _explicit_final_value_issues(text, contract))

    if attempt.status == "partial":
        missing_fields = contract["unresolved_fields"]
        descriptive = " ".join(
            x for x in (normalized, summary, trace)
            if isinstance(x, str)
        ).lower()
        if missing_fields and not any(
            field in descriptive or field.replace("_", " ") in descriptive
            for field in missing_fields
        ) and not any(word in descriptive for word in (
            "incomplete", "unresolved", "missing", "not returned",
            "not delivered", "partial",
        )):
            issues.append("partial_missing_field_not_acknowledged")
    if attempt.status == "missing" and contract["observed_operator_steps"]:
        # Stage-local check on truthful failed-acquisition description.
        if re.search(r"\b(?:completed|successfully|finished)\b.{0,75}"
                     r"\b(?:lookup|extraction|search|retrieval)\b",
                     prose, re.I):
            issues.append("missing_claims_completed_acquisition")
    return sorted(set(issues))


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
    if author is None:
        return base, {"status": "accepted", "mode": provenance,
                      "attempts": 0}
    # In LLM mode even a repaired retry whose WorkProduct equals the clean
    # Oracle is freshly authored: its *current* delegated instruction and
    # completed verification differ from the old Stage 2 trace.
    if critic is None:
        raise ValueError("subagent_naturalization_requires_critic")
    contract = _observed_contract(task, attempt, instruction)
    failures = []
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
                    candidate[field] = revised
            candidate, result_normalized = _prepare_result(
                candidate, attempt, contract,
            )
        else:
            result_normalized = False
        issues = _guard(candidate, contract, attempt)
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
        result = candidate.get("result")
        if result is not None and not isinstance(result, str):
            result, _ = _normalize_author_result(
                result,
                [row["values"] for row in relation_rows(attempt.product)],
                list(attempt.product.columns),
            )
        try:
            decision = critic.complete_json(
                _CRITIC_SYSTEM,
                json.dumps({
                    "observed_current_attempt": contract,
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
        if _review_confuses_intermediate_and_delivery(decision):
            scope_rereviewed = True
            try:
                decision = critic.complete_json(
                    _CRITIC_SYSTEM + (
                        "\\nSECOND REVIEW: Distinguish the authoritative "
                        "final actual_delivered_rows from earlier intermediate "
                        "operator outputs and sampled evidence atoms. "
                        "A returned value may differ from an intermediate; "
                        "do not rewrite the returned value as a correction. "
                        "Check whether the report claims verification or "
                        "only faithfully describes the return. If its "
                        "summary invents a successful match or operation, "
                        "still FAIL. Do not auto-pass this review.\\n"
                    ),
                    json.dumps({
                        "observed_current_attempt": contract,
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
                "review": decision,
                "critic_boundary_rereviewed": scope_rereviewed,
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
        "actual_workproduct_immutable": True,
    }
