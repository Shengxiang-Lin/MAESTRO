"""TRL reward bridge for the original MAESTRO Environment + Semantic Judge.

Only the reward changes. Policy rollout, Qwen3 chat template, TRL/DAPO loss,
8-completion grouping and Unsloth/vLLM remain the working Stage2 DA backend.
Private decision states are loaded from a SIDE-CAR keyed by the Stage4
candidate+decision round. Never concatenate them into the student prompt or
send them to the semantic API.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import pstdev
from typing import Any
from uuid import uuid4

from environment.untrusted_orchestration.rewards.decision_reward_state import SCHEMA
from environment.untrusted_orchestration.rewards.orchestration_reward import score_decision
from train_maestro_8b.policy.hybrid_reward import HybridReward

# Conservative normalization of Qwen3's reasoning envelope, *not* baseline's
# permissive "find the first brace anywhere" parser. Multiple JSON objects or
# arbitrary trailing commentary still fail the original hard legality gate.
_THINK = re.compile(r"^\s*<think>[\s\S]*?</think>\s*")


def normalize_action(text: str) -> tuple[str, bool]:
    """Return a standalone JSON Action when wrapped in one Qwen3 think/fence.

    Pure JSON stays byte-for-byte unchanged. Non-JSON remains invalid; do not
    convert a stream of multiple Answer/JSON blocks into an apparently legal
    Action.
    """
    if not isinstance(text, str):
        raise TypeError("Generated Action must be text")
    raw = text.strip()
    candidate = raw
    wrapper_removed = False
    if candidate.startswith("<think>"):
        match = _THINK.match(candidate)
        if match is None:
            return raw, False   # unfinished reasoning
        candidate = candidate[match.end():].strip()
        wrapper_removed = True
    fence = chr(96) * 3
    if candidate.startswith(fence):
        lines = candidate.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in (fence, fence + "json") or lines[-1].strip() != fence:
            return raw, False
        candidate = "\n".join(lines[1:-1]).strip()
        wrapper_removed = True
    try:
        obj = json.loads(candidate)
    except (TypeError, ValueError):
        return raw, False
    if not isinstance(obj, dict):
        return raw, False
    # No schema change: original score_decision still makes the final decision.
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")), wrapper_removed


def o1_contract_reason(text: str) -> str | None:
    """Check original Orchestra-o1 *structural* contract, not semantic binding.

    In O1 context and tools are optional. Wrong model/tool/capability and
    nonoptimal scheduling are for the global LLM Judge, not a JSON parser.
    """
    try:
        action = json.loads(text)
    except (ValueError, TypeError):
        return "invalid_json"
    if not isinstance(action, dict):
        return "invalid_json"
    kind, params = action.get("action"), action.get("params")
    if kind not in ("complete", "delegate_task") or not isinstance(params, dict):
        return "invalid_action_contract"
    if kind == "complete":
        answer = params.get("answer")
        return None if isinstance(answer, str) and answer.strip() else "missing_completion_answer"
    tasks = params.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return "invalid_delegate_params"
    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            return f"slot_{i}_invalid_subtask"
        if (not isinstance(task.get("task_instruction"), str)
                or not task["task_instruction"].strip()
                or not isinstance(task.get("model"), str)
                or not task["model"].strip()):
            return f"slot_{i}_invalid_subtask"
        if "context" in task and not isinstance(task["context"], str):
            return f"slot_{i}_invalid_context"
        if "tools" in task and (not isinstance(task["tools"], list)
                or any(not isinstance(tool, str) for tool in task["tools"])):
            return f"slot_{i}_invalid_tools"
    return None


def private_sample_id(entry: dict) -> str:
    task = entry.get("task_id")
    candidate = entry.get("round_sample_id")
    decision_round = entry.get("decision_round")
    if (not isinstance(task, str) or not isinstance(candidate, str)
            or not candidate.startswith(task + "-") or type(decision_round) is not int
            or decision_round < 1):
        raise ValueError("Invalid private Stage4 reward-state identity")
    return f"{candidate}-decision-round-{decision_round:02d}"


def load_private_states(path: str | Path) -> dict[str, dict]:
    """Load *only* Stage4 pre-action snapshots, rejecting mismatched schemas."""
    location = Path(path)
    if not location.is_file():
        raise FileNotFoundError(
            f"Missing Stage4 private decision reward states: {location}. "
            "Rebuild/backfill the Stage4 reward states before hybrid training."
        )
    mapping = {}
    with location.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            sample_id = private_sample_id(entry)
            state = entry.get("state")
            if not isinstance(state, dict) or state.get("schema") != SCHEMA:
                raise ValueError(f"{location}:{line_number}: invalid private state schema")
            if state.get("decision_round") != entry["decision_round"]:
                raise ValueError(f"{location}:{line_number}: mismatched decision round")
            if sample_id in mapping:
                raise ValueError("Duplicate private reward state for " + sample_id)
            mapping[sample_id] = state
    if not mapping:
        raise ValueError("No private decision reward states loaded")
    return mapping


def select_scorable_rows(rows: list[dict], private_states: dict[str, dict]) -> tuple[list[dict], dict]:
    """Fail on broken identifiers; explicitly report missing/unscorable rows."""
    selected, reasons, seen = [], Counter(), set()
    for row in rows:
        metadata = json.loads(row["extra_info_json"])
        sample_id = metadata.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Stage2 row without stable sample_id")
        if sample_id in seen:
            raise ValueError("Duplicate Stage2 data sample_id: " + sample_id)
        seen.add(sample_id)
        state = private_states.get(sample_id)
        if state is None:
            reasons["missing_private_state"] += 1
        elif state.get("scorable") is not True:
            reasons["unscorable_private_state"] += 1
        else:
            selected.append(row)
    if not selected:
        raise ValueError(
            "No aligned scorable Stage2 rows. "
            f"Input rows={len(rows)}, excluded={dict(reasons)}. "
            "Check Stage4 round_sample_id, decision_round and reward-state exports."
        )
    report = {
        "original_rows": len(rows), "selected_rows": len(selected),
        "excluded": dict(reasons),
        "private_state_count": len(private_states),
    }
    return selected, report


def prompt_lookup(rows: list[dict]) -> dict[str, str]:
    """Keep the original *public* policy prompt only, indexed by sample_id."""
    prompts = {}
    for row in rows:
        metadata = json.loads(row["extra_info_json"])
        sample_id = metadata["sample_id"]
        messages = row["prompt"]
        if (not isinstance(messages, list) or len(messages) != 1
                or messages[0].get("role") != "user"
                or not isinstance(messages[0].get("content"), str)):
            raise ValueError("Invalid O1 Main prompt in training rows")
        if sample_id in prompts:
            raise ValueError("Duplicate Main prompt sample_id")
        prompts[sample_id] = messages[0]["content"]
    return prompts


def diagnostic_legacy_reason(text: str, state: dict) -> str | None:
    """Old binding can fail even for valid O1 actions; do not gate new Judge."""
    try:
        return score_decision(text, state)["reason"]
    except (ValueError, KeyError, TypeError):
        return "legacy_scorer_unsupported"


def make_stage2_reward_function(*, private_states: dict[str, dict],
                                prompts: dict[str, str],
                                mode: str = "global",
                                judge=None, judge_weight: float = 0.4,
                                num_generations: int = 8,
                                diagnostics_path: str | Path | None = None,
                                global_contexts: dict[str, dict] | None = None):
    """Return one scalar for each TRL completion without changing trainer math.

    Hard illegal Action: -1. For valid hybrid Action: (1-w)*R_env+w*R_llm.
    In judge_only mode the Environment legality gate is still mandatory.
    """
    if mode not in ("global", "hybrid", "deterministic", "judge_only"):
        raise ValueError("Expected global/hybrid/deterministic/judge_only Stage2 reward")
    if (type(num_generations) is not int or num_generations < 2
            or not 0 <= judge_weight <= 1):
        raise ValueError("Invalid Stage2 GRPO group size / semantic weight")
    if mode != "deterministic" and judge is None:
        raise ValueError("Stage2 LLM reward requires a configured Judge")
    if mode == "global" and not global_contexts:
        raise ValueError("Global Judge requires decision-aligned evidence contexts")
    hybrid = (HybridReward(judge, mode=mode, judge_weight=judge_weight)
              if mode in ("hybrid", "judge_only") else None)
    path = Path(diagnostics_path) if diagnostics_path else None
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)

    def maestro_stage2_reward(
        completions: list,
        data_source: list[str],
        ground_truth: list[str],
        extra_info_json: list[str],
        trainer_state: Any = None,
        **_: Any,
    ) -> list[float]:
        from train_maestro_8b.policy.da_reward_adapter import completion_text
        n = len(completions)
        if not (n and n % num_generations == 0 and
                len(data_source) == len(ground_truth) == len(extra_info_json) == n):
            raise ValueError("TRL Stage2 candidate/metadata length mismatch")
        metadata = [json.loads(item) for item in extra_info_json]
        batch_id, step = uuid4().hex, getattr(trainer_state, "global_step", None)
        logs = []
        rewards = []
        for lo in range(0, n, num_generations):
            group_metadata = metadata[lo:lo + num_generations]
            sample_id = group_metadata[0].get("sample_id")
            if not isinstance(sample_id, str) or any(item.get("sample_id") != sample_id for item in group_metadata):
                raise ValueError("Mixed Stage4 candidate identities in GRPO reward group")
            if sample_id not in prompts or sample_id not in private_states:
                raise ValueError("Missing visible prompt or private reward state: " + sample_id)
            state = private_states[sample_id]
            if state.get("scorable") is not True:
                raise ValueError("Unexpected unscorable private state in training batch")
            raw_texts = [completion_text(item) for item in completions[lo:lo + num_generations]]
            normalized = [normalize_action(text) for text in raw_texts]
            texts = [item[0] for item in normalized]
            if mode == "global":
                from train_maestro_8b.policy.global_judge import build_global_judge_prompt
                if sample_id not in global_contexts:
                    raise ValueError("Missing exact global Judge context: " + sample_id)
                reference_prompt = build_global_judge_prompt(
                    prompts[sample_id], global_contexts[sample_id],
                )
                structural_reasons = [o1_contract_reason(text) for text in texts]
                eligible = [i for i, reason in enumerate(structural_reasons) if reason is None]
                semantic = judge.score_many(
                    reference_prompt, [texts[i] for i in eligible],
                ) if eligible else []
                if len(semantic) != len(eligible):
                    raise RuntimeError("Global Judge returned misaligned rewards")
                scored = [{
                    "reward": -1.0 if reason else None,
                    "valid": reason is None, "reason": reason,
                    "env_reward": None, "judge_reward": None,
                    "judge_dimensions": None, "judge_cached": None,
                    "judge_reason": None, "legacy_env_reason": None,
                } for reason in structural_reasons]
                for i, result in zip(eligible, semantic):
                    value = float(result["reward"])
                    if not 0 <= value <= 1:
                        raise ValueError("Global Judge reward outside [0,1]")
                    scored[i].update({
                        "reward": value, "judge_reward": value,
                        "judge_dimensions": result["dimensions"],
                        "judge_cached": bool(result.get("cached", False)),
                        "judge_reason": result.get("reason", ""),
                        # Legacy exact-string binding is DIAGNOSTIC ONLY.
                        # Never let its unsupported heuristic reject or stop
                        # a structurally legal globally judged candidate.
                        "legacy_env_reason": diagnostic_legacy_reason(texts[i], state),
                    })
            elif mode == "deterministic":
                scored = [score_decision(text, state) for text in texts]
                for value in scored:
                    value.update({
                        "env_reward": value["reward"], "judge_reward": None,
                        "judge_dimensions": None, "judge_cached": None,
                    })
            else:
                scored = hybrid.score_group(texts, {
                    "prompt": prompts[sample_id], "decision_state": state,
                })
            for offset, (result, raw, normalized_result) in enumerate(zip(scored, raw_texts, normalized)):
                rewards.append(float(result["reward"]))
                logs.append({
                    "batch_id": batch_id, "trainer_step": step,
                    "group_index": lo // num_generations, "candidate_index": offset,
                    "sample_id": sample_id, "reward_mode": mode,
                    "valid": bool(result["valid"]), "reason": result.get("reason"),
                    "env_reward": result["env_reward"], "judge_reward": result["judge_reward"],
                    "reward": result["reward"], "judge_dimensions": result["judge_dimensions"],
                    "judge_cached": result["judge_cached"],
                    "judge_reason": result.get("judge_reason"),
                    "legacy_env_reason": result.get("legacy_env_reason"),
                    "action_type": (
                        json.loads(normalized_result[0]).get("action")
                        if o1_contract_reason(normalized_result[0]) is None
                        else None
                    ),
                    "think_wrapper_normalized": normalized_result[1],
                    "raw_sha256": sha256(raw.encode("utf-8")).hexdigest(),
                })
            values = [x["reward"] for x in scored]
            reasons = Counter(x["reason"] for x in scored if not x["valid"])
            print("[MAESTRO/Stage2 reward] " + json.dumps({
                "step": step, "sample_id": sample_id, "mode": mode,
                "valid": sum(x["valid"] for x in scored),
                "judged": sum(x["judge_reward"] is not None for x in scored),
                "legacy_rejections_after_global_judge": (
                    sum(x.get("legacy_env_reason") is not None for x in scored if x["valid"])
                    if mode == "global" else None
                ),
                "reward_mean": round(sum(values) / len(values), 6),
                "reward_std": round(pstdev(values), 6),
                "invalid_reasons": dict(reasons),
            }, ensure_ascii=False), flush=True)
        if len(rewards) != n:
            raise RuntimeError("Reward alignment mismatch")
        if path is not None:
            with path.open("a", encoding="utf-8") as f:
                for entry in logs:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return rewards

    return maestro_stage2_reward
