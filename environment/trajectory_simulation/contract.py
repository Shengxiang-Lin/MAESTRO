"""Small O1 wire contracts for fragment-guided clean teacher trajectories.

LLMs never copy private graph handles or fact lists into public O1 actions.
Every delegated task is bound to a ready gold fragment by its OUTPUT INDEX.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .compiler import json_value
from .tools import tool_profile

TOOLS = tool_profile().available


def answer_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(json_value(value), ensure_ascii=False)


def _number(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    token = str(value).strip()
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", token):
        token = token.replace(",", "")
    result = Decimal(token)
    if not result.is_finite():
        raise InvalidOperation
    return result


def answers_equal(actual: Any, expected: Any, contract: Mapping[str, Any]) -> bool:
    if isinstance(expected, bool):
        if isinstance(actual, str) and actual.strip().lower() in {"true", "false"}:
            actual = actual.strip().lower() == "true"
        return isinstance(actual, bool) and actual == expected
    if isinstance(expected, (int, float)):
        try:
            return _number(actual) == _number(expected)
        except (InvalidOperation, ValueError, TypeError):
            return False
    if isinstance(expected, (list, tuple, set)):
        if isinstance(actual, str):
            try:
                actual = json.loads(actual)
            except json.JSONDecodeError:
                if contract.get("collection_mode") == "set" and len(expected) == 1:
                    actual = [actual]
        if not isinstance(actual, (list, tuple, set)):
            return False
        numerical = bool(expected) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in expected
        )
        def key(value: Any) -> str:
            return (str(_number(value).normalize()) if numerical
                    else json.dumps(json_value(value), sort_keys=True, ensure_ascii=False))
        try:
            left = [key(x) for x in actual]
            right = [key(x) for x in expected]
        except (ValueError, InvalidOperation, TypeError):
            return False
        mode = contract.get("collection_mode", "multiset")
        if mode == "set":
            return set(left) == set(right)
        if mode == "ordered":
            return left == right
        return sorted(left) == sorted(right)
    return actual == expected


def action_schema(guidance: Mapping[str, Any]) -> dict[str, Any]:
    text = {"type": "string"}
    if guidance["may_complete"]:
        return {
            "type": "object", "properties": {
                "action": {"type": "string", "enum": ["complete"]},
                "reasoning": text,
                "params": {"type": "object", "properties": {
                    "answer": {"type": "string"},
                }, "required": ["answer"], "additionalProperties": False},
            }, "required": ["action", "reasoning", "params"],
            "additionalProperties": False,
        }
    task = {
        "type": "object", "properties": {
            "task_instruction": text, "context": text,
            "model": {"type": "string", "enum": ["model_1"]},
            "tools": {"type": "array", "minItems": 1,
                      "items": {"type": "string", "enum": list(guidance.get("available_tools",TOOLS))}},
        }, "required": ["task_instruction", "context", "model", "tools"],
        "additionalProperties": False,
    }
    return {
        "type": "object", "properties": {
            "action": {"type": "string", "enum": ["delegate_task"]},
            "reasoning": text,
            "params": {"type": "object", "properties": {
                "tasks": {"type": "array", "items": task},
            }, "required": ["tasks"], "additionalProperties": False},
        }, "required": ["action", "reasoning", "params"], "additionalProperties": False,
    }


def feedback_schema() -> dict[str, Any]:
    return {
        "type": "object", "properties": {
            "summary": {"type": "string"},
            "trace_summary": {"type": "string"},
        }, "required": ["summary", "trace_summary"], "additionalProperties": False,
    }


def operation_steps(operations):
    """Translate the private DAG into an unambiguous task-writing reference."""
    local = {op["node_id"]: index + 1 for index, op in enumerate(operations)}
    steps = []
    for index, op in enumerate(operations, 1):
        name, args = op["op"], op["inputs"]
        if name == "FIND":
            instruction = "Identify the entity named " + args[0] + "."
        elif name == "FIND_ALL":
            instruction = "Start a NEW branch with all entities in the supplied records."
        elif name == "FILTER_TYPE":
            instruction = "Keep input entities of type " + args[0] + "."
        elif name in {"FILTER_STRING", "FILTER_NUMBER", "FILTER_YEAR", "FILTER_DATE"}:
            instruction = f"Keep input entities whose {args[0]} {args[1]} {args[2]}."
        elif name == "RELATE":
            direction = "objects back to subjects" if "backward" in args else "subjects to objects"
            instruction = f"Follow the {args[0]} relation from input {direction}."
        elif name in {"INTERSECT", "UNION"}:
            instruction = ("Intersect" if name == "INTERSECT" else "Union") + " the specified input sets."
        elif name == "COUNT":
            instruction = "Count the distinct entities in the input set."
        elif name == "QUERY_NAME":
            instruction = "Return the names of input entities."
        elif name == "QUERY_ATTR":
            instruction = "Retrieve the " + args[0] + " values of input entities."
        elif name.startswith("VERIFY_"):
            instruction = f"Return whether any input value {args[0]} {args[1]}."
        elif name == "QUERY_RELATION":
            instruction = "Return relation names linking the first input set to the second."
        elif name in {"SELECT_AMONG", "SELECT_BETWEEN"}:
            instruction = f"Select input entities by {args[0]} using comparison {args[1]}."
        else:
            raise ValueError("No task-writing reference for operation " + name)
        steps.append({
            "step": index, "instruction": instruction,
            "input_from": [
                f"step {local[parent]}" if parent in local else "completed predecessor result"
                for parent in op["dependencies"]
            ],
        })
    return steps


def source_evidence(observations):
    return [{
        "inspected_resource_id": observation["resource_id"],
        "exact_title_alias": observation["title"],
        "source_type": observation["source_type"],
        "authoritative_facts": [
            f"INSPECTED {observation['resource_id']} ({observation['title']}): "
            f"{claim['subject']} | {claim['predicate']} | {claim['object']}"
            for claim in observation["claims"]
        ],
    } for observation in observations]


def main_prompt(payload):
    from .realization import main_style_prompt
    return main_style_prompt(payload)

def feedback_prompt(payload):
    from .realization import feedback_style_prompt
    return feedback_style_prompt(payload)


def _hidden_identifier(text: str) -> bool:
    return bool(re.search(
        r"(?:fragment:\d+|fragment:terminal|subtask:|discover:|entity:[A-Za-z0-9_-]+"
        r"|\bProgramIR\b|\bSymbolicExecutor\b|\bsymbolic executor\b"
        r"|\bprivate (?:gold )?guidance\b|\bgold (?:fragment|operations|guidance)\b"
        r"|\bsynthesis controller\b|\bprivate fragment graph\b|\bauthoritative answer\b"
        r"|\bprevious (?:attempt|JSON|generation) (?:failed|was rejected)\b)",
        text,
        re.I,
    ))


def validate_action(value: Any, guidance: Mapping[str, Any],
                    oracle: Any, answer_contract: Mapping[str, Any],
                    *, public_state: Mapping[str, Any] | None = None) -> tuple[bool, list[str]]:
    if not isinstance(value, Mapping):
        return False, ["action_not_object"]
    issues: list[str] = []
    if set(value) != {"action", "reasoning", "params"}:
        issues.append("unexpected_action_fields")
    if not isinstance(value.get("reasoning"), str) or _hidden_identifier(value.get("reasoning", "")):
        issues.append("invalid_public_reasoning")
    params = value.get("params")
    if not isinstance(params, Mapping):
        return False, ["params_not_object"]
    public_sources = set(re.findall(
        r"source_\d+", json.dumps(public_state or {}, ensure_ascii=False)
    ))
    visible_text = str(value.get("reasoning", ""))
    if guidance["may_complete"]:
        if set(re.findall(r"source_\d+", visible_text)) - public_sources:
            issues.append("unobserved_resource_in_main_reasoning")
        if value.get("action") != "complete":
            issues.append("must_complete_after_gold_fragments")
        if not isinstance(params.get("answer"), str) or not params["answer"].strip():
            issues.append("complete_answer_must_be_nonempty_string")
        elif not answers_equal(params.get("answer"), oracle, answer_contract):
            issues.append("final_answer_mismatch")
        if set(params) != {"answer"}:
            issues.append("unexpected_complete_params")
        return not issues, issues
    if value.get("action") != "delegate_task":
        issues.append("must_delegate_ready_gold_fragments")
    tasks = params.get("tasks")
    ready = guidance["ready"]
    if not isinstance(tasks, list) or len(tasks) != len(ready):
        return False, issues + ["parallel_gold_fragment_count_mismatch"]
    if set(params) != {"tasks"}:
        issues.append("unexpected_delegate_params")
    for index, (task, fragment) in enumerate(zip(tasks, ready)):
        if not isinstance(task, Mapping):
            issues.append(f"task_{index}:not_object")
            continue
        if set(task) != {"task_instruction", "context", "model", "tools"}:
            issues.append(f"task_{index}:unexpected_fields")
        public_text = str(task.get("task_instruction", "")) + " " + str(task.get("context", ""))
        visible_text += " " + public_text
        for key in ("task_instruction", "context"):
            content = task.get(key)
            if not isinstance(content, str) or not content.strip() or _hidden_identifier(content):
                issues.append(f"task_{index}:invalid_{key}")
        if task.get("model") != "model_1":
            issues.append(f"task_{index}:invalid_model")
        tools = task.get("tools")
        if not isinstance(tools, list) or not tools or any(x not in guidance.get("available_tools",TOOLS) for x in tools):
            issues.append(f"task_{index}:invalid_tools")
        elif not set(fragment["suggested_tools"]).issubset(tools):
            issues.append(f"task_{index}:tools_not_grounded")
    if set(re.findall(r"source_\d+", visible_text)) - public_sources:
        issues.append("future_resource_id_in_public_action")
    return not issues, issues


def validate_feedback(value: Any, fragment: Mapping[str, Any], *, strict=False,
                      public_refs=()) -> tuple[bool, list[str]]:
    if strict:
        from .realization import validate_bound_feedback
        okay, bound_issues = validate_bound_feedback(value,fragment)
        if okay: return True, []
        _, legacy_issues = validate_feedback(value,fragment,strict=False,public_refs=public_refs)
        return False, sorted(set(legacy_issues + bound_issues))
    if not isinstance(value, Mapping):
        return False, ["feedback_not_object"]
    public_refs=tuple(ref for ref in public_refs if isinstance(ref,str))
    issues = []
    if set(value) != {"summary", "trace_summary"}:
        issues.append("unexpected_feedback_fields")
    supported = json.dumps({
        "answer": fragment["expected_answer"],
        "operations": [{"op": op["op"], "inputs": op["inputs"]}
                       for op in fragment["operations"]],
        "operation_answers": [item["answer"]
                              for item in fragment.get("operation_results", [])],
        "evidence": fragment["observations"],
        "known_evidence": fragment.get("known_observations", []),
    }, ensure_ascii=False)
    for key in ("summary", "trace_summary"):
        content = value.get(key)
        if not isinstance(content, str) or not content.strip():
            issues.append("empty_" + key)
            continue
        if len(content) > (600 if key == "summary" else 1400):
            issues.append(key + "_not_compressed")
        if _hidden_identifier(content):
            issues.append(key + "_leaks_private_fragment_id")
        # Prevent references to resources outside this task's scoped packet.
        mentioned = set(re.findall(r"source_\d+", content))
        assigned = {obs["resource_id"] for obs in [
            *fragment["observations"], *fragment.get("known_observations", []),
        ]}
        if mentioned - assigned:
            issues.append(key + "_unobserved_source")
        titles = {obs["title"].casefold() for obs in [
            *fragment["observations"], *fragment.get("known_observations", []),
        ]}
        for reference in re.findall(r"\bReference\s+\d+\b", content, re.I):
            if reference.casefold() not in titles:
                issues.append(key + "_unknown_resource_title:" + reference)
        observations = [*fragment["observations"], *fragment.get("known_observations", [])]
        # An explicit one-source assertion must refer to a fact in that source.
        # Ambiguous multi-source sentences stay with the language reviewer.
        for sentence in re.split(r"(?<=[.!?])\s+", content):
            lowered = sentence.casefold()
            referenced = {
                obs["resource_id"] for obs in observations
                if (obs["resource_id"].casefold() in lowered
                    or re.search(r"\b" + re.escape(obs["title"]) + r"\b", sentence, re.I))
            }
            if len(referenced) != 1 or not re.search(
                r"\b(?:confirmed|provided|lists?|recorded|states?|shows?|found|"
                r"retrieved|extracted|identified)\b", sentence, re.I,
            ) or re.search(r"\b(?:no|not|lacks?|missing|without)\b", sentence, re.I):
                continue
            source_id = next(iter(referenced))
            source_pairs = {
                (claim["subject"].casefold(), claim["predicate"].casefold())
                for obs in observations if obs["resource_id"] == source_id
                for claim in obs["claims"]
            }
            all_pairs = {
                (claim["subject"].casefold(), claim["predicate"].casefold())
                for obs in observations for claim in obs["claims"]
            }
            candidate_pairs = {
                (subject, predicate)
                for subject, predicate in all_pairs
                if subject in lowered and predicate in lowered
            }
            # The literal matcher cannot resolve grammatical ownership
            # across mixed clauses. Do not form the Cartesian product of
            # every entity/predicate in a multi-fact sentence and pretend
            # each pair was attributed to the named Source. The independent
            # semantic critic remains responsible for these complex claims.
            subjects = {subject for subject, _ in candidate_pairs}
            predicates = {predicate for _, predicate in candidate_pairs}
            if len(subjects) == 1 and len(predicates) == 1:
                for subject, predicate in candidate_pairs - source_pairs:
                    issues.append(
                        f"{key}_source_fact_mismatch:{source_id}:{subject}:{predicate}")
        # Numeric assertions may only reuse supplied observed/public values.
        def numeric(text: str) -> set[Decimal]:
            tokens = re.findall(r"(?<!\w)[+-]?\d+(?:,\d+)*(?:\.\d+)?", text)
            result = set()
            for token in tokens:
                try:
                    result.add(_number(token))
                except (InvalidOperation, ValueError, TypeError):
                    pass
            return result
        # Only labels of actually completed dependency results are metadata.
        # Do not mask arbitrary "Attempt N"/"subtask N" factual hallucinations.
        history_labels={
            label.casefold() for ref in public_refs
            for label in re.findall(r'\b(?:Attempt|subtask)\s+\d+\b',ref,re.I)
        }
        reference_pattern=r'\bAttempt\s+(\d+)\s*,?\s*subtask\s+(\d+)\b'
        known_pairs={pair for ref in public_refs
                     for pair in re.findall(reference_pattern,ref,re.I)}
        if set(re.findall(reference_pattern,content,re.I))-known_pairs:
            issues.append(key+"_unobserved_task_reference")
        # Source IDs are provenance handles, not numeric observations.
        masked = re.sub(r"source_\d+", "", content)
        masked = re.sub(
            r'\b(?:Attempt|subtask)\s+\d+\b',
            lambda match: '' if match[0].casefold() in history_labels else match[0],
            masked,flags=re.I)
        masked = re.sub(
            r"\b(?:step|operation)\s+(\d+)\b",
            lambda match: "" if 1 <= int(match[1]) <= len(fragment["operations"]) else match[0],
            masked, flags=re.I)
        if numeric(masked) - numeric(supported):
            issues.append(key + "_unsupported_numeric_claim")
    return not issues, sorted(set(issues))


def public_action(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ("action", "reasoning", "params")}


__all__ = [
    "TOOLS", "action_schema", "feedback_schema", "main_prompt", "feedback_prompt",
    "answer_string", "answers_equal", "validate_action", "validate_feedback",
    "public_action",
]
