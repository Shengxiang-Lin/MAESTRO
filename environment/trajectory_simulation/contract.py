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

TOOLS = (
    "GoogleSearchAction", "ExtractUrlContentAction", "ExecuteCodeAction",
    "ImageAnalysisAction", "AudioAnalysisAction", "VideoAnalysisAction",
)


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
                      "items": {"type": "string", "enum": list(TOOLS)}},
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


def main_prompt(payload: Mapping[str, Any]) -> str:
    public = payload["public_state"]
    guidance = payload["gold_guidance"]
    graph_index = {
        item["fragment_id"]: index
        for index, item in enumerate(guidance["fragment_graph"])
    }
    # Main needs the plan, not the controller's private execution handles.
    teacher_view = {
        "fragment_graph": [{
            "step": graph_index[item["fragment_id"]],
            "depends_on_steps": [graph_index[parent] for parent in item["depends_on"]],
            "steps": operation_steps(item["operations"]),
        } for item in guidance["fragment_graph"]],
        "ready": [{
            "task_index": index, "steps": operation_steps(item["operations"]),
            "source_types": sorted({resource["source_type"]
                                    for resource in item["required_resources"]}),
            "required_tools": item["suggested_tools"],
        } for index, item in enumerate(guidance["ready"])],
        "completed_step_count": len(guidance["completed_fragments"]),
        "terminal_operation": guidance["terminal_operation"],
        "may_complete": guidance["may_complete"],
        "completed_result_types": [{
            "answer": item["answer"], "answer_kind": item["answer_kind"],
        } for item in payload.get("completed_subtask_contracts", [])],
    }
    output = (
        "Return an O1 complete action with params.answer as a STRING. "
        "The last completed terminal subtask has already calculated the final result. "
        "Transport that verified public result without re-solving the original question."
        if guidance["may_complete"] else
        "Return an O1 delegate_task action with EXACTLY ONE task for EACH "
        "ready fragment, in the ORDER of ready[]. The tasks run in parallel. "
        "Use the private fragment operations and evidence descriptors to write "
        "specific actionable task_instruction/context and select appropriate tools. "
        "Do not invent, drop, merge or reorder fragments."
    )
    return (
        "You are producing ONE official Orchestra-o1 Main action at this point in "
        "the interaction, not a whole trajectory. The O1-visible state is limited "
        "to Question, completed Sub-Agent History and available tools. "
        "PRIVATE gold guidance is teacher supervision only; its fragment IDs, "
        "source selection and graph internals MUST NOT appear in the public action. "
        "Do not mention future observations or claim tasks are already completed. "
        "Use natural-language instructions for exactly the assigned operations, not the "
        "entire original question. '=' means equal, never greater than or equal. "
        "params is REQUIRED: tasks go inside params.tasks, never at the top level. "
        "Each task has task_instruction, context, model='model_1' and tools; no task IDs. "
        "Follow operation dependencies: separate branches remain separate until UNION or "
        "INTERSECT. FIND_ALL starts a new candidate set, not the previous branch's output. "
        "Do not apply Person/type or any other filters before their assigned operation. "
        "Never mention private guidance, graph steps, node IDs or opcode names in public text. "
        "Entity-set results contain public entity names as strings; they remain entity "
        "references that can be filtered or traversed, not untyped literal values. "
        "Each task.tools MUST include ALL its required_tools. They open the frozen evidence "
        "in its actual source modality; never substitute a search/code tool merely because "
        "the predicate looks structured. Additional registered tools are optional. "
        + output + " Follow required_output_schema. Return JSON only.\n"
        + json.dumps({
            "public_o1_state": public, "private_gold_guidance": teacher_view,
            "required_output_schema": action_schema(guidance),
        }, ensure_ascii=False, separators=(",", ":"))
    )


def feedback_prompt(payload: Mapping[str, Any]) -> str:
    """The answer is fixed by SymbolicExecutor; LLM writes narrative only."""
    return (
        "Simulate ONE clean Sub-Agent's concise textual report for the assigned "
        "composed reasoning subtask. Its correct answer and step outputs are supplied; "
        "DO NOT try to replace the answer or enumerate raw fact "
        "JSON in your response. Combine the supplied, source-scoped evidence with "
        "completed predecessor answers to describe the actual inference. "
        "Write summary as a brief, grounded interpretation of this SUBTASK answer; "
        "write trace_summary as a compressed account of evidence retrieval and the "
        "specified inference steps. Do not claim real tools were executed, "
        "invent sources, report future results, or solve unrelated tasks. "
        "Previously read source evidence may be reused without claiming a new retrieval. "
        "The authoritative answer is correct: never contradict it or infer missing values "
        "from an empty new-evidence packet when prior evidence is supplied. "
        "Do not apply extra original-question filters outside execution_steps. "
        "The per-operation results are authoritative. Respect their dependencies: a new "
        "FIND_ALL branch is independent, and UNION combines branches rather than applying "
        "all predicates to one candidate. Explain the supplied result, do not recompute it. "
        "Describe operations in natural language without private node IDs or opcode names. "
        "For attribution use the exact resource_id or title from its observation, never "
        "Reference numbers inferred from list order. Use exact observed years/values, not "
        "approximate century labels such as '1700s'. "
        "Relations may be multi-valued: a nationality/link to one object does not exclude "
        "another link supplied by a different source. Keep every fact's source attribution. "
        "Never mention ProgramIR, SymbolicExecutor, symbolic executor, private/gold guidance "
        "or the synthesis controller in public text. Report the Sub-Agent's inference, "
        "not how this training example was constructed. "
        "Return exactly JSON {\"summary\":str,\"trace_summary\":str}.\n"
        + json.dumps({
            "task_instruction": payload["task_instruction"],
            "dependency_results": payload["dependency_results"],
            "execution_steps": operation_steps(payload["fragment"]["operations"]),
            "verified_step_results": [{
                "step": index + 1, "answer": item["answer"],
            } for index, item in enumerate(payload["fragment"].get("operation_results", []))],
            "authoritative_subtask_answer": payload["fragment"]["expected_answer"],
            "assigned_source_observations": source_evidence(payload["fragment"]["observations"]),
            "previously_read_observations": source_evidence(
                payload["fragment"].get("known_observations", [])),
        }, ensure_ascii=False, separators=(",", ":"))
    )


def _hidden_identifier(text: str) -> bool:
    return bool(re.search(
        r"(?:fragment:\d+|fragment:terminal|subtask:|discover:|entity:[A-Za-z0-9_-]+"
        r"|\bProgramIR\b|\bSymbolicExecutor\b|\bsymbolic executor\b"
        r"|\bprivate (?:gold )?guidance\b|\bgold (?:fragment|operations|guidance)\b"
        r"|\bsynthesis controller\b)",
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
        if not isinstance(tools, list) or not tools or any(x not in TOOLS for x in tools):
            issues.append(f"task_{index}:invalid_tools")
        elif not set(fragment["suggested_tools"]).issubset(tools):
            issues.append(f"task_{index}:tools_not_grounded")
    if set(re.findall(r"source_\d+", visible_text)) - public_sources:
        issues.append("future_resource_id_in_public_action")
    return not issues, issues


def validate_feedback(value: Any, fragment: Mapping[str, Any]) -> tuple[bool, list[str]]:
    if not isinstance(value, Mapping):
        return False, ["feedback_not_object"]
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
            for subject, predicate in all_pairs:
                if (subject in lowered and predicate in lowered
                        and (subject, predicate) not in source_pairs):
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
        # Source IDs are provenance handles, not numeric observations.
        masked = re.sub(r"source_\d+", "", content)
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
